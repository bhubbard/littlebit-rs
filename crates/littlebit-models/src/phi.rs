//! Phi-3 and Phi-4 decoder model using [`PhiAttentionProjection`] (`PhiQKVSplitAttention`).

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, Var};
use rand_chacha::ChaCha8Rng;

use littlebit_core::{BiasMode, LayerShape, LittleBitConfig, TrainMode};

use crate::config::ModelConfig;
use crate::error::Result;
use crate::layer::{
    apply_activation, compute_attention, AttentionParams, Embedding, KvCache, LinearLayer,
    LinearLoadSpec, NormKind, RmsNorm, RotaryEmbedding,
};
use crate::model::{CausalLmOutput, HiddenStatesMode};
use crate::patch::{patch_linear_layer, PatchPolicy, PatchSummary};
use crate::phi_attn::PhiAttentionProjection;

#[derive(Debug, Clone)]
struct PhiAttention {
    qkv: PhiAttentionProjection,
    o_proj: LinearLayer,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl PhiAttention {
    fn forward(
        &self,
        hidden_states: &Tensor,
        cfg: &ModelConfig,
        rope: &RotaryEmbedding,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let pos_offset = cache.as_ref().map_or(0, |c| c.past_len(layer_idx));

        let (q_raw, k_raw, v_raw) = self.qkv.project(hidden_states, cfg)?;
        let q = q_raw
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k_raw
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v_raw
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = rope.apply(&q, pos_offset)?;
        let k = rope.apply(&k, pos_offset)?;

        let attn_out = compute_attention(
            &q,
            &k,
            &v,
            layer_idx,
            cache,
            AttentionParams {
                scale: 1.0 / (self.head_dim as f64).sqrt(),
                softcap: None,
                sliding_window: None,
            },
        )?;
        self.o_proj.forward(&attn_out)
    }
}

#[derive(Debug, Clone)]
struct PhiMlp {
    gate_up_proj: LinearLayer,
    down_proj: LinearLayer,
    intermediate_size: usize,
    hidden_act: String,
}

impl PhiMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up_states = self.gate_up_proj.forward(x)?;
        let last = up_states.rank() - 1;
        let gate = up_states.narrow(last, 0, self.intermediate_size)?;
        let up = up_states.narrow(last, self.intermediate_size, self.intermediate_size)?;
        let act = apply_activation(&gate, &self.hidden_act)?;
        self.down_proj.forward(&act.mul(&up)?)
    }
}

#[derive(Debug, Clone)]
struct PhiDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: PhiAttention,
    post_attention_layernorm: RmsNorm,
    mlp: PhiMlp,
}

/// Causal language model supporting Phi-3 and Phi-4 (`model_type = "phi3"`).
#[derive(Debug, Clone)]
pub struct PhiModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<PhiDecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<LinearLayer>,
    rope: RotaryEmbedding,
}

impl PhiModel {
    /// Load a Phi-3 / Phi-4 model from tensors (accepts either fused `qkv_proj` or split `q_proj/k_proj/v_proj`).
    ///
    /// # Errors
    /// Returns [`crate::ModelError::MissingWeight`] if required tensors are absent.
    pub fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let embed_tokens =
            Embedding::from_tensors("model.embed_tokens", tensors, None, spec.dtype, spec.device)?;
        let head_dim = config.effective_head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.effective_kv_heads();
        let inter = config.effective_intermediate_size();
        let attn_bias = BiasMode::from(config.attention_bias);
        let mlp_bias = BiasMode::from(config.mlp_bias);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let input_layernorm = RmsNorm::from_tensors(
                &format!("{p}.input_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;
            let qkv = PhiAttentionProjection::from_tensors(
                &format!("{p}.self_attn"),
                tensors,
                &config,
                spec.littlebit,
                spec,
            )?;
            let o_proj = LinearLayer::from_tensors(
                &format!("{p}.self_attn.o_proj"),
                tensors,
                LayerShape::new(num_heads * head_dim, config.hidden_size, attn_bias),
                LinearLoadSpec {
                    ratio_factor: 1.0,
                    ..spec
                },
            )?;
            let post_attention_layernorm = RmsNorm::from_tensors(
                &format!("{p}.post_attention_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;
            let gate_up_proj = LinearLayer::from_tensors(
                &format!("{p}.mlp.gate_up_proj"),
                tensors,
                LayerShape::new(config.hidden_size, 2 * inter, mlp_bias),
                LinearLoadSpec {
                    ratio_factor: 1.0,
                    ..spec
                },
            )?;
            let down_proj = LinearLayer::from_tensors(
                &format!("{p}.mlp.down_proj"),
                tensors,
                LayerShape::new(inter, config.hidden_size, mlp_bias),
                LinearLoadSpec {
                    ratio_factor: 1.0,
                    ..spec
                },
            )?;
            layers.push(PhiDecoderLayer {
                input_layernorm,
                self_attn: PhiAttention {
                    qkv,
                    o_proj,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                },
                post_attention_layernorm,
                mlp: PhiMlp {
                    gate_up_proj,
                    down_proj,
                    intermediate_size: inter,
                    hidden_act: config.hidden_act.clone(),
                },
            });
        }

        let norm = RmsNorm::from_tensors(
            "model.norm",
            tensors,
            config.rms_norm_eps,
            NormKind::Standard,
            spec.dtype,
            spec.device,
        )?;
        let lm_head = if !config.tie_word_embeddings && tensors.contains_key("lm_head.weight") {
            Some(LinearLayer::from_tensors(
                "lm_head",
                tensors,
                LayerShape::new(config.hidden_size, config.vocab_size, BiasMode::NoBias),
                LinearLoadSpec {
                    littlebit: None,
                    ratio_factor: 1.0,
                    ..spec
                },
            )?)
        } else {
            None
        };
        let rope = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings.max(256),
            config.rope_theta,
            spec.dtype,
            spec.device,
        )?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
        })
    }

    /// Initialize a random Phi-3 / Phi-4 model with fused `qkv_proj` for testing.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_random(
        config: ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let embed_tokens = Embedding::new_random(
            config.vocab_size,
            config.hidden_size,
            None,
            dtype,
            device,
            rng,
        )?;
        let head_dim = config.effective_head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.effective_kv_heads();
        let inter = config.effective_intermediate_size();
        let attn_bias = BiasMode::from(config.attention_bias);
        let mlp_bias = BiasMode::from(config.mlp_bias);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            let qkv = PhiAttentionProjection::new_random_fused(&config, dtype, device, rng)?;
            let o_proj = LinearLayer::new_random(
                LayerShape::new(num_heads * head_dim, config.hidden_size, attn_bias),
                dtype,
                device,
                rng,
            )?;
            let gate_up_proj = LinearLayer::new_random(
                LayerShape::new(config.hidden_size, 2 * inter, mlp_bias),
                dtype,
                device,
                rng,
            )?;
            let down_proj = LinearLayer::new_random(
                LayerShape::new(inter, config.hidden_size, mlp_bias),
                dtype,
                device,
                rng,
            )?;
            layers.push(PhiDecoderLayer {
                input_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                self_attn: PhiAttention {
                    qkv,
                    o_proj,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                },
                post_attention_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                mlp: PhiMlp {
                    gate_up_proj,
                    down_proj,
                    intermediate_size: inter,
                    hidden_act: config.hidden_act.clone(),
                },
            });
        }
        let norm = RmsNorm::new_init(
            config.hidden_size,
            config.rms_norm_eps,
            NormKind::Standard,
            dtype,
            device,
        )?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearLayer::new_random(
                LayerShape::new(config.hidden_size, config.vocab_size, BiasMode::NoBias),
                dtype,
                device,
                rng,
            )?)
        };
        let rope = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings.max(256),
            config.rope_theta,
            dtype,
            device,
        )?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
        })
    }

    /// Forward pass returning logits `(batch, seq_len, vocab_size)` and optional hidden states.
    ///
    /// # Errors
    /// Returns an error if layer forward passes fail.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        mut cache: Option<&mut KvCache>,
        hidden_mode: HiddenStatesMode,
    ) -> Result<CausalLmOutput> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let collect = hidden_mode == HiddenStatesMode::All;
        let mut all_hidden = if collect {
            Vec::with_capacity(self.layers.len() + 1)
        } else {
            Vec::new()
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            if collect {
                all_hidden.push(hidden.clone());
            }
            let residual = hidden.clone();
            let normed = layer.input_layernorm.forward(&hidden)?;
            let attn = layer.self_attn.forward(
                &normed,
                &self.config,
                &self.rope,
                idx,
                cache.as_deref_mut(),
            )?;
            let hidden_mid = residual.add(&attn)?;
            let normed_mid = layer.post_attention_layernorm.forward(&hidden_mid)?;
            let mlp_out = layer.mlp.forward(&normed_mid)?;
            hidden = hidden_mid.add(&mlp_out)?;
        }

        hidden = self.norm.forward(&hidden)?;
        if collect {
            all_hidden.push(hidden.clone());
        }
        let logits = match &self.lm_head {
            Some(head) => head.forward(&hidden)?,
            None => self.embed_tokens.forward_lm_head(&hidden)?,
        };
        Ok(CausalLmOutput {
            logits,
            hidden_states: all_hidden,
        })
    }

    /// Borrow attention projections for verification (e.g., verifying split QKV).
    #[must_use]
    pub fn layer_qkv_projection(&self, layer_idx: usize) -> Option<&PhiAttentionProjection> {
        self.layers.get(layer_idx).map(|l| &l.self_attn.qkv)
    }

    /// Split all fused `qkv_proj` attention projections into separate `q_proj`, `k_proj`, and `v_proj`
    /// dense layers (`PhiQKVSplitAttention` initialization step).
    ///
    /// # Errors
    /// Returns an error if splitting any layer fails.
    pub fn split_qkv(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.self_attn.qkv.split_qkv(&self.config)?;
        }
        Ok(())
    }

    pub(crate) fn apply_littlebit_patch(
        &mut self,
        lb_cfg: &LittleBitConfig,
        policy: &PatchPolicy,
        train_mode: TrainMode,
        base_seed: u64,
        summary: &mut PatchSummary,
    ) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("model.layers.{i}");
            layer.self_attn.qkv.apply_littlebit_patch(
                &format!("{p}.self_attn"),
                &self.config,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.self_attn.o_proj"),
                &mut layer.self_attn.o_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.mlp.gate_up_proj"),
                &mut layer.mlp.gate_up_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.mlp.down_proj"),
                &mut layer.mlp.down_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
        }
        if let Some(head) = &mut self.lm_head {
            patch_linear_layer(
                "lm_head", head, lb_cfg, policy, train_mode, base_seed, summary,
            )?;
        }
        Ok(())
    }

    pub(crate) fn convert_to_packed(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.self_attn.qkv.convert_to_packed()?;
            layer.self_attn.o_proj.convert_to_packed()?;
            layer.mlp.gate_up_proj.convert_to_packed()?;
            layer.mlp.down_proj.convert_to_packed()?;
        }
        if let Some(head) = &mut self.lm_head {
            head.convert_to_packed()?;
        }
        Ok(())
    }

    pub(crate) fn make_non_quantized_trainable(&mut self) -> Result<()> {
        self.embed_tokens.make_trainable()?;
        for layer in &mut self.layers {
            layer.input_layernorm.make_trainable()?;
            layer.post_attention_layernorm.make_trainable()?;
            layer.self_attn.o_proj.make_trainable()?;
            layer.mlp.gate_up_proj.make_trainable()?;
            layer.mlp.down_proj.make_trainable()?;
        }
        self.norm.make_trainable()?;
        if let Some(head) = &mut self.lm_head {
            head.make_trainable()?;
        }
        Ok(())
    }

    pub(crate) fn freeze_matching(&mut self, exclude: &[&str]) {
        let matches = |name: &str| exclude.iter().any(|k| name.contains(k));
        if matches("model.embed_tokens") {
            self.embed_tokens.freeze();
        }
        if matches("model.norm") {
            self.norm.freeze();
        }
        if let Some(head) = &mut self.lm_head {
            if matches("lm_head") {
                head.freeze();
            }
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("model.layers.{i}");
            if matches(&format!("{p}.input_layernorm")) {
                layer.input_layernorm.freeze();
            }
            if matches(&format!("{p}.post_attention_layernorm")) {
                layer.post_attention_layernorm.freeze();
            }
            if matches(&format!("{p}.self_attn")) {
                layer.self_attn.qkv.freeze();
                layer.self_attn.o_proj.freeze();
            }
            if matches(&format!("{p}.mlp")) {
                layer.mlp.gate_up_proj.freeze();
                layer.mlp.down_proj.freeze();
            }
        }
    }

    pub(crate) fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = self.embed_tokens.trainable_vars();
        for layer in &self.layers {
            vars.extend(layer.input_layernorm.trainable_vars());
            vars.extend(layer.self_attn.qkv.trainable_vars());
            vars.extend(layer.self_attn.o_proj.trainable_vars());
            vars.extend(layer.post_attention_layernorm.trainable_vars());
            vars.extend(layer.mlp.gate_up_proj.trainable_vars());
            vars.extend(layer.mlp.down_proj.trainable_vars());
        }
        vars.extend(self.norm.trainable_vars());
        if let Some(head) = &self.lm_head {
            vars.extend(head.trainable_vars());
        }
        vars
    }

    pub(crate) fn total_params(&self) -> usize {
        let mut count = self.embed_tokens.weight.elem_count() + self.norm.weight.elem_count();
        for layer in &self.layers {
            count += layer.input_layernorm.weight.elem_count()
                + layer.post_attention_layernorm.weight.elem_count()
                + layer.self_attn.qkv.param_count()
                + layer.self_attn.o_proj.param_count()
                + layer.mlp.gate_up_proj.param_count()
                + layer.mlp.down_proj.param_count();
        }
        if let Some(head) = &self.lm_head {
            count += head.param_count();
        }
        count
    }

    pub(crate) fn pack_state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        self.embed_tokens
            .pack_state_dict("model.embed_tokens", &mut out);
        for (i, layer) in self.layers.iter().enumerate() {
            let p = format!("model.layers.{i}");
            layer
                .input_layernorm
                .pack_state_dict(&format!("{p}.input_layernorm"), &mut out);
            layer
                .self_attn
                .qkv
                .pack_state_dict(&format!("{p}.self_attn"), &mut out)?;
            out.extend(
                layer
                    .self_attn
                    .o_proj
                    .pack_state_dict(&format!("{p}.self_attn.o_proj"))?,
            );
            layer
                .post_attention_layernorm
                .pack_state_dict(&format!("{p}.post_attention_layernorm"), &mut out);
            out.extend(
                layer
                    .mlp
                    .gate_up_proj
                    .pack_state_dict(&format!("{p}.mlp.gate_up_proj"))?,
            );
            out.extend(
                layer
                    .mlp
                    .down_proj
                    .pack_state_dict(&format!("{p}.mlp.down_proj"))?,
            );
        }
        self.norm.pack_state_dict("model.norm", &mut out);
        if let Some(head) = &self.lm_head {
            out.extend(head.pack_state_dict("lm_head")?);
        }
        Ok(out)
    }

    pub(crate) fn config(&self) -> &ModelConfig {
        &self.config
    }
}
