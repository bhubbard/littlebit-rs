//! Unified decoder implementation for Llama, Llama 2/3, Qwen2/2.5/QwQ, and Qwen3.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, Var};
use rand_chacha::ChaCha8Rng;

use littlebit_core::{BiasMode, LayerShape, LittleBitConfig, TrainMode, WeightState};

use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::Result;
use crate::layer::{
    apply_activation, compute_attention, AttentionParams, Embedding, KvCache, LinearLayer,
    LinearLoadSpec, NormKind, RmsNorm, RotaryEmbedding,
};
use crate::model::{CausalLmOutput, HiddenStatesMode};
use crate::patch::{patch_linear_layer, PatchPolicy, PatchSummary};

#[derive(Debug, Clone)]
struct LlamaAttention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LlamaAttention {
    #[allow(clippy::too_many_arguments)]
    fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        cfg: &ModelConfig,
        arch: ModelArchitecture,
        littlebit: Option<&LittleBitConfig>,
        weight_state: WeightState,
        inference_mode: littlebit_core::InferenceMode,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.effective_head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.effective_kv_heads();
        let kv_factor = littlebit.map_or(1.0, |c| c.kv_factor);

        let bias_for = |name: &str, default_bias: bool| -> BiasMode {
            BiasMode::from(default_bias || tensors.contains_key(&format!("{prefix}.{name}.bias")))
        };

        let q_bias = bias_for(
            "q_proj",
            cfg.attention_bias || arch == ModelArchitecture::Qwen2,
        );
        let k_bias = bias_for(
            "k_proj",
            cfg.attention_bias || arch == ModelArchitecture::Qwen2,
        );
        let v_bias = bias_for(
            "v_proj",
            cfg.attention_bias || arch == ModelArchitecture::Qwen2,
        );
        let o_bias = bias_for("o_proj", cfg.attention_bias);

        let base_spec = LinearLoadSpec {
            littlebit,
            ratio_factor: 1.0,
            weight_state,
            inference_mode,
            dtype,
            device,
        };
        let kv_spec = LinearLoadSpec {
            ratio_factor: kv_factor,
            ..base_spec
        };

        let q_proj = LinearLayer::from_tensors(
            &format!("{prefix}.q_proj"),
            tensors,
            LayerShape::new(cfg.hidden_size, num_heads * head_dim, q_bias),
            base_spec,
        )?;
        let k_proj = LinearLayer::from_tensors(
            &format!("{prefix}.k_proj"),
            tensors,
            LayerShape::new(cfg.hidden_size, num_kv_heads * head_dim, k_bias),
            kv_spec,
        )?;
        let v_proj = LinearLayer::from_tensors(
            &format!("{prefix}.v_proj"),
            tensors,
            LayerShape::new(cfg.hidden_size, num_kv_heads * head_dim, v_bias),
            kv_spec,
        )?;
        let o_proj = LinearLayer::from_tensors(
            &format!("{prefix}.o_proj"),
            tensors,
            LayerShape::new(num_heads * head_dim, cfg.hidden_size, o_bias),
            base_spec,
        )?;

        let q_norm = if tensors.contains_key(&format!("{prefix}.q_norm.weight")) {
            Some(RmsNorm::from_tensors(
                &format!("{prefix}.q_norm"),
                tensors,
                cfg.rms_norm_eps,
                NormKind::Standard,
                dtype,
                device,
            )?)
        } else {
            None
        };
        let k_norm = if tensors.contains_key(&format!("{prefix}.k_norm.weight")) {
            Some(RmsNorm::from_tensors(
                &format!("{prefix}.k_norm"),
                tensors,
                cfg.rms_norm_eps,
                NormKind::Standard,
                dtype,
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn new_random(
        cfg: &ModelConfig,
        arch: ModelArchitecture,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let head_dim = cfg.effective_head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.effective_kv_heads();
        let qkv_bias = BiasMode::from(cfg.attention_bias || arch == ModelArchitecture::Qwen2);
        let o_bias = BiasMode::from(cfg.attention_bias);

        let q_proj = LinearLayer::new_random(
            LayerShape::new(cfg.hidden_size, num_heads * head_dim, qkv_bias),
            dtype,
            device,
            rng,
        )?;
        let k_proj = LinearLayer::new_random(
            LayerShape::new(cfg.hidden_size, num_kv_heads * head_dim, qkv_bias),
            dtype,
            device,
            rng,
        )?;
        let v_proj = LinearLayer::new_random(
            LayerShape::new(cfg.hidden_size, num_kv_heads * head_dim, qkv_bias),
            dtype,
            device,
            rng,
        )?;
        let o_proj = LinearLayer::new_random(
            LayerShape::new(num_heads * head_dim, cfg.hidden_size, o_bias),
            dtype,
            device,
            rng,
        )?;

        let (q_norm, k_norm) = if arch == ModelArchitecture::Qwen3 {
            (
                Some(RmsNorm::new_init(
                    head_dim,
                    cfg.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?),
                Some(RmsNorm::new_init(
                    head_dim,
                    cfg.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let pos_offset = cache.as_ref().map_or(0, |c| c.past_len(layer_idx));

        let mut q = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        if let Some(qn) = &self.q_norm {
            q = qn.forward(&q)?;
        }
        if let Some(kn) = &self.k_norm {
            k = kn.forward(&k)?;
        }

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
struct LlamaMlp {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
    hidden_act: String,
}

impl LlamaMlp {
    fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        cfg: &ModelConfig,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let inter = cfg.effective_intermediate_size();
        let bias = BiasMode::from(cfg.mlp_bias);
        Ok(Self {
            gate_proj: LinearLayer::from_tensors(
                &format!("{prefix}.gate_proj"),
                tensors,
                LayerShape::new(cfg.hidden_size, inter, bias),
                spec,
            )?,
            up_proj: LinearLayer::from_tensors(
                &format!("{prefix}.up_proj"),
                tensors,
                LayerShape::new(cfg.hidden_size, inter, bias),
                spec,
            )?,
            down_proj: LinearLayer::from_tensors(
                &format!("{prefix}.down_proj"),
                tensors,
                LayerShape::new(inter, cfg.hidden_size, bias),
                spec,
            )?,
            hidden_act: cfg.hidden_act.clone(),
        })
    }

    fn new_random(
        cfg: &ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let inter = cfg.effective_intermediate_size();
        let bias = BiasMode::from(cfg.mlp_bias);
        Ok(Self {
            gate_proj: LinearLayer::new_random(
                LayerShape::new(cfg.hidden_size, inter, bias),
                dtype,
                device,
                rng,
            )?,
            up_proj: LinearLayer::new_random(
                LayerShape::new(cfg.hidden_size, inter, bias),
                dtype,
                device,
                rng,
            )?,
            down_proj: LinearLayer::new_random(
                LayerShape::new(inter, cfg.hidden_size, bias),
                dtype,
                device,
                rng,
            )?,
            hidden_act: cfg.hidden_act.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = apply_activation(&self.gate_proj.forward(x)?, &self.hidden_act)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }
}

#[derive(Debug, Clone)]
struct LlamaDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: LlamaAttention,
    post_attention_layernorm: RmsNorm,
    mlp: LlamaMlp,
}

impl LlamaDecoderLayer {
    fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let normed = self.input_layernorm.forward(hidden_states)?;
        let attn = self.self_attn.forward(&normed, rope, layer_idx, cache)?;
        let hidden = residual.add(&attn)?;

        let residual = &hidden;
        let normed = self.post_attention_layernorm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(residual.add(&mlp_out)?)
    }
}

/// Causal language model supporting Llama / Llama 2/3, Qwen2 / Qwen2.5 / QwQ, and Qwen3.
#[derive(Debug, Clone)]
pub struct LlamaModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<LinearLayer>,
    rope: RotaryEmbedding,
}

impl LlamaModel {
    /// Construct a Llama/Qwen2/Qwen3 model from a loaded state dictionary.
    ///
    /// # Errors
    /// Returns [`crate::ModelError::MissingWeight`] or tensor shape errors if checkpoint tensors are missing.
    pub fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let arch = config.detect_architecture()?;
        let embed_tokens =
            Embedding::from_tensors("model.embed_tokens", tensors, None, spec.dtype, spec.device)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let input_layernorm = RmsNorm::from_tensors(
                &format!("{prefix}.input_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;
            let self_attn = LlamaAttention::from_tensors(
                &format!("{prefix}.self_attn"),
                tensors,
                &config,
                arch,
                spec.littlebit,
                spec.weight_state,
                spec.inference_mode,
                spec.dtype,
                spec.device,
            )?;
            let post_attention_layernorm = RmsNorm::from_tensors(
                &format!("{prefix}.post_attention_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;
            let mlp = LlamaMlp::from_tensors(
                &format!("{prefix}.mlp"),
                tensors,
                &config,
                LinearLoadSpec {
                    ratio_factor: 1.0,
                    ..spec
                },
            )?;
            layers.push(LlamaDecoderLayer {
                input_layernorm,
                self_attn,
                post_attention_layernorm,
                mlp,
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
            config.effective_head_dim(),
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

    /// Initialize a random Llama/Qwen2/Qwen3 model (for testing and synthetic benchmarks).
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_random(
        config: ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let arch = config.detect_architecture()?;
        let embed_tokens = Embedding::new_random(
            config.vocab_size,
            config.hidden_size,
            None,
            dtype,
            device,
            rng,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer {
                input_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                self_attn: LlamaAttention::new_random(&config, arch, dtype, device, rng)?,
                post_attention_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                mlp: LlamaMlp::new_random(&config, dtype, device, rng)?,
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
            config.effective_head_dim(),
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
    /// Returns an error if tensor shapes are invalid or layer forward passes fail.
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

        let last_idx = self.layers.len().saturating_sub(1);
        for (idx, layer) in self.layers.iter().enumerate() {
            if collect {
                all_hidden.push(hidden.clone());
            }
            let c = cache.as_deref_mut();
            hidden = layer.forward(&hidden, &self.rope, idx, c)?;
            if collect && idx == last_idx {
                // Final entry after `self.norm` is appended below to preserve `L + 1` count.
            }
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
            patch_linear_layer(
                &format!("{p}.self_attn.q_proj"),
                &mut layer.self_attn.q_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.self_attn.k_proj"),
                &mut layer.self_attn.k_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.self_attn.v_proj"),
                &mut layer.self_attn.v_proj,
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
                &format!("{p}.mlp.gate_proj"),
                &mut layer.mlp.gate_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.mlp.up_proj"),
                &mut layer.mlp.up_proj,
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
            layer.self_attn.q_proj.convert_to_packed()?;
            layer.self_attn.k_proj.convert_to_packed()?;
            layer.self_attn.v_proj.convert_to_packed()?;
            layer.self_attn.o_proj.convert_to_packed()?;
            layer.mlp.gate_proj.convert_to_packed()?;
            layer.mlp.up_proj.convert_to_packed()?;
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
            layer.self_attn.q_proj.make_trainable()?;
            layer.self_attn.k_proj.make_trainable()?;
            layer.self_attn.v_proj.make_trainable()?;
            layer.self_attn.o_proj.make_trainable()?;
            if let Some(qn) = &mut layer.self_attn.q_norm {
                qn.make_trainable()?;
            }
            if let Some(kn) = &mut layer.self_attn.k_norm {
                kn.make_trainable()?;
            }
            layer.mlp.gate_proj.make_trainable()?;
            layer.mlp.up_proj.make_trainable()?;
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
            if matches(&format!("{p}.self_attn.q_proj")) {
                layer.self_attn.q_proj.freeze();
            }
            if matches(&format!("{p}.self_attn.k_proj")) {
                layer.self_attn.k_proj.freeze();
            }
            if matches(&format!("{p}.self_attn.v_proj")) {
                layer.self_attn.v_proj.freeze();
            }
            if matches(&format!("{p}.self_attn.o_proj")) {
                layer.self_attn.o_proj.freeze();
            }
            if matches(&format!("{p}.mlp.gate_proj")) {
                layer.mlp.gate_proj.freeze();
            }
            if matches(&format!("{p}.mlp.up_proj")) {
                layer.mlp.up_proj.freeze();
            }
            if matches(&format!("{p}.mlp.down_proj")) {
                layer.mlp.down_proj.freeze();
            }
        }
    }

    pub(crate) fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = self.embed_tokens.trainable_vars();
        for layer in &self.layers {
            vars.extend(layer.input_layernorm.trainable_vars());
            vars.extend(layer.self_attn.q_proj.trainable_vars());
            vars.extend(layer.self_attn.k_proj.trainable_vars());
            vars.extend(layer.self_attn.v_proj.trainable_vars());
            vars.extend(layer.self_attn.o_proj.trainable_vars());
            if let Some(qn) = &layer.self_attn.q_norm {
                vars.extend(qn.trainable_vars());
            }
            if let Some(kn) = &layer.self_attn.k_norm {
                vars.extend(kn.trainable_vars());
            }
            vars.extend(layer.post_attention_layernorm.trainable_vars());
            vars.extend(layer.mlp.gate_proj.trainable_vars());
            vars.extend(layer.mlp.up_proj.trainable_vars());
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
                + layer.self_attn.q_proj.param_count()
                + layer.self_attn.k_proj.param_count()
                + layer.self_attn.v_proj.param_count()
                + layer.self_attn.o_proj.param_count()
                + layer
                    .self_attn
                    .q_norm
                    .as_ref()
                    .map_or(0, |n| n.weight.elem_count())
                + layer
                    .self_attn
                    .k_norm
                    .as_ref()
                    .map_or(0, |n| n.weight.elem_count())
                + layer.mlp.gate_proj.param_count()
                + layer.mlp.up_proj.param_count()
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
            out.extend(
                layer
                    .self_attn
                    .q_proj
                    .pack_state_dict(&format!("{p}.self_attn.q_proj"))?,
            );
            out.extend(
                layer
                    .self_attn
                    .k_proj
                    .pack_state_dict(&format!("{p}.self_attn.k_proj"))?,
            );
            out.extend(
                layer
                    .self_attn
                    .v_proj
                    .pack_state_dict(&format!("{p}.self_attn.v_proj"))?,
            );
            out.extend(
                layer
                    .self_attn
                    .o_proj
                    .pack_state_dict(&format!("{p}.self_attn.o_proj"))?,
            );
            if let Some(qn) = &layer.self_attn.q_norm {
                qn.pack_state_dict(&format!("{p}.self_attn.q_norm"), &mut out);
            }
            if let Some(kn) = &layer.self_attn.k_norm {
                kn.pack_state_dict(&format!("{p}.self_attn.k_norm"), &mut out);
            }
            layer
                .post_attention_layernorm
                .pack_state_dict(&format!("{p}.post_attention_layernorm"), &mut out);
            out.extend(
                layer
                    .mlp
                    .gate_proj
                    .pack_state_dict(&format!("{p}.mlp.gate_proj"))?,
            );
            out.extend(
                layer
                    .mlp
                    .up_proj
                    .pack_state_dict(&format!("{p}.mlp.up_proj"))?,
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
