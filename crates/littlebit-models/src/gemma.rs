//! Gemma 2 and Gemma 3 decoder architectures.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, Var};
use rand_chacha::ChaCha8Rng;

use littlebit_core::{BiasMode, LayerShape, LittleBitConfig, TrainMode};

use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::Result;
use crate::layer::{
    apply_activation, compute_attention, AttentionParams, Embedding, KvCache, LinearLayer,
    LinearLoadSpec, NormKind, RmsNorm, RotaryEmbedding,
};
use crate::model::{CausalLmOutput, HiddenStatesMode};
use crate::patch::{patch_linear_layer, PatchPolicy, PatchSummary};

#[derive(Debug, Clone)]
struct GemmaAttention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    softcap: Option<f64>,
    sliding_window: Option<usize>,
    is_local: bool,
}

impl GemmaAttention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden_states: &Tensor,
        rope_global: &RotaryEmbedding,
        rope_local: Option<&RotaryEmbedding>,
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

        let rope = if self.is_local {
            rope_local.unwrap_or(rope_global)
        } else {
            rope_global
        };
        let q = rope.apply(&q, pos_offset)?;
        let k = rope.apply(&k, pos_offset)?;

        let attn_out = compute_attention(
            &q,
            &k,
            &v,
            layer_idx,
            cache,
            AttentionParams {
                scale: self.scale,
                softcap: self.softcap,
                sliding_window: self.sliding_window,
            },
        )?;
        self.o_proj.forward(&attn_out)
    }
}

#[derive(Debug, Clone)]
struct GemmaMlp {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
    hidden_act: String,
}

impl GemmaMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = apply_activation(&self.gate_proj.forward(x)?, &self.hidden_act)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }
}

#[derive(Debug, Clone)]
struct GemmaDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: GemmaAttention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: GemmaMlp,
    post_feedforward_layernorm: RmsNorm,
}

/// Causal language model supporting Gemma 2 and Gemma 3.
#[derive(Debug, Clone)]
pub struct GemmaModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    layers: Vec<GemmaDecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<LinearLayer>,
    rope_global: RotaryEmbedding,
    rope_local: Option<RotaryEmbedding>,
}

fn layer_window_and_local(
    cfg: &ModelConfig,
    arch: ModelArchitecture,
    layer_idx: usize,
) -> (Option<usize>, bool) {
    match arch {
        ModelArchitecture::Gemma2 => {
            let is_sliding = layer_idx.is_multiple_of(2);
            (if is_sliding { cfg.sliding_window } else { None }, false)
        }
        ModelArchitecture::Gemma3 => {
            let pattern = cfg.sliding_window_pattern.unwrap_or(6).max(1);
            let is_local = !(layer_idx + 1).is_multiple_of(pattern);
            (if is_local { cfg.sliding_window } else { None }, is_local)
        }
        _ => (cfg.sliding_window, false),
    }
}

impl GemmaModel {
    /// Load Gemma 2 / Gemma 3 weights from a state dictionary.
    ///
    /// # Errors
    /// Returns [`crate::ModelError::MissingWeight`] if expected tensors are absent.
    pub fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let arch = config.detect_architecture()?;
        let embed_scale = Some((config.hidden_size as f64).sqrt());
        let embed_tokens = Embedding::from_tensors(
            "model.embed_tokens",
            tensors,
            embed_scale,
            spec.dtype,
            spec.device,
        )?;
        let head_dim = config.effective_head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.effective_kv_heads();
        let inter = config.effective_intermediate_size();
        let attn_bias = BiasMode::from(config.attention_bias);
        let mlp_bias = BiasMode::from(config.mlp_bias);
        let kv_factor = spec.littlebit.map_or(1.0, |c| c.kv_factor);
        let scale = 1.0
            / config
                .query_pre_attn_scalar
                .unwrap_or(head_dim as f64)
                .sqrt();

        let kv_spec = LinearLoadSpec {
            ratio_factor: kv_factor,
            ..spec
        };
        let unit_spec = LinearLoadSpec {
            ratio_factor: 1.0,
            ..spec
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let (sliding_window, is_local) = layer_window_and_local(&config, arch, i);

            let input_layernorm = RmsNorm::from_tensors(
                &format!("{p}.input_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::PlusOne,
                spec.dtype,
                spec.device,
            )?;
            let q_proj = LinearLayer::from_tensors(
                &format!("{p}.self_attn.q_proj"),
                tensors,
                LayerShape::new(config.hidden_size, num_heads * head_dim, attn_bias),
                unit_spec,
            )?;
            let k_proj = LinearLayer::from_tensors(
                &format!("{p}.self_attn.k_proj"),
                tensors,
                LayerShape::new(config.hidden_size, num_kv_heads * head_dim, attn_bias),
                kv_spec,
            )?;
            let v_proj = LinearLayer::from_tensors(
                &format!("{p}.self_attn.v_proj"),
                tensors,
                LayerShape::new(config.hidden_size, num_kv_heads * head_dim, attn_bias),
                kv_spec,
            )?;
            let o_proj = LinearLayer::from_tensors(
                &format!("{p}.self_attn.o_proj"),
                tensors,
                LayerShape::new(num_heads * head_dim, config.hidden_size, attn_bias),
                unit_spec,
            )?;
            let q_norm = if tensors.contains_key(&format!("{p}.self_attn.q_norm.weight")) {
                Some(RmsNorm::from_tensors(
                    &format!("{p}.self_attn.q_norm"),
                    tensors,
                    config.rms_norm_eps,
                    NormKind::PlusOne,
                    spec.dtype,
                    spec.device,
                )?)
            } else {
                None
            };
            let k_norm = if tensors.contains_key(&format!("{p}.self_attn.k_norm.weight")) {
                Some(RmsNorm::from_tensors(
                    &format!("{p}.self_attn.k_norm"),
                    tensors,
                    config.rms_norm_eps,
                    NormKind::PlusOne,
                    spec.dtype,
                    spec.device,
                )?)
            } else {
                None
            };
            let post_attention_layernorm = RmsNorm::from_tensors(
                &format!("{p}.post_attention_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::PlusOne,
                spec.dtype,
                spec.device,
            )?;
            let pre_feedforward_layernorm = RmsNorm::from_tensors(
                &format!("{p}.pre_feedforward_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::PlusOne,
                spec.dtype,
                spec.device,
            )?;
            let gate_proj = LinearLayer::from_tensors(
                &format!("{p}.mlp.gate_proj"),
                tensors,
                LayerShape::new(config.hidden_size, inter, mlp_bias),
                unit_spec,
            )?;
            let up_proj = LinearLayer::from_tensors(
                &format!("{p}.mlp.up_proj"),
                tensors,
                LayerShape::new(config.hidden_size, inter, mlp_bias),
                unit_spec,
            )?;
            let down_proj = LinearLayer::from_tensors(
                &format!("{p}.mlp.down_proj"),
                tensors,
                LayerShape::new(inter, config.hidden_size, mlp_bias),
                unit_spec,
            )?;
            let post_feedforward_layernorm = RmsNorm::from_tensors(
                &format!("{p}.post_feedforward_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::PlusOne,
                spec.dtype,
                spec.device,
            )?;

            layers.push(GemmaDecoderLayer {
                input_layernorm,
                self_attn: GemmaAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    softcap: config.attn_logit_softcapping,
                    sliding_window,
                    is_local,
                },
                post_attention_layernorm,
                pre_feedforward_layernorm,
                mlp: GemmaMlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                    hidden_act: config.hidden_act.clone(),
                },
                post_feedforward_layernorm,
            });
        }

        let norm = RmsNorm::from_tensors(
            "model.norm",
            tensors,
            config.rms_norm_eps,
            NormKind::PlusOne,
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

        let rope_global = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings.max(256),
            config.rope_theta,
            spec.dtype,
            spec.device,
        )?;
        let rope_local = match config.rope_local_base_freq {
            Some(theta) => Some(RotaryEmbedding::new(
                head_dim,
                config.max_position_embeddings.max(256),
                theta,
                spec.dtype,
                spec.device,
            )?),
            None => None,
        };

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope_global,
            rope_local,
        })
    }

    /// Initialize a random Gemma 2 / Gemma 3 model for testing.
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
        let embed_scale = Some((config.hidden_size as f64).sqrt());
        let embed_tokens = Embedding::new_random(
            config.vocab_size,
            config.hidden_size,
            embed_scale,
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
        let scale = 1.0
            / config
                .query_pre_attn_scalar
                .unwrap_or(head_dim as f64)
                .sqrt();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let (sliding_window, is_local) = layer_window_and_local(&config, arch, i);
            let (q_norm, k_norm) = if arch == ModelArchitecture::Gemma3 {
                (
                    Some(RmsNorm::new_init(
                        head_dim,
                        config.rms_norm_eps,
                        NormKind::PlusOne,
                        dtype,
                        device,
                    )?),
                    Some(RmsNorm::new_init(
                        head_dim,
                        config.rms_norm_eps,
                        NormKind::PlusOne,
                        dtype,
                        device,
                    )?),
                )
            } else {
                (None, None)
            };
            layers.push(GemmaDecoderLayer {
                input_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::PlusOne,
                    dtype,
                    device,
                )?,
                self_attn: GemmaAttention {
                    q_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, num_heads * head_dim, attn_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    k_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, num_kv_heads * head_dim, attn_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    v_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, num_kv_heads * head_dim, attn_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    o_proj: LinearLayer::new_random(
                        LayerShape::new(num_heads * head_dim, config.hidden_size, attn_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    q_norm,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    softcap: config.attn_logit_softcapping,
                    sliding_window,
                    is_local,
                },
                post_attention_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::PlusOne,
                    dtype,
                    device,
                )?,
                pre_feedforward_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::PlusOne,
                    dtype,
                    device,
                )?,
                mlp: GemmaMlp {
                    gate_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, inter, mlp_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    up_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, inter, mlp_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    down_proj: LinearLayer::new_random(
                        LayerShape::new(inter, config.hidden_size, mlp_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    hidden_act: config.hidden_act.clone(),
                },
                post_feedforward_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::PlusOne,
                    dtype,
                    device,
                )?,
            });
        }

        let norm = RmsNorm::new_init(
            config.hidden_size,
            config.rms_norm_eps,
            NormKind::PlusOne,
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
        let rope_global = RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings.max(256),
            config.rope_theta,
            dtype,
            device,
        )?;
        let rope_local = match config.rope_local_base_freq {
            Some(theta) => Some(RotaryEmbedding::new(
                head_dim,
                config.max_position_embeddings.max(256),
                theta,
                dtype,
                device,
            )?),
            None => None,
        };
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope_global,
            rope_local,
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
                &self.rope_global,
                self.rope_local.as_ref(),
                idx,
                cache.as_deref_mut(),
            )?;
            let attn_normed = layer.post_attention_layernorm.forward(&attn)?;
            let h_mid = residual.add(&attn_normed)?;

            let pre_ff = layer.pre_feedforward_layernorm.forward(&h_mid)?;
            let ff_out = layer.mlp.forward(&pre_ff)?;
            let ff_normed = layer.post_feedforward_layernorm.forward(&ff_out)?;
            hidden = h_mid.add(&ff_normed)?;
        }

        hidden = self.norm.forward(&hidden)?;
        if collect {
            all_hidden.push(hidden.clone());
        }
        let mut logits = match &self.lm_head {
            Some(head) => head.forward(&hidden)?,
            None => self.embed_tokens.forward_lm_head(&hidden)?,
        };
        if let Some(cap) = self.config.final_logit_softcapping {
            logits = ((logits / cap)?.tanh()? * cap)?;
        }
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
            layer.pre_feedforward_layernorm.make_trainable()?;
            layer.post_feedforward_layernorm.make_trainable()?;
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
            if matches(&format!("{p}.pre_feedforward_layernorm")) {
                layer.pre_feedforward_layernorm.freeze();
            }
            if matches(&format!("{p}.post_feedforward_layernorm")) {
                layer.post_feedforward_layernorm.freeze();
            }
            if matches(&format!("{p}.self_attn")) {
                layer.self_attn.q_proj.freeze();
                layer.self_attn.k_proj.freeze();
                layer.self_attn.v_proj.freeze();
                layer.self_attn.o_proj.freeze();
            }
            if matches(&format!("{p}.mlp")) {
                layer.mlp.gate_proj.freeze();
                layer.mlp.up_proj.freeze();
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
            vars.extend(layer.pre_feedforward_layernorm.trainable_vars());
            vars.extend(layer.mlp.gate_proj.trainable_vars());
            vars.extend(layer.mlp.up_proj.trainable_vars());
            vars.extend(layer.mlp.down_proj.trainable_vars());
            vars.extend(layer.post_feedforward_layernorm.trainable_vars());
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
                + layer.pre_feedforward_layernorm.weight.elem_count()
                + layer.post_feedforward_layernorm.weight.elem_count()
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
            layer
                .pre_feedforward_layernorm
                .pack_state_dict(&format!("{p}.pre_feedforward_layernorm"), &mut out);
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
            layer
                .post_feedforward_layernorm
                .pack_state_dict(&format!("{p}.post_feedforward_layernorm"), &mut out);
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
