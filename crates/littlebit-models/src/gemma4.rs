//! Gemma 4 Dense and Mixture-of-Experts (MoE) decoder architectures.
//!
//! Implements Hugging Face `Gemma4TextModel` / `Gemma4ForConditionalGeneration` text backbone:
//! - Hybrid per-layer `sliding_attention` and `full_attention` schedules
//! - Heterogeneous sliding vs. global head dimensions and KV head counts
//! - Optional `attention_k_eq_v` on global (`full_attention`) layers (sharing `k_proj` for `V` prior to `k_norm`/RoPE + scaleless `v_norm`)
//! - Standard (non-offset) `Gemma4RMSNorm` (`NormKind::Standard`) plus weightless `RmsNormNoScale` for `v_norm` and router input norm
//! - Unit attention scaling (`scale = 1.0`) and Proportional RoPE (`p-RoPE`)
//! - Optional dual-path FFN combining a shared dense MLP with sparse top-k routed MoE (`Gemma4Router` + `Gemma4Experts`)
//! - Per-layer residual `layer_scalar` multiplier

use std::collections::{BTreeMap, HashMap};

use candle_core::{DType, Device, Tensor, Var, D};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use littlebit_core::{BiasMode, LayerShape, LittleBitConfig, Param, TrainMode};

use crate::config::{ModelConfig, ResolvedRopeParams};
use crate::error::{ModelError, Result};
use crate::layer::{
    apply_activation, compute_attention, move_cast, AttentionParams, Embedding, KvCache,
    LinearLayer, LinearLoadSpec, NormKind, RmsNorm, RmsNormNoScale, RopeMode, RotaryEmbedding,
};
use crate::model::{CausalLmOutput, HiddenStatesMode};
use crate::patch::{patch_linear_layer, PatchPolicy, PatchSummary};

/// Sparse expert routing decision emitted by [`Gemma4Router`].
#[derive(Debug, Clone)]
pub struct RouterDispatch {
    /// Selected expert indices per flattened token `[num_tokens][top_k]`.
    pub top_k_indices: Vec<Vec<usize>>,
    /// Normalized & scaled routing weights per flattened token `[num_tokens][top_k]`.
    pub top_k_weights: Vec<Vec<f32>>,
}

/// Top-k softmax router for Gemma 4 Mixture-of-Experts blocks.
///
/// The router linear projection (`proj`) always remains unquantized (`LinearLayer::Dense`)
/// so top-k routing fidelity is preserved at sub-1-bit expert compression.
#[derive(Debug, Clone)]
pub struct Gemma4Router {
    norm: RmsNormNoScale,
    scale: Param,
    /// Unquantized router linear projection `(hidden_size -> num_experts)`.
    pub proj: LinearLayer,
    per_expert_scale: Param,
    scalar_root_size: f64,
    num_experts: usize,
    top_k: usize,
}

impl Gemma4Router {
    /// Load [`Gemma4Router`] weights under prefix `{layer_prefix}.router`.
    ///
    /// # Errors
    /// Returns [`ModelError::MissingWeight`] if required router parameters are absent.
    pub fn from_tensors(
        prefix: &str,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let num_experts = config.num_experts.ok_or_else(|| {
            ModelError::InvalidArgument("Gemma 4 MoE requires config.num_experts".to_string())
        })?;
        let top_k = config.top_k_experts.unwrap_or(8).min(num_experts).max(1);

        let scale_key = format!("{prefix}.scale");
        let scale_t = tensors
            .get(&scale_key)
            .ok_or_else(|| ModelError::MissingWeight(scale_key.clone()))?;
        let scale = Param::Frozen(move_cast(scale_t, spec.dtype, spec.device)?);

        let per_expert_key = format!("{prefix}.per_expert_scale");
        let per_expert_t = tensors
            .get(&per_expert_key)
            .ok_or_else(|| ModelError::MissingWeight(per_expert_key.clone()))?;
        let per_expert_scale = Param::Frozen(move_cast(per_expert_t, spec.dtype, spec.device)?);

        // Keep router projection dense even during quantized checkpoint loading.
        let router_spec = LinearLoadSpec {
            littlebit: None,
            ratio_factor: 1.0,
            ..spec
        };
        let proj = LinearLayer::from_tensors(
            &format!("{prefix}.proj"),
            tensors,
            LayerShape::new(config.hidden_size, num_experts, BiasMode::NoBias),
            router_spec,
        )?;

        Ok(Self {
            norm: RmsNormNoScale::new(config.rms_norm_eps),
            scale,
            proj,
            per_expert_scale,
            scalar_root_size: (config.hidden_size as f64).powf(-0.5),
            num_experts,
            top_k,
        })
    }

    /// Initialize random [`Gemma4Router`] weights for tests.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_random(
        config: &ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let num_experts = config.num_experts.unwrap_or(8).max(1);
        let top_k = config.top_k_experts.unwrap_or(2).min(num_experts).max(1);
        Ok(Self {
            norm: RmsNormNoScale::new(config.rms_norm_eps),
            scale: Param::Frozen(Tensor::ones(config.hidden_size, dtype, device)?),
            proj: LinearLayer::new_random(
                LayerShape::new(config.hidden_size, num_experts, BiasMode::NoBias),
                dtype,
                device,
                rng,
            )?,
            per_expert_scale: Param::Frozen(Tensor::ones(num_experts, dtype, device)?),
            scalar_root_size: (config.hidden_size as f64).powf(-0.5),
            num_experts,
            top_k,
        })
    }

    /// Total number of experts supported by this router.
    #[must_use]
    pub const fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Number of top-ranked experts activated per token.
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    /// Compute top-k expert assignments and scaled probabilities for `hidden_flat` `(num_tokens, hidden_size)`.
    ///
    /// # Errors
    /// Returns an error if tensor shapes are invalid or projection fails.
    pub fn forward(&self, hidden_flat: &Tensor) -> Result<RouterDispatch> {
        let normed = self.norm.forward(hidden_flat)?;
        let scale_raw = self.scale.as_tensor();
        let scale_t = if scale_raw.dtype() == normed.dtype() {
            scale_raw.clone()
        } else {
            scale_raw.to_dtype(normed.dtype())?
        };
        let scaled = (normed * self.scalar_root_size)?.broadcast_mul(&scale_t)?;
        let logits = self.proj.forward(&scaled)?.to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;

        let probs_vec = probs.to_vec2::<f32>()?;
        let per_expert = self
            .per_expert_scale
            .as_tensor()
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        let mut top_k_indices = Vec::with_capacity(probs_vec.len());
        let mut top_k_weights = Vec::with_capacity(probs_vec.len());

        for row in probs_vec {
            let mut ranked: Vec<(usize, f32)> = row.into_iter().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            ranked.truncate(self.top_k);

            let denom: f32 = ranked.iter().map(|&(_, p)| p).sum();
            let mut idxs = Vec::with_capacity(self.top_k);
            let mut weights = Vec::with_capacity(self.top_k);
            for (eid, p) in ranked {
                let norm_w = if denom > 0.0 {
                    p / denom
                } else {
                    1.0 / (self.top_k as f32)
                };
                let expert_scale = per_expert.get(eid).copied().unwrap_or(1.0);
                idxs.push(eid);
                weights.push(norm_w * expert_scale);
            }
            top_k_indices.push(idxs);
            top_k_weights.push(weights);
        }

        Ok(RouterDispatch {
            top_k_indices,
            top_k_weights,
        })
    }
}

/// Single routed SwiGLU / GeGLU expert MLP inside [`Gemma4Experts`].
#[derive(Debug, Clone)]
pub struct Gemma4Expert {
    /// Fused `[gate_proj; up_proj]` projection `(hidden_size -> 2 * moe_intermediate_size)`.
    pub gate_up_proj: LinearLayer,
    /// Down projection `(moe_intermediate_size -> hidden_size)`.
    pub down_proj: LinearLayer,
}

/// Collection of routed Mixture-of-Experts feed-forward networks with sparse dispatch.
#[derive(Debug, Clone)]
pub struct Gemma4Experts {
    /// Individual routed experts (`len == num_experts`).
    pub experts: Vec<Gemma4Expert>,
    intermediate_size: usize,
    hidden_act: String,
}

impl Gemma4Experts {
    /// Load experts from either per-expert keys (`{prefix}.{e}.gate_up_proj.*`) or
    /// Hugging Face fused 3D parameter tensors (`{prefix}.gate_up_proj` / `{prefix}.down_proj`).
    ///
    /// # Errors
    /// Returns an error if expert weights are missing or shapes do not match `config`.
    pub fn from_tensors(
        prefix: &str,
        config: &ModelConfig,
        tensors: &HashMap<String, Tensor>,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let num_experts = config.num_experts.ok_or_else(|| {
            ModelError::InvalidArgument("Gemma 4 MoE requires config.num_experts".to_string())
        })?;
        let inter = config
            .moe_intermediate_size
            .unwrap_or_else(|| config.effective_intermediate_size());

        let has_per_expert_keys = tensors
            .contains_key(&format!("{prefix}.0.gate_up_proj.U_packed"))
            || tensors.contains_key(&format!("{prefix}.0.gate_up_proj.U"))
            || tensors.contains_key(&format!("{prefix}.0.gate_up_proj.weight"));

        let mut experts = Vec::with_capacity(num_experts);
        if has_per_expert_keys {
            for e in 0..num_experts {
                let gate_up_proj = LinearLayer::from_tensors(
                    &format!("{prefix}.{e}.gate_up_proj"),
                    tensors,
                    LayerShape::new(config.hidden_size, 2 * inter, BiasMode::NoBias),
                    spec,
                )?;
                let down_proj = LinearLayer::from_tensors(
                    &format!("{prefix}.{e}.down_proj"),
                    tensors,
                    LayerShape::new(inter, config.hidden_size, BiasMode::NoBias),
                    spec,
                )?;
                experts.push(Gemma4Expert {
                    gate_up_proj,
                    down_proj,
                });
            }
        } else {
            let gu_key = format!("{prefix}.gate_up_proj");
            let dn_key = format!("{prefix}.down_proj");
            let gu_3d = tensors
                .get(&gu_key)
                .ok_or_else(|| ModelError::MissingWeight(gu_key.clone()))?;
            let dn_3d = tensors
                .get(&dn_key)
                .ok_or_else(|| ModelError::MissingWeight(dn_key.clone()))?;

            let gu_dims = gu_3d.dims();
            let dn_dims = dn_3d.dims();
            if gu_dims.len() != 3 || gu_dims[0] != num_experts {
                return Err(ModelError::InvalidArgument(format!(
                    "expected `{gu_key}` with shape [{num_experts}, 2*{inter}, {}], got {gu_dims:?}",
                    config.hidden_size
                )));
            }
            if dn_dims.len() != 3 || dn_dims[0] != num_experts {
                return Err(ModelError::InvalidArgument(format!(
                    "expected `{dn_key}` with shape [{num_experts}, {}, {inter}], got {dn_dims:?}",
                    config.hidden_size
                )));
            }

            for e in 0..num_experts {
                let w_gu = move_cast(&gu_3d.get(e)?.contiguous()?, spec.dtype, spec.device)?;
                let w_dn = move_cast(&dn_3d.get(e)?.contiguous()?, spec.dtype, spec.device)?;
                experts.push(Gemma4Expert {
                    gate_up_proj: LinearLayer::Dense {
                        weight: Param::Frozen(w_gu),
                        bias: None,
                    },
                    down_proj: LinearLayer::Dense {
                        weight: Param::Frozen(w_dn),
                        bias: None,
                    },
                });
            }
        }

        Ok(Self {
            experts,
            intermediate_size: inter,
            hidden_act: config.effective_hidden_act().to_string(),
        })
    }

    /// Initialize random [`Gemma4Experts`] for tests.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_random(
        config: &ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let num_experts = config.num_experts.unwrap_or(8).max(1);
        let inter = config
            .moe_intermediate_size
            .unwrap_or_else(|| config.effective_intermediate_size());
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(Gemma4Expert {
                gate_up_proj: LinearLayer::new_random(
                    LayerShape::new(config.hidden_size, 2 * inter, BiasMode::NoBias),
                    dtype,
                    device,
                    rng,
                )?,
                down_proj: LinearLayer::new_random(
                    LayerShape::new(inter, config.hidden_size, BiasMode::NoBias),
                    dtype,
                    device,
                    rng,
                )?,
            });
        }
        Ok(Self {
            experts,
            intermediate_size: inter,
            hidden_act: config.effective_hidden_act().to_string(),
        })
    }

    /// Execute ONLY the experts selected in `dispatch` across their assigned tokens.
    ///
    /// # Errors
    /// Returns an error if an assigned expert ID is out of bounds or tensor ops fail.
    pub fn forward(&self, hidden_flat: &Tensor, dispatch: &RouterDispatch) -> Result<Tensor> {
        let (num_tokens, _) = hidden_flat.dims2()?;
        if num_tokens == 0 {
            return Ok(Tensor::zeros_like(hidden_flat)?);
        }

        let mut buckets: BTreeMap<usize, Vec<(u32, f32)>> = BTreeMap::new();
        for (tok_idx, (eids, weights)) in dispatch
            .top_k_indices
            .iter()
            .zip(dispatch.top_k_weights.iter())
            .enumerate()
        {
            let tok_u32 = tok_idx as u32;
            for (&eid, &w) in eids.iter().zip(weights.iter()) {
                buckets.entry(eid).or_default().push((tok_u32, w));
            }
        }

        let device = hidden_flat.device();
        let dtype = hidden_flat.dtype();
        let mut acc = Tensor::zeros_like(hidden_flat)?;

        for (expert_id, entries) in buckets {
            let expert = self.experts.get(expert_id).ok_or_else(|| {
                ModelError::InvalidArgument(format!(
                    "routed expert index {expert_id} out of bounds (num_experts = {})",
                    self.experts.len()
                ))
            })?;
            let n_tok = entries.len();
            let tok_indices: Vec<u32> = entries.iter().map(|&(t, _)| t).collect();
            let weights: Vec<f32> = entries.iter().map(|&(_, w)| w).collect();

            let tok_t = Tensor::from_vec(tok_indices, n_tok, device)?;
            let w_col = Tensor::from_vec(weights, (n_tok, 1), device)?.to_dtype(dtype)?;

            let x_e = hidden_flat.index_select(&tok_t, 0)?;
            let gate_up = expert.gate_up_proj.forward(&x_e)?;
            let gate = gate_up.narrow(1, 0, self.intermediate_size)?;
            let up = gate_up.narrow(1, self.intermediate_size, self.intermediate_size)?;
            let act = apply_activation(&gate, &self.hidden_act)?.mul(&up)?;
            let y_e = expert.down_proj.forward(&act)?.broadcast_mul(&w_col)?;

            acc = acc.index_add(&tok_t, &y_e, 0)?;
        }

        Ok(acc)
    }
}

/// Complete Mixture-of-Experts sub-block inside a [`Gemma4Model`] decoder layer.
#[derive(Debug, Clone)]
pub struct Gemma4MoE {
    /// Post-normalization on the parallel dense shared MLP output.
    pub post_feedforward_layernorm_1: RmsNorm,
    /// Pre-normalization before routed expert execution.
    pub pre_feedforward_layernorm_2: RmsNorm,
    /// Unquantized top-k router.
    pub router: Gemma4Router,
    /// Sparse expert ensemble (`Dense`, `LittleBit`, or `PackedLittleBitLinear`).
    pub experts: Gemma4Experts,
    /// Post-normalization on the summed sparse expert output.
    pub post_feedforward_layernorm_2: RmsNorm,
}

#[derive(Debug, Clone)]
struct Gemma4Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: Option<LinearLayer>,
    o_proj: LinearLayer,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    v_norm: RmsNormNoScale,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softcap: Option<f64>,
    sliding_window: Option<usize>,
    is_sliding: bool,
}

impl Gemma4Attention {
    fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        // Use cumulative sequence position so sliding-window eviction does not stall RoPE positions.
        let pos_offset = cache.as_ref().map_or(0, |c| c.seq_pos(layer_idx));

        let q_raw = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k_raw = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // When `attention_k_eq_v` is enabled (`v_proj` is None), clone `k_raw` into `v_raw`
        // BEFORE `k_norm` and RoPE, matching Hugging Face `Gemma4TextAttention`.
        let v_raw = match &self.v_proj {
            Some(vp) => vp
                .forward(hidden_states)?
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?,
            None => k_raw.clone(),
        };

        let q = rope.apply(&self.q_norm.forward(&q_raw)?, pos_offset)?;
        let k = rope.apply(&self.k_norm.forward(&k_raw)?, pos_offset)?;
        let v = self.v_norm.forward(&v_raw)?;

        let attn_out = compute_attention(
            &q,
            &k,
            &v,
            layer_idx,
            cache,
            AttentionParams {
                scale: 1.0,
                softcap: self.softcap,
                sliding_window: self.sliding_window,
            },
        )?;
        self.o_proj.forward(&attn_out)
    }
}

#[derive(Debug, Clone)]
struct Gemma4Mlp {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
    hidden_act: String,
}

impl Gemma4Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = apply_activation(&self.gate_proj.forward(x)?, &self.hidden_act)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&gate.mul(&up)?)
    }
}

#[derive(Debug, Clone)]
struct Gemma4DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Gemma4Attention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    mlp: Gemma4Mlp,
    moe: Option<Gemma4MoE>,
    post_feedforward_layernorm: RmsNorm,
    layer_scalar: Param,
}

/// Causal language model supporting Gemma 4 Dense and Mixture-of-Experts checkpoints.
#[derive(Debug, Clone)]
pub struct Gemma4Model {
    config: ModelConfig,
    weight_prefix: String,
    embed_tokens: Embedding,
    layers: Vec<Gemma4DecoderLayer>,
    norm: RmsNorm,
    lm_head: Option<LinearLayer>,
    rope_full: RotaryEmbedding,
    rope_sliding: Option<RotaryEmbedding>,
}

fn detect_weight_prefix(tensors: &HashMap<String, Tensor>) -> String {
    if tensors.contains_key("model.language_model.embed_tokens.weight") {
        "model.language_model".to_string()
    } else if tensors.contains_key("language_model.model.embed_tokens.weight") {
        "language_model.model".to_string()
    } else {
        "model".to_string()
    }
}

fn build_rope(
    head_dim: usize,
    max_seq: usize,
    params: ResolvedRopeParams,
    dtype: DType,
    device: &Device,
) -> Result<RotaryEmbedding> {
    let mode = if params.is_proportional {
        RopeMode::Proportional
    } else {
        RopeMode::Default
    };
    RotaryEmbedding::new_with_config(
        head_dim,
        max_seq,
        params.rope_theta,
        params.partial_rotary_factor,
        params.scale_factor,
        mode,
        dtype,
        device,
    )
}

fn load_layer_scalar(
    key: &str,
    tensors: &HashMap<String, Tensor>,
    dtype: DType,
    device: &Device,
) -> Result<Param> {
    match tensors.get(key) {
        Some(t) => Ok(Param::Frozen(move_cast(t, dtype, device)?)),
        None => Ok(Param::Frozen(Tensor::ones((1,), dtype, device)?)),
    }
}

fn broadcast_scalar_mul(x: &Tensor, scalar: &Param) -> Result<Tensor> {
    let s = scalar.as_tensor();
    let s = if s.dtype() == x.dtype() {
        s.clone()
    } else {
        s.to_dtype(x.dtype())?
    };
    Ok(x.broadcast_mul(&s)?)
}

impl Gemma4Model {
    /// Load Gemma 4 Dense or MoE weights from a state dictionary.
    ///
    /// Automatically supports both standalone (`model.*`) and multimodal (`model.language_model.*` /
    /// `language_model.model.*`) weight prefixes, as well as fused 3D MoE experts and LittleBit packed experts.
    ///
    /// # Errors
    /// Returns [`ModelError::MissingWeight`] or [`ModelError::UnsupportedArchitecture`] on incompatible configs.
    pub fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        if config.num_kv_shared_layers.unwrap_or(0) > 0 {
            return Err(ModelError::UnsupportedArchitecture(
                "Gemma 4 KV-shared edge layers (num_kv_shared_layers > 0) are not supported"
                    .to_string(),
            ));
        }

        let weight_prefix = detect_weight_prefix(tensors);
        let embed_scale = Some((config.hidden_size as f64).sqrt());
        let embed_tokens = Embedding::from_tensors(
            &format!("{weight_prefix}.embed_tokens"),
            tensors,
            embed_scale,
            spec.dtype,
            spec.device,
        )?;

        let num_heads = config.num_attention_heads;
        let inter = config.effective_intermediate_size();
        let attn_bias = BiasMode::from(config.attention_bias);
        let mlp_bias = BiasMode::from(config.mlp_bias);
        let kv_factor = spec.littlebit.map_or(1.0, |c| c.kv_factor);
        let hidden_act = config.effective_hidden_act().to_string();
        let layer_types = config.effective_layer_types();
        let enable_moe = config.enable_moe_block.unwrap_or(false);

        let kv_spec = LinearLoadSpec {
            ratio_factor: kv_factor,
            ..spec
        };
        let unit_spec = LinearLoadSpec {
            ratio_factor: 1.0,
            ..spec
        };

        let mut any_sliding = false;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for (i, layer_type) in layer_types.iter().enumerate() {
            let p = format!("{weight_prefix}.layers.{i}");
            let is_sliding = layer_type == "sliding_attention";
            if is_sliding {
                any_sliding = true;
            }
            let head_dim = if is_sliding {
                config.effective_head_dim()
            } else {
                config.effective_global_head_dim()
            };
            let num_kv_heads = if is_sliding {
                config.effective_kv_heads()
            } else {
                config.effective_global_kv_heads()
            };
            let sliding_window = if is_sliding {
                config.sliding_window
            } else {
                None
            };
            let use_k_eq_v = !is_sliding && config.attention_k_eq_v.unwrap_or(false);

            let input_layernorm = RmsNorm::from_tensors(
                &format!("{p}.input_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
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
            let v_proj = if use_k_eq_v {
                None
            } else {
                Some(LinearLayer::from_tensors(
                    &format!("{p}.self_attn.v_proj"),
                    tensors,
                    LayerShape::new(config.hidden_size, num_kv_heads * head_dim, attn_bias),
                    kv_spec,
                )?)
            };
            let o_proj = LinearLayer::from_tensors(
                &format!("{p}.self_attn.o_proj"),
                tensors,
                LayerShape::new(num_heads * head_dim, config.hidden_size, attn_bias),
                unit_spec,
            )?;
            let q_norm = RmsNorm::from_tensors(
                &format!("{p}.self_attn.q_norm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;
            let k_norm = RmsNorm::from_tensors(
                &format!("{p}.self_attn.k_norm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;
            let v_norm = RmsNormNoScale::new(config.rms_norm_eps);

            let post_attention_layernorm = RmsNorm::from_tensors(
                &format!("{p}.post_attention_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;
            let pre_feedforward_layernorm = RmsNorm::from_tensors(
                &format!("{p}.pre_feedforward_layernorm"),
                tensors,
                config.rms_norm_eps,
                NormKind::Standard,
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
                NormKind::Standard,
                spec.dtype,
                spec.device,
            )?;

            let moe = if enable_moe {
                Some(Gemma4MoE {
                    post_feedforward_layernorm_1: RmsNorm::from_tensors(
                        &format!("{p}.post_feedforward_layernorm_1"),
                        tensors,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        spec.dtype,
                        spec.device,
                    )?,
                    pre_feedforward_layernorm_2: RmsNorm::from_tensors(
                        &format!("{p}.pre_feedforward_layernorm_2"),
                        tensors,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        spec.dtype,
                        spec.device,
                    )?,
                    router: Gemma4Router::from_tensors(
                        &format!("{p}.router"),
                        &config,
                        tensors,
                        unit_spec,
                    )?,
                    experts: Gemma4Experts::from_tensors(
                        &format!("{p}.experts"),
                        &config,
                        tensors,
                        unit_spec,
                    )?,
                    post_feedforward_layernorm_2: RmsNorm::from_tensors(
                        &format!("{p}.post_feedforward_layernorm_2"),
                        tensors,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        spec.dtype,
                        spec.device,
                    )?,
                })
            } else {
                None
            };

            let layer_scalar = load_layer_scalar(
                &format!("{p}.layer_scalar"),
                tensors,
                spec.dtype,
                spec.device,
            )?;

            layers.push(Gemma4DecoderLayer {
                input_layernorm,
                self_attn: Gemma4Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    v_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    softcap: config.attn_logit_softcapping,
                    sliding_window,
                    is_sliding,
                },
                post_attention_layernorm,
                pre_feedforward_layernorm,
                mlp: Gemma4Mlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                    hidden_act: hidden_act.clone(),
                },
                moe,
                post_feedforward_layernorm,
                layer_scalar,
            });
        }

        let norm = RmsNorm::from_tensors(
            &format!("{weight_prefix}.norm"),
            tensors,
            config.rms_norm_eps,
            NormKind::Standard,
            spec.dtype,
            spec.device,
        )?;
        let lm_head_key = if tensors.contains_key("lm_head.weight") {
            Some("lm_head")
        } else if tensors.contains_key("language_model.lm_head.weight") {
            Some("language_model.lm_head")
        } else {
            None
        };
        let lm_head = match lm_head_key {
            Some(prefix) if !config.tie_word_embeddings => Some(LinearLayer::from_tensors(
                prefix,
                tensors,
                LayerShape::new(config.hidden_size, config.vocab_size, BiasMode::NoBias),
                LinearLoadSpec {
                    littlebit: None,
                    ratio_factor: 1.0,
                    ..spec
                },
            )?),
            _ => None,
        };

        let max_seq = config.max_position_embeddings.max(256);
        let rope_full = build_rope(
            config.effective_global_head_dim(),
            max_seq,
            config.gemma4_rope_params("full_attention"),
            spec.dtype,
            spec.device,
        )?;
        let rope_sliding = if any_sliding {
            Some(build_rope(
                config.effective_head_dim(),
                max_seq,
                config.gemma4_rope_params("sliding_attention"),
                spec.dtype,
                spec.device,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            weight_prefix,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope_full,
            rope_sliding,
        })
    }

    /// Initialize a random Gemma 4 Dense or MoE model for testing and benchmarking.
    ///
    /// # Errors
    /// Returns an error if tensor allocation or configuration validation fails.
    pub fn new_random(
        config: ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let embed_scale = Some((config.hidden_size as f64).sqrt());
        let embed_tokens = Embedding::new_random(
            config.vocab_size,
            config.hidden_size,
            embed_scale,
            dtype,
            device,
            rng,
        )?;

        let num_heads = config.num_attention_heads;
        let inter = config.effective_intermediate_size();
        let attn_bias = BiasMode::from(config.attention_bias);
        let mlp_bias = BiasMode::from(config.mlp_bias);
        let hidden_act = config.effective_hidden_act().to_string();
        let layer_types = config.effective_layer_types();
        let enable_moe = config.enable_moe_block.unwrap_or(false);

        let scalar_dist = Normal::new(1.0f32, 0.05f32).map_err(|err| {
            ModelError::InvalidArgument(format!("failed to construct scalar normal dist: {err}"))
        })?;

        let mut any_sliding = false;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for layer_type in &layer_types {
            let is_sliding = layer_type == "sliding_attention";
            if is_sliding {
                any_sliding = true;
            }
            let head_dim = if is_sliding {
                config.effective_head_dim()
            } else {
                config.effective_global_head_dim()
            };
            let num_kv_heads = if is_sliding {
                config.effective_kv_heads()
            } else {
                config.effective_global_kv_heads()
            };
            let sliding_window = if is_sliding {
                config.sliding_window
            } else {
                None
            };
            let use_k_eq_v = !is_sliding && config.attention_k_eq_v.unwrap_or(false);

            let v_proj = if use_k_eq_v {
                None
            } else {
                Some(LinearLayer::new_random(
                    LayerShape::new(config.hidden_size, num_kv_heads * head_dim, attn_bias),
                    dtype,
                    device,
                    rng,
                )?)
            };

            let moe = if enable_moe {
                Some(Gemma4MoE {
                    post_feedforward_layernorm_1: RmsNorm::new_init(
                        config.hidden_size,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        dtype,
                        device,
                    )?,
                    pre_feedforward_layernorm_2: RmsNorm::new_init(
                        config.hidden_size,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        dtype,
                        device,
                    )?,
                    router: Gemma4Router::new_random(&config, dtype, device, rng)?,
                    experts: Gemma4Experts::new_random(&config, dtype, device, rng)?,
                    post_feedforward_layernorm_2: RmsNorm::new_init(
                        config.hidden_size,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        dtype,
                        device,
                    )?,
                })
            } else {
                None
            };

            let scalar_val = scalar_dist.sample(rng);
            let layer_scalar =
                Param::Frozen(Tensor::from_slice(&[scalar_val], (1,), device)?.to_dtype(dtype)?);

            layers.push(Gemma4DecoderLayer {
                input_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                self_attn: Gemma4Attention {
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
                    v_proj,
                    o_proj: LinearLayer::new_random(
                        LayerShape::new(num_heads * head_dim, config.hidden_size, attn_bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    q_norm: RmsNorm::new_init(
                        head_dim,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        dtype,
                        device,
                    )?,
                    k_norm: RmsNorm::new_init(
                        head_dim,
                        config.rms_norm_eps,
                        NormKind::Standard,
                        dtype,
                        device,
                    )?,
                    v_norm: RmsNormNoScale::new(config.rms_norm_eps),
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    softcap: config.attn_logit_softcapping,
                    sliding_window,
                    is_sliding,
                },
                post_attention_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                pre_feedforward_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                mlp: Gemma4Mlp {
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
                    hidden_act: hidden_act.clone(),
                },
                moe,
                post_feedforward_layernorm: RmsNorm::new_init(
                    config.hidden_size,
                    config.rms_norm_eps,
                    NormKind::Standard,
                    dtype,
                    device,
                )?,
                layer_scalar,
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

        let max_seq = config.max_position_embeddings.max(256);
        let rope_full = build_rope(
            config.effective_global_head_dim(),
            max_seq,
            config.gemma4_rope_params("full_attention"),
            dtype,
            device,
        )?;
        let rope_sliding = if any_sliding {
            Some(build_rope(
                config.effective_head_dim(),
                max_seq,
                config.gemma4_rope_params("sliding_attention"),
                dtype,
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            weight_prefix: "model".to_string(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope_full,
            rope_sliding,
        })
    }

    /// Borrow a decoder layer's MoE block (useful for testing router / expert quantization states).
    #[must_use]
    pub fn layer_moe(&self, layer_idx: usize) -> Option<&Gemma4MoE> {
        self.layers.get(layer_idx).and_then(|l| l.moe.as_ref())
    }

    /// Mutably borrow a decoder layer's MoE block (useful in unit tests).
    #[must_use]
    pub fn layer_moe_mut(&mut self, layer_idx: usize) -> Option<&mut Gemma4MoE> {
        self.layers.get_mut(layer_idx).and_then(|l| l.moe.as_mut())
    }

    /// Execute causal forward pass over `input_ids` `(batch, seq_len)`.
    ///
    /// # Errors
    /// Returns an error if layer computation fails.
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
            let rope = if layer.self_attn.is_sliding {
                self.rope_sliding.as_ref().unwrap_or(&self.rope_full)
            } else {
                &self.rope_full
            };
            let attn = layer
                .self_attn
                .forward(&normed, rope, idx, cache.as_deref_mut())?;
            let attn_normed = layer.post_attention_layernorm.forward(&attn)?;
            let h_mid = residual.add(&attn_normed)?;

            let mut next_hidden = match &layer.moe {
                None => {
                    let pre_ff = layer.pre_feedforward_layernorm.forward(&h_mid)?;
                    let ff_out = layer.mlp.forward(&pre_ff)?;
                    let ff_normed = layer.post_feedforward_layernorm.forward(&ff_out)?;
                    h_mid.add(&ff_normed)?
                }
                Some(moe) => {
                    let pre_ff = layer.pre_feedforward_layernorm.forward(&h_mid)?;
                    let mlp_out = layer.mlp.forward(&pre_ff)?;
                    let h1 = moe.post_feedforward_layernorm_1.forward(&mlp_out)?;

                    let (b, s, d) = h_mid.dims3()?;
                    let flat = h_mid.reshape((b * s, d))?;
                    let dispatch = moe.router.forward(&flat)?;
                    let h2_in = moe.pre_feedforward_layernorm_2.forward(&flat)?;
                    let h2_experts = moe.experts.forward(&h2_in, &dispatch)?.reshape((b, s, d))?;
                    let h2 = moe.post_feedforward_layernorm_2.forward(&h2_experts)?;

                    let ff_normed = layer.post_feedforward_layernorm.forward(&h1.add(&h2)?)?;
                    h_mid.add(&ff_normed)?
                }
            };

            next_hidden = broadcast_scalar_mul(&next_hidden, &layer.layer_scalar)?;
            hidden = next_hidden;
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
            let p = format!("{}.layers.{i}", self.weight_prefix);
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
            if let Some(vp) = &mut layer.self_attn.v_proj {
                patch_linear_layer(
                    &format!("{p}.self_attn.v_proj"),
                    vp,
                    lb_cfg,
                    policy,
                    train_mode,
                    base_seed,
                    summary,
                )?;
            }
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
            if let Some(moe) = &mut layer.moe {
                // Passes through policy (`router` is excluded -> stays unquantized dense).
                patch_linear_layer(
                    &format!("{p}.router.proj"),
                    &mut moe.router.proj,
                    lb_cfg,
                    policy,
                    train_mode,
                    base_seed,
                    summary,
                )?;
                for (e, expert) in moe.experts.experts.iter_mut().enumerate() {
                    patch_linear_layer(
                        &format!("{p}.experts.{e}.gate_up_proj"),
                        &mut expert.gate_up_proj,
                        lb_cfg,
                        policy,
                        train_mode,
                        base_seed,
                        summary,
                    )?;
                    patch_linear_layer(
                        &format!("{p}.experts.{e}.down_proj"),
                        &mut expert.down_proj,
                        lb_cfg,
                        policy,
                        train_mode,
                        base_seed,
                        summary,
                    )?;
                }
            }
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
            if let Some(vp) = &mut layer.self_attn.v_proj {
                vp.convert_to_packed()?;
            }
            layer.self_attn.o_proj.convert_to_packed()?;
            layer.mlp.gate_proj.convert_to_packed()?;
            layer.mlp.up_proj.convert_to_packed()?;
            layer.mlp.down_proj.convert_to_packed()?;
            if let Some(moe) = &mut layer.moe {
                // Keep moe.router.proj unquantized (Dense); pack all LittleBit experts.
                for expert in &mut moe.experts.experts {
                    expert.gate_up_proj.convert_to_packed()?;
                    expert.down_proj.convert_to_packed()?;
                }
            }
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
            layer.self_attn.q_norm.make_trainable()?;
            layer.self_attn.k_norm.make_trainable()?;
            layer.post_attention_layernorm.make_trainable()?;
            layer.pre_feedforward_layernorm.make_trainable()?;
            layer.post_feedforward_layernorm.make_trainable()?;
            if let Param::Frozen(t) = &layer.layer_scalar {
                layer.layer_scalar = Param::Trainable(Var::from_tensor(t)?);
            }
            if let Some(moe) = &mut layer.moe {
                moe.post_feedforward_layernorm_1.make_trainable()?;
                moe.pre_feedforward_layernorm_2.make_trainable()?;
                moe.post_feedforward_layernorm_2.make_trainable()?;
                if let Param::Frozen(t) = &moe.router.scale {
                    moe.router.scale = Param::Trainable(Var::from_tensor(t)?);
                }
                if let Param::Frozen(t) = &moe.router.per_expert_scale {
                    moe.router.per_expert_scale = Param::Trainable(Var::from_tensor(t)?);
                }
                moe.router.proj.make_trainable()?;
            }
        }
        self.norm.make_trainable()?;
        if let Some(head) = &mut self.lm_head {
            head.make_trainable()?;
        }
        Ok(())
    }

    pub(crate) fn freeze_matching(&mut self, exclude: &[&str]) {
        let matches = |path: &str| exclude.iter().any(|pat| path.contains(pat));
        let wp = &self.weight_prefix;
        if matches("embed_tokens") {
            self.embed_tokens.freeze();
        }
        if matches("lm_head") {
            if let Some(head) = &mut self.lm_head {
                head.freeze();
            }
        }
        if matches("norm") {
            self.norm.freeze();
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("{wp}.layers.{i}");
            if matches(&format!("{p}.input_layernorm")) || matches("layernorm") {
                layer.input_layernorm.freeze();
            }
            if matches(&format!("{p}.post_attention_layernorm")) || matches("layernorm") {
                layer.post_attention_layernorm.freeze();
            }
            if matches(&format!("{p}.pre_feedforward_layernorm")) || matches("layernorm") {
                layer.pre_feedforward_layernorm.freeze();
            }
            if matches(&format!("{p}.post_feedforward_layernorm")) || matches("layernorm") {
                layer.post_feedforward_layernorm.freeze();
            }
            if matches(&format!("{p}.self_attn.q_norm")) || matches("norm") {
                layer.self_attn.q_norm.freeze();
            }
            if matches(&format!("{p}.self_attn.k_norm")) || matches("norm") {
                layer.self_attn.k_norm.freeze();
            }
            if matches(&format!("{p}.layer_scalar")) || matches("layer_scalar") {
                layer.layer_scalar.freeze();
            }
            if matches(&format!("{p}.self_attn")) {
                layer.self_attn.q_proj.freeze();
                layer.self_attn.k_proj.freeze();
                if let Some(vp) = &mut layer.self_attn.v_proj {
                    vp.freeze();
                }
                layer.self_attn.o_proj.freeze();
            }
            if matches(&format!("{p}.mlp")) {
                layer.mlp.gate_proj.freeze();
                layer.mlp.up_proj.freeze();
                layer.mlp.down_proj.freeze();
            }
            if let Some(moe) = &mut layer.moe {
                if matches(&format!("{p}.post_feedforward_layernorm_1")) || matches("layernorm") {
                    moe.post_feedforward_layernorm_1.freeze();
                }
                if matches(&format!("{p}.pre_feedforward_layernorm_2")) || matches("layernorm") {
                    moe.pre_feedforward_layernorm_2.freeze();
                }
                if matches(&format!("{p}.post_feedforward_layernorm_2")) || matches("layernorm") {
                    moe.post_feedforward_layernorm_2.freeze();
                }
                if matches(&format!("{p}.router")) {
                    moe.router.scale.freeze();
                    moe.router.per_expert_scale.freeze();
                    moe.router.proj.freeze();
                }
                if matches(&format!("{p}.experts")) {
                    for expert in &mut moe.experts.experts {
                        expert.gate_up_proj.freeze();
                        expert.down_proj.freeze();
                    }
                }
            }
        }
    }

    pub(crate) fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = self.embed_tokens.trainable_vars();
        for layer in &self.layers {
            vars.extend(layer.input_layernorm.trainable_vars());
            vars.extend(layer.self_attn.q_proj.trainable_vars());
            vars.extend(layer.self_attn.k_proj.trainable_vars());
            if let Some(vp) = &layer.self_attn.v_proj {
                vars.extend(vp.trainable_vars());
            }
            vars.extend(layer.self_attn.o_proj.trainable_vars());
            vars.extend(layer.self_attn.q_norm.trainable_vars());
            vars.extend(layer.self_attn.k_norm.trainable_vars());
            vars.extend(layer.post_attention_layernorm.trainable_vars());
            vars.extend(layer.pre_feedforward_layernorm.trainable_vars());
            vars.extend(layer.mlp.gate_proj.trainable_vars());
            vars.extend(layer.mlp.up_proj.trainable_vars());
            vars.extend(layer.mlp.down_proj.trainable_vars());
            if let Some(moe) = &layer.moe {
                vars.extend(moe.post_feedforward_layernorm_1.trainable_vars());
                vars.extend(moe.pre_feedforward_layernorm_2.trainable_vars());
                if let Some(v) = moe.router.scale.as_var() {
                    vars.push(v.clone());
                }
                if let Some(v) = moe.router.per_expert_scale.as_var() {
                    vars.push(v.clone());
                }
                vars.extend(moe.router.proj.trainable_vars());
                for expert in &moe.experts.experts {
                    vars.extend(expert.gate_up_proj.trainable_vars());
                    vars.extend(expert.down_proj.trainable_vars());
                }
                vars.extend(moe.post_feedforward_layernorm_2.trainable_vars());
            }
            vars.extend(layer.post_feedforward_layernorm.trainable_vars());
            if let Some(v) = layer.layer_scalar.as_var() {
                vars.push(v.clone());
            }
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
                + layer.layer_scalar.elem_count()
                + layer.self_attn.q_proj.param_count()
                + layer.self_attn.k_proj.param_count()
                + layer
                    .self_attn
                    .v_proj
                    .as_ref()
                    .map_or(0, LinearLayer::param_count)
                + layer.self_attn.o_proj.param_count()
                + layer.self_attn.q_norm.weight.elem_count()
                + layer.self_attn.k_norm.weight.elem_count()
                + layer.mlp.gate_proj.param_count()
                + layer.mlp.up_proj.param_count()
                + layer.mlp.down_proj.param_count();
            if let Some(moe) = &layer.moe {
                count += moe.post_feedforward_layernorm_1.weight.elem_count()
                    + moe.pre_feedforward_layernorm_2.weight.elem_count()
                    + moe.post_feedforward_layernorm_2.weight.elem_count()
                    + moe.router.scale.elem_count()
                    + moe.router.per_expert_scale.elem_count()
                    + moe.router.proj.param_count();
                for expert in &moe.experts.experts {
                    count += expert.gate_up_proj.param_count() + expert.down_proj.param_count();
                }
            }
        }
        if let Some(head) = &self.lm_head {
            count += head.param_count();
        }
        count
    }

    pub(crate) fn pack_state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        let wp = &self.weight_prefix;
        self.embed_tokens
            .pack_state_dict(&format!("{wp}.embed_tokens"), &mut out);
        for (i, layer) in self.layers.iter().enumerate() {
            let p = format!("{wp}.layers.{i}");
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
            if let Some(vp) = &layer.self_attn.v_proj {
                out.extend(vp.pack_state_dict(&format!("{p}.self_attn.v_proj"))?);
            }
            out.extend(
                layer
                    .self_attn
                    .o_proj
                    .pack_state_dict(&format!("{p}.self_attn.o_proj"))?,
            );
            layer
                .self_attn
                .q_norm
                .pack_state_dict(&format!("{p}.self_attn.q_norm"), &mut out);
            layer
                .self_attn
                .k_norm
                .pack_state_dict(&format!("{p}.self_attn.k_norm"), &mut out);
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
            if let Some(moe) = &layer.moe {
                moe.post_feedforward_layernorm_1
                    .pack_state_dict(&format!("{p}.post_feedforward_layernorm_1"), &mut out);
                moe.pre_feedforward_layernorm_2
                    .pack_state_dict(&format!("{p}.pre_feedforward_layernorm_2"), &mut out);
                out.insert(
                    format!("{p}.router.scale"),
                    moe.router.scale.as_tensor().detach(),
                );
                out.insert(
                    format!("{p}.router.per_expert_scale"),
                    moe.router.per_expert_scale.as_tensor().detach(),
                );
                out.extend(
                    moe.router
                        .proj
                        .pack_state_dict(&format!("{p}.router.proj"))?,
                );
                for (e, expert) in moe.experts.experts.iter().enumerate() {
                    out.extend(
                        expert
                            .gate_up_proj
                            .pack_state_dict(&format!("{p}.experts.{e}.gate_up_proj"))?,
                    );
                    out.extend(
                        expert
                            .down_proj
                            .pack_state_dict(&format!("{p}.experts.{e}.down_proj"))?,
                    );
                }
                moe.post_feedforward_layernorm_2
                    .pack_state_dict(&format!("{p}.post_feedforward_layernorm_2"), &mut out);
            }
            layer
                .post_feedforward_layernorm
                .pack_state_dict(&format!("{p}.post_feedforward_layernorm"), &mut out);
            out.insert(
                format!("{p}.layer_scalar"),
                layer.layer_scalar.as_tensor().detach(),
            );
        }
        self.norm.pack_state_dict(&format!("{wp}.norm"), &mut out);
        if let Some(head) = &self.lm_head {
            out.extend(head.pack_state_dict("lm_head")?);
        }
        Ok(out)
    }

    pub(crate) fn config(&self) -> &ModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelArchitecture;
    use crate::hub::{LittleBitModel, LoadOptions};
    use crate::model::CausalLm;
    use littlebit_core::{CastPolicy, InferenceMode};
    use rand::SeedableRng;
    use tempfile::TempDir;

    #[test]
    fn multimodal_gemma4_config_unwraps_text_config_and_prefixes() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let cfg_json = serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "tie_word_embeddings": true,
            "text_config": {
                "vocab_size": 128,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "global_head_dim": 32,
                "num_global_key_value_heads": 1,
                "attention_k_eq_v": true,
                "sliding_window": 4,
                "layer_types": ["sliding_attention", "full_attention"],
                "rms_norm_eps": 1e-6,
                "hidden_activation": "gelu_pytorch_tanh",
                "rope_parameters": {
                    "sliding_attention": { "rope_type": "default", "rope_theta": 10000.0 },
                    "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0
                    }
                }
            }
        });
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string_pretty(&cfg_json)?,
        )?;
        let parsed = ModelConfig::from_file(&dir.path().join("config.json"))?;
        assert_eq!(parsed.detect_architecture()?, ModelArchitecture::Gemma4);
        assert!(parsed.tie_word_embeddings);
        assert_eq!(parsed.effective_global_head_dim(), 32);
        assert_eq!(parsed.effective_global_kv_heads(), 1);
        assert_eq!(parsed.effective_hidden_act(), "gelu_pytorch_tanh");

        let full_rope = parsed.gemma4_rope_params("full_attention");
        assert!(full_rope.is_proportional);
        assert!((full_rope.partial_rotary_factor - 0.25).abs() < 1e-12);
        assert!((full_rope.rope_theta - 1_000_000.0).abs() < 1e-12);

        // Verify multimodal weight prefix (`model.language_model.*`) loads cleanly via remapped keys.
        let dev = Device::Cpu;
        let base_model = CausalLm::new_random(parsed.clone(), DType::F32, &dev, 9)?;
        let raw_sd = base_model.pack_state_dict()?;
        let mut mm_sd = HashMap::new();
        for (k, v) in raw_sd {
            let new_k = if let Some(rest) = k.strip_prefix("model.") {
                format!("model.language_model.{rest}")
            } else {
                k
            };
            mm_sd.insert(new_k, v);
        }
        let mm_model = CausalLm::from_tensors(
            parsed,
            &mm_sd,
            None,
            littlebit_core::WeightState::Raw,
            InferenceMode::Unpacked,
            DType::F32,
            &dev,
        )?;
        let ids = Tensor::from_slice(&[1u32, 3, 5], (1, 3), &dev)?;
        let out_a = base_model.forward(&ids, None, HiddenStatesMode::None)?;
        let out_b = mm_model.forward(&ids, None, HiddenStatesMode::None)?;
        let diff = out_a
            .logits
            .sub(&out_b.logits)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-6, "multimodal prefix drift: {diff}");
        Ok(())
    }

    #[test]
    fn proportional_rope_preserves_nope_subspace_and_scales_angles_by_full_head_dim() -> Result<()>
    {
        let dev = Device::Cpu;
        let head_dim = 16;
        let rope = RotaryEmbedding::new_with_config(
            head_dim,
            32,
            10_000.0,
            0.25,
            1.0,
            RopeMode::Proportional,
            DType::F32,
            &dev,
        )?;
        let x = Tensor::ones((1, 2, 5, head_dim), DType::F32, &dev)?;
        let out = rope.apply(&x, 3)?;

        let half = head_dim / 2;
        let rope_angles = 2; // 0.25 * 16 / 2
        let vals = out.get(0)?.to_vec3::<f32>()?;
        let inv_freq_1 = 1.0f64 / 10_000.0f64.powf(2.0 / 16.0);
        for head in &vals {
            for (t, token) in head.iter().enumerate() {
                // Non-rotated NOPE pairs (`rope_angles..half` and `half+rope_angles..head_dim`) stay 1.0.
                for d in rope_angles..half {
                    assert!(
                        (token[d] - 1.0).abs() < 1e-6,
                        "NOPE dim {d} modified: {}",
                        token[d]
                    );
                    assert!(
                        (token[half + d] - 1.0).abs() < 1e-6,
                        "NOPE pair dim {} modified: {}",
                        half + d,
                        token[half + d]
                    );
                }
                // Check angle index `i = 1` at `pos = 3 + t` uses `denom = head_dim (16)`, not `rotary_dim (4)`.
                let pos = (3 + t) as f64;
                let expected_dim1 = (pos * inv_freq_1).cos() - (pos * inv_freq_1).sin();
                assert!(
                    (f64::from(token[1]) - expected_dim1).abs() < 1e-5,
                    "p-RoPE angle mismatch at pos {pos}: got {}, expected {expected_dim1}",
                    token[1]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn gemma4_dense_k_eq_v_and_sliding_window_kv_cache_parity() -> Result<()> {
        let dev = Device::Cpu;
        let mut cfg = ModelConfig::tiny_gemma4_dense(96, 64, 2);
        // Set small sliding window = 4 and test 7 tokens to exercise sliding eviction + seq_pos RoPE!
        cfg.sliding_window = Some(4);

        let mut model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 41)?;
        if let CausalLm::Gemma4(ref g4) = model {
            assert!(g4.layers[0].self_attn.is_sliding);
            assert!(g4.layers[0].self_attn.v_proj.is_some());
            assert!(!g4.layers[1].self_attn.is_sliding);
            assert!(
                g4.layers[1].self_attn.v_proj.is_none(),
                "full_attention with attention_k_eq_v must omit v_proj"
            );
        } else {
            panic!("expected Gemma4");
        }

        let lb_cfg = LittleBitConfig {
            eff_bit: Some(1.0),
            use_itq: true,
            itq_n_iter: 5,
            ..Default::default()
        };
        model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 19)?;

        let prompt = [2u32, 7, 15, 3, 21, 9, 11];
        let full_t = Tensor::from_slice(&prompt, (1, prompt.len()), &dev)?;
        let full_logits = model
            .forward(&full_t, None, HiddenStatesMode::None)?
            .logits
            .get(0)?
            .get(prompt.len() - 1)?;

        let mut cache = KvCache::new(cfg.num_hidden_layers);
        let mut step_logits = None;
        for &tok in &prompt {
            let tok_t = Tensor::from_slice(&[tok], (1, 1), &dev)?;
            let out = model.forward(&tok_t, Some(&mut cache), HiddenStatesMode::None)?;
            step_logits = Some(out.logits.get(0)?.get(0)?);
        }
        assert_eq!(cache.seq_pos(0), prompt.len());
        assert_eq!(cache.past_len(0), 4);
        assert_eq!(cache.past_len(1), prompt.len());

        let diff = full_logits
            .sub(&step_logits.expect("last step logits"))?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-4,
            "Gemma 4 sliding+full KV cache parity failed beyond window: diff={diff}"
        );
        Ok(())
    }

    #[test]
    fn gemma4_moe_sparse_dispatch_matches_dense_reference_and_keeps_router_dense() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::tiny_gemma4_moe(96, 64, 2, 16, 8);
        let mut model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 77)?;

        // Verify sparse top-k dispatch mathematically against explicit dense all-expert loop.
        if let CausalLm::Gemma4(ref g4) = model {
            let moe = g4.layer_moe(0).expect("moe layer 0");
            let x = Tensor::randn(0.0f32, 0.5, (5, cfg.hidden_size), &dev)?;
            let dispatch = moe.router.forward(&x)?;
            assert_eq!(dispatch.top_k_indices.len(), 5);
            assert!(dispatch.top_k_indices.iter().all(|row| row.len() == 8));

            let sparse_out = moe.experts.forward(&x, &dispatch)?;

            let inter = moe.experts.intermediate_size;
            let mut ref_rows = Vec::with_capacity(5);
            for (t, (eids, weights)) in dispatch
                .top_k_indices
                .iter()
                .zip(dispatch.top_k_weights.iter())
                .enumerate()
            {
                let x_tok = x.narrow(0, t, 1)?;
                let mut row_sum = Tensor::zeros_like(&x_tok)?;
                for (&eid, &w) in eids.iter().zip(weights.iter()) {
                    let exp = &moe.experts.experts[eid];
                    let gu = exp.gate_up_proj.forward(&x_tok)?;
                    let g = gu.narrow(1, 0, inter)?;
                    let u = gu.narrow(1, inter, inter)?;
                    let act = apply_activation(&g, &moe.experts.hidden_act)?.mul(&u)?;
                    let y = (exp.down_proj.forward(&act)? * f64::from(w))?;
                    row_sum = row_sum.add(&y)?;
                }
                ref_rows.push(row_sum);
            }
            let dense_ref = Tensor::cat(&ref_rows, 0)?;
            let diff = sparse_out
                .sub(&dense_ref)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-5, "sparse vs reference expert sum diff: {diff}");
        }

        // Quantize and verify router stays LinearLayer::Dense while experts become LittleBit / Packed!
        let lb_cfg = LittleBitConfig {
            eff_bit: Some(1.0),
            residual: true,
            use_itq: true,
            itq_n_iter: 5,
            ..Default::default()
        };
        let summary = model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 23)?;
        assert!(summary.patched_layers > 0);
        assert!(
            summary.skipped_layers >= cfg.num_hidden_layers,
            "all router projections must be recorded as skipped"
        );
        if let CausalLm::Gemma4(ref g4) = model {
            for layer_idx in 0..cfg.num_hidden_layers {
                let moe = g4.layer_moe(layer_idx).expect("moe");
                assert!(
                    matches!(moe.router.proj, LinearLayer::Dense { .. }),
                    "router at layer {layer_idx} must remain Dense"
                );
                assert!(
                    matches!(
                        moe.experts.experts[0].gate_up_proj,
                        LinearLayer::LittleBit(_)
                    ),
                    "experts must be quantized to LittleBit"
                );
            }
        }

        let input = Tensor::from_slice(&[1u32, 4, 9, 12], (1, 4), &dev)?;
        let out_unpacked = model.forward(&input, None, HiddenStatesMode::None)?;

        // Save, reload both Unpacked and Packed, verify bit-tight logits match and KV-cache decode parity.
        let dir = TempDir::new().expect("tempdir");
        let wrapper = LittleBitModel::new(model, lb_cfg, Some("test/gemma4-moe".to_string()), None);
        wrapper.save_pretrained(dir.path(), None, CastPolicy::Preserve)?;

        let reloaded_packed = LittleBitModel::from_pretrained(
            dir.path().to_str().expect("utf8"),
            &LoadOptions {
                inference_mode: InferenceMode::Packed,
                ..Default::default()
            },
        )?;
        if let CausalLm::Gemma4(ref g4) = reloaded_packed.model {
            let moe = g4.layer_moe(0).expect("moe");
            assert!(matches!(moe.router.proj, LinearLayer::Dense { .. }));
            assert!(matches!(
                moe.experts.experts[0].gate_up_proj,
                LinearLayer::Packed(_)
            ));
        }

        let out_packed = reloaded_packed
            .model
            .forward(&input, None, HiddenStatesMode::None)?;
        let drift = out_unpacked
            .logits
            .sub(&out_packed.logits)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(drift < 1e-4, "Gemma 4 MoE packed reload drift: {drift}");

        // KV cache step-by-step parity on Packed Gemma 4 MoE.
        let last_full = out_packed.logits.get(0)?.get(3)?;
        let mut cache = KvCache::new(cfg.num_hidden_layers);
        let mut last_step = None;
        for tok in [1u32, 4, 9, 12] {
            let t = Tensor::from_slice(&[tok], (1, 1), &dev)?;
            last_step = Some(
                reloaded_packed
                    .model
                    .forward(&t, Some(&mut cache), HiddenStatesMode::None)?
                    .logits
                    .get(0)?
                    .get(0)?,
            );
        }
        let kv_drift = last_full
            .sub(&last_step.expect("last step"))?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(kv_drift < 1e-4, "Packed Gemma 4 MoE KV drift: {kv_drift}");

        Ok(())
    }

    #[test]
    fn gemma4_moe_loads_hf_fused_3d_expert_tensors_and_skips_unrouted_experts() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::tiny_gemma4_moe(64, 32, 1, 6, 2);
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let model = Gemma4Model::new_random(cfg.clone(), DType::F32, &dev, &mut rng)?;
        let mut sd = model.pack_state_dict()?;

        // Collapse per-expert keys into Hugging Face 3D fused `experts.gate_up_proj` & `experts.down_proj`.
        let mut gu_slices = Vec::new();
        let mut dn_slices = Vec::new();
        for e in 0..6 {
            let gu = sd
                .remove(&format!("model.layers.0.experts.{e}.gate_up_proj.weight"))
                .expect("expert gu");
            let dn = sd
                .remove(&format!("model.layers.0.experts.{e}.down_proj.weight"))
                .expect("expert dn");
            gu_slices.push(gu);
            dn_slices.push(dn);
        }
        sd.insert(
            "model.layers.0.experts.gate_up_proj".to_string(),
            Tensor::stack(&gu_slices, 0)?,
        );
        sd.insert(
            "model.layers.0.experts.down_proj".to_string(),
            Tensor::stack(&dn_slices, 0)?,
        );

        let loaded = Gemma4Model::from_tensors(
            cfg.clone(),
            &sd,
            LinearLoadSpec {
                littlebit: None,
                ratio_factor: 1.0,
                weight_state: littlebit_core::WeightState::Raw,
                inference_mode: InferenceMode::Unpacked,
                dtype: DType::F32,
                device: &dev,
            },
        )?;

        let ids = Tensor::from_slice(&[1u32, 3, 5], (1, 3), &dev)?;
        let out_orig = model.forward(&ids, None, HiddenStatesMode::None)?;
        let out_3d = loaded.forward(&ids, None, HiddenStatesMode::None)?;
        let diff = out_orig
            .logits
            .sub(&out_3d.logits)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-6, "HF 3D fused experts load mismatch: {diff}");

        // Also verify unrouted experts are never touched: inject NaNs into an expert that is not
        // selected for a single token forward and check finite output.
        let mut model_sparse = loaded;
        let single_tok =
            Tensor::from_vec(vec![0.25f32; cfg.hidden_size], (1, cfg.hidden_size), &dev)?;
        let moe = model_sparse.layer_moe(0).expect("moe");
        let dispatch = moe.router.forward(&single_tok)?;
        let chosen = &dispatch.top_k_indices[0];
        let unused = (0..6)
            .find(|eid| !chosen.contains(eid))
            .expect("at least one unchosen expert");

        let moe_mut = model_sparse.layer_moe_mut(0).expect("moe mut");
        let nan_shape = (
            2 * cfg.moe_intermediate_size.expect("moe inter"),
            cfg.hidden_size,
        );
        let nan_w = Tensor::from_vec(vec![f32::NAN; nan_shape.0 * nan_shape.1], nan_shape, &dev)?;
        moe_mut.experts.experts[unused].gate_up_proj = LinearLayer::Dense {
            weight: Param::Frozen(nan_w),
            bias: None,
        };
        let sparse_out = moe_mut.experts.forward(&single_tok, &dispatch)?;
        let sum_val = sparse_out.sum_all()?.to_scalar::<f32>()?;
        assert!(
            sum_val.is_finite(),
            "sparse dispatch touched unrouted expert {unused} (produced non-finite output)"
        );

        Ok(())
    }
}
