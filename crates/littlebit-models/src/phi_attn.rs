// -----------------------------------------------------------------------------
// Derived from quantization/modules/attention.py in SamsungLabs/LittleBit,
// originally adapted under the Apache-2.0 License:
// Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
// -----------------------------------------------------------------------------

//! Phi-3 / Phi-4 QKV splitting projection (`PhiQKVSplitAttention` parity).

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Tensor, Var};

use littlebit_core::{BiasMode, LayerShape, LittleBitConfig, Param, TrainMode};

use crate::config::ModelConfig;
use crate::error::{ModelError, Result};
use crate::layer::{LinearLayer, LinearLoadSpec};
use crate::patch::{patch_linear_layer, PatchPolicy, PatchSummary};

/// Attention QKV projection for Phi-3 / Phi-4 supporting both fused dense checkpoints
/// (`qkv_proj`) and split LittleBit checkpoints (`q_proj`, `k_proj`, `v_proj`).
#[derive(Debug, Clone)]
pub enum PhiAttentionProjection {
    /// Original fused Hugging Face Phi-3/Phi-4 `qkv_proj` dense layer.
    Fused {
        /// Combined `(q_dim + 2 * kv_dim, hidden_size)` linear projection.
        qkv_proj: Box<LinearLayer>,
    },
    /// Split Q/K/V projections produced by `PhiQKVSplitAttention`, allowing separate
    /// `kv_factor` scaling on `k_proj` and `v_proj`.
    Split {
        /// Query projection `(num_heads * head_dim, hidden_size)`.
        q_proj: Box<LinearLayer>,
        /// Key projection `(num_kv_heads * head_dim, hidden_size)`.
        k_proj: Box<LinearLayer>,
        /// Value projection `(num_kv_heads * head_dim, hidden_size)`.
        v_proj: Box<LinearLayer>,
    },
}

impl PhiAttentionProjection {
    /// Load either split Q/K/V weights (when present in a quantized checkpoint) or fused `qkv_proj`.
    ///
    /// # Errors
    /// Returns [`ModelError::MissingWeight`] if neither split nor fused projections exist.
    pub fn from_tensors(
        attn_prefix: &str,
        tensors: &HashMap<String, Tensor>,
        cfg: &ModelConfig,
        littlebit_cfg: Option<&LittleBitConfig>,
        spec_base: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let head_dim = cfg.effective_head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.effective_kv_heads() * head_dim;
        let bias_mode = BiasMode::from(cfg.attention_bias);

        let has_split = tensors.contains_key(&format!("{attn_prefix}.q_proj.U_packed"))
            || tensors.contains_key(&format!("{attn_prefix}.q_proj.U"))
            || tensors.contains_key(&format!("{attn_prefix}.q_proj.weight"));

        if has_split {
            let kv_factor = littlebit_cfg.map_or(1.0, |c| c.kv_factor);
            let q_spec = LinearLoadSpec {
                ratio_factor: 1.0,
                ..spec_base
            };
            let kv_spec = LinearLoadSpec {
                ratio_factor: kv_factor,
                ..spec_base
            };
            let q_proj = LinearLayer::from_tensors(
                &format!("{attn_prefix}.q_proj"),
                tensors,
                LayerShape::new(cfg.hidden_size, q_dim, bias_mode),
                q_spec,
            )?;
            let k_proj = LinearLayer::from_tensors(
                &format!("{attn_prefix}.k_proj"),
                tensors,
                LayerShape::new(cfg.hidden_size, kv_dim, bias_mode),
                kv_spec,
            )?;
            let v_proj = LinearLayer::from_tensors(
                &format!("{attn_prefix}.v_proj"),
                tensors,
                LayerShape::new(cfg.hidden_size, kv_dim, bias_mode),
                kv_spec,
            )?;
            Ok(Self::Split {
                q_proj: Box::new(q_proj),
                k_proj: Box::new(k_proj),
                v_proj: Box::new(v_proj),
            })
        } else {
            let qkv_proj = LinearLayer::from_tensors(
                &format!("{attn_prefix}.qkv_proj"),
                tensors,
                LayerShape::new(cfg.hidden_size, q_dim + 2 * kv_dim, bias_mode),
                LinearLoadSpec {
                    littlebit: None,
                    ratio_factor: 1.0,
                    ..spec_base
                },
            )?;
            Ok(Self::Fused {
                qkv_proj: Box::new(qkv_proj),
            })
        }
    }

    /// Project `hidden_states` into `(query, key, value)` tensors of shape `[..., dim]`.
    ///
    /// # Errors
    /// Returns an error if underlying linear projections or slicing fail.
    pub fn project(
        &self,
        hidden_states: &Tensor,
        cfg: &ModelConfig,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let head_dim = cfg.effective_head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.effective_kv_heads() * head_dim;

        match self {
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => Ok((
                q_proj.forward(hidden_states)?,
                k_proj.forward(hidden_states)?,
                v_proj.forward(hidden_states)?,
            )),
            Self::Fused { qkv_proj } => {
                let qkv = qkv_proj.forward(hidden_states)?;
                let last = qkv.rank() - 1;
                let q = qkv.narrow(last, 0, q_dim)?;
                let k = qkv.narrow(last, q_dim, kv_dim)?;
                let v = qkv.narrow(last, q_dim + kv_dim, kv_dim)?;
                Ok((q, k, v))
            }
        }
    }

    /// Convert fused `qkv_proj` into separate `q_proj`, `k_proj`, and `v_proj` layers
    /// matching upstream `PhiQKVSplitAttention._initialize_parameters`.
    ///
    /// # Errors
    /// Returns an error if slicing weights/biases fails or if `qkv_proj` is already quantized.
    pub fn split_qkv(&mut self, cfg: &ModelConfig) -> Result<()> {
        let Self::Fused { qkv_proj } = self else {
            return Ok(());
        };
        let (weight, bias) = match qkv_proj.as_ref() {
            LinearLayer::Dense { weight, bias } => (weight.as_tensor().clone(), bias.clone()),
            _ => {
                return Err(ModelError::InvalidArgument(
                    "cannot split non-dense qkv_proj".to_string(),
                ));
            }
        };

        let head_dim = cfg.effective_head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.effective_kv_heads() * head_dim;

        let qw = weight.i(0..q_dim)?.contiguous()?;
        let kw = weight.i(q_dim..q_dim + kv_dim)?.contiguous()?;
        let vw = weight.i(q_dim + kv_dim..q_dim + 2 * kv_dim)?.contiguous()?;

        let (qb, kb, vb) = if let Some(b) = bias {
            let bt = b.as_tensor();
            (
                Some(Param::Frozen(bt.i(0..q_dim)?.contiguous()?)),
                Some(Param::Frozen(bt.i(q_dim..q_dim + kv_dim)?.contiguous()?)),
                Some(Param::Frozen(
                    bt.i(q_dim + kv_dim..q_dim + 2 * kv_dim)?.contiguous()?,
                )),
            )
        } else {
            (None, None, None)
        };

        *self = Self::Split {
            q_proj: Box::new(LinearLayer::Dense {
                weight: Param::Frozen(qw),
                bias: qb,
            }),
            k_proj: Box::new(LinearLayer::Dense {
                weight: Param::Frozen(kw),
                bias: kb,
            }),
            v_proj: Box::new(LinearLayer::Dense {
                weight: Param::Frozen(vw),
                bias: vb,
            }),
        };
        Ok(())
    }

    /// Split fused `qkv_proj` if needed and patch `q_proj`, `k_proj`, `v_proj` with LittleBit.
    ///
    /// # Errors
    /// Returns an error if splitting or LittleBit factorization fails.
    pub fn apply_littlebit_patch(
        &mut self,
        attn_prefix: &str,
        cfg: &ModelConfig,
        lb_cfg: &LittleBitConfig,
        policy: &PatchPolicy,
        train_mode: TrainMode,
        base_seed: u64,
        summary: &mut PatchSummary,
    ) -> Result<()> {
        self.split_qkv(cfg)?;
        if let Self::Split {
            q_proj,
            k_proj,
            v_proj,
        } = self
        {
            patch_linear_layer(
                &format!("{attn_prefix}.q_proj"),
                q_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{attn_prefix}.k_proj"),
                k_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{attn_prefix}.v_proj"),
                v_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
        }
        Ok(())
    }

    /// Convert any `LittleBit` projections inside to compact 1-bit `Packed` representation.
    ///
    /// # Errors
    /// Returns an error if packing fails.
    pub fn convert_to_packed(&mut self) -> Result<()> {
        match self {
            Self::Fused { qkv_proj } => qkv_proj.convert_to_packed(),
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => {
                q_proj.convert_to_packed()?;
                k_proj.convert_to_packed()?;
                v_proj.convert_to_packed()
            }
        }
    }

    /// Export weights under `{attn_prefix}.{qkv_proj|q_proj,k_proj,v_proj}.*`.
    ///
    /// # Errors
    /// Returns an error if tensor export fails.
    pub fn pack_state_dict(
        &self,
        attn_prefix: &str,
        out: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        match self {
            Self::Fused { qkv_proj } => {
                out.extend(qkv_proj.pack_state_dict(&format!("{attn_prefix}.qkv_proj"))?);
            }
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => {
                out.extend(q_proj.pack_state_dict(&format!("{attn_prefix}.q_proj"))?);
                out.extend(k_proj.pack_state_dict(&format!("{attn_prefix}.k_proj"))?);
                out.extend(v_proj.pack_state_dict(&format!("{attn_prefix}.v_proj"))?);
            }
        }
        Ok(())
    }

    /// Collect trainable variables.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        match self {
            Self::Fused { qkv_proj } => qkv_proj.trainable_vars(),
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => {
                let mut vars = q_proj.trainable_vars();
                vars.extend(k_proj.trainable_vars());
                vars.extend(v_proj.trainable_vars());
                vars
            }
        }
    }

    /// Freeze all trainable parameters.
    pub fn freeze(&mut self) {
        match self {
            Self::Fused { qkv_proj } => qkv_proj.freeze(),
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => {
                q_proj.freeze();
                k_proj.freeze();
                v_proj.freeze();
            }
        }
    }

    /// Total scalar parameters across Q/K/V projections.
    #[must_use]
    pub fn param_count(&self) -> usize {
        match self {
            Self::Fused { qkv_proj } => qkv_proj.param_count(),
            Self::Split {
                q_proj,
                k_proj,
                v_proj,
            } => q_proj.param_count() + k_proj.param_count() + v_proj.param_count(),
        }
    }

    /// Ensure dtype and device helper for tests.
    #[allow(dead_code)]
    pub(crate) fn new_random_fused(
        cfg: &ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> Result<Self> {
        let head_dim = cfg.effective_head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.effective_kv_heads() * head_dim;
        let bias = BiasMode::from(cfg.attention_bias);
        Ok(Self::Fused {
            qkv_proj: Box::new(LinearLayer::new_random(
                LayerShape::new(cfg.hidden_size, q_dim + 2 * kv_dim, bias),
                dtype,
                device,
                rng,
            )?),
        })
    }
}
