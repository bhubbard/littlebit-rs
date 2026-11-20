//! Differentiable neural network building blocks for quantized and dense LLMs.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, Var, D};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

use littlebit_core::{
    BiasMode, InferenceMode, LayerShape, LittleBitConfig, LittleBitLinear, PackedLittleBitLinear,
    Param, TrainMode, WeightState,
};

use crate::error::{ModelError, Result};

/// Bundle of parameters specifying how to load a [`LinearLayer`] from a state dict.
#[derive(Debug, Clone, Copy)]
pub struct LinearLoadSpec<'a> {
    /// Optional LittleBit quantization config when loading quantized checkpoints.
    pub littlebit: Option<&'a LittleBitConfig>,
    /// Per-layer rank multiplier (`1.0` default, `kv_factor` for `k_proj`/`v_proj`).
    pub ratio_factor: f64,
    /// Whether unpacked `U`/`V` weights are already binarized.
    pub weight_state: WeightState,
    /// Whether to keep `U`/`V` packed at inference (`Packed`) or unpacked floats (`Unpacked`).
    pub inference_mode: InferenceMode,
    /// Target dtype for float parameters.
    pub dtype: DType,
    /// Target compute device.
    pub device: &'a Device,
}

/// Unified linear projection supporting dense weights, differentiable LittleBit QAT layers,
/// and compact 1-bit packed inference layers.
#[derive(Debug, Clone)]
pub enum LinearLayer {
    /// Standard dense linear transformation `x @ W^T + b`.
    Dense {
        /// Weight tensor `(out_features, in_features)`.
        weight: Param,
        /// Optional bias tensor `(out_features,)`.
        bias: Option<Param>,
    },
    /// Differentiable or unpacked LittleBit factorized linear layer.
    LittleBit(Box<LittleBitLinear>),
    /// Memory-compact 1-bit packed LittleBit inference layer.
    Packed(Box<PackedLittleBitLinear>),
}

impl LinearLayer {
    /// Construct a random dense linear layer with normal initialization (`std = 0.02`).
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_random(
        shape: LayerShape,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let dist = Normal::new(0.0f32, 0.02f32).map_err(|err| {
            ModelError::InvalidArgument(format!("failed to construct normal dist: {err}"))
        })?;
        let n = shape.out_features * shape.in_features;
        let data: Vec<f32> = (0..n).map(|_| dist.sample(rng)).collect();
        let w = Tensor::from_vec(data, (shape.out_features, shape.in_features), device)?
            .to_dtype(dtype)?;
        let bias = if shape.bias == BiasMode::WithBias {
            Some(Param::Frozen(Tensor::zeros(
                shape.out_features,
                dtype,
                device,
            )?))
        } else {
            None
        };
        Ok(Self::Dense {
            weight: Param::Frozen(w),
            bias,
        })
    }

    /// Load a linear layer from a state dictionary, detecting packed LittleBit, unpacked
    /// LittleBit, or dense weights automatically.
    ///
    /// # Errors
    /// Returns [`ModelError::MissingWeight`] if expected checkpoint keys are missing.
    pub fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        shape: LayerShape,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let packed_key = format!("{prefix}.U_packed");
        let unpacked_key = format!("{prefix}.U");

        if let Some(lb_cfg) = spec.littlebit {
            let effective_cfg = lb_cfg.effective_for_module(prefix)?;
            if spec.inference_mode == InferenceMode::Packed && tensors.contains_key(&packed_key) {
                let packed = PackedLittleBitLinear::from_packed_tensors(
                    prefix,
                    tensors,
                    shape,
                    &effective_cfg,
                    spec.ratio_factor,
                )?;
                return Ok(Self::Packed(Box::new(packed)));
            }
            if tensors.contains_key(&unpacked_key) {
                let lb = LittleBitLinear::from_state_dict(
                    prefix,
                    tensors,
                    shape,
                    &effective_cfg,
                    spec.ratio_factor,
                    spec.weight_state,
                    spec.dtype,
                    spec.device,
                )?;
                return Ok(Self::LittleBit(Box::new(lb)));
            }
        }

        let w_key = format!("{prefix}.weight");
        let w = tensors
            .get(&w_key)
            .ok_or_else(|| ModelError::MissingWeight(w_key.clone()))?;
        let w = move_cast(w, spec.dtype, spec.device)?;

        let bias = if shape.bias == BiasMode::WithBias {
            let b_key = format!("{prefix}.bias");
            match tensors.get(&b_key) {
                Some(b) => Some(Param::Frozen(move_cast(b, spec.dtype, spec.device)?)),
                None => None,
            }
        } else {
            None
        };

        Ok(Self::Dense {
            weight: Param::Frozen(w),
            bias,
        })
    }

    /// Input feature dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        match self {
            Self::Dense { weight, .. } => weight.as_tensor().dims().get(1).copied().unwrap_or(0),
            Self::LittleBit(lb) => lb.in_features(),
            Self::Packed(p) => p.budget().in_features,
        }
    }

    /// Output feature dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        match self {
            Self::Dense { weight, .. } => weight.as_tensor().dims().first().copied().unwrap_or(0),
            Self::LittleBit(lb) => lb.out_features(),
            Self::Packed(p) => p.budget().out_features,
        }
    }

    /// Return whether this layer currently carries a bias parameter.
    #[must_use]
    pub fn bias_mode(&self) -> BiasMode {
        match self {
            Self::Dense { bias, .. } => BiasMode::from(bias.is_some()),
            Self::LittleBit(lb) => BiasMode::from(lb.bias().is_some()),
            Self::Packed(_) => BiasMode::NoBias,
        }
    }

    /// Execute linear forward pass on `[..., in_features]`.
    ///
    /// # Errors
    /// Returns an error if shapes mismatch or tensor math fails.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense { weight, bias } => {
                let dims = x.dims();
                let in_f = self.in_features();
                let out_f = self.out_features();
                if dims.is_empty() || dims[dims.len() - 1] != in_f {
                    return Err(ModelError::Core(
                        littlebit_core::LittleBitError::InvalidShape {
                            expected: format!("[..., {in_f}]"),
                            actual: format!("{dims:?}"),
                        },
                    ));
                }
                let mut out_dims = dims[..dims.len() - 1].to_vec();
                let batch_rows: usize = out_dims.iter().product::<usize>().max(1);
                out_dims.push(out_f);

                let wt = weight.as_tensor();
                let wt = if wt.dtype() == x.dtype() {
                    wt.clone()
                } else {
                    wt.to_dtype(x.dtype())?
                };
                let wt_t = wt.t()?.contiguous()?;
                let mut y = x.reshape((batch_rows, in_f))?.matmul(&wt_t)?;
                if let Some(b) = bias {
                    let bt = b.as_tensor();
                    let bt = if bt.dtype() == y.dtype() {
                        bt.clone()
                    } else {
                        bt.to_dtype(y.dtype())?
                    };
                    y = y.broadcast_add(&bt)?;
                }
                Ok(y.reshape(out_dims)?)
            }
            Self::LittleBit(lb) => Ok(lb.forward(x)?),
            Self::Packed(p) => Ok(p.forward(x)?),
        }
    }

    /// Convert a dense layer into [`LinearLayer::LittleBit`] via SVD (and optional Joint-ITQ).
    ///
    /// # Errors
    /// Returns an error if called on an already-packed layer or if factorization fails.
    pub fn convert_to_littlebit(
        &mut self,
        config: &LittleBitConfig,
        ratio_factor: f64,
        train_mode: TrainMode,
        seed: u64,
    ) -> Result<()> {
        match self {
            Self::Dense { weight, bias } => {
                let b_ref = bias.as_ref().map(Param::as_tensor);
                let lb = LittleBitLinear::from_dense_weight(
                    weight.as_tensor(),
                    b_ref,
                    config,
                    ratio_factor,
                    train_mode,
                    seed,
                )?;
                *self = Self::LittleBit(Box::new(lb));
                Ok(())
            }
            Self::LittleBit(_) => Ok(()),
            Self::Packed(_) => Err(ModelError::InvalidArgument(
                "cannot re-factorize an already packed linear layer".to_string(),
            )),
        }
    }

    /// Convert a [`LinearLayer::LittleBit`] layer to compact 1-bit [`LinearLayer::Packed`].
    ///
    /// # Errors
    /// Returns an error if packing fails.
    pub fn convert_to_packed(&mut self) -> Result<()> {
        if let Self::LittleBit(lb) = self {
            *self = Self::Packed(Box::new(lb.to_packed()?));
        }
        Ok(())
    }

    /// Export layer weights in packed LittleBit or dense format under `prefix`.
    ///
    /// # Errors
    /// Returns an error if serialization conversion fails.
    pub fn pack_state_dict(&self, prefix: &str) -> Result<HashMap<String, Tensor>> {
        match self {
            Self::Dense { weight, bias } => {
                let mut map = HashMap::new();
                map.insert(format!("{prefix}.weight"), weight.as_tensor().detach());
                if let Some(b) = bias {
                    map.insert(format!("{prefix}.bias"), b.as_tensor().detach());
                }
                Ok(map)
            }
            Self::LittleBit(lb) => Ok(lb.pack_state_dict(prefix)?),
            Self::Packed(p) => Ok(p.pack_state_dict(prefix, &Device::Cpu)?),
        }
    }

    /// Collect trainable [`Var`] parameters inside this layer.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        match self {
            Self::Dense { weight, bias } => {
                let mut out = Vec::new();
                if let Some(v) = weight.as_var() {
                    out.push(v.clone());
                }
                if let Some(b) = bias {
                    if let Some(v) = b.as_var() {
                        out.push(v.clone());
                    }
                }
                out
            }
            Self::LittleBit(lb) => lb.trainable_vars(),
            Self::Packed(_) => Vec::new(),
        }
    }

    /// Convert all dense weights into trainable [`Var`]s (no-op for `Packed`).
    ///
    /// # Errors
    /// Returns an error if converting tensors to `Var` fails.
    pub fn make_trainable(&mut self) -> Result<()> {
        if let Self::Dense { weight, bias } = self {
            *weight = Param::Trainable(Var::from_tensor(weight.as_tensor())?);
            if let Some(b) = bias {
                *b = Param::Trainable(Var::from_tensor(b.as_tensor())?);
            }
        }
        Ok(())
    }

    /// Freeze all trainable parameters in place.
    pub fn freeze(&mut self) {
        match self {
            Self::Dense { weight, bias } => {
                weight.freeze();
                if let Some(b) = bias {
                    b.freeze();
                }
            }
            Self::LittleBit(lb) => lb.freeze(),
            Self::Packed(_) => {}
        }
    }

    /// Total scalar parameter elements stored in this layer.
    #[must_use]
    pub fn param_count(&self) -> usize {
        match self {
            Self::Dense { weight, bias } => {
                weight.elem_count() + bias.as_ref().map_or(0, Param::elem_count)
            }
            Self::LittleBit(lb) => lb.param_count(),
            Self::Packed(p) => {
                let b = p.budget();
                let mult = if b.residual == littlebit_core::ResidualMode::Dual {
                    2
                } else {
                    1
                };
                mult * (b.out_features * b.split_dim
                    + b.split_dim * b.in_features
                    + b.out_features
                    + b.in_features
                    + 2 * b.split_dim)
            }
        }
    }
}

/// RMSNorm variant (standard vs. Gemma `+1.0` offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NormKind {
    /// Standard RMSNorm scaling by `weight`.
    Standard,
    /// Gemma RMSNorm scaling by `(1.0 + weight)`.
    PlusOne,
}

/// Differentiable RMSNorm layer using `candle_nn::ops::rms_norm_slow` so QAT backprop works.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    /// Scale weight `(dim,)`.
    pub weight: Param,
    /// Numerical stability epsilon.
    pub eps: f32,
    /// Standard vs. `+1.0` weight offset.
    pub kind: NormKind,
}

impl RmsNorm {
    /// Load RMSNorm weight `{prefix}.weight`.
    ///
    /// # Errors
    /// Returns [`ModelError::MissingWeight`] if absent.
    pub fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        eps: f64,
        kind: NormKind,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let key = format!("{prefix}.weight");
        let w = tensors
            .get(&key)
            .ok_or_else(|| ModelError::MissingWeight(key.clone()))?;
        Ok(Self {
            weight: Param::Frozen(move_cast(w, dtype, device)?),
            eps: eps as f32,
            kind,
        })
    }

    /// Construct unit-initialized RMSNorm (zero-initialized for [`NormKind::PlusOne`]).
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_init(
        dim: usize,
        eps: f64,
        kind: NormKind,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let weight = match kind {
            NormKind::Standard => Tensor::ones(dim, dtype, device)?,
            NormKind::PlusOne => Tensor::zeros(dim, dtype, device)?,
        };
        Ok(Self {
            weight: Param::Frozen(weight),
            eps: eps as f32,
            kind,
        })
    }

    /// Forward pass using differentiable `rms_norm_slow`.
    ///
    /// # Errors
    /// Returns an error if norm computation fails.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.as_tensor();
        let w = if w.dtype() == x.dtype() {
            w.clone()
        } else {
            w.to_dtype(x.dtype())?
        };
        let alpha = match self.kind {
            NormKind::Standard => w,
            NormKind::PlusOne => (w + 1.0)?,
        };
        Ok(candle_nn::ops::rms_norm_slow(x, &alpha, self.eps)?)
    }

    /// Collect trainable `Var` if unfrozen.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        self.weight.as_var().into_iter().cloned().collect()
    }

    /// Enable gradients on this norm weight.
    ///
    /// # Errors
    /// Returns an error if creating `Var` fails.
    pub fn make_trainable(&mut self) -> Result<()> {
        if let Param::Frozen(t) = &self.weight {
            self.weight = Param::Trainable(Var::from_tensor(t)?);
        }
        Ok(())
    }

    /// Freeze norm weight.
    pub fn freeze(&mut self) {
        self.weight.freeze();
    }

    /// Export `{prefix}.weight`.
    pub fn pack_state_dict(&self, prefix: &str, out: &mut HashMap<String, Tensor>) {
        out.insert(format!("{prefix}.weight"), self.weight.as_tensor().detach());
    }
}

/// Weightless RMSNorm used for Gemma 4 value normalization (`v_norm`) and router input norm.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormNoScale {
    /// Numerical stability epsilon.
    pub eps: f32,
}

impl RmsNormNoScale {
    /// Construct a weightless RMSNorm with epsilon `eps`.
    #[must_use]
    pub fn new(eps: f64) -> Self {
        Self { eps: eps as f32 }
    }

    /// Normalize along the last dimension in F32 (`x * (mean(x^2) + eps)^-0.5`), casting back to `x.dtype()`.
    ///
    /// # Errors
    /// Returns an error if tensor operations fail.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dtype = x.dtype();
        let x_f32 = if orig_dtype == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let mean_sq = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_rms = (mean_sq + f64::from(self.eps))?.powf(-0.5)?;
        let out = x_f32.broadcast_mul(&inv_rms)?;
        if orig_dtype == DType::F32 {
            Ok(out)
        } else {
            Ok(out.to_dtype(orig_dtype)?)
        }
    }
}

/// Differentiable LayerNorm using `candle_nn::ops::layer_norm_slow` (for OPT).
#[derive(Debug, Clone)]
pub struct LayerNorm {
    /// Optional affine weight `(dim,)`.
    pub weight: Option<Param>,
    /// Optional affine bias `(dim,)`.
    pub bias: Option<Param>,
    /// Hidden size.
    pub dim: usize,
    /// Stability epsilon.
    pub eps: f32,
}

impl LayerNorm {
    /// Load LayerNorm `{prefix}.weight` and `{prefix}.bias` when `affine` is true.
    ///
    /// # Errors
    /// Returns [`ModelError::MissingWeight`] if expected weights are absent.
    pub fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        dim: usize,
        eps: f64,
        affine: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if !affine {
            return Ok(Self {
                weight: None,
                bias: None,
                dim,
                eps: eps as f32,
            });
        }
        let w_key = format!("{prefix}.weight");
        let b_key = format!("{prefix}.bias");
        let w = tensors
            .get(&w_key)
            .ok_or_else(|| ModelError::MissingWeight(w_key.clone()))?;
        let b = tensors
            .get(&b_key)
            .ok_or_else(|| ModelError::MissingWeight(b_key.clone()))?;
        Ok(Self {
            weight: Some(Param::Frozen(move_cast(w, dtype, device)?)),
            bias: Some(Param::Frozen(move_cast(b, dtype, device)?)),
            dim,
            eps: eps as f32,
        })
    }

    /// Construct standard unit-weight, zero-bias LayerNorm.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_init(
        dim: usize,
        eps: f64,
        affine: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let (weight, bias) = if affine {
            (
                Some(Param::Frozen(Tensor::ones(dim, dtype, device)?)),
                Some(Param::Frozen(Tensor::zeros(dim, dtype, device)?)),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            weight,
            bias,
            dim,
            eps: eps as f32,
        })
    }

    /// Forward pass using differentiable `layer_norm_slow`.
    ///
    /// # Errors
    /// Returns an error if normalization fails.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let alpha = match &self.weight {
            Some(w) => {
                let wt = w.as_tensor();
                if wt.dtype() == x.dtype() {
                    wt.clone()
                } else {
                    wt.to_dtype(x.dtype())?
                }
            }
            None => Tensor::ones(self.dim, x.dtype(), x.device())?,
        };
        let beta = match &self.bias {
            Some(b) => {
                let bt = b.as_tensor();
                if bt.dtype() == x.dtype() {
                    bt.clone()
                } else {
                    bt.to_dtype(x.dtype())?
                }
            }
            None => Tensor::zeros(self.dim, x.dtype(), x.device())?,
        };
        Ok(candle_nn::ops::layer_norm_slow(x, &alpha, &beta, self.eps)?)
    }

    /// Collect trainable variables.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        if let Some(w) = &self.weight {
            if let Some(v) = w.as_var() {
                vars.push(v.clone());
            }
        }
        if let Some(b) = &self.bias {
            if let Some(v) = b.as_var() {
                vars.push(v.clone());
            }
        }
        vars
    }

    /// Enable gradients on affine parameters.
    ///
    /// # Errors
    /// Returns an error if creating `Var` fails.
    pub fn make_trainable(&mut self) -> Result<()> {
        if let Some(Param::Frozen(w)) = &self.weight {
            self.weight = Some(Param::Trainable(Var::from_tensor(w)?));
        }
        if let Some(Param::Frozen(b)) = &self.bias {
            self.bias = Some(Param::Trainable(Var::from_tensor(b)?));
        }
        Ok(())
    }

    /// Freeze affine parameters.
    pub fn freeze(&mut self) {
        if let Some(w) = &mut self.weight {
            w.freeze();
        }
        if let Some(b) = &mut self.bias {
            b.freeze();
        }
    }

    /// Export `{prefix}.weight` and `{prefix}.bias`.
    pub fn pack_state_dict(&self, prefix: &str, out: &mut HashMap<String, Tensor>) {
        if let Some(w) = &self.weight {
            out.insert(format!("{prefix}.weight"), w.as_tensor().detach());
        }
        if let Some(b) = &self.bias {
            out.insert(format!("{prefix}.bias"), b.as_tensor().detach());
        }
    }

    /// Scalar parameter count.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.weight.as_ref().map_or(0, Param::elem_count)
            + self.bias.as_ref().map_or(0, Param::elem_count)
    }
}

/// Token or positional embedding table `(vocab_size, embed_dim)`.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Weight table `(num_embeddings, embed_dim)`.
    pub weight: Param,
    /// Optional multiplicative embedding scale (e.g. `sqrt(hidden_size)` for Gemma).
    pub scale: Option<f64>,
}

impl Embedding {
    /// Load `{prefix}.weight`.
    ///
    /// # Errors
    /// Returns [`ModelError::MissingWeight`] if absent.
    pub fn from_tensors(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        scale: Option<f64>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let key = format!("{prefix}.weight");
        let w = tensors
            .get(&key)
            .ok_or_else(|| ModelError::MissingWeight(key.clone()))?;
        Ok(Self {
            weight: Param::Frozen(move_cast(w, dtype, device)?),
            scale,
        })
    }

    /// Random initialization `(num_embeddings, embed_dim)` with `std = 0.02`.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_random(
        num_embeddings: usize,
        embed_dim: usize,
        scale: Option<f64>,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let dist = Normal::new(0.0f32, 0.02f32).map_err(|err| {
            ModelError::InvalidArgument(format!("failed to construct normal dist: {err}"))
        })?;
        let data: Vec<f32> = (0..num_embeddings * embed_dim)
            .map(|_| dist.sample(rng))
            .collect();
        let w = Tensor::from_vec(data, (num_embeddings, embed_dim), device)?.to_dtype(dtype)?;
        Ok(Self {
            weight: Param::Frozen(w),
            scale,
        })
    }

    /// Embed token indices `[..., seq_len]` into `[..., seq_len, embed_dim]`.
    ///
    /// # Errors
    /// Returns an error if indexing or reshaping fails.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let mut out_dims = ids.dims().to_vec();
        let embed_dim = self.weight.as_tensor().dims().get(1).copied().unwrap_or(0);
        out_dims.push(embed_dim);

        let flat_ids = ids.flatten_all()?.to_dtype(DType::U32)?;
        let selected = self
            .weight
            .as_tensor()
            .index_select(&flat_ids, 0)?
            .reshape(out_dims)?;
        match self.scale {
            Some(s) => Ok((selected * s)?),
            None => Ok(selected),
        }
    }

    /// Project hidden states `[..., embed_dim]` through tied embedding weights -> `[..., num_embeddings]`.
    ///
    /// # Errors
    /// Returns an error if matrix multiplication fails.
    pub fn forward_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.dims();
        let embed_dim = self.weight.as_tensor().dims().get(1).copied().unwrap_or(0);
        let vocab_size = self.weight.as_tensor().dims().first().copied().unwrap_or(0);
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        let batch_rows: usize = out_dims.iter().product::<usize>().max(1);
        out_dims.push(vocab_size);

        let w = self.weight.as_tensor();
        let w = if w.dtype() == hidden.dtype() {
            w.clone()
        } else {
            w.to_dtype(hidden.dtype())?
        };
        let wt = w.t()?.contiguous()?;
        let out = hidden.reshape((batch_rows, embed_dim))?.matmul(&wt)?;
        Ok(out.reshape(out_dims)?)
    }

    /// Freeze embedding weights.
    pub fn freeze(&mut self) {
        self.weight.freeze();
    }

    /// Collect trainable `Var` if unfrozen.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        self.weight.as_var().into_iter().cloned().collect()
    }

    /// Enable gradients on embedding weights.
    ///
    /// # Errors
    /// Returns an error if creating `Var` fails.
    pub fn make_trainable(&mut self) -> Result<()> {
        if let Param::Frozen(w) = &self.weight {
            self.weight = Param::Trainable(Var::from_tensor(w)?);
        }
        Ok(())
    }

    /// Export `{prefix}.weight`.
    pub fn pack_state_dict(&self, prefix: &str, out: &mut HashMap<String, Tensor>) {
        out.insert(format!("{prefix}.weight"), self.weight.as_tensor().detach());
    }
}

/// Rotary Position Embedding frequency parameterization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeMode {
    /// Standard RoPE denominator `2 * rope_angles` (equal to `head_dim` when `partial_rotary_factor == 1.0`).
    Default,
    /// Gemma 4 proportional RoPE (p-RoPE): always uses full `head_dim` as exponent denominator
    /// even when `partial_rotary_factor < 1.0`, zero-padding non-rotated NOPE pairs.
    Proportional,
}

/// Precomputed Rotary Position Embedding (`cos`, `sin`) supporting differentiable `rope_slow`
/// as well as partial and proportional RoPE (`p-RoPE`).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    head_dim: usize,
    rope_theta: f64,
    partial_rotary_factor: f64,
    scale_factor: f64,
    mode: RopeMode,
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    /// Precompute `(max_seq, head_dim / 2)` cosine and sine tables using standard full-head RoPE.
    ///
    /// # Errors
    /// Returns an error if `head_dim` is odd or tensor creation fails.
    pub fn new(
        head_dim: usize,
        max_seq: usize,
        rope_theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_config(
            head_dim,
            max_seq,
            rope_theta,
            1.0,
            1.0,
            RopeMode::Default,
            dtype,
            device,
        )
    }

    /// Precompute `(max_seq, head_dim / 2)` cosine and sine tables with partial/proportional RoPE options.
    ///
    /// # Errors
    /// Returns an error if `head_dim` is odd, `partial_rotary_factor` is outside `(0.0, 1.0]`,
    /// `scale_factor <= 0.0`, or tensor creation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_config(
        head_dim: usize,
        max_seq: usize,
        rope_theta: f64,
        partial_rotary_factor: f64,
        scale_factor: f64,
        mode: RopeMode,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if !head_dim.is_multiple_of(2) {
            return Err(ModelError::InvalidArgument(format!(
                "rotary embedding head_dim must be even, got {head_dim}"
            )));
        }
        if !partial_rotary_factor.is_finite()
            || partial_rotary_factor <= 0.0
            || partial_rotary_factor > 1.0
        {
            return Err(ModelError::InvalidArgument(format!(
                "partial_rotary_factor must be in (0.0, 1.0], got {partial_rotary_factor}"
            )));
        }
        if !scale_factor.is_finite() || scale_factor <= 0.0 {
            return Err(ModelError::InvalidArgument(format!(
                "rope scale_factor must be finite and positive, got {scale_factor}"
            )));
        }
        let (cos, sin) = compute_rope_tables_with_config(
            head_dim,
            max_seq,
            rope_theta,
            partial_rotary_factor,
            scale_factor,
            mode,
            dtype,
            device,
        )?;
        Ok(Self {
            head_dim,
            rope_theta,
            partial_rotary_factor,
            scale_factor,
            mode,
            cos,
            sin,
        })
    }

    /// Apply RoPE to `(batch, heads, seq_len, head_dim)` starting at `pos_offset`.
    ///
    /// # Errors
    /// Returns an error if `rope_slow` or dynamic table recomputation fails.
    pub fn apply(&self, x: &Tensor, pos_offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = x.dims4()?;
        let needed = pos_offset + seq_len;
        let (cos_full, sin_full) = if needed <= self.cos.dims()[0] {
            (self.cos.clone(), self.sin.clone())
        } else {
            compute_rope_tables_with_config(
                self.head_dim,
                needed,
                self.rope_theta,
                self.partial_rotary_factor,
                self.scale_factor,
                self.mode,
                x.dtype(),
                x.device(),
            )?
        };
        let cos = cos_full
            .narrow(0, pos_offset, seq_len)?
            .to_dtype(x.dtype())?;
        let sin = sin_full
            .narrow(0, pos_offset, seq_len)?
            .to_dtype(x.dtype())?;
        Ok(candle_nn::rotary_emb::rope_slow(x, &cos, &sin)?)
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_rope_tables_with_config(
    head_dim: usize,
    max_seq: usize,
    rope_theta: f64,
    partial_rotary_factor: f64,
    scale_factor: f64,
    mode: RopeMode,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let rope_angles =
        (((partial_rotary_factor * head_dim as f64) / 2.0).floor() as usize).min(half_dim);
    let denom = match mode {
        RopeMode::Proportional => head_dim as f64,
        RopeMode::Default => (2 * rope_angles).max(2) as f64,
    };
    let mut inv_freq = vec![0.0f32; half_dim];
    for (i, slot) in inv_freq.iter_mut().take(rope_angles).enumerate() {
        *slot = (1.0f64 / rope_theta.powf((2 * i) as f64 / denom) / scale_factor) as f32;
    }
    let inv_freq_t = Tensor::from_vec(inv_freq, (1, half_dim), device)?;
    let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
    let pos_t = Tensor::from_vec(positions, (max_seq, 1), device)?;
    let freqs = pos_t.matmul(&inv_freq_t)?;
    let cos = freqs.cos()?.to_dtype(dtype)?;
    let sin = freqs.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Layer-indexed key/value cache for autoregressive generation.
#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    seen_tokens: Vec<usize>,
}

impl KvCache {
    /// Allocate an empty cache for `num_layers` transformer blocks.
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            seen_tokens: vec![0; num_layers],
        }
    }

    /// Number of tokens currently retained in the cache window at `layer_idx`.
    #[must_use]
    pub fn past_len(&self, layer_idx: usize) -> usize {
        self.layers
            .get(layer_idx)
            .and_then(|entry| entry.as_ref())
            .and_then(|(k, _)| k.dims().get(2).copied())
            .unwrap_or(0)
    }

    /// Total number of sequence positions processed so far at `layer_idx` (unaffected by sliding-window eviction).
    #[must_use]
    pub fn seq_pos(&self, layer_idx: usize) -> usize {
        self.seen_tokens.get(layer_idx).copied().unwrap_or(0)
    }

    /// Append `(k, v)` of shape `(batch, kv_heads, seq_len, head_dim)` and return the concatenated history.
    ///
    /// # Errors
    /// Returns an error if `layer_idx` is out of bounds or concatenation fails.
    pub fn append(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        sliding_window: Option<usize>,
    ) -> Result<(Tensor, Tensor)> {
        let added = k.dims()[2];
        let slot = self.layers.get_mut(layer_idx).ok_or_else(|| {
            ModelError::InvalidArgument(format!("kv cache layer index {layer_idx} out of bounds"))
        })?;
        let (mut next_k, mut next_v) = match slot.take() {
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[&prev_k, k], 2)?,
                Tensor::cat(&[&prev_v, v], 2)?,
            ),
            None => (k.clone(), v.clone()),
        };

        if let Some(seen) = self.seen_tokens.get_mut(layer_idx) {
            *seen += added;
        }

        if let Some(win) = sliding_window {
            let cur_len = next_k.dims()[2];
            if cur_len > win {
                next_k = next_k.narrow(2, cur_len - win, win)?;
                next_v = next_v.narrow(2, cur_len - win, win)?;
            }
        }

        *slot = Some((next_k.clone(), next_v.clone()));
        Ok((next_k, next_v))
    }

    /// Reset all cached key/value tensors and processed position counters.
    pub fn clear(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
        self.seen_tokens.fill(0);
    }
}

/// Repeat key/value heads `n_rep` times along dimension 1 for Grouped-Query Attention (GQA).
///
/// # Errors
/// Returns an error if tensor is not rank-4 or reshape fails.
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, kv_heads, seq_len, head_dim) = x.dims4()?;
    let expanded = x
        .unsqueeze(2)?
        .expand((batch, kv_heads, n_rep, seq_len, head_dim))?
        .reshape((batch, kv_heads * n_rep, seq_len, head_dim))?;
    Ok(expanded)
}

/// Build a causal additive attention mask `(1, 1, seq_len, total_len)` with `-1e9` on masked positions.
///
/// # Errors
/// Returns an error if tensor creation fails.
pub fn causal_mask(
    seq_len: usize,
    past_len: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total_len = past_len + seq_len;
    let mut mask = vec![0.0f32; seq_len * total_len];
    for q in 0..seq_len {
        let q_pos = past_len + q;
        for k in 0..total_len {
            let masked_future = k > q_pos;
            let masked_window = match sliding_window {
                Some(win) if win > 0 => q_pos >= win && k <= q_pos - win,
                _ => false,
            };
            if masked_future || masked_window {
                mask[q * total_len + k] = -1.0e9;
            }
        }
    }
    let t = Tensor::from_vec(mask, (1, 1, seq_len, total_len), device)?;
    if dtype == DType::F32 {
        Ok(t)
    } else {
        Ok(t.to_dtype(dtype)?)
    }
}

/// Apply activation function matching Hugging Face `hidden_act`.
///
/// # Errors
/// Returns [`ModelError::UnsupportedArchitecture`] for unknown activation names.
pub fn apply_activation(x: &Tensor, name: &str) -> Result<Tensor> {
    match name.to_ascii_lowercase().as_str() {
        "silu" | "swish" => Ok(candle_nn::ops::silu(x)?),
        "gelu" => Ok(x.gelu_erf()?),
        "gelu_pytorch_tanh" | "gelu_new" | "gelu_fast" => Ok(x.gelu()?),
        "relu" => Ok(x.relu()?),
        other => Err(ModelError::UnsupportedArchitecture(format!(
            "unsupported activation function `{other}`"
        ))),
    }
}

/// Parameters controlling scaled dot-product attention inside decoder layers.
#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    /// Multiplicative logit scale (typically `1.0 / sqrt(head_dim)`).
    pub scale: f64,
    /// Optional logit softcap (`softcap * tanh(scores / softcap)`), used by Gemma 2/3.
    pub softcap: Option<f64>,
    /// Optional sliding window attention span.
    pub sliding_window: Option<usize>,
}

/// Differentiable multi-head / grouped-query causal scaled dot-product attention.
///
/// Expects `q` `(batch, num_heads, seq_len, head_dim)` and `k, v` `(batch, num_kv_heads, seq_len, head_dim)`.
/// Returns `(batch, seq_len, num_heads * head_dim)`.
///
/// # Errors
/// Returns an error if GQA head counts are incompatible or tensor ops fail.
pub(crate) fn compute_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    layer_idx: usize,
    cache: Option<&mut KvCache>,
    params: AttentionParams,
) -> Result<Tensor> {
    let (batch, num_heads, seq_len, head_dim) = q.dims4()?;
    let (_, num_kv_heads, _, _) = k.dims4()?;
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(ModelError::InvalidArgument(format!(
            "num_attention_heads ({num_heads}) must be divisible by num_key_value_heads ({num_kv_heads})"
        )));
    }
    let n_rep = num_heads / num_kv_heads;

    let (k_all, v_all) = match cache {
        Some(c) => c.append(layer_idx, k, v, params.sliding_window)?,
        None => (k.clone(), v.clone()),
    };
    let effective_past_len = k_all.dims()[2].saturating_sub(seq_len);

    let k_rep = repeat_kv(&k_all, n_rep)?;
    let v_rep = repeat_kv(&v_all, n_rep)?;

    let mut scores = (q.matmul(&k_rep.t()?.contiguous()?)? * params.scale)?;
    if let Some(cap) = params.softcap {
        scores = ((scores / cap)?.tanh()? * cap)?;
    }
    if seq_len > 1 {
        let mask = causal_mask(
            seq_len,
            effective_past_len,
            params.sliding_window,
            scores.dtype(),
            scores.device(),
        )?;
        scores = scores.broadcast_add(&mask)?;
    }

    // Use differentiable softmax (`candle_nn::ops::softmax`) rather than `softmax_last_dim`.
    let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
    let ctx = probs.matmul(&v_rep.contiguous()?)?;
    Ok(ctx
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch, seq_len, num_heads * head_dim))?)
}

pub(crate) fn move_cast(t: &Tensor, dtype: DType, device: &Device) -> Result<Tensor> {
    let moved = if t.device().same_device(device) {
        t.clone()
    } else {
        t.to_device(device)?
    };
    if moved.dtype() == dtype {
        Ok(moved)
    } else {
        Ok(moved.to_dtype(dtype)?)
    }
}
