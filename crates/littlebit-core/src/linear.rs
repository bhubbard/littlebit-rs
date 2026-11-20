//! `LittleBitLinear` and compact 1-bit `PackedLittleBitLinear` modules.

use std::collections::HashMap;
use std::fmt;

use candle_core::{DType, Device, Tensor, Var};
use faer::Mat;

use crate::binary::binarize_sign_no_grad;
use crate::config::{
    BiasMode, BitBudget, LayerShape, LittleBitConfig, QuantFunc, ResidualMode, TrainMode,
    WeightState,
};
use crate::error::{LittleBitError, Result};
use crate::itq::apply_joint_itq;
use crate::packer::{binary_packer, PackedBinaryMatrix};
use crate::svd::{rank_one_decompose, svd_lowrank, DEFAULT_LOWRANK_NITER};

/// Parameter storage that supports both differentiable QAT training and frozen inference.
#[derive(Debug, Clone)]
pub enum Param {
    /// Differentiable leaf variable tracked by Candle autograd.
    Trainable(Var),
    /// Frozen non-differentiable tensor.
    Frozen(Tensor),
}

impl Param {
    /// Wrap a tensor as [`Param::Trainable`] or [`Param::Frozen`] according to `mode`.
    ///
    /// # Errors
    /// Returns a Candle error if constructing a [`Var`] fails.
    pub fn from_tensor(tensor: Tensor, mode: TrainMode) -> Result<Self> {
        match mode {
            TrainMode::Train => Ok(Self::Trainable(Var::from_tensor(&tensor)?)),
            TrainMode::Eval => Ok(Self::Frozen(tensor.detach())),
        }
    }

    /// Borrow the underlying tensor regardless of mutability mode.
    #[must_use]
    pub fn as_tensor(&self) -> &Tensor {
        match self {
            Self::Trainable(v) => v.as_tensor(),
            Self::Frozen(t) => t,
        }
    }

    /// Return the underlying [`Var`] when trainable.
    #[must_use]
    pub fn as_var(&self) -> Option<&Var> {
        match self {
            Self::Trainable(v) => Some(v),
            Self::Frozen(_) => None,
        }
    }

    /// Convert to a frozen detached parameter in place.
    pub fn freeze(&mut self) {
        if let Self::Trainable(v) = self {
            *self = Self::Frozen(v.as_tensor().detach());
        }
    }

    /// Return true when backed by a trainable [`Var`].
    #[must_use]
    pub fn is_trainable(&self) -> bool {
        matches!(self, Self::Trainable(_))
    }

    /// Element count of the underlying tensor.
    #[must_use]
    pub fn elem_count(&self) -> usize {
        self.as_tensor().elem_count()
    }
}

/// One latent factorization branch `(U, V, u1, u2, v1, v2)`.
#[derive(Debug, Clone)]
pub struct FactorBranch {
    /// Left binary-quantized factor `(out_features, split_dim)`.
    pub u: Param,
    /// Right binary-quantized factor `(split_dim, in_features)`.
    pub v: Param,
    /// Output row scale `(1, out_features)`.
    pub u1: Param,
    /// Output latent rank scale `(1, split_dim)`.
    pub u2: Param,
    /// Input latent rank scale `(1, split_dim)`.
    pub v1: Param,
    /// Input column scale `(1, in_features)`.
    pub v2: Param,
}

/// Maximum finite magnitude safely representable in IEEE-754 binary16 (`f16`).
pub const FP16_MAX_SAFE: f64 = 65_504.0;

/// Compute the smallest power of two `>= val` (clamped to `>= 1.0`), matching the
/// `pow2_at_least` helper from `SamsungLabs/LittleBit` Issue #24.
#[inline]
#[must_use]
pub fn pow2_at_least(val: f64) -> f64 {
    if !val.is_finite() || val <= 1.0 {
        1.0
    } else {
        2.0_f64.powf(val.log2().ceil())
    }
}

impl FactorBranch {
    fn from_raw(
        raw: RawDecomposedBranch,
        dtype: DType,
        device: &Device,
        mode: TrainMode,
        out_features: usize,
        in_features: usize,
        split_dim: usize,
    ) -> Result<Self> {
        let make = |data: Vec<f32>, shape: (usize, usize)| -> Result<Param> {
            let t = Tensor::from_vec(data, shape, device)?;
            let t = if dtype == DType::F32 {
                t
            } else {
                t.to_dtype(dtype)?
            };
            Param::from_tensor(t, mode)
        };
        Ok(Self {
            u: make(raw.u, (out_features, split_dim))?,
            v: make(raw.v, (split_dim, in_features))?,
            u1: make(raw.u1, (1, out_features))?,
            u2: make(raw.u2, (1, split_dim))?,
            v1: make(raw.v1, (1, split_dim))?,
            v2: make(raw.v2, (1, in_features))?,
        })
    }

    fn empty(
        out_features: usize,
        in_features: usize,
        split_dim: usize,
        dtype: DType,
        device: &Device,
        mode: TrainMode,
    ) -> Result<Self> {
        let make = |shape: (usize, usize)| -> Result<Param> {
            let t = Tensor::zeros(shape, dtype, device)?;
            Param::from_tensor(t, mode)
        };
        Ok(Self {
            u: make((out_features, split_dim))?,
            v: make((split_dim, in_features))?,
            u1: make((1, out_features))?,
            u2: make((1, split_dim))?,
            v1: make((1, split_dim))?,
            v2: make((1, in_features))?,
        })
    }

    fn forward(&self, x_2d: &Tensor, quant_func: QuantFunc, binarized: bool) -> Result<Tensor> {
        let x_dtype = x_2d.dtype();
        let cast = |t: &Tensor| -> Result<Tensor> {
            if t.dtype() == x_dtype {
                Ok(t.clone())
            } else {
                Ok(t.to_dtype(x_dtype)?)
            }
        };

        let v = cast(self.v.as_tensor())?;
        let u = cast(self.u.as_tensor())?;
        let vq = if binarized { v } else { quant_func.apply(&v)? };
        let uq = if binarized { u } else { quant_func.apply(&u)? };

        let vq_t = vq.t()?.contiguous()?;
        let uq_t = uq.t()?.contiguous()?;

        if x_dtype == DType::F16 {
            let x_f32 = x_2d.to_dtype(DType::F32)?;
            let v2_f32 = self.v2.as_tensor().to_dtype(DType::F32)?;
            let v1_f32 = self.v1.as_tensor().to_dtype(DType::F32)?;
            let u2_f32 = self.u2.as_tensor().to_dtype(DType::F32)?;
            let u1_f32 = self.u1.as_tensor().to_dtype(DType::F32)?;
            let v1u2_f32 = v1_f32.mul(&u2_f32)?;

            if x_2d.device().is_cpu() {
                let vq_t_f32 = vq_t.to_dtype(DType::F32)?;
                let uq_t_f32 = uq_t.to_dtype(DType::F32)?;
                let h = x_f32.broadcast_mul(&v2_f32)?.matmul(&vq_t_f32)?;
                let g = h.broadcast_mul(&v1u2_f32)?;
                let out = g
                    .matmul(&uq_t_f32)?
                    .broadcast_mul(&u1_f32)?
                    .to_dtype(DType::F16)?;
                return Ok(out);
            }

            let a = x_f32.broadcast_mul(&v2_f32)?;
            let a_max = f64::from(a.abs()?.max_all()?.to_scalar::<f32>()?);
            let in_features = vq_t.dims()[0] as f64;
            let s1 = pow2_at_least(a_max * in_features / FP16_MAX_SAFE);
            let a_scaled = if s1 > 1.0 { (a / s1)? } else { a };

            let h = a_scaled
                .to_dtype(DType::F16)?
                .matmul(&vq_t)?
                .to_dtype(DType::F32)?;
            let g = h.broadcast_mul(&v1u2_f32)?;
            let g_max = f64::from(g.abs()?.max_all()?.to_scalar::<f32>()?);
            let split_dim = uq_t.dims()[0] as f64;
            let s2 = pow2_at_least(g_max * split_dim / FP16_MAX_SAFE);
            let g_scaled = if s2 > 1.0 { (g / s2)? } else { g };

            let y = g_scaled
                .to_dtype(DType::F16)?
                .matmul(&uq_t)?
                .to_dtype(DType::F32)?;
            let total_scale = s1 * s2;
            let u1_scaled = if total_scale > 1.0 {
                (u1_f32 * total_scale)?
            } else {
                u1_f32
            };
            let out = y.broadcast_mul(&u1_scaled)?.to_dtype(DType::F16)?;
            return Ok(out);
        }

        let v2 = cast(self.v2.as_tensor())?;
        let v1 = cast(self.v1.as_tensor())?;
        let u2 = cast(self.u2.as_tensor())?;
        let u1 = cast(self.u1.as_tensor())?;

        let v1u2 = v1.mul(&u2)?;

        // ((((x * v2) @ Vq^T) * (v1 * u2)) @ Uq^T) * u1
        let h = x_2d.broadcast_mul(&v2)?.matmul(&vq_t)?;
        let h = h.broadcast_mul(&v1u2)?;
        let out = h.matmul(&uq_t)?.broadcast_mul(&u1)?;
        Ok(out)
    }

    fn push_vars(&self, vars: &mut Vec<Var>) {
        for p in [&self.u, &self.v, &self.u1, &self.u2, &self.v1, &self.v2] {
            if let Some(v) = p.as_var() {
                vars.push(v.clone());
            }
        }
    }

    fn freeze(&mut self) {
        self.u.freeze();
        self.v.freeze();
        self.u1.freeze();
        self.u2.freeze();
        self.v1.freeze();
        self.v2.freeze();
    }

    fn warm_start(&mut self, latent_mag: f64) -> Result<()> {
        let u_scaled = (self.u.as_tensor().detach() * latent_mag)?;
        let v_scaled = (self.v.as_tensor().detach() * latent_mag)?;
        self.u = Param::Trainable(Var::from_tensor(&u_scaled)?);
        self.v = Param::Trainable(Var::from_tensor(&v_scaled)?);
        self.u1 = Param::Trainable(Var::from_tensor(&self.u1.as_tensor().detach())?);
        self.u2 = Param::Trainable(Var::from_tensor(&self.u2.as_tensor().detach())?);
        self.v1 = Param::Trainable(Var::from_tensor(&self.v1.as_tensor().detach())?);
        self.v2 = Param::Trainable(Var::from_tensor(&self.v2.as_tensor().detach())?);
        Ok(())
    }

    fn elem_count(&self) -> usize {
        self.u.elem_count()
            + self.v.elem_count()
            + self.u1.elem_count()
            + self.u2.elem_count()
            + self.v1.elem_count()
            + self.v2.elem_count()
    }
}

struct RawDecomposedBranch {
    u: Vec<f32>,
    v: Vec<f32>,
    u1: Vec<f32>,
    u2: Vec<f32>,
    v1: Vec<f32>,
    v2: Vec<f32>,
}

fn decompose_matrix_host(
    weight_f32: &[f32],
    out_features: usize,
    in_features: usize,
    split_dim: usize,
    config: &LittleBitConfig,
    seed: u64,
) -> Result<RawDecomposedBranch> {
    let svd = svd_lowrank(
        weight_f32,
        out_features,
        in_features,
        split_dim,
        DEFAULT_LOWRANK_NITER,
        seed,
    )?;

    let r = svd.rank;
    let sqrt_s: Vec<f32> = svd.s.iter().map(|&s| s.max(0.0).sqrt()).collect();

    // U: (out_features, split_dim), zero-padded if r < split_dim.
    let mut u = vec![0.0f32; out_features * split_dim];
    for row in 0..out_features {
        for k in 0..r {
            u[row * split_dim + k] = svd.u[row * r + k] * sqrt_s[k];
        }
    }

    // V: (split_dim, in_features), zero-padded if r < split_dim.
    let mut v = vec![0.0f32; split_dim * in_features];
    for k in 0..r {
        let s_root = sqrt_s[k];
        for col in 0..in_features {
            v[k * in_features + col] = s_root * svd.vt[k * in_features + col];
        }
    }

    if config.use_itq {
        apply_joint_itq(
            &mut u,
            &mut v,
            out_features,
            in_features,
            split_dim,
            config.itq_n_iter,
            seed.wrapping_add(1),
        )?;
    }

    let abs_v: Vec<f32> = v.iter().map(|&x| x.abs()).collect();
    let (v1, v2) = rank_one_decompose(&abs_v, split_dim, in_features, seed.wrapping_add(2))?;

    let abs_u: Vec<f32> = u.iter().map(|&x| x.abs()).collect();
    let (u1, u2) = rank_one_decompose(&abs_u, out_features, split_dim, seed.wrapping_add(3))?;

    Ok(RawDecomposedBranch {
        u,
        v,
        u1,
        u2,
        v1,
        v2,
    })
}

fn compute_residual_matrix_host(
    orig_w: &[f32],
    branch: &RawDecomposedBranch,
    out_features: usize,
    in_features: usize,
    split_dim: usize,
) -> Vec<f32> {
    let u_scaled = Mat::from_fn(out_features, split_dim, |r, k| {
        let sign = if branch.u[r * split_dim + k] < 0.0 {
            -1.0f32
        } else {
            1.0f32
        };
        sign * (branch.u1[r] * branch.u2[k])
    });
    let v_scaled = Mat::from_fn(split_dim, in_features, |k, c| {
        let sign = if branch.v[k * in_features + c] < 0.0 {
            -1.0f32
        } else {
            1.0f32
        };
        sign * (branch.v1[k] * branch.v2[c])
    });
    let approx = &u_scaled * &v_scaled;
    let mut residual = Vec::with_capacity(orig_w.len());
    for r in 0..out_features {
        for c in 0..in_features {
            residual.push(orig_w[r * in_features + c] - approx[(r, c)]);
        }
    }
    residual
}

/// Sub-1-bit factorized linear layer matching upstream `quantization.modules.LittleBitLinear`.
#[derive(Debug, Clone)]
pub struct LittleBitLinear {
    in_features: usize,
    out_features: usize,
    budget: BitBudget,
    quant_func: QuantFunc,
    binarized: bool,
    main: FactorBranch,
    residual: Option<FactorBranch>,
    bias: Option<Param>,
}

impl LittleBitLinear {
    /// Decompose a dense weight `(out_features, in_features)` and optional bias `(out_features,)`
    /// into a `LittleBitLinear` module via low-rank SVD (and Joint-ITQ when configured).
    ///
    /// # Errors
    /// Returns an error on invalid config, non-2D weight, shape mismatch, or SVD failure.
    pub fn from_dense_weight(
        weight: &Tensor,
        bias: Option<&Tensor>,
        config: &LittleBitConfig,
        ratio_factor: f64,
        train_mode: TrainMode,
        seed: u64,
    ) -> Result<Self> {
        config.validate()?;
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(LittleBitError::InvalidShape {
                expected: "2D weight (out_features, in_features)".to_string(),
                actual: format!("{dims:?}"),
            });
        }
        let (out_features, in_features) = (dims[0], dims[1]);
        if let Some(b) = bias {
            if b.dims() != [out_features] {
                return Err(LittleBitError::InvalidShape {
                    expected: format!("[{out_features}]"),
                    actual: format!("{:?}", b.dims()),
                });
            }
        }

        let target_dtype = weight.dtype();
        let target_device = weight.device().clone();
        let budget = BitBudget::resolve(in_features, out_features, config, ratio_factor);

        let weight_f32 = weight
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape(out_features * in_features)?
            .to_vec1::<f32>()?;

        let main_raw = decompose_matrix_host(
            &weight_f32,
            out_features,
            in_features,
            budget.split_dim,
            config,
            seed,
        )?;

        let residual_branch = if budget.residual == ResidualMode::Dual {
            let residual_w = compute_residual_matrix_host(
                &weight_f32,
                &main_raw,
                out_features,
                in_features,
                budget.split_dim,
            );
            let res_raw = decompose_matrix_host(
                &residual_w,
                out_features,
                in_features,
                budget.split_dim,
                config,
                seed.wrapping_add(10_000),
            )?;
            Some(FactorBranch::from_raw(
                res_raw,
                target_dtype,
                &target_device,
                train_mode,
                out_features,
                in_features,
                budget.split_dim,
            )?)
        } else {
            None
        };

        let main = FactorBranch::from_raw(
            main_raw,
            target_dtype,
            &target_device,
            train_mode,
            out_features,
            in_features,
            budget.split_dim,
        )?;

        let bias_param = match bias {
            Some(b) => Some(Param::from_tensor(
                b.to_device(&target_device)?.to_dtype(target_dtype)?,
                train_mode,
            )?),
            None => None,
        };

        Ok(Self {
            in_features,
            out_features,
            budget,
            quant_func: config.quant_func,
            binarized: false,
            main,
            residual: residual_branch,
            bias: bias_param,
        })
    }

    /// Allocate zero-initialized factor tensors without performing SVD (for checkpoint loading).
    ///
    /// # Errors
    /// Returns an error if config validation or tensor allocation fails.
    pub fn empty(
        shape: LayerShape,
        config: &LittleBitConfig,
        ratio_factor: f64,
        train_mode: TrainMode,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let (in_features, out_features) = (shape.in_features, shape.out_features);
        let budget = BitBudget::resolve(in_features, out_features, config, ratio_factor);
        let main = FactorBranch::empty(
            out_features,
            in_features,
            budget.split_dim,
            dtype,
            device,
            train_mode,
        )?;
        let residual = if budget.residual == ResidualMode::Dual {
            Some(FactorBranch::empty(
                out_features,
                in_features,
                budget.split_dim,
                dtype,
                device,
                train_mode,
            )?)
        } else {
            None
        };
        let bias = if shape.bias == BiasMode::WithBias {
            Some(Param::from_tensor(
                Tensor::zeros(out_features, dtype, device)?,
                train_mode,
            )?)
        } else {
            None
        };
        Ok(Self {
            in_features,
            out_features,
            budget,
            quant_func: config.quant_func,
            binarized: false,
            main,
            residual,
            bias,
        })
    }

    /// Reconstruct a [`LittleBitLinear`] layer from an unpacked state dictionary.
    ///
    /// # Errors
    /// Returns [`LittleBitError::MissingTensor`] or shape mismatch errors if any factor tensor
    /// is absent or malformed.
    #[allow(clippy::too_many_arguments)]
    pub fn from_state_dict(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        shape: LayerShape,
        config: &LittleBitConfig,
        ratio_factor: f64,
        weight_state: WeightState,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let (in_features, out_features) = (shape.in_features, shape.out_features);
        let mut budget = BitBudget::resolve(in_features, out_features, config, ratio_factor);

        let load_tensor = |suffix: &str| -> Result<Tensor> {
            let key = format!("{prefix}.{suffix}");
            let t = tensors
                .get(&key)
                .ok_or_else(|| LittleBitError::MissingTensor(key.clone()))?;
            let t = if t.device().same_device(device) {
                t.clone()
            } else {
                t.to_device(device)?
            };
            if t.dtype() == dtype {
                Ok(t)
            } else {
                Ok(t.to_dtype(dtype)?)
            }
        };

        let u = load_tensor("U")?;
        let v = load_tensor("V")?;
        let actual_split =
            u.dims()
                .get(1)
                .copied()
                .ok_or_else(|| LittleBitError::InvalidShape {
                    expected: "2D U tensor".to_string(),
                    actual: format!("{:?}", u.dims()),
                })?;
        if actual_split != budget.split_dim {
            budget.split_dim = actual_split;
            budget.eff_bit_actual = crate::config::compute_eff_bits(
                in_features,
                out_features,
                actual_split,
                budget.residual,
            );
            budget.total_bit_usage =
                budget.eff_bit_actual * (in_features as f64) * (out_features as f64);
        }

        let main = FactorBranch {
            u: Param::Frozen(u),
            v: Param::Frozen(v),
            u1: Param::Frozen(load_tensor("u1")?),
            u2: Param::Frozen(load_tensor("u2")?),
            v1: Param::Frozen(load_tensor("v1")?),
            v2: Param::Frozen(load_tensor("v2")?),
        };

        let residual = if budget.residual == ResidualMode::Dual {
            Some(FactorBranch {
                u: Param::Frozen(load_tensor("U_R")?),
                v: Param::Frozen(load_tensor("V_R")?),
                u1: Param::Frozen(load_tensor("u1_R")?),
                u2: Param::Frozen(load_tensor("u2_R")?),
                v1: Param::Frozen(load_tensor("v1_R")?),
                v2: Param::Frozen(load_tensor("v2_R")?),
            })
        } else {
            None
        };

        let bias = if shape.bias == BiasMode::WithBias {
            let b_key = format!("{prefix}.bias");
            if tensors.contains_key(&b_key) {
                Some(Param::Frozen(load_tensor("bias")?))
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            budget,
            quant_func: config.quant_func,
            binarized: weight_state == WeightState::PreBinarized,
            main,
            residual,
            bias,
        })
    }

    /// Forward pass accepting any leading shape `[..., in_features]` and returning `[..., out_features]`.
    ///
    /// # Errors
    /// Returns an error if input trailing dimension differs from `in_features` or tensor ops fail.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.is_empty() || dims[dims.len() - 1] != self.in_features {
            return Err(LittleBitError::InvalidShape {
                expected: format!("[..., {}]", self.in_features),
                actual: format!("{dims:?}"),
            });
        }

        let mut out_dims = dims[..dims.len() - 1].to_vec();
        let batch_rows: usize = out_dims.iter().product::<usize>().max(1);
        out_dims.push(self.out_features);

        let x_2d = x.reshape((batch_rows, self.in_features))?;
        let mut y = self.main.forward(&x_2d, self.quant_func, self.binarized)?;
        if let Some(res) = &self.residual {
            let r = res.forward(&x_2d, self.quant_func, self.binarized)?;
            y = y.add(&r)?;
        }
        if let Some(b) = &self.bias {
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

    /// Binarize and pack `U`, `V` (and `U_R`, `V_R` when dual) into upstream safetensors keys.
    ///
    /// # Errors
    /// Returns an error if binarization or packing fails.
    pub fn pack_state_dict(&self, prefix: &str) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        let mut insert_branch = |branch: &FactorBranch, suffix: &str| -> Result<()> {
            let (u_packed, u_shape) = binary_packer(branch.u.as_tensor())?;
            let (v_packed, v_shape) = binary_packer(branch.v.as_tensor())?;
            out.insert(format!("{prefix}.U{suffix}_packed"), u_packed);
            out.insert(format!("{prefix}.U{suffix}_shape"), u_shape);
            out.insert(format!("{prefix}.V{suffix}_packed"), v_packed);
            out.insert(format!("{prefix}.V{suffix}_shape"), v_shape);
            out.insert(
                format!("{prefix}.u1{suffix}"),
                branch.u1.as_tensor().detach(),
            );
            out.insert(
                format!("{prefix}.u2{suffix}"),
                branch.u2.as_tensor().detach(),
            );
            out.insert(
                format!("{prefix}.v1{suffix}"),
                branch.v1.as_tensor().detach(),
            );
            out.insert(
                format!("{prefix}.v2{suffix}"),
                branch.v2.as_tensor().detach(),
            );
            Ok(())
        };

        insert_branch(&self.main, "")?;
        if let Some(res) = &self.residual {
            insert_branch(res, "_R")?;
        }
        if let Some(b) = &self.bias {
            out.insert(format!("{prefix}.bias"), b.as_tensor().detach());
        }

        let dev = self.main.u.as_tensor().device();
        let target_val = self.budget.eff_bit_target.unwrap_or(-1.0) as f32;
        out.insert(
            format!("{prefix}._eff_bit_target"),
            Tensor::from_slice(&[target_val], (), dev)?,
        );
        out.insert(
            format!("{prefix}._split_dim_final"),
            Tensor::from_slice(&[self.budget.split_dim as i64], (), dev)?,
        );
        out.insert(
            format!("{prefix}._eff_bit_actual"),
            Tensor::from_slice(&[self.budget.eff_bit_actual as f32], (), dev)?,
        );
        Ok(out)
    }

    /// Convert this layer into a memory-compact 1-bit [`PackedLittleBitLinear`] runtime module.
    ///
    /// # Errors
    /// Returns an error if binarization or host packing fails.
    pub fn to_packed(&self) -> Result<PackedLittleBitLinear> {
        let pack_branch = |branch: &FactorBranch| -> Result<PackedFactorBranch> {
            let (u_packed_t, _) = binary_packer(branch.u.as_tensor())?;
            let (v_packed_t, _) = binary_packer(branch.v.as_tensor())?;
            let u_words = u_packed_t
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<i32>()?;
            let v_words = v_packed_t
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<i32>()?;
            let u =
                PackedBinaryMatrix::from_words(self.out_features, self.budget.split_dim, u_words)?;
            let v =
                PackedBinaryMatrix::from_words(self.budget.split_dim, self.in_features, v_words)?;
            let to_vec_f32 = |t: &Tensor| -> Result<Vec<f32>> {
                Ok(t.to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?)
            };
            let u1 = to_vec_f32(branch.u1.as_tensor())?;
            let u2 = to_vec_f32(branch.u2.as_tensor())?;
            let v1 = to_vec_f32(branch.v1.as_tensor())?;
            let v2 = to_vec_f32(branch.v2.as_tensor())?;
            let v1u2: Vec<f32> = v1.iter().zip(u2.iter()).map(|(&a, &b)| a * b).collect();
            Ok(PackedFactorBranch {
                u,
                v,
                u1,
                u2,
                v1,
                v2,
                v1u2,
            })
        };

        let main = pack_branch(&self.main)?;
        let residual = match &self.residual {
            Some(r) => Some(pack_branch(r)?),
            None => None,
        };
        let bias = match &self.bias {
            Some(b) => Some(
                b.as_tensor()
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            ),
            None => None,
        };

        Ok(PackedLittleBitLinear {
            in_features: self.in_features,
            out_features: self.out_features,
            budget: self.budget,
            main,
            residual,
            bias,
        })
    }

    /// Input feature dimension.
    #[must_use]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature dimension.
    #[must_use]
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Resolved bit budget and latent rank geometry.
    #[must_use]
    pub fn budget(&self) -> &BitBudget {
        &self.budget
    }

    /// Selected latent rank `split_dim`.
    #[must_use]
    pub fn split_dim(&self) -> usize {
        self.budget.split_dim
    }

    /// Whether weights are already pre-binarized (`_binarized` inference mode).
    #[must_use]
    pub fn is_binarized(&self) -> bool {
        self.binarized
    }

    /// Switch module to pre-binarized inference mode, converting `U` and `V` in place to `{-1, +1}`.
    ///
    /// # Errors
    /// Returns an error if binarization fails.
    pub fn binarize_in_place(&mut self) -> Result<()> {
        let bin_branch = |branch: &mut FactorBranch| -> Result<()> {
            branch.u = Param::Frozen(binarize_sign_no_grad(branch.u.as_tensor())?);
            branch.v = Param::Frozen(binarize_sign_no_grad(branch.v.as_tensor())?);
            branch.freeze();
            Ok(())
        };
        bin_branch(&mut self.main)?;
        if let Some(res) = &mut self.residual {
            bin_branch(res)?;
        }
        if let Some(b) = &mut self.bias {
            b.freeze();
        }
        self.binarized = true;
        Ok(())
    }

    /// Collect all trainable [`Var`] parameters inside this layer.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = Vec::new();
        self.main.push_vars(&mut vars);
        if let Some(res) = &self.residual {
            res.push_vars(&mut vars);
        }
        if let Some(b) = &self.bias {
            if let Some(v) = b.as_var() {
                vars.push(v.clone());
            }
        }
        vars
    }

    /// Freeze all trainable parameters in this layer.
    pub fn freeze(&mut self) {
        self.main.freeze();
        if let Some(res) = &mut self.residual {
            res.freeze();
        }
        if let Some(b) = &mut self.bias {
            b.freeze();
        }
    }

    /// Promote a finished/pre-binarized layer back into trainable latent QAT factors by scaling
    /// binary signs `U`/`V` (and residual `U_R`/`V_R`) by `latent_mag` (`--warm_mag`), promoting
    /// scales and bias to trainable `Var`s, and disabling `binarized` mode.
    ///
    /// Matches `littlebit-bench/patches/volta-fp16.patch::warm_start`.
    ///
    /// # Errors
    /// Returns [`LittleBitError::InvalidConfig`] when `latent_mag` is not finite and positive, or
    /// Candle tensor/variable errors if scaling or allocation fails.
    pub fn warm_start_from_binarized(&mut self, latent_mag: f64) -> Result<()> {
        if !latent_mag.is_finite() || latent_mag <= 0.0 {
            return Err(LittleBitError::InvalidConfig(format!(
                "warm_start latent_mag must be finite and positive, got {latent_mag}"
            )));
        }
        self.main.warm_start(latent_mag)?;
        if let Some(res) = &mut self.residual {
            res.warm_start(latent_mag)?;
        }
        if let Some(b) = &mut self.bias {
            *b = Param::Trainable(Var::from_tensor(&b.as_tensor().detach())?);
        }
        self.binarized = false;
        Ok(())
    }

    /// Total scalar parameter elements stored in this layer.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.main.elem_count()
            + self.residual.as_ref().map_or(0, FactorBranch::elem_count)
            + self.bias.as_ref().map_or(0, Param::elem_count)
    }

    /// Borrow the main latent branch.
    #[must_use]
    pub fn main_branch(&self) -> &FactorBranch {
        &self.main
    }

    /// Borrow the optional residual latent branch.
    #[must_use]
    pub fn residual_branch(&self) -> Option<&FactorBranch> {
        self.residual.as_ref()
    }

    /// Borrow the optional bias parameter.
    #[must_use]
    pub fn bias(&self) -> Option<&Param> {
        self.bias.as_ref()
    }

    /// Format module summary string matching upstream `LittleBitLinear.extra_repr()`.
    #[must_use]
    pub fn extra_repr(&self) -> String {
        let target = match self.budget.eff_bit_target {
            Some(v) => format!("{v:.4}"),
            None => "N/A".to_string(),
        };
        format!(
            "in_features={}, out_features={}, bias={}, split_dim={}, eff_bit_target={}, eff_bit_actual={:.4}, residual={}, total_bit_usage={:.0}",
            self.in_features,
            self.out_features,
            self.bias.is_some(),
            self.budget.split_dim,
            target,
            self.budget.eff_bit_actual,
            self.budget.residual == ResidualMode::Dual,
            self.budget.total_bit_usage,
        )
    }
}

impl fmt::Display for LittleBitLinear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LittleBitLinear({})", self.extra_repr())
    }
}

#[derive(Debug, Clone)]
struct PackedFactorBranch {
    u: PackedBinaryMatrix,
    v: PackedBinaryMatrix,
    u1: Vec<f32>,
    u2: Vec<f32>,
    v1: Vec<f32>,
    v2: Vec<f32>,
    v1u2: Vec<f32>,
}

impl PackedFactorBranch {
    fn forward_f32(&self, x: &[f32], batch_rows: usize, in_features: usize) -> Result<Vec<f32>> {
        let split_dim = self.v.rows();
        let out_features = self.u.rows();

        let mut x_scaled = Vec::with_capacity(batch_rows * in_features);
        for row in x.chunks_exact(in_features) {
            for (&xv, &sv) in row.iter().zip(self.v2.iter()) {
                x_scaled.push(xv * sv);
            }
        }

        let mut hidden = self.v.matmul_transposed_f32(&x_scaled, batch_rows)?;
        for h_row in hidden.chunks_exact_mut(split_dim) {
            for (hv, &sv) in h_row.iter_mut().zip(self.v1u2.iter()) {
                *hv *= sv;
            }
        }

        let mut out = self.u.matmul_transposed_f32(&hidden, batch_rows)?;
        for o_row in out.chunks_exact_mut(out_features) {
            for (ov, &sv) in o_row.iter_mut().zip(self.u1.iter()) {
                *ov *= sv;
            }
        }
        Ok(out)
    }

    fn memory_bytes(&self) -> usize {
        self.u.memory_bytes()
            + self.v.memory_bytes()
            + (self.u1.len() + self.u2.len() + self.v1.len() + self.v2.len() + self.v1u2.len())
                * std::mem::size_of::<f32>()
    }
}

/// Memory-compact 1-bit inference representation keeping `U`, `V` packed as `i32` bitfields.
#[derive(Debug, Clone)]
pub struct PackedLittleBitLinear {
    in_features: usize,
    out_features: usize,
    budget: BitBudget,
    main: PackedFactorBranch,
    residual: Option<PackedFactorBranch>,
    bias: Option<Vec<f32>>,
}

impl PackedLittleBitLinear {
    /// Construct directly from packed checkpoint tensors without allocating dense float matrices.
    ///
    /// # Errors
    /// Returns [`LittleBitError::MissingTensor`] or shape mismatch errors if packed entries
    /// are missing or invalid.
    pub fn from_packed_tensors(
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
        shape: LayerShape,
        config: &LittleBitConfig,
        ratio_factor: f64,
    ) -> Result<Self> {
        config.validate()?;
        let (in_features, out_features) = (shape.in_features, shape.out_features);
        let mut budget = BitBudget::resolve(in_features, out_features, config, ratio_factor);

        let load_vec_f32 = |suffix: &str| -> Result<Vec<f32>> {
            let key = format!("{prefix}.{suffix}");
            let t = tensors
                .get(&key)
                .ok_or_else(|| LittleBitError::MissingTensor(key.clone()))?;
            Ok(t.to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?)
        };

        let load_packed = |name: &str| -> Result<(usize, usize, Vec<i32>)> {
            let p_key = format!("{prefix}.{name}_packed");
            let s_key = format!("{prefix}.{name}_shape");
            let packed_t = tensors
                .get(&p_key)
                .ok_or_else(|| LittleBitError::MissingTensor(p_key.clone()))?;
            let shape_t = tensors
                .get(&s_key)
                .ok_or_else(|| LittleBitError::MissingTensor(s_key.clone()))?;
            let shape_vals = shape_t
                .to_device(&Device::Cpu)?
                .to_dtype(DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?;
            if shape_vals.len() != 2 || shape_vals[0] < 0 || shape_vals[1] < 0 {
                return Err(LittleBitError::InvalidShape {
                    expected: "non-negative [rows, cols]".to_string(),
                    actual: format!("{shape_vals:?}"),
                });
            }
            let (rows, cols) = (shape_vals[0] as usize, shape_vals[1] as usize);
            let words = packed_t
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<i32>()?;
            Ok((rows, cols, words))
        };

        let load_branch = |suffix: &str| -> Result<(usize, PackedFactorBranch)> {
            let (u_rows, u_cols, u_words) = load_packed(&format!("U{suffix}"))?;
            let (v_rows, v_cols, v_words) = load_packed(&format!("V{suffix}"))?;
            if u_rows != out_features || v_cols != in_features || u_cols != v_rows {
                return Err(LittleBitError::InvalidShape {
                    expected: format!("U=({out_features}, s), V=(s, {in_features})"),
                    actual: format!("U=({u_rows}, {u_cols}), V=({v_rows}, {v_cols})"),
                });
            }
            let u = PackedBinaryMatrix::from_words(u_rows, u_cols, u_words)?;
            let v = PackedBinaryMatrix::from_words(v_rows, v_cols, v_words)?;
            let u1 = load_vec_f32(&format!("u1{suffix}"))?;
            let u2 = load_vec_f32(&format!("u2{suffix}"))?;
            let v1 = load_vec_f32(&format!("v1{suffix}"))?;
            let v2 = load_vec_f32(&format!("v2{suffix}"))?;
            let v1u2: Vec<f32> = v1.iter().zip(u2.iter()).map(|(&a, &b)| a * b).collect();
            Ok((
                u_cols,
                PackedFactorBranch {
                    u,
                    v,
                    u1,
                    u2,
                    v1,
                    v2,
                    v1u2,
                },
            ))
        };

        let (actual_split, main) = load_branch("")?;
        if actual_split != budget.split_dim {
            budget.split_dim = actual_split;
            budget.eff_bit_actual = crate::config::compute_eff_bits(
                in_features,
                out_features,
                actual_split,
                budget.residual,
            );
            budget.total_bit_usage =
                budget.eff_bit_actual * (in_features as f64) * (out_features as f64);
        }

        let residual = if budget.residual == ResidualMode::Dual {
            let (_, res_branch) = load_branch("_R")?;
            Some(res_branch)
        } else {
            None
        };

        let bias = if shape.bias == BiasMode::WithBias
            && tensors.contains_key(&format!("{prefix}.bias"))
        {
            Some(load_vec_f32("bias")?)
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            budget,
            main,
            residual,
            bias,
        })
    }

    /// Execute direct 1-bit packed forward pass and return result on `x`'s device and dtype.
    ///
    /// # Errors
    /// Returns an error if input dimensions do not match `in_features` or tensor transfer fails.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.is_empty() || dims[dims.len() - 1] != self.in_features {
            return Err(LittleBitError::InvalidShape {
                expected: format!("[..., {}]", self.in_features),
                actual: format!("{dims:?}"),
            });
        }
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        let batch_rows: usize = out_dims.iter().product::<usize>().max(1);
        out_dims.push(self.out_features);

        let x_flat = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .reshape(batch_rows * self.in_features)?
            .to_vec1::<f32>()?;

        let mut out = self
            .main
            .forward_f32(&x_flat, batch_rows, self.in_features)?;
        if let Some(res) = &self.residual {
            let r = res.forward_f32(&x_flat, batch_rows, self.in_features)?;
            for (dst, rv) in out.iter_mut().zip(r.iter()) {
                *dst += *rv;
            }
        }
        if let Some(bias) = &self.bias {
            for row in out.chunks_exact_mut(self.out_features) {
                for (dst, &bv) in row.iter_mut().zip(bias.iter()) {
                    *dst += bv;
                }
            }
        }

        let out_t = Tensor::from_vec(out, out_dims, x.device())?;
        if x.dtype() == DType::F32 {
            Ok(out_t)
        } else {
            Ok(out_t.to_dtype(x.dtype())?)
        }
    }

    /// Export packed tensors for checkpoint serialization.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn pack_state_dict(
        &self,
        prefix: &str,
        device: &Device,
    ) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        let mut insert_branch = |branch: &PackedFactorBranch, suffix: &str| -> Result<()> {
            let (u_packed, u_shape) = branch.u.to_tensors(device)?;
            let (v_packed, v_shape) = branch.v.to_tensors(device)?;
            out.insert(format!("{prefix}.U{suffix}_packed"), u_packed);
            out.insert(format!("{prefix}.U{suffix}_shape"), u_shape);
            out.insert(format!("{prefix}.V{suffix}_packed"), v_packed);
            out.insert(format!("{prefix}.V{suffix}_shape"), v_shape);
            out.insert(
                format!("{prefix}.u1{suffix}"),
                Tensor::from_slice(&branch.u1, (1, self.out_features), device)?,
            );
            out.insert(
                format!("{prefix}.u2{suffix}"),
                Tensor::from_slice(&branch.u2, (1, self.budget.split_dim), device)?,
            );
            out.insert(
                format!("{prefix}.v1{suffix}"),
                Tensor::from_slice(&branch.v1, (1, self.budget.split_dim), device)?,
            );
            out.insert(
                format!("{prefix}.v2{suffix}"),
                Tensor::from_slice(&branch.v2, (1, self.in_features), device)?,
            );
            Ok(())
        };

        insert_branch(&self.main, "")?;
        if let Some(res) = &self.residual {
            insert_branch(res, "_R")?;
        }
        if let Some(b) = &self.bias {
            out.insert(
                format!("{prefix}.bias"),
                Tensor::from_slice(b, self.out_features, device)?,
            );
        }
        let target_val = self.budget.eff_bit_target.unwrap_or(-1.0) as f32;
        out.insert(
            format!("{prefix}._eff_bit_target"),
            Tensor::from_slice(&[target_val], (), device)?,
        );
        out.insert(
            format!("{prefix}._split_dim_final"),
            Tensor::from_slice(&[self.budget.split_dim as i64], (), device)?,
        );
        out.insert(
            format!("{prefix}._eff_bit_actual"),
            Tensor::from_slice(&[self.budget.eff_bit_actual as f32], (), device)?,
        );
        Ok(out)
    }

    /// Total heap memory footprint in bytes for the 1-bit packed layer.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.main.memory_bytes()
            + self
                .residual
                .as_ref()
                .map_or(0, PackedFactorBranch::memory_bytes)
            + self
                .bias
                .as_ref()
                .map_or(0, |b| b.len() * std::mem::size_of::<f32>())
    }

    /// Resolved bit budget and rank geometry.
    #[must_use]
    pub fn budget(&self) -> &BitBudget {
        &self.budget
    }
}

/// Reconstruct dense approximated weight `(out_features, in_features)` from `U` and `V` factors
/// for diagnostics and error evaluation.
///
/// # Errors
/// Returns an error on shape mismatch.
pub fn reconstruct_branch_matrix(
    u: &Tensor,
    v: &Tensor,
    u1: &Tensor,
    u2: &Tensor,
    v1: &Tensor,
    v2: &Tensor,
    quant_func: QuantFunc,
) -> Result<Tensor> {
    let uq = quant_func.apply(&u.to_dtype(DType::F32)?)?;
    let vq = quant_func.apply(&v.to_dtype(DType::F32)?)?;
    let u_scale = u1
        .to_dtype(DType::F32)?
        .t()?
        .matmul(&u2.to_dtype(DType::F32)?)?;
    let v_scale = v1
        .to_dtype(DType::F32)?
        .t()?
        .matmul(&v2.to_dtype(DType::F32)?)?;
    Ok(uq.mul(&u_scale)?.matmul(&vq.mul(&v_scale)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_tensor(rows: usize, cols: usize, seed: u64) -> Result<Tensor> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        Ok(Tensor::from_vec(data, (rows, cols), &Device::Cpu)?)
    }

    #[test]
    fn forward_matches_packed_and_direct_matrix_product() -> Result<()> {
        let out_f = 48;
        let in_f = 64;
        let w = random_tensor(out_f, in_f, 7)?;
        let bias = Tensor::from_vec(
            (0..out_f).map(|i| (i as f32) * 0.01 - 0.2).collect(),
            out_f,
            &Device::Cpu,
        )?;
        let cfg = LittleBitConfig {
            quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
            eff_bit: Some(0.5),
            residual: true,
            use_itq: true,
            itq_n_iter: 20,
            ..Default::default()
        };

        let layer =
            LittleBitLinear::from_dense_weight(&w, Some(&bias), &cfg, 1.0, TrainMode::Eval, 11)?;
        let packed = layer.to_packed()?;

        assert!(packed.memory_bytes() < out_f * in_f * 4);

        let x = random_tensor(6, in_f, 99)?.reshape((2, 3, in_f))?;
        let y_layer = layer.forward(&x)?;
        let y_packed = packed.forward(&x)?;
        assert_eq!(y_layer.dims(), &[2, 3, out_f]);
        assert_eq!(y_packed.dims(), &[2, 3, out_f]);

        // Compare against explicit matrix reconstruction W_hat = W_main + W_res.
        let main = layer.main_branch();
        let res = layer.residual_branch().expect("residual branch");
        let w_main = reconstruct_branch_matrix(
            main.u.as_tensor(),
            main.v.as_tensor(),
            main.u1.as_tensor(),
            main.u2.as_tensor(),
            main.v1.as_tensor(),
            main.v2.as_tensor(),
            cfg.quant_func,
        )?;
        let w_res = reconstruct_branch_matrix(
            res.u.as_tensor(),
            res.v.as_tensor(),
            res.u1.as_tensor(),
            res.u2.as_tensor(),
            res.v1.as_tensor(),
            res.v2.as_tensor(),
            cfg.quant_func,
        )?;
        let w_hat = w_main.add(&w_res)?;
        let y_ref = x
            .reshape((6, in_f))?
            .matmul(&w_hat.t()?)?
            .broadcast_add(&bias)?
            .reshape((2, 3, out_f))?;

        let diff_packed = y_layer
            .sub(&y_packed)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let diff_ref = y_layer.sub(&y_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff_packed < 1e-4, "packed drift: {diff_packed}");
        assert!(diff_ref < 1e-4, "reference drift: {diff_ref}");
        Ok(())
    }

    #[test]
    fn autograd_flows_through_all_factors_for_both_quant_funcs() -> Result<()> {
        for qf in [QuantFunc::SteBinary, QuantFunc::SmoothSign { alpha: 50.0 }] {
            let w = random_tensor(32, 32, 19)?;
            let bias = Tensor::zeros(32, DType::F32, &Device::Cpu)?;
            let cfg = LittleBitConfig {
                quant_func: qf,
                eff_bit: None,
                split_dim: 8,
                residual: true,
                use_itq: true,
                itq_n_iter: 10,
                ..Default::default()
            };
            let layer = LittleBitLinear::from_dense_weight(
                &w,
                Some(&bias),
                &cfg,
                1.0,
                TrainMode::Train,
                3,
            )?;
            let x = random_tensor(4, 32, 23)?;
            let loss = layer.forward(&x)?.sqr()?.mean_all()?;
            let grads = loss.backward()?;

            let vars = layer.trainable_vars();
            assert_eq!(vars.len(), 13);
            for var in vars {
                let g = grads
                    .get(var.as_tensor())
                    .expect("every factor var must receive a gradient");
                let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()?;
                assert!(norm.is_finite() && norm > 0.0, "zero or NaN grad: {norm}");
            }
        }
        Ok(())
    }
}
