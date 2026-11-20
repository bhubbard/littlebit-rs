//! Binary activation/weight quantization functions (`STEBinary` and `SmoothSign`).

use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result as CandleResult, Shape, Tensor};
use half::{bf16, f16};

use crate::config::QuantFunc;
use crate::error::Result;

/// Binarize a floating-point tensor without tracking gradients:
/// values `>= 0` map to `+1`, values `< 0` map to `-1`.
///
/// Matches PyTorch `y = x.sign(); y[y == 0] = 1`.
///
/// # Errors
/// Returns a [`crate::LittleBitError::Candle`] error if the tensor operation fails.
pub fn binarize_sign_no_grad(x: &Tensor) -> Result<Tensor> {
    let x = x.detach();
    let zero = x.zeros_like()?;
    let ones = x.ones_like()?;
    let neg_ones = ones.neg()?;
    Ok(x.ge(&zero)?.where_cond(&ones, &neg_ones)?)
}

/// Apply `STEBinary` (`sign` forward with zero mapped to `+1`, clipped identity backward for `|x| < 1`).
///
/// # Errors
/// Returns an error if the tensor dtype is unsupported or an internal Candle op fails.
pub fn ste_binary(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() {
        let contiguous = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        Ok(contiguous.apply_op1(SteBinaryOp)?)
    } else {
        // Portable straight-through estimator on GPU backends.
        let y_hard = binarize_sign_no_grad(x)?;
        let y_soft = x.clamp(-1.0f64, 1.0f64)?;
        Ok((y_hard.sub(&y_soft)?.detach() + y_soft)?)
    }
}

/// Apply `SmoothSign` (`sign` forward with zero mapped to `+1`,
/// derivative `alpha * (1 - tanh(alpha * x)^2)` backward).
///
/// # Errors
/// Returns an error if the tensor dtype is unsupported or an internal Candle op fails.
pub fn smooth_sign(x: &Tensor, alpha: f64) -> Result<Tensor> {
    if x.device().is_cpu() {
        let contiguous = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        Ok(contiguous.apply_op1(SmoothSignOp { alpha })?)
    } else {
        let work_dtype = if x.dtype() == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let y_hard = binarize_sign_no_grad(x)?;
        let y_soft = (x.to_dtype(work_dtype)? * alpha)?
            .tanh()?
            .to_dtype(x.dtype())?;
        Ok((y_hard.sub(&y_soft)?.detach() + y_soft)?)
    }
}

impl QuantFunc {
    /// Apply this quantization function to `x`, preserving differentiable gradients.
    ///
    /// # Errors
    /// Returns an error if the tensor dtype or backend operation fails.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        match *self {
            Self::SteBinary => ste_binary(x),
            Self::SmoothSign { alpha } => smooth_sign(x, alpha),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SteBinaryOp;

#[derive(Debug, Clone, Copy)]
struct SmoothSignOp {
    alpha: f64,
}

#[inline]
fn sign_plus_one_f64(v: f64) -> f64 {
    if v.is_nan() {
        f64::NAN
    } else if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

#[inline]
fn sign_plus_one_f32(v: f32) -> f32 {
    if v.is_nan() {
        f32::NAN
    } else if v < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn strided_offsets(layout: &Layout) -> Vec<usize> {
    if let Some((start, end)) = layout.contiguous_offsets() {
        return (start..end).collect();
    }
    let dims = layout.dims();
    let stride = layout.stride();
    let total = layout.shape().elem_count();
    let mut offsets = Vec::with_capacity(total);
    if total == 0 {
        return offsets;
    }
    if dims.is_empty() {
        offsets.push(layout.start_offset());
        return offsets;
    }
    let mut coords = vec![0usize; dims.len()];
    for _ in 0..total {
        let mut idx = layout.start_offset();
        for (c, &s) in coords.iter().zip(stride.iter()) {
            idx += c * s;
        }
        offsets.push(idx);
        for d in (0..dims.len()).rev() {
            coords[d] += 1;
            if coords[d] < dims[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    offsets
}

fn cpu_sign_forward(storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
    let shape = layout.shape().clone();
    if let Some((start, end)) = layout.contiguous_offsets() {
        let out = match storage {
            CpuStorage::F32(vs) => CpuStorage::F32(
                vs[start..end]
                    .iter()
                    .map(|&v| sign_plus_one_f32(v))
                    .collect(),
            ),
            CpuStorage::F64(vs) => CpuStorage::F64(
                vs[start..end]
                    .iter()
                    .map(|&v| sign_plus_one_f64(v))
                    .collect(),
            ),
            CpuStorage::F16(vs) => CpuStorage::F16(
                vs[start..end]
                    .iter()
                    .map(|&v| f16::from_f32(sign_plus_one_f32(v.to_f32())))
                    .collect(),
            ),
            CpuStorage::BF16(vs) => CpuStorage::BF16(
                vs[start..end]
                    .iter()
                    .map(|&v| bf16::from_f32(sign_plus_one_f32(v.to_f32())))
                    .collect(),
            ),
            other => {
                candle_core::bail!(
                    "binary quantization requires floating-point storage, got {:?}",
                    other
                );
            }
        };
        return Ok((out, shape));
    }

    let offsets = strided_offsets(layout);
    let out = match storage {
        CpuStorage::F32(vs) => {
            CpuStorage::F32(offsets.iter().map(|&i| sign_plus_one_f32(vs[i])).collect())
        }
        CpuStorage::F64(vs) => {
            CpuStorage::F64(offsets.iter().map(|&i| sign_plus_one_f64(vs[i])).collect())
        }
        CpuStorage::F16(vs) => CpuStorage::F16(
            offsets
                .iter()
                .map(|&i| f16::from_f32(sign_plus_one_f32(vs[i].to_f32())))
                .collect(),
        ),
        CpuStorage::BF16(vs) => CpuStorage::BF16(
            offsets
                .iter()
                .map(|&i| bf16::from_f32(sign_plus_one_f32(vs[i].to_f32())))
                .collect(),
        ),
        other => {
            candle_core::bail!(
                "binary quantization requires floating-point storage, got {:?}",
                other
            );
        }
    };
    Ok((out, shape))
}

impl CustomOp1 for SteBinaryOp {
    fn name(&self) -> &'static str {
        "littlebit_ste_binary"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        cpu_sign_forward(storage, layout)
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> CandleResult<Option<Tensor>> {
        let work_dtype = if arg.dtype() == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let x = arg.to_dtype(work_dtype)?;
        let pos_one = x.ones_like()?;
        let neg_one = pos_one.neg()?;
        let mask = x
            .gt(&neg_one)?
            .mul(&x.lt(&pos_one)?)?
            .to_dtype(grad_res.dtype())?;
        Ok(Some(grad_res.mul(&mask)?))
    }
}

impl CustomOp1 for SmoothSignOp {
    fn name(&self) -> &'static str {
        "littlebit_smooth_sign"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CandleResult<(CpuStorage, Shape)> {
        cpu_sign_forward(storage, layout)
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> CandleResult<Option<Tensor>> {
        let work_dtype = if arg.dtype() == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let x = arg.to_dtype(work_dtype)?;
        let t = (x * self.alpha)?.tanh()?;
        let one_minus_t2 = (t.sqr()?.neg()? + 1.0)?;
        let deriv = (one_minus_t2 * self.alpha)?.to_dtype(grad_res.dtype())?;
        Ok(Some(grad_res.mul(&deriv)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn forward_maps_zero_and_neg_zero_to_plus_one() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[-2.5f32, -1e-6, -0.0, 0.0, 1e-6, 3.0], 6, &dev)?;
        let y_ste = ste_binary(&x)?.to_vec1::<f32>()?;
        let y_smooth = smooth_sign(&x, 100.0)?.to_vec1::<f32>()?;
        let want = vec![-1.0f32, -1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(y_ste, want);
        assert_eq!(y_smooth, want);
        Ok(())
    }

    #[test]
    fn ste_binary_backward_clips_outside_unit_interval() -> Result<()> {
        let dev = Device::Cpu;
        let x = Var::from_slice(&[-1.5f32, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], 7, &dev)?;
        let weights = Tensor::from_slice(&[2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 7, &dev)?;
        let y = ste_binary(x.as_tensor())?;
        let loss = (y * weights)?.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads
            .get(x.as_tensor())
            .expect("x gradient")
            .to_vec1::<f32>()?;
        assert_eq!(grad, vec![0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn smooth_sign_backward_matches_analytical_tanh_derivative() -> Result<()> {
        let dev = Device::Cpu;
        let alpha = 5.0_f64;
        let vals = [-0.8f32, -0.2, 0.0, 0.3, 1.1];
        let g_out = [1.5f32, -2.0, 0.75, 4.0, -0.5];

        let x = Var::from_slice(&vals, vals.len(), &dev)?;
        let w = Tensor::from_slice(&g_out, g_out.len(), &dev)?;
        let y = smooth_sign(x.as_tensor(), alpha)?;
        let loss = (y * w)?.sum_all()?;
        let grads = loss.backward()?;
        let grad = grads
            .get(x.as_tensor())
            .expect("x gradient")
            .to_vec1::<f32>()?;

        for i in 0..vals.len() {
            let t = (alpha * vals[i] as f64).tanh();
            let expected = (g_out[i] as f64 * alpha * (1.0 - t * t)) as f32;
            assert!(
                (grad[i] - expected).abs() < 1e-5,
                "idx {i}: got {}, want {expected}",
                grad[i]
            );
        }
        Ok(())
    }
}
