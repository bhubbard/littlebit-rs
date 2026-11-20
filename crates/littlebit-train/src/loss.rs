//! Knowledge distillation KL and layer-to-layer hidden-state MSE losses (`utils/kd_utils.py`).

use candle_core::{DType, Tensor, D};

use littlebit_models::CausalLmOutput;

use crate::error::{Result, TrainError};

/// Scalar + differentiable breakdown returned by [`compute_kd_loss`].
#[derive(Debug)]
pub struct KdLossBreakdown {
    /// Differentiable total loss `kd_loss + l2l_loss_scale * sum_l mse(student_l, teacher_l)`.
    pub total: Tensor,
    /// Detached logit KL-divergence scalar (`kd_loss`).
    pub kd_loss: f64,
    /// Detached scaled layer-to-layer MSE scalar (`l2l_loss`).
    pub l2l_loss: f64,
}

/// Compute `F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction="batchmean")`
/// in `F32` for numerical stability.
///
/// # Errors
/// Returns an error if shapes mismatch or tensor ops fail.
pub fn kl_divergence_batchmean(student_logits: &Tensor, teacher_logits: &Tensor) -> Result<Tensor> {
    if student_logits.dims() != teacher_logits.dims() {
        return Err(TrainError::InvalidArgument(format!(
            "logits shape mismatch: student {:?} vs teacher {:?}",
            student_logits.dims(),
            teacher_logits.dims()
        )));
    }
    let batch_size = student_logits.dims().first().copied().unwrap_or(0);
    if batch_size == 0 {
        return Err(TrainError::InvalidArgument(
            "batch size must be > 0 for KL divergence".to_string(),
        ));
    }

    let s = student_logits.to_dtype(DType::F32)?;
    let t = teacher_logits.detach().to_dtype(DType::F32)?;

    let log_p = candle_nn::ops::log_softmax(&s, D::Minus1)?;
    let log_q = candle_nn::ops::log_softmax(&t, D::Minus1)?;
    let q = log_q.exp()?;

    let elem_kl = q.mul(&log_q.sub(&log_p)?)?;
    Ok((elem_kl.sum_all()? / (batch_size as f64))?)
}

/// Compute layer-to-layer MSE loss summed over aligned transformer hidden states and scaled by `scale`.
///
/// Matches `KDTrainer.mse_loss` accumulation in `utils/kd_utils.py`.
///
/// # Errors
/// Returns an error if hidden state counts or tensor shapes mismatch.
pub fn l2l_mse_loss(
    student_reps: &[Tensor],
    teacher_reps: &[Tensor],
    scale: f64,
) -> Result<Tensor> {
    if student_reps.len() != teacher_reps.len() {
        return Err(TrainError::InvalidArgument(format!(
            "hidden state count mismatch: student {} vs teacher {}",
            student_reps.len(),
            teacher_reps.len()
        )));
    }
    if student_reps.is_empty() {
        return Err(TrainError::InvalidArgument(
            "expected non-empty hidden states for L2L MSE loss".to_string(),
        ));
    }

    let mut total: Option<Tensor> = None;
    for (s_rep, t_rep) in student_reps.iter().zip(teacher_reps.iter()) {
        let s = s_rep.to_dtype(DType::F32)?;
        let t = t_rep.detach().to_dtype(DType::F32)?;
        let layer_mse = candle_nn::loss::mse(&s, &t)?;
        total = Some(match total {
            None => layer_mse,
            Some(acc) => acc.add(&layer_mse)?,
        });
    }

    let summed = total.ok_or_else(|| {
        TrainError::InvalidArgument("missing hidden states for L2L MSE loss".to_string())
    })?;
    Ok((summed * scale)?)
}

/// Compute combined Knowledge Distillation loss (`KD + L2L MSE`) from student and teacher outputs.
///
/// Slices `&hidden_states[1..]` to skip embedding outputs, matching upstream `outputs.hidden_states[1:]`.
///
/// # Errors
/// Returns an error if `hidden_states` are missing or loss calculation fails.
pub fn compute_kd_loss(
    student_out: &CausalLmOutput,
    teacher_out: &CausalLmOutput,
    l2l_loss_scale: f64,
) -> Result<KdLossBreakdown> {
    let kd = kl_divergence_batchmean(&student_out.logits, &teacher_out.logits)?;
    let kd_scalar = kd.to_scalar::<f32>()? as f64;

    let student_reps = student_out.hidden_states.get(1..).unwrap_or(&[]);
    let teacher_reps = teacher_out.hidden_states.get(1..).unwrap_or(&[]);
    let l2l = if !student_reps.is_empty() && l2l_loss_scale > 0.0 {
        l2l_mse_loss(student_reps, teacher_reps, l2l_loss_scale)?
    } else {
        Tensor::zeros((), DType::F32, kd.device())?
    };
    let l2l_scalar = l2l.to_scalar::<f32>()? as f64;
    let total = kd.add(&l2l)?;

    Ok(KdLossBreakdown {
        total,
        kd_loss: kd_scalar,
        l2l_loss: l2l_scalar,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn kl_divergence_is_zero_when_distributions_match() -> Result<()> {
        let dev = Device::Cpu;
        let logits = Tensor::from_slice(&[1.0f32, -0.5, 2.0, 0.3, -1.2, 0.8], (2, 1, 3), &dev)?;
        let kl = kl_divergence_batchmean(&logits, &logits)?.to_scalar::<f32>()?;
        assert!(kl.abs() < 1e-6, "expected 0 KL, got {kl}");
        Ok(())
    }

    #[test]
    fn kl_divergence_matches_analytical_value_and_propagates_grad() -> Result<()> {
        let dev = Device::Cpu;
        let s = Var::from_slice(&[0.0f32, 0.0], (1, 1, 2), &dev)?;
        let t = Tensor::from_slice(&[1.0f32, 0.0], (1, 1, 2), &dev)?;
        let kl = kl_divergence_batchmean(s.as_tensor(), &t)?;
        let got = kl.to_scalar::<f32>()?;

        // q = [e/(e+1), 1/(e+1)], p = [0.5, 0.5]
        let e = 1.0f64.exp();
        let q0 = e / (e + 1.0);
        let q1 = 1.0 / (e + 1.0);
        let expected = (q0 * (q0 / 0.5).ln() + q1 * (q1 / 0.5).ln()) as f32;
        assert!((got - expected).abs() < 1e-5, "got {got}, want {expected}");

        let grads = kl.backward()?;
        let grad = grads
            .get(s.as_tensor())
            .expect("student logit grad")
            .flatten_all()?
            .to_vec1::<f32>()?;
        // d(KL)/d(s_i) = p_i - q_i
        assert!((grad[0] - (0.5 - q0 as f32)).abs() < 1e-5);
        assert!((grad[1] - (0.5 - q1 as f32)).abs() < 1e-5);
        Ok(())
    }
}
