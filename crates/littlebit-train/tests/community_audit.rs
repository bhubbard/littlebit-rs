//! Quality-pilot loss, bit-flip popcount, FP16 loss precision, and seeded training
//! reproducibility tests ported from `timurista/littlebit-deployment-audit` and
//! `christopherrobertbrooks-tech/littlebit-bench`.

use candle_core::{DType, Device, Tensor, Var};
use littlebit_core::{pack_i8_slice, LittleBitConfig, QuantFunc, TrainMode};
use littlebit_models::{CausalLm, ModelConfig};
use littlebit_train::{
    kl_divergence_batchmean, l2l_mse_loss, KdTrainer, Result, TrainError, TrainingConfig,
};

fn popcount_flips(a: &[i32], b: &[i32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x ^ y) as u32).count_ones() as usize)
        .sum()
}

#[test]
fn pilot_kl_positive_zero_and_teacher_detached() -> Result<()> {
    let dev = Device::Cpu;
    let logits = Tensor::from_slice(&[0.4f32, -1.2, 0.9, 1.5, -0.3, 0.2], (1, 2, 3), &dev)?;
    let zero_kl = kl_divergence_batchmean(&logits, &logits)?.to_scalar::<f32>()?;
    assert!(zero_kl.abs() < 1e-6, "expected ~0 KL, got {zero_kl}");

    let s = Var::from_slice(&[0.2f32, -0.4, 1.1, 0.5, 0.0, -0.8], (1, 2, 3), &dev)?;
    let t = Var::from_slice(&[1.4f32, 0.2, -0.9, -0.2, 1.1, 0.3], (1, 2, 3), &dev)?;
    let kl = kl_divergence_batchmean(s.as_tensor(), t.as_tensor())?;
    let kl_val = kl.to_scalar::<f32>()?;
    assert!(kl_val > 0.0, "expected positive KL, got {kl_val}");

    let kl_grads = kl.backward()?;
    assert!(kl_grads.get(s.as_tensor()).is_some());
    assert!(kl_grads.get(t.as_tensor()).is_none());

    // Teacher hidden state must likewise remain detached in l2l_mse_loss.
    let s_h = Var::from_slice(&[0.1f32, 0.2, -0.3, 0.4], (1, 1, 4), &dev)?;
    let t_h = Var::from_slice(&[0.5f32, -0.1, 0.7, -0.2], (1, 1, 4), &dev)?;
    let mse = l2l_mse_loss(&[s_h.as_tensor().clone()], &[t_h.as_tensor().clone()], 10.0)?;
    let mse_grads = mse.backward()?;
    assert!(mse_grads.get(s_h.as_tensor()).is_some());
    assert!(mse_grads.get(t_h.as_tensor()).is_none());

    assert!(matches!(
        l2l_mse_loss(&[], &[], 1.0),
        Err(TrainError::InvalidArgument(_))
    ));
    Ok(())
}

#[test]
fn pilot_popcount_flips_on_packed_words() -> littlebit_core::Result<()> {
    assert_eq!(popcount_flips(&[0i32, -1i32], &[1i32, 0i32]), 33);

    let mut base = vec![1i8; 64];
    base[0] = -1;
    base[31] = -1;
    let mut flipped = base.clone();
    for idx in [2usize, 7, 15, 33, 63] {
        flipped[idx] = -flipped[idx];
    }

    let p_base = pack_i8_slice(&base, 1, 64)?;
    let p_flipped = pack_i8_slice(&flipped, 1, 64)?;
    assert_eq!(popcount_flips(p_base.words(), p_flipped.words()), 5);
    Ok(())
}

#[test]
fn bench_fp16_kd_and_l2l_losses_avoid_overflow() -> Result<()> {
    let dev = Device::Cpu;
    let s_f32 = Tensor::from_slice(&[-20.0f32, 0.0, 20.0, 5.0, -5.0, 1.0], (1, 2, 3), &dev)?;
    let t_f32 = Tensor::from_slice(&[20.0f32, -10.0, 0.0, -5.0, 5.0, 2.0], (1, 2, 3), &dev)?;
    let s_f16 = s_f32.to_dtype(DType::F16)?;
    let t_f16 = t_f32.to_dtype(DType::F16)?;

    let kl32 = kl_divergence_batchmean(&s_f32, &t_f32)?.to_scalar::<f32>()?;
    let kl16 = kl_divergence_batchmean(&s_f16, &t_f16)?.to_scalar::<f32>()?;
    assert!(kl16.is_finite());
    assert!((kl32 - kl16).abs() / kl32.abs().max(1.0) < 1e-2);

    // 300 - (-300) = 600 -> 600^2 = 360,000 > 65,504 (overflows if computed in FP16).
    let hs_f32 = Tensor::from_slice(&[300.0f32, -300.0, 280.0, -280.0], (1, 2, 2), &dev)?;
    let ht_f32 = Tensor::from_slice(&[-300.0f32, 300.0, -280.0, 280.0], (1, 2, 2), &dev)?;
    let hs_f16 = hs_f32.to_dtype(DType::F16)?;
    let ht_f16 = ht_f32.to_dtype(DType::F16)?;

    let l2l32 = l2l_mse_loss(&[hs_f32], &[ht_f32], 1.0)?.to_scalar::<f32>()?;
    let l2l16 = l2l_mse_loss(&[hs_f16], &[ht_f16], 1.0)?.to_scalar::<f32>()?;
    assert!(l2l16.is_finite());
    assert!((l2l32 - l2l16).abs() / l2l32.abs().max(1.0) < 1e-2);
    Ok(())
}

#[test]
fn bench_seeded_training_reproducibility() -> Result<()> {
    let dev = Device::Cpu;
    let model_cfg = ModelConfig::tiny_llama(32, 32, 1);
    let lb_cfg = LittleBitConfig {
        quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
        eff_bit: None,
        split_dim: 8,
        use_itq: false,
        ..Default::default()
    };
    let blocks = vec![
        vec![1u32, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10, 11, 12],
        vec![13, 14, 15, 16],
    ];
    let train_cfg = TrainingConfig {
        seed: 42,
        num_train_epochs: 2.0,
        batch_size: 2,
        gradient_accumulation_steps: 1,
        warmup_ratio: 0.25,
        lr: 5e-4,
        l2l_loss_scale: 10.0,
        ..Default::default()
    };

    let run_once = || -> Result<Vec<littlebit_train::StepMetrics>> {
        let teacher = CausalLm::new_random(model_cfg.clone(), DType::F32, &dev, 99)?;
        let mut student = CausalLm::new_random(model_cfg.clone(), DType::F32, &dev, 99)?;
        student.apply_littlebit_patch(&lb_cfg, TrainMode::Train, 7)?;
        let mut trainer = KdTrainer::new(&mut student, train_cfg.clone())?;
        Ok(trainer.train(&student, &teacher, &blocks, &dev)?.history)
    };

    let h1 = run_once()?;
    let h2 = run_once()?;
    assert_eq!(h1.len(), 4);
    assert_eq!(h1.len(), h2.len());
    for (m1, m2) in h1.iter().zip(h2.iter()) {
        assert_eq!(m1.step, m2.step);
        assert!((m1.total_loss - m2.total_loss).abs() < 1e-6);
        assert!((m1.kd_loss - m2.kd_loss).abs() < 1e-6);
        assert!((m1.l2l_loss - m2.l2l_loss).abs() < 1e-6);
        assert!((m1.lr - m2.lr).abs() < 1e-9);
    }
    Ok(())
}
