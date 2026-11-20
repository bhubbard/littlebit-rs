//! QAT Knowledge Distillation trainer (`main.py` & `utils/kd_utils.py` parity).

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tracing::info;

use littlebit_core::{CastPolicy, LittleBitConfig};
use littlebit_models::{CausalLm, HiddenStatesMode, LittleBitModel, ParameterStats};

use crate::error::{Result, TrainError};
use crate::loss::compute_kd_loss;
use crate::scheduler::CosineWarmupScheduler;

/// Hyperparameters for [`KdTrainer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Random seed for batch shuffling.
    pub seed: u64,
    /// Number of training epochs (supports fractional epochs like `0.5` or `5.0`).
    pub num_train_epochs: f64,
    /// Per-step micro-batch size.
    pub batch_size: usize,
    /// Number of micro-batches accumulated per optimizer step.
    pub gradient_accumulation_steps: usize,
    /// Fraction of total optimizer steps spent warming up LR linearly.
    pub warmup_ratio: f64,
    /// Peak AdamW learning rate.
    pub lr: f64,
    /// AdamW beta1.
    pub adam_beta1: f64,
    /// AdamW beta2.
    pub adam_beta2: f64,
    /// AdamW numerical epsilon.
    pub adam_eps: f64,
    /// AdamW weight decay.
    pub weight_decay: f64,
    /// Global gradient L2 clip threshold (`0.0` disables clipping).
    pub max_grad_norm: f64,
    /// Minimum cosine LR ratio at end of decay (`cos_min_ratio`).
    pub cos_min_ratio: f64,
    /// Multiplier for summed layer-to-layer hidden-state MSE loss (`10.0` default).
    pub l2l_loss_scale: f64,
    /// Output directory where checkpoints and metrics logs are saved (optional).
    pub save_dir: Option<PathBuf>,
    /// Experiment name metadata.
    pub exp_name: String,
    /// Run name metadata.
    pub run_name: String,
    /// Module name substrings excluded from training (`["lm_head", "embed"]` default).
    pub exclude_from_grad: Vec<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            num_train_epochs: 3.0,
            batch_size: 4,
            gradient_accumulation_steps: 1,
            warmup_ratio: 0.03,
            lr: 2e-5,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_eps: 1e-8,
            weight_decay: 0.0,
            max_grad_norm: 1.0,
            cos_min_ratio: 0.0001,
            l2l_loss_scale: 10.0,
            save_dir: None,
            exp_name: "LittleBit".to_string(),
            run_name: "littlebit_run".to_string(),
            exclude_from_grad: vec!["lm_head".to_string(), "embed".to_string()],
        }
    }
}

/// Metrics logged for each optimizer step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// 1-based optimizer step index.
    pub step: usize,
    /// Fractional epoch progress.
    pub epoch: f64,
    /// Combined `kd_loss + l2l_loss`.
    pub total_loss: f64,
    /// KL divergence logit loss.
    pub kd_loss: f64,
    /// Scaled layer-to-layer MSE loss.
    pub l2l_loss: f64,
    /// Active learning rate used for this step.
    pub lr: f64,
}

/// Summary returned after [`KdTrainer::train`] completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// Total optimizer steps executed.
    pub total_steps: usize,
    /// First step total loss (if at least 1 step executed).
    pub initial_loss: Option<f64>,
    /// Final step total loss (if at least 1 step executed).
    pub final_loss: Option<f64>,
    /// Final step KL divergence loss.
    pub final_kd_loss: Option<f64>,
    /// Final step L2L MSE loss.
    pub final_l2l_loss: Option<f64>,
    /// Per-step optimization metrics history.
    pub history: Vec<StepMetrics>,
}

/// Quantization-Aware Knowledge Distillation trainer matching `KDTrainer` in `utils/kd_utils.py`.
#[derive(Debug)]
pub struct KdTrainer {
    config: TrainingConfig,
    trainable_vars: Vec<Var>,
    optimizer: AdamW,
    param_stats: ParameterStats,
}

impl KdTrainer {
    /// Prepare `student` for training (freezing `exclude_from_grad`) and initialize AdamW.
    ///
    /// # Errors
    /// Returns an error if no trainable parameters exist or optimizer creation fails.
    pub fn new(student: &mut CausalLm, config: TrainingConfig) -> Result<Self> {
        let exclude_refs: Vec<&str> = config
            .exclude_from_grad
            .iter()
            .map(String::as_str)
            .collect();
        student.prepare_for_training(&exclude_refs)?;

        let trainable_vars = student.trainable_vars();
        if trainable_vars.is_empty() {
            return Err(TrainError::InvalidArgument(
                "student model has zero trainable variables".to_string(),
            ));
        }

        let param_stats = student.parameter_stats();
        info!("{param_stats}");

        let optimizer = AdamW::new(
            trainable_vars.clone(),
            ParamsAdamW {
                lr: config.lr,
                beta1: config.adam_beta1,
                beta2: config.adam_beta2,
                eps: config.adam_eps,
                weight_decay: config.weight_decay,
            },
        )?;

        Ok(Self {
            config,
            trainable_vars,
            optimizer,
            param_stats,
        })
    }

    /// Borrow parameter statistics logged at trainer initialization.
    #[must_use]
    pub fn parameter_stats(&self) -> ParameterStats {
        self.param_stats
    }

    /// Run knowledge distillation training on `token_blocks` (`&[Vec<u32>]`) against frozen `teacher`.
    ///
    /// # Errors
    /// Returns an error on invalid dataset/batch sizes, forward/backward failures, or I/O logging errors.
    pub fn train(
        &mut self,
        student: &CausalLm,
        teacher: &CausalLm,
        token_blocks: &[Vec<u32>],
        device: &Device,
    ) -> Result<TrainingSummary> {
        if token_blocks.is_empty() {
            return Err(TrainError::InvalidArgument(
                "training dataset contains 0 token blocks".to_string(),
            ));
        }
        let batch_size = self.config.batch_size.max(1);
        let grad_accum = self.config.gradient_accumulation_steps.max(1);

        let micro_batches_per_epoch = token_blocks.len().div_ceil(batch_size);
        let total_micro_batches =
            ((micro_batches_per_epoch as f64) * self.config.num_train_epochs).ceil() as usize;
        let total_micro_batches = total_micro_batches.max(1);
        let total_opt_steps = total_micro_batches.div_ceil(grad_accum).max(1);

        let scheduler = CosineWarmupScheduler::new(
            self.config.lr,
            total_opt_steps,
            self.config.warmup_ratio,
            self.config.cos_min_ratio,
        );

        let mut log_file = if let Some(dir) = &self.config.save_dir {
            fs::create_dir_all(dir)?;
            Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join("metrics.jsonl"))?,
            )
        } else {
            None
        };

        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed);
        let mut indices: Vec<usize> = (0..token_blocks.len()).collect();
        let mut history = Vec::with_capacity(total_opt_steps);

        let mut micro_idx = 0usize;
        let mut opt_step = 0usize;
        let mut accum_grads: Option<GradStore> = None;
        let mut accum_kd = 0.0_f64;
        let mut accum_l2l = 0.0_f64;
        let mut accum_count = 0usize;

        while micro_idx < total_micro_batches {
            if micro_idx.is_multiple_of(micro_batches_per_epoch) {
                indices.shuffle(&mut rng);
            }
            let epoch_offset = micro_idx % micro_batches_per_epoch;
            let start = epoch_offset * batch_size;
            let end = (start + batch_size).min(token_blocks.len());
            let batch_indices = &indices[start..end];

            let seq_len = token_blocks[batch_indices[0]].len();
            let mut flat_ids = Vec::with_capacity(batch_indices.len() * seq_len);
            for &idx in batch_indices {
                flat_ids.extend_from_slice(&token_blocks[idx]);
            }
            let input_ids = Tensor::from_vec(flat_ids, (batch_indices.len(), seq_len), device)?;

            let teacher_out = teacher.forward(&input_ids, None, HiddenStatesMode::All)?;
            let student_out = student.forward(&input_ids, None, HiddenStatesMode::All)?;

            let loss_bd = compute_kd_loss(&student_out, &teacher_out, self.config.l2l_loss_scale)?;
            let scaled_loss = (loss_bd.total / (grad_accum as f64))?;
            let step_grads = scaled_loss.backward()?;

            match &mut accum_grads {
                None => accum_grads = Some(step_grads),
                Some(acc) => acc.extend(step_grads)?,
            }

            accum_kd += loss_bd.kd_loss;
            accum_l2l += loss_bd.l2l_loss;
            accum_count += 1;
            micro_idx += 1;

            let ready = accum_count == grad_accum || micro_idx == total_micro_batches;
            if ready {
                if let Some(mut grads) = accum_grads.take() {
                    if self.config.max_grad_norm > 0.0 {
                        clip_grad_norm(
                            &mut grads,
                            &self.trainable_vars,
                            self.config.max_grad_norm,
                        )?;
                    }
                    let lr = scheduler.lr_at_step(opt_step);
                    self.optimizer.set_learning_rate(lr);
                    self.optimizer.step(&grads)?;

                    opt_step += 1;
                    let kd_avg = accum_kd / (accum_count as f64);
                    let l2l_avg = accum_l2l / (accum_count as f64);
                    let total_avg = kd_avg + l2l_avg;
                    let epoch_progress = (micro_idx as f64) / (micro_batches_per_epoch as f64);

                    let metric = StepMetrics {
                        step: opt_step,
                        epoch: epoch_progress,
                        total_loss: total_avg,
                        kd_loss: kd_avg,
                        l2l_loss: l2l_avg,
                        lr,
                    };
                    info!(
                        step = metric.step,
                        epoch = format!("{:.3}", metric.epoch),
                        loss = format!("{:.6}", metric.total_loss),
                        kd_loss = format!("{:.6}", metric.kd_loss),
                        l2l_loss = format!("{:.6}", metric.l2l_loss),
                        lr = format!("{:.3e}", metric.lr),
                        "training step"
                    );
                    if let Some(file) = &mut log_file {
                        writeln!(file, "{}", serde_json::to_string(&metric)?)?;
                    }
                    history.push(metric);
                }
                accum_kd = 0.0;
                accum_l2l = 0.0;
                accum_count = 0;
            }
        }

        let initial_loss = history.first().map(|m| m.total_loss);
        let final_m = history.last();
        Ok(TrainingSummary {
            total_steps: opt_step,
            initial_loss,
            final_loss: final_m.map(|m| m.total_loss),
            final_kd_loss: final_m.map(|m| m.kd_loss),
            final_l2l_loss: final_m.map(|m| m.l2l_loss),
            history,
        })
    }

    /// Save final packed LittleBit checkpoint + tokenizer artifacts matching `main.py::save_artifacts`.
    ///
    /// # Errors
    /// Returns an error if saving config or safetensors fails.
    pub fn save_artifacts(
        student: &CausalLm,
        lb_cfg: &LittleBitConfig,
        base_model_id: Option<&str>,
        save_dir: &std::path::Path,
        tokenizer_src: Option<&std::path::Path>,
    ) -> Result<()> {
        let wrapper = LittleBitModel::new(
            student.clone(),
            lb_cfg.clone(),
            base_model_id.map(str::to_string),
            tokenizer_src.map(std::path::Path::to_path_buf),
        );
        wrapper.save_pretrained(save_dir, tokenizer_src, CastPolicy::ToBf16)?;
        Ok(())
    }
}

fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<()> {
    let mut norm_sq = 0.0_f64;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            let s = g
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()? as f64;
            norm_sq += s;
        }
    }
    let total_norm = norm_sq.sqrt();
    if total_norm > max_norm && total_norm > 0.0 {
        let scale = max_norm / (total_norm + 1e-6);
        for var in vars {
            if let Some(g) = grads.get(var.as_tensor()).cloned() {
                grads.insert(var.as_tensor(), (g * scale)?);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use littlebit_core::{InferenceMode, QuantFunc, TrainMode};
    use littlebit_models::{LoadOptions, ModelConfig};
    use tempfile::TempDir;

    #[test]
    fn kd_trainer_reduces_loss_with_frozen_embed_and_lm_head() -> Result<()> {
        let dev = Device::Cpu;
        let model_cfg = ModelConfig::tiny_llama(48, 32, 2);

        let teacher = CausalLm::new_random(model_cfg.clone(), DType::F32, &dev, 1)?;
        let mut student = CausalLm::new_random(model_cfg.clone(), DType::F32, &dev, 1)?;

        let lb_cfg = LittleBitConfig {
            quant_func: QuantFunc::SmoothSign { alpha: 20.0 },
            eff_bit: None,
            split_dim: 8,
            residual: false,
            use_itq: true,
            itq_n_iter: 10,
            ..Default::default()
        };
        student.apply_littlebit_patch(&lb_cfg, TrainMode::Train, 7)?;

        // Snapshot embed_tokens and lm_head before training to verify freeze invariant.
        let before_state = student.pack_state_dict()?;
        let embed_before = before_state["model.embed_tokens.weight"].to_vec2::<f32>()?;
        let head_before = before_state["lm_head.weight"].to_vec2::<f32>()?;

        let save_dir = TempDir::new().expect("tempdir");
        let train_cfg = TrainingConfig {
            seed: 42,
            num_train_epochs: 6.0,
            batch_size: 2,
            gradient_accumulation_steps: 2,
            warmup_ratio: 0.1,
            lr: 5e-2,
            l2l_loss_scale: 1.0,
            save_dir: Some(save_dir.path().to_path_buf()),
            ..Default::default()
        };

        let mut trainer = KdTrainer::new(&mut student, train_cfg)?;
        let blocks = vec![
            vec![1u32, 2, 3, 4, 5, 6],
            vec![7u32, 8, 9, 10, 11, 12],
            vec![13u32, 14, 15, 16, 17, 18],
            vec![19u32, 20, 21, 22, 23, 24],
        ];
        let summary = trainer.train(&student, &teacher, &blocks, &dev)?;

        assert!(summary.total_steps >= 6);
        let first = summary.initial_loss.expect("initial loss");
        let last = summary.final_loss.expect("final loss");
        assert!(
            last < first,
            "KD training should lower total loss: initial={first}, final={last}"
        );

        let after_state = student.pack_state_dict()?;
        let embed_after = after_state["model.embed_tokens.weight"].to_vec2::<f32>()?;
        let head_after = after_state["lm_head.weight"].to_vec2::<f32>()?;
        assert_eq!(embed_before, embed_after, "embed_tokens must remain frozen");
        assert_eq!(head_before, head_after, "lm_head must remain frozen");

        KdTrainer::save_artifacts(
            &student,
            &lb_cfg,
            Some("test/tiny-llama"),
            save_dir.path(),
            None,
        )?;
        assert!(save_dir.path().join("metrics.jsonl").exists());

        let reloaded = LittleBitModel::from_pretrained(
            save_dir.path().to_str().expect("utf8 path"),
            &LoadOptions {
                inference_mode: InferenceMode::Packed,
                ..Default::default()
            },
        )?;
        let test_ids = Tensor::from_slice(&[1u32, 2, 3, 4], (1, 4), &dev)?;
        let logits = reloaded
            .model
            .forward(&test_ids, None, HiddenStatesMode::None)?
            .logits;
        assert_eq!(logits.dims(), &[1, 4, model_cfg.vocab_size]);
        Ok(())
    }
}
