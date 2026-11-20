//! Cosine learning rate schedule with linear warmup (`WarmupCosineLR`).

use std::f64::consts::PI;

/// Learning rate scheduler implementing linear warmup followed by cosine decay.
#[derive(Debug, Clone, Copy)]
pub struct CosineWarmupScheduler {
    base_lr: f64,
    total_steps: usize,
    warmup_steps: usize,
    min_lr_ratio: f64,
}

impl CosineWarmupScheduler {
    /// Construct a scheduler from base LR, total optimizer steps, warmup ratio, and `cos_min_ratio`.
    #[must_use]
    pub fn new(base_lr: f64, total_steps: usize, warmup_ratio: f64, min_lr_ratio: f64) -> Self {
        let warmup_steps = ((total_steps as f64) * warmup_ratio.clamp(0.0, 1.0)).ceil() as usize;
        Self {
            base_lr,
            total_steps: total_steps.max(1),
            warmup_steps,
            min_lr_ratio: min_lr_ratio.clamp(0.0, 1.0),
        }
    }

    /// Compute the learning rate at optimizer step `step` (0-indexed).
    #[must_use]
    pub fn lr_at_step(&self, step: usize) -> f64 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * ((step + 1) as f64 / self.warmup_steps as f64);
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps);
        if decay_steps == 0 {
            return self.base_lr;
        }
        let progress =
            ((step.saturating_sub(self.warmup_steps)) as f64 / decay_steps as f64).clamp(0.0, 1.0);
        let cosine = 0.5 * (1.0 + (PI * progress).cos());
        let factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine;
        self.base_lr * factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_warmup_hits_expected_milestones() {
        let sched = CosineWarmupScheduler::new(1e-4, 100, 0.1, 0.0001);
        assert!((sched.lr_at_step(0) - 1e-5).abs() < 1e-9);
        assert!((sched.lr_at_step(9) - 1e-4).abs() < 1e-9);
        let mid = sched.lr_at_step(55);
        assert!(mid > 1e-5 && mid < 1e-4);
        assert!((sched.lr_at_step(100) - 1e-8).abs() < 1e-10);
    }
}
