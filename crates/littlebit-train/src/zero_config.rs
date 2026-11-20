//! Parser for upstream DeepSpeed configuration files (`configs/zero*.json`).

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// BF16 section inside a DeepSpeed config JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroBf16Config {
    /// Whether bfloat16 training/inference is enabled.
    #[serde(default)]
    pub enabled: bool,
}

/// Scheduler parameters inside a DeepSpeed config JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroSchedulerParams {
    /// Minimum cosine decay ratio (`cos_min_ratio`).
    pub cos_min_ratio: Option<f64>,
    /// Linear warmup min ratio.
    pub warmup_min_ratio: Option<f64>,
    /// Warmup schedule type (`linear`, etc.).
    pub warmup_type: Option<String>,
}

/// Scheduler block inside a DeepSpeed config JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroSchedulerConfig {
    /// Scheduler class name (e.g. `WarmupCosineLR`).
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Scheduler parameters.
    #[serde(default)]
    pub params: ZeroSchedulerParams,
}

/// ZeRO optimization block inside a DeepSpeed config JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroOptimizationConfig {
    /// ZeRO stage (`0`, `1`, `2`, or `3`).
    #[serde(default)]
    pub stage: u8,
}

/// Single-node compatible representation of `configs/zero*.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ZeroConfig {
    /// Mixed-precision bfloat16 toggle.
    #[serde(default)]
    pub bf16: ZeroBf16Config,
    /// Scheduler configuration.
    pub scheduler: Option<ZeroSchedulerConfig>,
    /// ZeRO stage settings (logged for single-node Rust compatibility).
    pub zero_optimization: Option<ZeroOptimizationConfig>,
    /// Optional numeric gradient clipping threshold (`"auto"` ignored).
    #[serde(default)]
    pub gradient_clipping: serde_json::Value,
    /// Optional numeric gradient accumulation steps (`"auto"` ignored).
    #[serde(default)]
    pub gradient_accumulation_steps: serde_json::Value,
}

impl ZeroConfig {
    /// Load and parse a `zero*.json` configuration file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or fails JSON parsing.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&raw)?;
        Ok(cfg)
    }

    /// Return `cos_min_ratio` if specified, defaulting to `0.0001`.
    #[must_use]
    pub fn cos_min_ratio(&self) -> f64 {
        self.scheduler
            .as_ref()
            .and_then(|s| s.params.cos_min_ratio)
            .unwrap_or(0.0001)
    }

    /// Return numeric `gradient_clipping` if configured (ignoring `"auto"`).
    #[must_use]
    pub fn max_grad_norm(&self) -> Option<f64> {
        self.gradient_clipping.as_f64()
    }

    /// Return numeric `gradient_accumulation_steps` if configured (ignoring `"auto"`).
    #[must_use]
    pub fn grad_accum_steps(&self) -> Option<usize> {
        self.gradient_accumulation_steps
            .as_u64()
            .map(|n| n.max(1) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_all_reference_zero_configs() -> Result<()> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("configs");
        for name in ["zero.json", "zero3.json", "zero3_inference.json"] {
            let cfg = ZeroConfig::from_file(&root.join(name))?;
            assert!(cfg.bf16.enabled, "{name} should enable bf16");
        }
        let z3 = ZeroConfig::from_file(&root.join("zero3.json"))?;
        assert!((z3.cos_min_ratio() - 0.0001).abs() < 1e-9);
        Ok(())
    }
}
