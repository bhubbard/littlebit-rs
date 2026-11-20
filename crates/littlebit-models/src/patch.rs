//! Dynamic LittleBit patching matching upstream `quantization/utils/quant_util.py`.

use std::fmt;

use regex::Regex;

use littlebit_core::{LittleBitConfig, TrainMode};

use crate::error::{ModelError, Result};
use crate::layer::LinearLayer;

/// Pattern rules for per-layer rank scaling (`KV_PATTERN`) and exclusion (`lm_head`).
#[derive(Debug, Clone)]
pub struct PatchPolicy {
    /// Regex patterns matching K/V projections that receive `config.kv_factor`.
    kv_patterns: Vec<Regex>,
    /// Substring patterns for module names excluded from quantization (`["lm_head"]` default).
    pub exclude_names: Vec<String>,
}

impl Default for PatchPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl PatchPolicy {
    /// Build default upstream policy (`\.k_proj$` and `\.v_proj$` receive `kv_factor`;
    /// `lm_head` and MoE `router` projections are excluded).
    #[must_use]
    pub fn new() -> Self {
        let kv_patterns = vec![
            Regex::new(r"\.k_proj$").expect("valid k_proj regex"),
            Regex::new(r"\.v_proj$").expect("valid v_proj regex"),
        ];
        Self {
            kv_patterns,
            exclude_names: vec!["lm_head".to_string(), "router".to_string()],
        }
    }

    /// Return true if `module_name` matches any excluded substring.
    #[must_use]
    pub fn is_excluded(&self, module_name: &str) -> bool {
        self.exclude_names
            .iter()
            .any(|ex| module_name.contains(ex.as_str()))
    }

    /// Compute the rank multiplier (`kv_factor` for `.k_proj` / `.v_proj`, otherwise `1.0`).
    #[must_use]
    pub fn ratio_factor(&self, module_name: &str, config: &LittleBitConfig) -> f64 {
        if self.kv_patterns.iter().any(|re| re.is_match(module_name)) {
            config.kv_factor
        } else {
            1.0
        }
    }
}

/// Deterministic FNV-1a seed derivation from `base_seed` and module name.
#[must_use]
pub fn derive_layer_seed(base_seed: u64, module_name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ base_seed;
    for byte in module_name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Summary statistics emitted after applying LittleBit quantization patch across a model.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchSummary {
    /// Number of linear modules quantized to `LittleBitLinear`.
    pub patched_layers: usize,
    /// Number of linear modules skipped due to exclusion policy (`lm_head`).
    pub skipped_layers: usize,
    /// Sum of `in_features * out_features` across all patched layers.
    pub total_quantized_weights: usize,
    /// Sum of actual bit usage across all patched layers.
    pub total_quantized_bits: f64,
    /// Average effective bits per weight across patched layers.
    pub avg_eff_bits: f64,
}

impl PatchSummary {
    pub(crate) fn record_budget(&mut self, budget: &littlebit_core::BitBudget) {
        self.patched_layers += 1;
        self.total_quantized_weights += budget.in_features * budget.out_features;
        self.total_quantized_bits += budget.total_bit_usage;
        if self.total_quantized_weights > 0 {
            self.avg_eff_bits = self.total_quantized_bits / (self.total_quantized_weights as f64);
        }
    }

    pub(crate) fn record_patched(&mut self, layer: &LinearLayer) {
        if let LinearLayer::LittleBit(lb) = layer {
            self.record_budget(lb.budget());
        }
    }

    pub(crate) fn record_skipped(&mut self) {
        self.skipped_layers += 1;
    }
}

impl Default for PatchSummary {
    fn default() -> Self {
        Self {
            patched_layers: 0,
            skipped_layers: 0,
            total_quantized_weights: 0,
            total_quantized_bits: 0.0,
            avg_eff_bits: 0.0,
        }
    }
}

impl fmt::Display for PatchSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "patched={} skipped={} weights={} avg_eff_bits={:.4}",
            self.patched_layers,
            self.skipped_layers,
            self.total_quantized_weights,
            self.avg_eff_bits
        )
    }
}

/// Patch a named [`LinearLayer`] according to `policy` and update `summary`.
///
/// # Errors
/// Returns an error if SVD/ITQ decomposition fails.
pub fn patch_linear_layer(
    module_name: &str,
    layer: &mut LinearLayer,
    config: &LittleBitConfig,
    policy: &PatchPolicy,
    train_mode: TrainMode,
    base_seed: u64,
    summary: &mut PatchSummary,
) -> Result<()> {
    config.validate().map_err(ModelError::Core)?;
    if policy.is_excluded(module_name) {
        summary.record_skipped();
        return Ok(());
    }
    let effective_cfg = config
        .effective_for_module(module_name)
        .map_err(ModelError::Core)?;
    let ratio = policy.ratio_factor(module_name, &effective_cfg);
    let seed = derive_layer_seed(base_seed, module_name);
    layer.convert_to_littlebit(&effective_cfg, ratio, train_mode, seed)?;
    summary.record_patched(layer);
    Ok(())
}
