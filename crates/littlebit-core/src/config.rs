//! Quantization configuration, algebraic modes, and bit-budget calculation.

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{LittleBitError, Result};

/// Default sharpness hyperparameter `alpha` for [`QuantFunc::SmoothSign`].
pub const SMOOTH_SIGN_DEFAULT_ALPHA: f64 = 100.0;

/// Binarization function and straight-through gradient estimator.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum QuantFunc {
    /// Sign forward pass with clipped identity STE (`|x| < 1`), matching `STEBinary`.
    #[default]
    SteBinary,
    /// Sign forward pass with hyperbolic-tangent derivative (`alpha * (1 - tanh(alpha * x)^2)`),
    /// matching `SmoothSign`.
    SmoothSign {
        /// Sharpness scale factor (`100.0` in upstream LittleBit).
        alpha: f64,
    },
}

impl QuantFunc {
    /// Canonical upstream wire identifier (`"STEBinary"` or `"SmoothSign"`).
    #[must_use]
    pub fn as_wire_name(self) -> &'static str {
        match self {
            Self::SteBinary => "STEBinary",
            Self::SmoothSign { .. } => "SmoothSign",
        }
    }
}

impl fmt::Display for QuantFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_name())
    }
}

impl FromStr for QuantFunc {
    type Err = LittleBitError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "stebinary" | "ste_binary" | "ste" => Ok(Self::SteBinary),
            "smoothsign" | "smooth_sign" | "smooth" => Ok(Self::SmoothSign {
                alpha: SMOOTH_SIGN_DEFAULT_ALPHA,
            }),
            other => Err(LittleBitError::InvalidConfig(format!(
                "unknown quant_func `{other}` (expected `STEBinary` or `SmoothSign`)"
            ))),
        }
    }
}

impl Serialize for QuantFunc {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_wire_name())
    }
}

impl<'de> Deserialize<'de> for QuantFunc {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(serde::de::Error::custom)
    }
}

/// Residual factorization mode for a [`crate::LittleBitLinear`] layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualMode {
    /// Single latent factorization branch `(U, V, u1, u2, v1, v2)`.
    Single,
    /// Dual latent factorization adding a residual branch `(U_R, V_R, u1_R, u2_R, v1_R, v2_R)`.
    Dual,
}

impl From<bool> for ResidualMode {
    fn from(residual: bool) -> Self {
        if residual {
            Self::Dual
        } else {
            Self::Single
        }
    }
}

/// LittleBit-2 Joint Iterative Quantization (Joint-ITQ) rotation configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItqMode {
    /// Original LittleBit SVD-only initialization (`use_itq = false`).
    Disabled,
    /// LittleBit-2 internal latent rotation optimizing `min || B - X R ||_F^2` (`use_itq = true`).
    Joint {
        /// Number of alternating Procrustes / sign updates.
        n_iter: usize,
    },
}

/// Whether parameters inside a quantized layer hold autograd variables or frozen tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrainMode {
    /// Wrap factor parameters in differentiable `candle_core::Var`s for QAT.
    Train,
    /// Store frozen `candle_core::Tensor`s for evaluation or inference.
    Eval,
}

/// Weight initialization mode when constructing [`crate::LittleBitLinear`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InitMode {
    /// Decompose a dense weight matrix via low-rank SVD (+ optional Joint-ITQ).
    Decompose,
    /// Allocate zero/empty placeholder factors before loading checkpoint tensors.
    Empty,
}

/// Runtime weight representation when loading quantized checkpoints for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferenceMode {
    /// Unpack 1-bit matrices into dense float tensors (`{-1, +1}` in target dtype).
    Unpacked,
    /// Keep 1-bit weights packed as row-major 32-bit words (`PackedBinaryMatrix`).
    Packed,
}

/// Whether a linear layer includes a bias vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiasMode {
    /// Layer includes a bias vector of length `out_features`.
    WithBias,
    /// Layer omits bias.
    NoBias,
}

impl From<bool> for BiasMode {
    fn from(has_bias: bool) -> Self {
        if has_bias {
            Self::WithBias
        } else {
            Self::NoBias
        }
    }
}

/// Whether factor weights `U`, `V` in a loaded `LittleBitLinear` are already binarized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightState {
    /// Latent weights are continuous and should pass through `QuantFunc` during forward.
    Raw,
    /// Latent weights are already binarized (`{-1, +1}`) and used directly during forward.
    PreBinarized,
}

/// Linear layer dimension specification `(in_features, out_features, bias)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerShape {
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Whether the layer includes an output bias.
    pub bias: BiasMode,
}

impl LayerShape {
    /// Construct a layer shape specification.
    #[must_use]
    pub const fn new(in_features: usize, out_features: usize, bias: BiasMode) -> Self {
        Self {
            in_features,
            out_features,
            bias,
        }
    }
}

/// Serialization policy for checkpoint float tensors when saving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CastPolicy {
    /// Preserve existing tensor dtypes verbatim.
    Preserve,
    /// Cast `F32` scale/norm/head tensors to `BF16` while keeping packed `I32`/`I64` untouched.
    ToBf16,
}

/// Quantization configuration matching upstream `littlebit_config.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LittleBitConfig {
    /// Binarization estimator (`"STEBinary"` or `"SmoothSign"`).
    pub quant_func: QuantFunc,
    /// Target average effective bits per weight (`None` falls back to `split_dim`).
    pub eff_bit: Option<f64>,
    /// Optional per-decoder-block effective bit overrides (e.g., `"0-7,27:2.0;12:1.5"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eff_bit_layers: Option<String>,
    /// Fallback inner latent rank when `eff_bit` is omitted.
    pub split_dim: usize,
    /// Whether to include a second residual latent factorization branch.
    pub residual: bool,
    /// Rank multiplier applied to attention `k_proj` and `v_proj` layers.
    pub kv_factor: f64,
    /// Minimum inner rank floor (must be positive; rounded in multiples of 8 when computed).
    pub min_split_dim: usize,
    /// Quantized module identifier (`"LittleBitLinear"`).
    pub quant_mod: String,
    /// Opt-in flag for LittleBit-2 Joint-ITQ initialization.
    pub use_itq: bool,
    /// Number of Joint-ITQ iterations when `use_itq` is enabled.
    pub itq_n_iter: usize,
}

impl Default for LittleBitConfig {
    fn default() -> Self {
        Self {
            quant_func: QuantFunc::SteBinary,
            eff_bit: Some(1.0),
            eff_bit_layers: None,
            split_dim: 1024,
            residual: false,
            kv_factor: 1.0,
            min_split_dim: 8,
            quant_mod: "LittleBitLinear".to_string(),
            use_itq: false,
            itq_n_iter: 50,
        }
    }
}

/// Optional CLI or API overrides applied with highest priority over config files.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LittleBitConfigOverrides {
    /// Override `quant_func`.
    pub quant_func: Option<QuantFunc>,
    /// Override `eff_bit` (`Some(Some(x))` sets target, `Some(None)` clears target).
    pub eff_bit: Option<Option<f64>>,
    /// Override `eff_bit_layers` (`Some(Some(spec))` sets spec, `Some(None)` clears spec).
    pub eff_bit_layers: Option<Option<String>>,
    /// Override `split_dim`.
    pub split_dim: Option<usize>,
    /// Override `residual`.
    pub residual: Option<bool>,
    /// Override `kv_factor`.
    pub kv_factor: Option<f64>,
    /// Override `min_split_dim`.
    pub min_split_dim: Option<usize>,
    /// Override `quant_mod`.
    pub quant_mod: Option<String>,
    /// Override `use_itq`.
    pub use_itq: Option<bool>,
    /// Override `itq_n_iter`.
    pub itq_n_iter: Option<usize>,
}

impl LittleBitConfig {
    /// Return the residual factorization mode.
    #[must_use]
    pub fn residual_mode(&self) -> ResidualMode {
        ResidualMode::from(self.residual)
    }

    /// Return the LittleBit-2 ITQ initialization mode.
    #[must_use]
    pub fn itq_mode(&self) -> ItqMode {
        if self.use_itq {
            ItqMode::Joint {
                n_iter: self.itq_n_iter,
            }
        } else {
            ItqMode::Disabled
        }
    }

    /// Validate hyperparameter invariants.
    ///
    /// # Errors
    /// Returns [`LittleBitError::InvalidConfig`] when any value is invalid or non-finite.
    pub fn validate(&self) -> Result<()> {
        if self.min_split_dim == 0 {
            return Err(LittleBitError::InvalidConfig(
                "min_split_dim must be greater than 0".to_string(),
            ));
        }
        if !self.kv_factor.is_finite() || self.kv_factor <= 0.0 {
            return Err(LittleBitError::InvalidConfig(format!(
                "kv_factor must be finite and positive, got {}",
                self.kv_factor
            )));
        }
        if let Some(bit) = self.eff_bit {
            if !bit.is_finite() || bit <= 0.0 {
                return Err(LittleBitError::InvalidConfig(format!(
                    "eff_bit must be finite and positive, got {bit}"
                )));
            }
        } else if self.split_dim == 0 {
            return Err(LittleBitError::InvalidConfig(
                "split_dim must be greater than 0 when eff_bit is None".to_string(),
            ));
        }
        if let QuantFunc::SmoothSign { alpha } = self.quant_func {
            if !alpha.is_finite() || alpha <= 0.0 {
                return Err(LittleBitError::InvalidConfig(format!(
                    "smooth_sign alpha must be finite and positive, got {alpha}"
                )));
            }
        }
        if self.use_itq && self.itq_n_iter == 0 {
            return Err(LittleBitError::InvalidConfig(
                "itq_n_iter must be greater than 0 when use_itq is enabled".to_string(),
            ));
        }
        parse_eff_bit_layers(self.eff_bit_layers.as_deref())?;
        Ok(())
    }

    /// Resolve the effective bits target for `module_name` by applying matching
    /// [`LittleBitConfig::eff_bit_layers`] overrides over [`LittleBitConfig::eff_bit`].
    ///
    /// # Errors
    /// Returns [`LittleBitError::InvalidConfig`] if `eff_bit_layers` is malformed.
    pub fn eff_bit_for_module(&self, module_name: &str) -> Result<Option<f64>> {
        let mut matched = None;
        if let Some(after) = module_name
            .find(".layers.")
            .map(|idx| &module_name[idx + ".layers.".len()..])
        {
            if let Some(seg) = after.split('.').next() {
                if let Ok(layer_idx) = seg.parse::<usize>() {
                    for (blocks, bits) in parse_eff_bit_layers(self.eff_bit_layers.as_deref())? {
                        if blocks.binary_search(&layer_idx).is_ok() {
                            matched = Some(bits);
                        }
                    }
                }
            }
        }
        Ok(matched.or(self.eff_bit))
    }

    /// Return a config scoped to `module_name`, applying any per-block `eff_bit_layers` override.
    ///
    /// # Errors
    /// Returns [`LittleBitError::InvalidConfig`] if `eff_bit_layers` is malformed.
    pub fn effective_for_module(&self, module_name: &str) -> Result<Self> {
        if self.eff_bit_layers.is_none() {
            return Ok(self.clone());
        }
        let resolved = self.eff_bit_for_module(module_name)?;
        let mut cloned = self.clone();
        cloned.eff_bit = resolved;
        Ok(cloned)
    }

    /// Load configuration using upstream priority:
    /// 1. Explicit `overrides`
    /// 2. `littlebit_config.json` in `dir`
    /// 3. `config.json` fallback fields for legacy checkpoints
    /// 4. [`LittleBitConfig::default`]
    ///
    /// # Errors
    /// Returns an error if a present config file cannot be read or parsed, or if the
    /// resulting configuration fails [`LittleBitConfig::validate`].
    pub fn load_with_priority(dir: &Path, overrides: &LittleBitConfigOverrides) -> Result<Self> {
        let default = Self::default();
        let littlebit_path = dir.join("littlebit_config.json");
        let mut config = if littlebit_path.exists() {
            let raw = fs::read_to_string(&littlebit_path)?;
            serde_json::from_str::<Self>(&raw)?
        } else {
            default.clone()
        };

        let model_cfg_path = dir.join("config.json");
        if model_cfg_path.exists() {
            let raw = fs::read_to_string(&model_cfg_path)?;
            let value: serde_json::Value = serde_json::from_str(&raw)?;
            if let Some(obj) = value.as_object() {
                apply_legacy_fallback(&mut config, &default, obj)?;
            }
        }

        apply_overrides(&mut config, overrides);
        config.validate()?;
        Ok(config)
    }
}

fn apply_legacy_fallback(
    config: &mut LittleBitConfig,
    default: &LittleBitConfig,
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    if config.quant_func == default.quant_func {
        if let Some(v) = obj.get("quant_func").and_then(serde_json::Value::as_str) {
            config.quant_func = v.parse()?;
        }
    }
    if config.eff_bit == default.eff_bit {
        if let Some(val) = obj.get("eff_bit") {
            if val.is_null() {
                config.eff_bit = None;
            } else if let Some(b) = val.as_f64() {
                config.eff_bit = Some(b);
            }
        }
    }
    if config.eff_bit_layers.is_none() {
        if let Some(spec) = obj
            .get("eff_bit_layers")
            .and_then(serde_json::Value::as_str)
        {
            config.eff_bit_layers = Some(spec.to_string());
        }
    }
    if config.split_dim == default.split_dim {
        if let Some(s) = obj.get("split_dim").and_then(serde_json::Value::as_u64) {
            config.split_dim = s as usize;
        }
    }
    if config.residual == default.residual {
        if let Some(r) = obj.get("residual").and_then(serde_json::Value::as_bool) {
            config.residual = r;
        }
    }
    if config.quant_mod == default.quant_mod {
        if let Some(m) = obj.get("quant_mod").and_then(serde_json::Value::as_str) {
            config.quant_mod = m.to_string();
        }
    }
    if (config.kv_factor - default.kv_factor).abs() < f64::EPSILON {
        if let Some(k) = obj.get("kv_factor").and_then(serde_json::Value::as_f64) {
            config.kv_factor = k;
        }
    }
    if config.min_split_dim == default.min_split_dim {
        if let Some(m) = obj.get("min_split_dim").and_then(serde_json::Value::as_u64) {
            config.min_split_dim = m as usize;
        }
    }
    if config.use_itq == default.use_itq {
        if let Some(u) = obj.get("use_itq").and_then(serde_json::Value::as_bool) {
            config.use_itq = u;
        }
    }
    if config.itq_n_iter == default.itq_n_iter {
        if let Some(n) = obj.get("itq_n_iter").and_then(serde_json::Value::as_u64) {
            config.itq_n_iter = n as usize;
        }
    }
    Ok(())
}

fn apply_overrides(config: &mut LittleBitConfig, overrides: &LittleBitConfigOverrides) {
    if let Some(qf) = overrides.quant_func {
        config.quant_func = qf;
    }
    if let Some(eff_bit) = overrides.eff_bit {
        config.eff_bit = eff_bit;
    }
    if let Some(spec) = &overrides.eff_bit_layers {
        config.eff_bit_layers = spec.clone();
    }
    if let Some(split_dim) = overrides.split_dim {
        config.split_dim = split_dim;
    }
    if let Some(residual) = overrides.residual {
        config.residual = residual;
    }
    if let Some(kv_factor) = overrides.kv_factor {
        config.kv_factor = kv_factor;
    }
    if let Some(min_split_dim) = overrides.min_split_dim {
        config.min_split_dim = min_split_dim;
    }
    if let Some(quant_mod) = &overrides.quant_mod {
        config.quant_mod = quant_mod.clone();
    }
    if let Some(use_itq) = overrides.use_itq {
        config.use_itq = use_itq;
    }
    if let Some(itq_n_iter) = overrides.itq_n_iter {
        config.itq_n_iter = itq_n_iter;
    }
}

/// Parse a per-block effective-bit specification such as `"0-7,27:2.0;12:1.5"`.
///
/// Matches `littlebit-bench/patches/mixed-bits.patch::parse_eff_bit_layers`.
/// Each `;`-delimited group has `<blocks>:<eff_bit>` where `<blocks>` is a comma-separated
/// list of single indices or inclusive `a-b` ranges. Each group's block indices are
/// deduplicated and sorted ascending. Returns an empty vector for `None` or blank strings.
///
/// # Errors
/// Returns [`LittleBitError::InvalidConfig`] if any group lacks `:`, contains an invalid or
/// reversed block range, contains an empty block list, or specifies a non-positive/non-finite bit target.
pub fn parse_eff_bit_layers(spec: Option<&str>) -> Result<Vec<(Vec<usize>, f64)>> {
    let Some(raw) = spec else {
        return Ok(Vec::new());
    };
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for raw_group in raw.split(';') {
        let group = raw_group.trim();
        if group.is_empty() {
            continue;
        }
        let (blocks_s, bits_s) = group.rsplit_once(':').ok_or_else(|| {
            LittleBitError::InvalidConfig(format!(
                "invalid eff_bit_layers group `{group}` (expected `<blocks>:<eff_bit>`)"
            ))
        })?;
        let bits: f64 = bits_s.trim().parse().map_err(|_| {
            LittleBitError::InvalidConfig(format!(
                "invalid eff_bit value `{bits_s}` in eff_bit_layers group `{group}`"
            ))
        })?;
        if !bits.is_finite() || bits <= 0.0 {
            return Err(LittleBitError::InvalidConfig(format!(
                "eff_bit in eff_bit_layers must be finite and positive, got {bits}"
            )));
        }

        let mut blocks = BTreeSet::new();
        for raw_part in blocks_s.split(',') {
            let part = raw_part.trim();
            if part.is_empty() {
                return Err(LittleBitError::InvalidConfig(format!(
                    "empty block specification in eff_bit_layers group `{group}`"
                )));
            }
            if let Some((a_s, b_s)) = part.split_once('-') {
                let start: usize = a_s.trim().parse().map_err(|_| {
                    LittleBitError::InvalidConfig(format!(
                        "invalid range start `{a_s}` in eff_bit_layers group `{group}`"
                    ))
                })?;
                let end: usize = b_s.trim().parse().map_err(|_| {
                    LittleBitError::InvalidConfig(format!(
                        "invalid range end `{b_s}` in eff_bit_layers group `{group}`"
                    ))
                })?;
                if start > end {
                    return Err(LittleBitError::InvalidConfig(format!(
                        "invalid descending range `{part}` in eff_bit_layers group `{group}`"
                    )));
                }
                for idx in start..=end {
                    blocks.insert(idx);
                }
            } else {
                let idx: usize = part.parse().map_err(|_| {
                    LittleBitError::InvalidConfig(format!(
                        "invalid block index `{part}` in eff_bit_layers group `{group}`"
                    ))
                })?;
                blocks.insert(idx);
            }
        }

        if blocks.is_empty() {
            return Err(LittleBitError::InvalidConfig(format!(
                "no block indices found in eff_bit_layers group `{group}`"
            )));
        }
        out.push((blocks.into_iter().collect(), bits));
    }

    Ok(out)
}

/// Resolved bit-budget and rank geometry for a single quantized linear layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitBudget {
    /// Input feature dimension `a`.
    pub in_features: usize,
    /// Output feature dimension `b`.
    pub out_features: usize,
    /// Requested effective bits per weight (`None` when fixed `split_dim` was used).
    pub eff_bit_target: Option<f64>,
    /// Final quantized inner rank `s` (rounded to a multiple of 8 with `min_split_dim` floor).
    pub split_dim: usize,
    /// Actual effective bits per weight achieved by `(in_features, out_features, split_dim, residual)`.
    pub eff_bit_actual: f64,
    /// Total bit footprint `eff_bit_actual * in_features * out_features`.
    pub total_bit_usage: f64,
    /// Single vs. dual residual factorization mode.
    pub residual: ResidualMode,
}

impl BitBudget {
    /// Compute the exact layer rank and bit budget using the upstream LittleBit formula.
    #[must_use]
    pub fn resolve(
        in_features: usize,
        out_features: usize,
        config: &LittleBitConfig,
        ratio_factor: f64,
    ) -> Self {
        let residual = config.residual_mode();
        let split_calc_float =
            estimate_split_dim(in_features, out_features, config.eff_bit, residual)
                .map(|v| v * ratio_factor);
        let split_dim =
            finalize_split_dim(split_calc_float, config.split_dim, config.min_split_dim);
        let eff_bit_actual = compute_eff_bits(in_features, out_features, split_dim, residual);
        let total_bit_usage = eff_bit_actual * (in_features as f64) * (out_features as f64);
        Self {
            in_features,
            out_features,
            eff_bit_target: config.eff_bit,
            split_dim,
            eff_bit_actual,
            total_bit_usage,
            residual,
        }
    }
}

/// Estimate the floating-point latent rank `split_dim` needed to match `eff_bit_target`.
///
/// Matches `LittleBitLinear._estimate_split_dim`.
#[must_use]
pub fn estimate_split_dim(
    in_features: usize,
    out_features: usize,
    eff_bit_target: Option<f64>,
    residual: ResidualMode,
) -> Option<f64> {
    let target = eff_bit_target?;
    let a = in_features as f64;
    let b = out_features as f64;
    if in_features == 0 || out_features == 0 {
        return None;
    }

    let base = a + b + 16.0;
    let (numerator, denominator) = match residual {
        ResidualMode::Dual => (a * b * target - 32.0 * (a + b), 2.0 * base),
        ResidualMode::Single => (a * b * target - 16.0 * (a + b), base),
    };
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

/// Round `split_float` (or `split_default`) down to a multiple of 8 with a minimum floor.
///
/// Matches `LittleBitLinear._finalize_split_dim`.
#[must_use]
pub fn finalize_split_dim(
    split_float: Option<f64>,
    split_default: usize,
    min_split_dim: usize,
) -> usize {
    let cand = match split_float {
        Some(v) => v as isize,
        None => split_default as isize,
    };
    let mut rounded = cand.div_euclid(8) * 8;
    if rounded == 0 {
        rounded = min_split_dim as isize;
    }
    rounded.max(min_split_dim as isize) as usize
}

/// Compute actual effective bits per weight for a layer geometry.
///
/// Matches `LittleBitLinear._compute_eff_bits`:
/// - `Single`: `(s * (a + b + 16) + 16 * (a + b)) / (a * b)`
/// - `Dual`: `(s * 2 * (a + b + 16) + 32 * (a + b)) / (a * b)`
#[must_use]
pub fn compute_eff_bits(
    in_features: usize,
    out_features: usize,
    split_dim: usize,
    residual: ResidualMode,
) -> f64 {
    if in_features == 0 || out_features == 0 {
        return f64::INFINITY;
    }
    let a = in_features as f64;
    let b = out_features as f64;
    let s = split_dim as f64;
    let num = match residual {
        ResidualMode::Dual => s * 2.0 * (a + b + 16.0) + 32.0 * (a + b),
        ResidualMode::Single => s * (a + b + 16.0) + 16.0 * (a + b),
    };
    num / (a * b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn bit_budget_matches_upstream_llama7b_shapes() {
        let cfg = LittleBitConfig {
            eff_bit: Some(0.1),
            residual: true,
            min_split_dim: 8,
            ..Default::default()
        };
        // 4096 x 4096 at 0.1 bpw with residual=True
        let b1 = BitBudget::resolve(4096, 4096, &cfg, 1.0);
        let raw = (4096.0_f64 * 4096.0 * 0.1 - 32.0 * 8192.0) / (2.0 * (8192.0 + 16.0));
        let expected_s = ((raw as isize).div_euclid(8) * 8).max(8) as usize;
        assert_eq!(b1.split_dim, expected_s);
        assert!(b1.eff_bit_actual <= 0.11);

        // kv_factor scales the rank before rounding
        let b_kv = BitBudget::resolve(4096, 1024, &cfg, 2.0);
        assert!(b_kv.split_dim >= 8);
        assert_eq!(b_kv.split_dim % 8, 0);
    }

    #[test]
    fn finalize_split_dim_floors_zero_and_negatives() {
        assert_eq!(finalize_split_dim(Some(-42.5), 1024, 8), 8);
        assert_eq!(finalize_split_dim(Some(7.9), 1024, 8), 8);
        assert_eq!(finalize_split_dim(Some(63.9), 1024, 8), 56);
        assert_eq!(finalize_split_dim(None, 1024, 8), 1024);
    }

    #[test]
    fn config_priority_follows_overrides_then_littlebit_then_legacy() {
        let dir = TempDir::new().expect("tempdir");
        fs::write(
            dir.path().join("config.json"),
            r#"{"quant_func": "SmoothSign", "split_dim": 512, "use_itq": true}"#,
        )
        .expect("write config.json");
        fs::write(
            dir.path().join("littlebit_config.json"),
            r#"{"eff_bit": 0.25, "residual": true}"#,
        )
        .expect("write littlebit_config.json");

        let overrides = LittleBitConfigOverrides {
            split_dim: Some(256),
            ..Default::default()
        };
        let resolved =
            LittleBitConfig::load_with_priority(dir.path(), &overrides).expect("load config");
        assert_eq!(
            resolved.quant_func,
            QuantFunc::SmoothSign {
                alpha: SMOOTH_SIGN_DEFAULT_ALPHA
            }
        );
        assert_eq!(resolved.eff_bit, Some(0.25));
        assert!(resolved.residual);
        assert!(resolved.use_itq);
        assert_eq!(resolved.split_dim, 256);
    }
}
