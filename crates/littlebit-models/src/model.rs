//! Unified [`CausalLm`] abstraction dispatching across all 6 supported LLM families.

use std::collections::HashMap;
use std::fmt;

use candle_core::{DType, Device, Tensor, Var};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use littlebit_core::{InferenceMode, LittleBitConfig, TrainMode, WeightState};

use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::Result;
use crate::gemma::GemmaModel;
use crate::layer::{KvCache, LinearLoadSpec};
use crate::llama::LlamaModel;
use crate::opt::OptModel;
use crate::patch::{PatchPolicy, PatchSummary};
use crate::phi::PhiModel;

/// Whether [`CausalLm::forward`] should collect per-layer hidden states (used during KD training).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HiddenStatesMode {
    /// Return only output logits (`hidden_states` vector is empty).
    None,
    /// Collect embeddings and per-layer hidden outputs (`len == num_hidden_layers + 1`).
    All,
}

/// Output returned by [`CausalLm::forward`].
#[derive(Debug, Clone)]
pub struct CausalLmOutput {
    /// Unnormalized vocabulary logits `(batch, seq_len, vocab_size)`.
    pub logits: Tensor,
    /// Optional hidden states `[embed, layer_0, ..., final_norm]` when [`HiddenStatesMode::All`].
    pub hidden_states: Vec<Tensor>,
}

/// Trainable vs. total parameter summary matching `utils/utils.py::print_trainable_parameters`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterStats {
    /// Count of scalar elements inside trainable `Var`s.
    pub trainable_params: usize,
    /// Approximate total scalar parameters across the model.
    pub all_params: usize,
}

impl ParameterStats {
    /// Percentage of parameters that are trainable (`100 * trainable / all`).
    #[must_use]
    pub fn trainable_percent(self) -> f64 {
        if self.all_params == 0 {
            0.0
        } else {
            100.0 * (self.trainable_params as f64) / (self.all_params as f64)
        }
    }
}

impl fmt::Display for ParameterStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "trainable params: {} || all params: {} || trainable%: {:.4}",
            self.trainable_params,
            self.all_params,
            self.trainable_percent()
        )
    }
}

/// Sampling and termination configuration for [`CausalLm::generate`].
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of new tokens to decode.
    pub max_new_tokens: usize,
    /// Sampling temperature (`<= 0.0` triggers greedy argmax decoding).
    pub temperature: f64,
    /// Nucleus (top-p) probability threshold in `(0.0, 1.0]`.
    pub top_p: f64,
    /// Deterministic RNG seed for stochastic sampling.
    pub seed: u64,
    /// Optional stop token IDs halting generation early.
    pub eos_token_ids: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            eos_token_ids: Vec::new(),
        }
    }
}

/// Unified causal language model supporting all architectures in `SamsungLabs/LittleBit`.
#[derive(Debug, Clone)]
pub enum CausalLm {
    /// Llama, Llama 2/3, Qwen2/2.5/QwQ, and Qwen3.
    LlamaFamily(LlamaModel),
    /// Phi-3 and Phi-4 (including `PhiQKVSplitAttention`).
    Phi(PhiModel),
    /// Gemma 2 and Gemma 3.
    Gemma(GemmaModel),
    /// Meta OPT.
    Opt(Box<OptModel>),
}

impl CausalLm {
    /// Construct a model from a parsed [`ModelConfig`] and loaded checkpoint tensors.
    ///
    /// # Errors
    /// Returns an error if the architecture is unsupported or required weights are missing.
    pub fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        littlebit: Option<&LittleBitConfig>,
        weight_state: WeightState,
        inference_mode: InferenceMode,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let config = config.resolve_effective();
        let arch = config.detect_architecture()?;
        let spec = LinearLoadSpec {
            littlebit,
            ratio_factor: 1.0,
            weight_state,
            inference_mode,
            dtype,
            device,
        };
        match arch {
            ModelArchitecture::Llama | ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3 => Ok(
                Self::LlamaFamily(LlamaModel::from_tensors(config, tensors, spec)?),
            ),
            ModelArchitecture::Phi3 => {
                Ok(Self::Phi(PhiModel::from_tensors(config, tensors, spec)?))
            }
            ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3 => Ok(Self::Gemma(
                GemmaModel::from_tensors(config, tensors, spec)?,
            )),
            ModelArchitecture::Opt => Ok(Self::Opt(Box::new(OptModel::from_tensors(
                config, tensors, spec,
            )?))),
        }
    }

    /// Initialize a compact or full-sized random model for testing and benchmarking.
    ///
    /// # Errors
    /// Returns an error if architecture detection or tensor allocation fails.
    pub fn new_random(
        config: ModelConfig,
        dtype: DType,
        device: &Device,
        seed: u64,
    ) -> Result<Self> {
        let config = config.resolve_effective();
        let arch = config.detect_architecture()?;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match arch {
            ModelArchitecture::Llama | ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3 => Ok(
                Self::LlamaFamily(LlamaModel::new_random(config, dtype, device, &mut rng)?),
            ),
            ModelArchitecture::Phi3 => Ok(Self::Phi(PhiModel::new_random(
                config, dtype, device, &mut rng,
            )?)),
            ModelArchitecture::Gemma2 | ModelArchitecture::Gemma3 => Ok(Self::Gemma(
                GemmaModel::new_random(config, dtype, device, &mut rng)?,
            )),
            ModelArchitecture::Opt => Ok(Self::Opt(Box::new(OptModel::new_random(
                config, dtype, device, &mut rng,
            )?))),
        }
    }

    /// Execute causal forward pass over `input_ids` `(batch, seq_len)`.
    ///
    /// # Errors
    /// Returns an error if layer computation fails.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        cache: Option<&mut KvCache>,
        hidden_mode: HiddenStatesMode,
    ) -> Result<CausalLmOutput> {
        match self {
            Self::LlamaFamily(m) => m.forward(input_ids, cache, hidden_mode),
            Self::Phi(m) => m.forward(input_ids, cache, hidden_mode),
            Self::Gemma(m) => m.forward(input_ids, cache, hidden_mode),
            Self::Opt(m) => m.forward(input_ids, cache, hidden_mode),
        }
    }

    /// Apply LittleBit / LittleBit-2 quantization patch across all linear projections
    /// (excluding `lm_head`, splitting `qkv_proj` on Phi, and scaling `k_proj`/`v_proj` by `kv_factor`).
    ///
    /// # Errors
    /// Returns an error if config validation or SVD/ITQ decomposition fails.
    pub fn apply_littlebit_patch(
        &mut self,
        lb_cfg: &LittleBitConfig,
        train_mode: TrainMode,
        base_seed: u64,
    ) -> Result<PatchSummary> {
        let policy = PatchPolicy::new();
        let mut summary = PatchSummary::default();
        match self {
            Self::LlamaFamily(m) => {
                m.apply_littlebit_patch(lb_cfg, &policy, train_mode, base_seed, &mut summary)?;
            }
            Self::Phi(m) => {
                m.apply_littlebit_patch(lb_cfg, &policy, train_mode, base_seed, &mut summary)?;
            }
            Self::Gemma(m) => {
                m.apply_littlebit_patch(lb_cfg, &policy, train_mode, base_seed, &mut summary)?;
            }
            Self::Opt(m) => {
                m.apply_littlebit_patch(lb_cfg, &policy, train_mode, base_seed, &mut summary)?;
            }
        }
        Ok(summary)
    }

    /// Convert all `LittleBitLinear` layers into memory-compact 1-bit `PackedLittleBitLinear` modules.
    ///
    /// # Errors
    /// Returns an error if packing fails.
    pub fn convert_to_packed(&mut self) -> Result<()> {
        match self {
            Self::LlamaFamily(m) => m.convert_to_packed(),
            Self::Phi(m) => m.convert_to_packed(),
            Self::Gemma(m) => m.convert_to_packed(),
            Self::Opt(m) => m.convert_to_packed(),
        }
    }

    /// Prepare model parameters for QAT training, making all non-excluded parameters trainable
    /// and freezing parameters whose module path contains any pattern in `exclude`.
    ///
    /// Matches upstream `utils/utils.py::prepare_model_for_training`.
    ///
    /// # Errors
    /// Returns an error if wrapping parameters in [`Var`] fails.
    pub fn prepare_for_training(&mut self, exclude: &[&str]) -> Result<()> {
        match self {
            Self::LlamaFamily(m) => m.make_non_quantized_trainable()?,
            Self::Phi(m) => m.make_non_quantized_trainable()?,
            Self::Gemma(m) => m.make_non_quantized_trainable()?,
            Self::Opt(m) => m.make_non_quantized_trainable()?,
        }
        self.freeze_matching(exclude);
        Ok(())
    }

    /// Freeze all parameters matching any substring in `exclude`.
    pub fn freeze_matching(&mut self, exclude: &[&str]) {
        match self {
            Self::LlamaFamily(m) => m.freeze_matching(exclude),
            Self::Phi(m) => m.freeze_matching(exclude),
            Self::Gemma(m) => m.freeze_matching(exclude),
            Self::Opt(m) => m.freeze_matching(exclude),
        }
    }

    /// Collect all active trainable [`Var`] parameters across the model.
    #[must_use]
    pub fn trainable_vars(&self) -> Vec<Var> {
        match self {
            Self::LlamaFamily(m) => m.trainable_vars(),
            Self::Phi(m) => m.trainable_vars(),
            Self::Gemma(m) => m.trainable_vars(),
            Self::Opt(m) => m.trainable_vars(),
        }
    }

    /// Compute trainable and total scalar parameter counts.
    #[must_use]
    pub fn parameter_stats(&self) -> ParameterStats {
        let trainable_params: usize = self.trainable_vars().iter().map(|v| v.elem_count()).sum();
        let all_params = match self {
            Self::LlamaFamily(m) => m.total_params(),
            Self::Phi(m) => m.total_params(),
            Self::Gemma(m) => m.total_params(),
            Self::Opt(m) => m.total_params(),
        };
        ParameterStats {
            trainable_params,
            all_params,
        }
    }

    /// Export full model state dictionary with LittleBit layers binarized & 1-bit packed.
    ///
    /// # Errors
    /// Returns an error if packing any layer fails.
    pub fn pack_state_dict(&self) -> Result<HashMap<String, Tensor>> {
        match self {
            Self::LlamaFamily(m) => m.pack_state_dict(),
            Self::Phi(m) => m.pack_state_dict(),
            Self::Gemma(m) => m.pack_state_dict(),
            Self::Opt(m) => m.pack_state_dict(),
        }
    }

    /// Borrow underlying Hugging Face [`ModelConfig`].
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        match self {
            Self::LlamaFamily(m) => m.config(),
            Self::Phi(m) => m.config(),
            Self::Gemma(m) => m.config(),
            Self::Opt(m) => m.config(),
        }
    }

    /// Autoregressive text generation using [`KvCache`] given initial `prompt_ids`.
    ///
    /// # Errors
    /// Returns an error if `prompt_ids` is empty or forward/sampling fails.
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        cfg: &GenerateConfig,
        device: &Device,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Err(crate::error::ModelError::InvalidArgument(
                "prompt_ids must not be empty".to_string(),
            ));
        }

        let mut cache = KvCache::new(self.config().num_hidden_layers);
        let mut generated = prompt_ids.to_vec();
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

        let prompt_t = Tensor::from_slice(prompt_ids, (1, prompt_ids.len()), device)?;
        let out = self.forward(&prompt_t, Some(&mut cache), HiddenStatesMode::None)?;
        let last_pos = prompt_ids.len() - 1;
        let mut next_logits = out.logits.get(0)?.get(last_pos)?;

        for _ in 0..cfg.max_new_tokens {
            let next_id = sample_token(&next_logits, cfg.temperature, cfg.top_p, &mut rng)?;
            generated.push(next_id);
            if cfg.eos_token_ids.contains(&next_id) {
                break;
            }
            let step_t = Tensor::from_slice(&[next_id], (1, 1), device)?;
            let step_out = self.forward(&step_t, Some(&mut cache), HiddenStatesMode::None)?;
            next_logits = step_out.logits.get(0)?.get(0)?;
        }

        Ok(generated)
    }
}

fn sample_token(
    logits: &Tensor,
    temperature: f64,
    top_p: f64,
    rng: &mut ChaCha8Rng,
) -> Result<u32> {
    let probs_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if probs_vec.is_empty() {
        return Err(crate::error::ModelError::InvalidArgument(
            "empty vocabulary logits".to_string(),
        ));
    }

    if temperature <= 0.0 {
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in probs_vec.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return Ok(best_idx as u32);
    }

    let inv_temp = 1.0 / temperature;
    let max_logit = probs_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut scaled: Vec<(usize, f64)> = probs_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v as f64 - max_logit) * inv_temp).exp()))
        .collect();
    let sum: f64 = scaled.iter().map(|&(_, p)| p).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Ok(0);
    }
    for (_, p) in &mut scaled {
        *p /= sum;
    }

    if top_p < 1.0 && top_p > 0.0 {
        scaled.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut cum = 0.0_f64;
        let mut cutoff = scaled.len();
        for (idx, &(_, p)) in scaled.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                cutoff = idx + 1;
                break;
            }
        }
        scaled.truncate(cutoff.max(1));
        let norm: f64 = scaled.iter().map(|&(_, p)| p).sum();
        for (_, p) in &mut scaled {
            *p /= norm;
        }
    }

    let draw: f64 = rng.random();
    let mut acc = 0.0_f64;
    for &(idx, p) in &scaled {
        acc += p;
        if draw <= acc {
            return Ok(idx as u32);
        }
    }
    Ok(scaled.last().map_or(0, |&(idx, _)| idx as u32))
}
