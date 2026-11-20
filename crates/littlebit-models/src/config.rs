//! Hugging Face `config.json` parsing and architecture normalization across all supported LittleBit families.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Deserializer, Serialize};

use crate::error::{ModelError, Result};

/// Supported LLM architecture families matching `SamsungLabs/LittleBit` plus Gemma 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Llama, Llama 2, and Llama 3 (`model_type = "llama"`).
    Llama,
    /// Qwen2, Qwen2.5, and QwQ (`model_type = "qwen2"`).
    Qwen2,
    /// Qwen3 (`model_type = "qwen3"`).
    Qwen3,
    /// Phi-3 and Phi-4 (`model_type = "phi3"`).
    Phi3,
    /// Gemma 2 (`model_type = "gemma2"`).
    Gemma2,
    /// Gemma 3 (`model_type = "gemma3" | "gemma3_text"`).
    Gemma3,
    /// Gemma 4 Dense and MoE (`model_type = "gemma4" | "gemma4_text"`).
    Gemma4,
    /// Meta OPT (`model_type = "opt"`).
    Opt,
}

/// Resolved RoPE configuration for a specific attention layer type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedRopeParams {
    /// Inverse frequency base (`rope_theta`).
    pub rope_theta: f64,
    /// Fraction of head dimensions rotated (`(0.0, 1.0]`).
    pub partial_rotary_factor: f64,
    /// Frequency scaling divisor (`factor`, `1.0` when unscaled).
    pub scale_factor: f64,
    /// Whether `rope_type == "proportional"` (full `head_dim` denominator in frequencies).
    pub is_proportional: bool,
}

/// Token ID specification allowing either a single ID or a list (as in Llama 3 `eos_token_id`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum TokenIdSpec {
    /// Single token ID.
    Single(u32),
    /// Multiple valid token IDs (e.g., EOS + EOT).
    Multiple(Vec<u32>),
}

impl TokenIdSpec {
    /// Primary token ID (`Single(id)` or `Multiple[0]`).
    #[must_use]
    pub fn primary(&self) -> Option<u32> {
        match self {
            Self::Single(id) => Some(*id),
            Self::Multiple(ids) => ids.first().copied(),
        }
    }

    /// Return true if `id` is contained in this token spec.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        match self {
            Self::Single(v) => *v == id,
            Self::Multiple(vs) => vs.contains(&id),
        }
    }
}

impl<'de> Deserialize<'de> for TokenIdSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            One(u32),
            Many(Vec<u32>),
        }
        match Raw::deserialize(deserializer)? {
            Raw::One(v) => Ok(Self::Single(v)),
            Raw::Many(vs) => Ok(Self::Multiple(vs)),
        }
    }
}

/// Unified Hugging Face `config.json` model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Architecture identifier (`llama`, `qwen2`, `qwen3`, `phi3`, `gemma2`, `gemma3`, `opt`).
    pub model_type: String,
    /// Hugging Face architecture class names.
    pub architectures: Vec<String>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Transformer hidden dimension.
    pub hidden_size: usize,
    /// Feed-forward intermediate dimension (`ffn_dim` for OPT).
    pub intermediate_size: usize,
    /// OPT alias for `intermediate_size`.
    pub ffn_dim: Option<usize>,
    /// OPT projection dimension between embeddings and decoder hidden states.
    pub word_embed_proj_dim: Option<usize>,
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value attention heads (GQA when smaller than `num_attention_heads`).
    pub num_key_value_heads: Option<usize>,
    /// Explicit per-head dimension (defaults to `hidden_size / num_attention_heads`).
    pub head_dim: Option<usize>,
    /// Maximum context length.
    pub max_position_embeddings: usize,
    /// Legacy context length alias (`n_ctx`).
    pub n_ctx: Option<usize>,
    /// RMSNorm / LayerNorm epsilon.
    pub rms_norm_eps: f64,
    /// RoPE base frequency (`rope_theta`).
    pub rope_theta: f64,
    /// Local sliding-window RoPE base frequency for Gemma 3.
    pub rope_local_base_freq: Option<f64>,
    /// Whether `lm_head` shares weights with `embed_tokens`.
    pub tie_word_embeddings: bool,
    /// Whether Q/K/V/O attention projections carry biases.
    pub attention_bias: bool,
    /// Whether MLP projections carry biases.
    pub mlp_bias: bool,
    /// OPT bias toggle (`enable_bias`).
    pub enable_bias: Option<bool>,
    /// OPT pre-norm vs. post-norm toggle (`do_layer_norm_before`).
    pub do_layer_norm_before: Option<bool>,
    /// OPT affine toggle on LayerNorm.
    pub layer_norm_elementwise_affine: Option<bool>,
    /// Hidden activation function name (`silu`, `gelu`, `gelu_pytorch_tanh`, `relu`).
    pub hidden_act: String,
    /// Gemma 2/3 attention logit softcapping threshold.
    pub attn_logit_softcapping: Option<f64>,
    /// Gemma 2/3 final logit softcapping threshold.
    pub final_logit_softcapping: Option<f64>,
    /// Gemma 2/3 attention scaling denominator (`1 / sqrt(query_pre_attn_scalar)`).
    pub query_pre_attn_scalar: Option<f64>,
    /// Sliding window size (Gemma 2/3, Phi).
    pub sliding_window: Option<usize>,
    /// Gemma 3 sliding window periodicity (every Nth layer is global).
    pub sliding_window_pattern: Option<usize>,
    /// Per-layer attention type schedule (`"sliding_attention"` vs `"full_attention"`), used by Gemma 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    /// Global (`full_attention`) head dimension when larger than sliding `head_dim` (Gemma 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_head_dim: Option<usize>,
    /// Number of key/value heads for global (`full_attention`) layers when distinct from sliding layers (Gemma 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_global_key_value_heads: Option<usize>,
    /// Whether global attention layers share keys and values (`V = K`, omitting `v_proj`), used by Gemma 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_k_eq_v: Option<bool>,
    /// Per-layer-type RoPE parameters (`rope_type`, `rope_theta`, `partial_rotary_factor`, `factor`) for Gemma 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_parameters: Option<serde_json::Value>,
    /// Default fraction of head dims rotated when partial RoPE is enabled without nested `rope_parameters`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_rotary_factor: Option<f64>,
    /// Alternative activation key (`hidden_activation` alias for `hidden_act`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_activation: Option<String>,
    /// Enable Mixture-of-Experts FFN block alongside the shared dense MLP (Gemma 4 MoE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_moe_block: Option<bool>,
    /// Total number of routed experts in MoE layers (Gemma 4 MoE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<usize>,
    /// Number of top-ranked experts activated per token (e.g., 8 in Gemma 4 MoE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k_experts: Option<usize>,
    /// Intermediate dimension of each routed MoE expert MLP (Gemma 4 MoE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_intermediate_size: Option<usize>,
    /// Number of trailing KV-shared edge layers (Gemma 4 E2B/E4B; `0` for Dense 31B / MoE 26B).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_kv_shared_layers: Option<usize>,
    /// Double-wide MLP flag for KV-shared layers (Gemma 4 edge variants).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_double_wide_mlp: Option<bool>,
    /// Optional nested `text_config` used by Gemma 3 / Gemma 4 multimodal checkpoints.
    pub text_config: Option<Box<ModelConfig>>,
    /// Beginning-of-sequence token ID.
    pub bos_token_id: Option<TokenIdSpec>,
    /// End-of-sequence token ID(s).
    pub eos_token_id: Option<TokenIdSpec>,
    /// Padding token ID.
    pub pad_token_id: Option<TokenIdSpec>,
    /// Preserved extra `config.json` keys for lossless round-trip serialization.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_type: "llama".to_string(),
            architectures: Vec::new(),
            vocab_size: 32000,
            hidden_size: 0,
            intermediate_size: 0,
            ffn_dim: None,
            word_embed_proj_dim: None,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: None,
            head_dim: None,
            max_position_embeddings: 2048,
            n_ctx: None,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_local_base_freq: None,
            tie_word_embeddings: false,
            attention_bias: false,
            mlp_bias: false,
            enable_bias: None,
            do_layer_norm_before: None,
            layer_norm_elementwise_affine: None,
            hidden_act: "silu".to_string(),
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            sliding_window: None,
            sliding_window_pattern: None,
            layer_types: None,
            global_head_dim: None,
            num_global_key_value_heads: None,
            attention_k_eq_v: None,
            rope_parameters: None,
            partial_rotary_factor: None,
            hidden_activation: None,
            enable_moe_block: None,
            num_experts: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            num_kv_shared_layers: None,
            use_double_wide_mlp: None,
            text_config: None,
            bos_token_id: None,
            eos_token_id: None,
            pad_token_id: None,
            extra: serde_json::Map::new(),
        }
    }
}

impl ModelConfig {
    /// Load and parse a Hugging Face `config.json` file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or fails JSON deserialization.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&raw)?;
        Ok(cfg.resolve_effective())
    }

    /// Unwrap nested `text_config` when present (e.g., Gemma 3 / Gemma 4 multimodal checkpoints).
    #[must_use]
    pub fn resolve_effective(mut self) -> Self {
        if self.hidden_size == 0 {
            if let Some(inner) = self.text_config.take() {
                let mut resolved = *inner;
                if resolved.model_type.is_empty()
                    || (resolved.model_type == "llama"
                        && !self.model_type.eq_ignore_ascii_case("llama"))
                {
                    resolved.model_type = self.model_type;
                }
                if resolved.architectures.is_empty() {
                    resolved.architectures = self.architectures;
                }
                if !resolved.tie_word_embeddings && self.tie_word_embeddings {
                    resolved.tie_word_embeddings = true;
                }
                if resolved.bos_token_id.is_none() {
                    resolved.bos_token_id = self.bos_token_id;
                }
                if resolved.eos_token_id.is_none() {
                    resolved.eos_token_id = self.eos_token_id;
                }
                if resolved.pad_token_id.is_none() {
                    resolved.pad_token_id = self.pad_token_id;
                }
                return resolved.resolve_effective();
            }
        }
        if self.intermediate_size == 0 {
            if let Some(ffn) = self.ffn_dim {
                self.intermediate_size = ffn;
            }
        }
        self
    }

    /// Detect the canonical [`ModelArchitecture`] family.
    ///
    /// # Errors
    /// Returns [`ModelError::UnsupportedArchitecture`] for unrecognized `model_type`.
    pub fn detect_architecture(&self) -> Result<ModelArchitecture> {
        let mt = self.model_type.to_ascii_lowercase();
        match mt.as_str() {
            "llama" => Ok(ModelArchitecture::Llama),
            "qwen2" => Ok(ModelArchitecture::Qwen2),
            "qwen3" => Ok(ModelArchitecture::Qwen3),
            "phi3" => Ok(ModelArchitecture::Phi3),
            "gemma2" => Ok(ModelArchitecture::Gemma2),
            "gemma3" | "gemma3_text" => Ok(ModelArchitecture::Gemma3),
            "gemma4" | "gemma4_text" => Ok(ModelArchitecture::Gemma4),
            "opt" => Ok(ModelArchitecture::Opt),
            _ => Err(ModelError::UnsupportedArchitecture(format!(
                "unsupported model_type `{}`; supported: opt, llama, phi3, qwen2, qwen3, gemma2, gemma3, gemma4",
                self.model_type
            ))),
        }
    }

    /// Effective hidden activation identifier (`hidden_activation` when set, otherwise `hidden_act`).
    #[must_use]
    pub fn effective_hidden_act(&self) -> &str {
        match self.hidden_activation.as_deref() {
            Some(act) if !act.is_empty() => act,
            _ => &self.hidden_act,
        }
    }

    /// Effective per-head dimension.
    #[must_use]
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim.unwrap_or_else(|| {
            self.hidden_size
                .checked_div(self.num_attention_heads)
                .unwrap_or(0)
        })
    }

    /// Effective global (`full_attention`) head dimension for Gemma 4.
    #[must_use]
    pub fn effective_global_head_dim(&self) -> usize {
        self.global_head_dim
            .unwrap_or_else(|| self.effective_head_dim())
    }

    /// Effective number of key/value heads.
    #[must_use]
    pub fn effective_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Effective number of key/value heads for global (`full_attention`) layers.
    #[must_use]
    pub fn effective_global_kv_heads(&self) -> usize {
        self.num_global_key_value_heads
            .unwrap_or_else(|| self.effective_kv_heads())
    }

    /// Per-layer attention type schedule (`"sliding_attention"` vs. `"full_attention"`).
    #[must_use]
    pub fn effective_layer_types(&self) -> Vec<String> {
        if let Some(ref lt) = self.layer_types {
            if lt.len() == self.num_hidden_layers {
                return lt.clone();
            }
        }
        if let Some(pattern) = self.sliding_window_pattern {
            if pattern > 0 {
                return (0..self.num_hidden_layers)
                    .map(|i| {
                        if (i + 1).is_multiple_of(pattern) {
                            "full_attention".to_string()
                        } else {
                            "sliding_attention".to_string()
                        }
                    })
                    .collect();
            }
        }
        vec!["full_attention".to_string(); self.num_hidden_layers]
    }

    /// Resolve Gemma 4 RoPE hyperparameters for a specific attention layer type.
    #[must_use]
    pub fn gemma4_rope_params(&self, layer_type: &str) -> ResolvedRopeParams {
        use serde_json::Value;

        let default_theta = if layer_type == "sliding_attention" {
            self.rope_local_base_freq.unwrap_or(self.rope_theta)
        } else {
            self.rope_theta
        };
        let default_partial = self.partial_rotary_factor.unwrap_or(1.0);

        let entry = self
            .rope_parameters
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|map| {
                map.get(layer_type).and_then(Value::as_object).or_else(|| {
                    if map.contains_key("rope_type")
                        || map.contains_key("rope_theta")
                        || map.contains_key("partial_rotary_factor")
                    {
                        Some(map)
                    } else {
                        None
                    }
                })
            });

        match entry {
            Some(obj) => ResolvedRopeParams {
                rope_theta: obj
                    .get("rope_theta")
                    .and_then(Value::as_f64)
                    .unwrap_or(default_theta),
                partial_rotary_factor: obj
                    .get("partial_rotary_factor")
                    .and_then(Value::as_f64)
                    .unwrap_or(default_partial),
                scale_factor: obj.get("factor").and_then(Value::as_f64).unwrap_or(1.0),
                is_proportional: obj
                    .get("rope_type")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s.eq_ignore_ascii_case("proportional")),
            },
            None => ResolvedRopeParams {
                rope_theta: default_theta,
                partial_rotary_factor: default_partial,
                scale_factor: 1.0,
                is_proportional: false,
            },
        }
    }

    /// Effective sequence length matching `eval.py` fallback (`n_ctx` -> `max_position_embeddings` -> `2048`).
    #[must_use]
    pub fn effective_seqlen(&self) -> usize {
        self.n_ctx.unwrap_or(if self.max_position_embeddings > 0 {
            self.max_position_embeddings
        } else {
            2048
        })
    }

    /// Effective intermediate size (`ffn_dim` for OPT or `intermediate_size` otherwise).
    #[must_use]
    pub fn effective_intermediate_size(&self) -> usize {
        if self.intermediate_size > 0 {
            self.intermediate_size
        } else {
            self.ffn_dim.unwrap_or(0)
        }
    }

    /// Build a compact synthetic Llama config suitable for unit tests and quick prototyping.
    #[must_use]
    pub fn tiny_llama(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        Self {
            model_type: "llama".to_string(),
            architectures: vec!["LlamaForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            ..Default::default()
        }
    }

    /// Build a compact synthetic Qwen2 config.
    #[must_use]
    pub fn tiny_qwen2(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        Self {
            model_type: "qwen2".to_string(),
            architectures: vec!["Qwen2ForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            attention_bias: true,
            ..Default::default()
        }
    }

    /// Build a compact synthetic Qwen3 config.
    #[must_use]
    pub fn tiny_qwen3(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        Self {
            model_type: "qwen3".to_string(),
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(hidden_size / 4),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            ..Default::default()
        }
    }

    /// Build a compact synthetic Phi-3/Phi-4 config.
    #[must_use]
    pub fn tiny_phi3(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        Self {
            model_type: "phi3".to_string(),
            architectures: vec!["Phi3ForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_bias: false,
            ..Default::default()
        }
    }

    /// Build a compact synthetic Gemma 2 config.
    #[must_use]
    pub fn tiny_gemma2(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        Self {
            model_type: "gemma2".to_string(),
            architectures: vec!["Gemma2ForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(hidden_size / 4),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: true,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            attn_logit_softcapping: Some(50.0),
            final_logit_softcapping: Some(30.0),
            query_pre_attn_scalar: Some((hidden_size / 4) as f64),
            sliding_window: Some(64),
            ..Default::default()
        }
    }

    /// Build a compact synthetic Gemma 3 config.
    #[must_use]
    pub fn tiny_gemma3(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        Self {
            model_type: "gemma3_text".to_string(),
            architectures: vec!["Gemma3ForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(hidden_size / 4),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: Some(10000.0),
            tie_word_embeddings: true,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            query_pre_attn_scalar: Some((hidden_size / 4) as f64),
            sliding_window: Some(64),
            sliding_window_pattern: Some(2),
            ..Default::default()
        }
    }

    /// Build a compact synthetic Dense Gemma 4 config (`sliding_attention` + `full_attention`, p-RoPE, `k_eq_v`).
    #[must_use]
    pub fn tiny_gemma4_dense(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        let layer_types = (0..layers)
            .map(|i| {
                if (i + 1).is_multiple_of(2) {
                    "full_attention".to_string()
                } else {
                    "sliding_attention".to_string()
                }
            })
            .collect();
        let head_dim = (hidden_size / 4).max(2);
        let global_head_dim = (hidden_size / 2).max(8);
        Self {
            model_type: "gemma4_text".to_string(),
            architectures: vec!["Gemma4ForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            num_hidden_layers: layers,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(head_dim),
            global_head_dim: Some(global_head_dim),
            num_global_key_value_heads: Some(1),
            attention_k_eq_v: Some(true),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: Some(10_000.0),
            tie_word_embeddings: true,
            hidden_act: "gelu_pytorch_tanh".to_string(),
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            final_logit_softcapping: Some(30.0),
            sliding_window: Some(64),
            layer_types: Some(layer_types),
            rope_parameters: Some(serde_json::json!({
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0
                }
            })),
            ..Default::default()
        }
    }

    /// Build a compact synthetic MoE Gemma 4 config (dual shared MLP + top-k routed experts).
    #[must_use]
    pub fn tiny_gemma4_moe(
        vocab_size: usize,
        hidden_size: usize,
        layers: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Self {
        let mut cfg = Self::tiny_gemma4_dense(vocab_size, hidden_size, layers);
        cfg.enable_moe_block = Some(true);
        cfg.num_experts = Some(num_experts);
        cfg.top_k_experts = Some(top_k);
        cfg.moe_intermediate_size = Some(hidden_size);
        cfg
    }

    /// Build a compact synthetic OPT config.
    #[must_use]
    pub fn tiny_opt(vocab_size: usize, hidden_size: usize, layers: usize) -> Self {
        Self {
            model_type: "opt".to_string(),
            architectures: vec!["OPTForCausalLM".to_string()],
            vocab_size,
            hidden_size,
            intermediate_size: hidden_size * 2,
            ffn_dim: Some(hidden_size * 2),
            word_embed_proj_dim: Some(hidden_size),
            num_hidden_layers: layers,
            num_attention_heads: 4,
            max_position_embeddings: 256,
            tie_word_embeddings: true,
            enable_bias: Some(true),
            do_layer_norm_before: Some(true),
            layer_norm_elementwise_affine: Some(true),
            ..Default::default()
        }
    }
}
