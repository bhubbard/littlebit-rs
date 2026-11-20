//! Hugging Face Hub integration and checkpoint packaging (`quantization/hub.py` parity).

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use hf_hub::{split_id, HFClient, RepoTypeModel};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;

use littlebit_core::{
    load_and_unpack_state_dict, load_raw_safetensors_dir, save_packed_safetensors, CastPolicy,
    CheckpointFormat, InferenceMode, LittleBitConfig, LittleBitConfigOverrides, WeightState,
};

use crate::config::ModelConfig;
use crate::error::{ModelError, Result};
use crate::model::CausalLm;

const TOKENIZER_ARTIFACT_FILES: [&str; 6] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
];

#[derive(Debug, Serialize, Deserialize)]
struct BaseModelMeta {
    model_id: String,
}

/// Options controlling how a checkpoint or dense model is loaded from disk or Hub.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Target compute dtype (`DType::BF16` or `DType::F32`).
    pub dtype: DType,
    /// Target compute device (`Device::Cpu`, Metal, or CUDA).
    pub device: Device,
    /// Unpack binary weights into float matrices (`Unpacked`) or keep 1-bit packed (`Packed`).
    pub inference_mode: InferenceMode,
    /// Explicit CLI/API overrides applied with highest priority over config files.
    pub overrides: LittleBitConfigOverrides,
    /// Optional custom Hugging Face cache directory.
    pub cache_dir: Option<PathBuf>,
    /// Optional Hugging Face API token.
    pub token: Option<String>,
    /// Use only local cached files when resolving a Hub repo ID.
    pub local_files_only: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            dtype: DType::F32,
            device: Device::Cpu,
            inference_mode: InferenceMode::Unpacked,
            overrides: LittleBitConfigOverrides::default(),
            cache_dir: None,
            token: None,
            local_files_only: false,
        }
    }
}

/// Options controlling Hub publication via [`LittleBitModel::push_to_hub`].
#[derive(Debug, Clone)]
pub struct PushOptions {
    /// Commit message for the Hub upload.
    pub commit_message: String,
    /// Whether to create the repository as private.
    pub private: bool,
    /// Optional Hugging Face API token (defaults to `HF_TOKEN` / cached login).
    pub token: Option<String>,
    /// Optional local staging directory (if omitted, uses a temporary directory).
    pub local_dir: Option<PathBuf>,
    /// Optional directory containing tokenizer artifacts to bundle before uploading.
    pub tokenizer_dir: Option<PathBuf>,
}

impl Default for PushOptions {
    fn default() -> Self {
        Self {
            commit_message: "Push LittleBit quantized model to Hub".to_string(),
            private: false,
            token: None,
            local_dir: None,
            tokenizer_dir: None,
        }
    }
}

/// LittleBit quantized model wrapper supporting local persistence and Hugging Face Hub upload/download.
#[derive(Debug, Clone)]
pub struct LittleBitModel {
    /// Underlying causal language model.
    pub model: CausalLm,
    /// Resolved LittleBit quantization hyperparameters.
    pub littlebit_config: LittleBitConfig,
    /// Optional source base model identifier (e.g., `meta-llama/Llama-2-7b-hf`).
    pub base_model_id: Option<String>,
    /// Local directory from which the model or tokenizer was resolved.
    pub source_dir: Option<PathBuf>,
}

/// Resolve a local directory or download a model repository snapshot from Hugging Face Hub.
///
/// # Errors
/// Returns an error if the Hub client cannot be built or snapshot download fails.
pub fn resolve_model_dir(
    model_id_or_path: &str,
    cache_dir: Option<&Path>,
    token: Option<&str>,
    local_files_only: bool,
) -> Result<PathBuf> {
    let as_path = Path::new(model_id_or_path);
    if as_path.is_dir() {
        return Ok(as_path.to_path_buf());
    }

    let mut builder = HFClient::builder();
    if let Some(tok) = token {
        builder = builder.token(tok);
    }
    if let Some(cache) = cache_dir {
        builder = builder.cache_dir(cache);
    }
    let client = builder.build_sync()?;
    let (owner, name) = split_id(model_id_or_path);
    let path = client
        .model(owner, name)
        .snapshot_download()
        .local_files_only(local_files_only)
        .send()?;
    Ok(path)
}

/// Load a dense (unquantized) Hugging Face causal LM from a local directory or Hub repo ID.
///
/// # Errors
/// Returns an error if config or weights cannot be resolved or loaded.
pub fn load_dense_model(
    model_id_or_path: &str,
    options: &LoadOptions,
) -> Result<(CausalLm, PathBuf)> {
    let dir = resolve_model_dir(
        model_id_or_path,
        options.cache_dir.as_deref(),
        options.token.as_deref(),
        options.local_files_only,
    )?;
    let cfg = ModelConfig::from_file(&dir.join("config.json"))?;
    let tensors = load_raw_safetensors_dir(&dir, &options.device)?;
    let model = CausalLm::from_tensors(
        cfg,
        &tensors,
        None,
        WeightState::Raw,
        InferenceMode::Unpacked,
        options.dtype,
        &options.device,
    )?;
    Ok((model, dir))
}

impl LittleBitModel {
    /// Wrap a quantized [`CausalLm`] alongside its [`LittleBitConfig`] and optional base model ID.
    #[must_use]
    pub fn new(
        model: CausalLm,
        littlebit_config: LittleBitConfig,
        base_model_id: Option<String>,
        source_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            model,
            littlebit_config,
            base_model_id,
            source_dir,
        }
    }

    /// Load a pre-quantized LittleBit model from a local path or Hugging Face Hub repository ID.
    ///
    /// Implements upstream priority:
    /// 1. `options.overrides`
    /// 2. `littlebit_config.json`
    /// 3. `config.json` fallback
    /// 4. Defaults
    ///
    /// # Errors
    /// Returns an error if downloading, config resolution, or weight loading fails.
    pub fn from_pretrained(model_id_or_path: &str, options: &LoadOptions) -> Result<Self> {
        let dir = resolve_model_dir(
            model_id_or_path,
            options.cache_dir.as_deref(),
            options.token.as_deref(),
            options.local_files_only,
        )?;
        let lb_cfg = LittleBitConfig::load_with_priority(&dir, &options.overrides)?;
        let model_cfg = ModelConfig::from_file(&dir.join("config.json"))?;

        let (tensors, weight_state) = match options.inference_mode {
            InferenceMode::Packed => {
                let raw = load_raw_safetensors_dir(&dir, &Device::Cpu)?;
                if raw.keys().any(|k| k.ends_with("_packed")) {
                    (raw, WeightState::PreBinarized)
                } else {
                    let unpacked =
                        load_and_unpack_state_dict(&dir, options.dtype, &options.device)?;
                    let ws = match unpacked.format {
                        CheckpointFormat::Packed => WeightState::PreBinarized,
                        CheckpointFormat::LegacyUnpacked => WeightState::Raw,
                    };
                    (unpacked.tensors, ws)
                }
            }
            InferenceMode::Unpacked => {
                let unpacked = load_and_unpack_state_dict(&dir, options.dtype, &options.device)?;
                let ws = match unpacked.format {
                    CheckpointFormat::Packed => WeightState::PreBinarized,
                    CheckpointFormat::LegacyUnpacked => WeightState::Raw,
                };
                (unpacked.tensors, ws)
            }
        };

        let model = CausalLm::from_tensors(
            model_cfg,
            &tensors,
            Some(&lb_cfg),
            weight_state,
            options.inference_mode,
            options.dtype,
            &options.device,
        )?;

        let base_model_path = dir.join("base_model.json");
        let base_model_id = if base_model_path.exists() {
            let raw = fs::read_to_string(&base_model_path)?;
            serde_json::from_str::<BaseModelMeta>(&raw)
                .ok()
                .map(|m| m.model_id)
        } else {
            None
        };

        Ok(Self {
            model,
            littlebit_config: lb_cfg,
            base_model_id,
            source_dir: Some(dir),
        })
    }

    /// Save `littlebit_config.json`, `base_model.json`, `config.json`, packed `model.safetensors`,
    /// `model.safetensors.index.json`, and tokenizer files (when found) to `save_dir`.
    ///
    /// # Errors
    /// Returns an error if packing weights or writing files fails.
    pub fn save_pretrained(
        &self,
        save_dir: &Path,
        tokenizer_src: Option<&Path>,
        cast_policy: CastPolicy,
    ) -> Result<()> {
        fs::create_dir_all(save_dir)?;

        let lb_json = serde_json::to_string_pretty(&self.littlebit_config)?;
        fs::write(save_dir.join("littlebit_config.json"), lb_json)?;

        if let Some(base_id) = &self.base_model_id {
            let meta = serde_json::to_string_pretty(&BaseModelMeta {
                model_id: base_id.clone(),
            })?;
            fs::write(save_dir.join("base_model.json"), meta)?;
        }

        let mut model_cfg_val = serde_json::to_value(self.model.config())?;
        if let Some(obj) = model_cfg_val.as_object_mut() {
            obj.insert(
                "auto_map".to_string(),
                serde_json::json!({ "AutoModelForCausalLM": "quantization.hub.LittleBitModel" }),
            );
            obj.insert(
                "quant_func".to_string(),
                serde_json::json!(self.littlebit_config.quant_func.as_wire_name()),
            );
            obj.insert(
                "eff_bit".to_string(),
                serde_json::json!(self.littlebit_config.eff_bit),
            );
            obj.insert(
                "split_dim".to_string(),
                serde_json::json!(self.littlebit_config.split_dim),
            );
            obj.insert(
                "residual".to_string(),
                serde_json::json!(self.littlebit_config.residual),
            );
            obj.insert(
                "kv_factor".to_string(),
                serde_json::json!(self.littlebit_config.kv_factor),
            );
            obj.insert(
                "min_split_dim".to_string(),
                serde_json::json!(self.littlebit_config.min_split_dim),
            );
            obj.insert(
                "quant_mod".to_string(),
                serde_json::json!(self.littlebit_config.quant_mod),
            );
            obj.insert(
                "use_itq".to_string(),
                serde_json::json!(self.littlebit_config.use_itq),
            );
            obj.insert(
                "itq_n_iter".to_string(),
                serde_json::json!(self.littlebit_config.itq_n_iter),
            );
            if let Some(ref layers) = self.littlebit_config.eff_bit_layers {
                obj.insert("eff_bit_layers".to_string(), serde_json::json!(layers));
            }
        }
        fs::write(
            save_dir.join("config.json"),
            serde_json::to_string_pretty(&model_cfg_val)?,
        )?;

        let state_dict = self.model.pack_state_dict()?;
        save_packed_safetensors(save_dir, &state_dict, cast_policy)?;

        if let Some(tok_dir) = tokenizer_src.or(self.source_dir.as_deref()) {
            for filename in TOKENIZER_ARTIFACT_FILES {
                let src = tok_dir.join(filename);
                if src.is_file() {
                    let dst = save_dir.join(filename);
                    if src != dst {
                        fs::copy(&src, &dst)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Generate a Hugging Face Hub model card matching upstream `LittleBitModel._generate_readme`.
    #[must_use]
    pub fn generate_readme(&self, repo_id: &str) -> String {
        let model_name = repo_id.rsplit('/').next().unwrap_or(repo_id);
        let cfg = self.model.config();
        let lb = &self.littlebit_config;
        let eff_bit_str = lb
            .eff_bit
            .map_or_else(|| "None".to_string(), |v| format!("{v}"));
        let original_section = match &self.base_model_id {
            Some(base) => {
                format!("## Original Model\n\nBase model: [{base}](https://huggingface.co/{base})")
            }
            None => "## Original Model\n\n- Not specified".to_string(),
        };

        [
            format!(
                "---\nlicense: other\ntags:\n- quantization\n- littlebit\n- {}\n---",
                cfg.model_type
            ),
            format!("# {model_name}\n\nA LittleBit quantized model."),
            format!(
                "## Quantization Config\n\n\
                - **Quantization Function**: {}\n\
                - **Effective Bits**: {eff_bit_str}\n\
                - **Split Dim**: {}\n\
                - **Residual**: {}\n\
                - **KV Factor**: {}\n\
                - **Use ITQ (LittleBit-2)**: {}",
                lb.quant_func, lb.split_dim, lb.residual, lb.kv_factor, lb.use_itq
            ),
            format!(
                "## Usage\n\nLoad with `littlebit` CLI:\n\n```bash\nlittlebit eval --model-id \"{repo_id}\"\n```"
            ),
            original_section,
            format!(
                "## Model Details\n\n\
                - **Model Type**: {}\n\
                - **Hidden Size**: {}\n\
                - **Num Attention Heads**: {}\n\
                - **Num Hidden Layers**: {}",
                cfg.model_type, cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers
            ),
            "## Citation\n\nIf you use this model, please cite:\n\n```bibtex\n\
@inproceedings{lee2026littlebit2,\n\
  title={LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},\n\
  author={Lee, Banseok and Kim, Youngmin},\n\
  booktitle={ICML},\n\
  year={2026}\n\
}\n\n\
@inproceedings{lee2025littlebit,\n\
  title={LittleBit: Ultra Low-Bit Quantization via Latent Factorization},\n\
  author={Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},\n\
  booktitle={NeurIPS},\n\
  year={2025}\n\
}\n\
```"
                .to_string(),
        ]
        .join("\n\n")
    }

    /// Save the quantized checkpoint and upload it to Hugging Face Hub.
    ///
    /// # Errors
    /// Returns [`ModelError::Hub`] or I/O errors if staging or uploading fails.
    pub fn push_to_hub(&self, repo_id: &str, options: &PushOptions) -> Result<String> {
        let _temp_guard: Option<TempDir>;
        let stage_dir: PathBuf = match &options.local_dir {
            Some(dir) => {
                _temp_guard = None;
                fs::create_dir_all(dir)?;
                dir.clone()
            }
            None => {
                let tmp = TempDir::new()?;
                let p = tmp.path().to_path_buf();
                _temp_guard = Some(tmp);
                p
            }
        };

        self.save_pretrained(
            &stage_dir,
            options.tokenizer_dir.as_deref(),
            CastPolicy::ToBf16,
        )?;
        fs::write(stage_dir.join("README.md"), self.generate_readme(repo_id))?;

        let mut builder = HFClient::builder();
        if let Some(tok) = &options.token {
            builder = builder.token(tok);
        }
        let client = builder.build_sync()?;
        let _ = client
            .create_repository()
            .repo_id(repo_id)
            .repo_type(RepoTypeModel)
            .private(options.private)
            .exist_ok(true)
            .send();

        let (owner, name) = split_id(repo_id);
        if owner.is_empty() {
            return Err(ModelError::InvalidArgument(format!(
                "repo_id must be in `owner/name` format, got `{repo_id}`"
            )));
        }
        let commit = client
            .model(owner, name)
            .upload_folder()
            .folder_path(&stage_dir)
            .commit_message(&options.commit_message)
            .send()?;

        Ok(commit
            .commit_url
            .unwrap_or_else(|| format!("https://huggingface.co/{repo_id}")))
    }
}
