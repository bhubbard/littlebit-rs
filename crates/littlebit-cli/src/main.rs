//! Command-line interface for LittleBit and LittleBit-2 sub-1-bit LLM compression.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use tracing_subscriber::EnvFilter;

use littlebit_core::{
    inspect_checkpoint, load_and_unpack_state_dict, CastPolicy, InferenceMode, LittleBitConfig,
    LittleBitConfigOverrides, QuantFunc, TrainMode,
};
use littlebit_models::{
    load_dense_model, resolve_model_dir, stream_quantize_checkpoint, GenerateConfig,
    LittleBitModel, LoadOptions, PushOptions, StreamingQuantizeOptions,
};
use littlebit_train::{
    evaluate_multiple_choice, evaluate_perplexity, load_eval_tokens, load_multiple_choice_jsonl,
    load_tokenizer, prepare_dataset, DatasetKind, KdTrainer, PplResult, TrainingConfig, ZeroConfig,
    ZeroShotTaskResult, DEFAULT_BLOCK_SIZE,
};

#[derive(Debug, Parser)]
#[command(
    name = "littlebit",
    version,
    about = "Pure-Rust implementation of LittleBit (NeurIPS 2025) & LittleBit-2 (ICML 2026)"
)]
struct Cli {
    #[command(flatten)]
    global: GlobalOpts,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, Args)]
struct GlobalOpts {
    /// Compute device: cpu, metal, or cuda.
    #[arg(long, global = true, default_value = "cpu")]
    device: String,

    /// Tensor compute dtype: f32, bf16, or f16.
    #[arg(long, global = true, default_value = "f32")]
    dtype: String,

    /// Custom Hugging Face cache directory.
    #[arg(long, global = true, alias = "cache_dir")]
    cache_dir: Option<PathBuf>,

    /// Hugging Face Hub authentication token.
    #[arg(long, global = true, env = "HF_TOKEN")]
    token: Option<String>,

    /// Restrict Hub resolution to already cached local files.
    #[arg(long, global = true)]
    local_files_only: bool,
}

#[derive(Debug, Clone, Args)]
struct QuantOverridesArgs {
    /// Quantization function: STEBinary or SmoothSign.
    #[arg(long, alias = "quant_func")]
    quant_func: Option<String>,

    /// Target effective bits per weight (e.g. 1.0, 0.5, 0.1).
    #[arg(long, alias = "eff_bit")]
    eff_bit: Option<f64>,

    /// Per-decoder-block effective bit overrides (e.g. "0-7,27:2.0;12:1.5").
    #[arg(long, alias = "eff_bit_layers")]
    eff_bit_layers: Option<String>,

    /// Explicit inner latent rank when eff-bit is unset.
    #[arg(long, alias = "split_dim")]
    split_dim: Option<usize>,

    /// Enable dual residual latent factorization.
    #[arg(long, num_args = 0..=1, default_missing_value = "true", value_parser = parse_bool_flag)]
    residual: Option<bool>,

    /// Rank multiplier applied to attention k_proj and v_proj layers.
    #[arg(long, alias = "kv_factor")]
    kv_factor: Option<f64>,

    /// Minimum latent rank floor (multiple of 8 recommended).
    #[arg(long, alias = "min_split_dim")]
    min_split_dim: Option<usize>,

    /// Enable LittleBit-2 Joint-ITQ latent rotation initialization.
    #[arg(long, alias = "use_itq", num_args = 0..=1, default_missing_value = "true", value_parser = parse_bool_flag)]
    use_itq: Option<bool>,

    /// Number of Joint-ITQ iterations when --use-itq is enabled.
    #[arg(long, alias = "itq_n_iter")]
    itq_n_iter: Option<usize>,
}

impl QuantOverridesArgs {
    fn to_overrides(&self) -> Result<LittleBitConfigOverrides> {
        let quant_func = match &self.quant_func {
            Some(s) => Some(s.parse::<QuantFunc>()?),
            None => None,
        };
        Ok(LittleBitConfigOverrides {
            quant_func,
            eff_bit: self.eff_bit.map(Some),
            eff_bit_layers: self.eff_bit_layers.clone().map(Some),
            split_dim: self.split_dim,
            residual: self.residual,
            kv_factor: self.kv_factor,
            min_split_dim: self.min_split_dim,
            quant_mod: None,
            use_itq: self.use_itq,
            itq_n_iter: self.itq_n_iter,
        })
    }

    fn to_config_with_defaults(&self) -> Result<LittleBitConfig> {
        let mut cfg = LittleBitConfig::default();
        if let Some(s) = &self.quant_func {
            cfg.quant_func = s.parse()?;
        }
        if let Some(bit) = self.eff_bit {
            cfg.eff_bit = Some(bit);
        } else if self.split_dim.is_some() {
            cfg.eff_bit = None;
        }
        if let Some(spec) = &self.eff_bit_layers {
            cfg.eff_bit_layers = Some(spec.clone());
        }
        if let Some(s) = self.split_dim {
            cfg.split_dim = s;
        }
        if let Some(r) = self.residual {
            cfg.residual = r;
        }
        if let Some(kv) = self.kv_factor {
            cfg.kv_factor = kv;
        }
        if let Some(m) = self.min_split_dim {
            cfg.min_split_dim = m;
        }
        if let Some(itq) = self.use_itq {
            cfg.use_itq = itq;
        }
        if let Some(it) = self.itq_n_iter {
            cfg.itq_n_iter = it;
        }
        cfg.validate()?;
        Ok(cfg)
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Post-training LittleBit / LittleBit-2 SVD+ITQ factorization and 1-bit packing.
    Quantize {
        /// Local model path or Hugging Face Hub model ID.
        #[arg(long, alias = "model_id")]
        model_id: String,

        /// Destination directory for packed quantized artifacts.
        #[arg(long, alias = "save_dir")]
        save_dir: PathBuf,

        /// Deterministic RNG seed for randomized SVD and Joint-ITQ.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Preserve F32 scale/head tensors instead of casting to BF16 on save.
        #[arg(long)]
        preserve_f32: bool,

        /// Stream one transformer block at a time from safetensors to keep peak RAM minimal.
        #[arg(long)]
        streaming: bool,

        /// Maximum output shard size in MiB when `--streaming` is enabled.
        #[arg(long, default_value_t = 2048)]
        max_shard_mb: usize,

        /// Skip multimodal vision/audio tower weights when streaming multimodal checkpoints.
        #[arg(long)]
        text_only: bool,

        /// Print output summary as JSON.
        #[arg(long)]
        json: bool,

        #[command(flatten)]
        quant: QuantOverridesArgs,
    },

    /// Quantization-Aware Knowledge Distillation training (`main.py` parity).
    Train {
        /// Base dense model path or Hugging Face Hub model ID.
        #[arg(long, alias = "model_id")]
        model_id: String,

        /// Root directory for tokenized dataset caches.
        #[arg(long, alias = "data_root", default_value = ".")]
        data_root: PathBuf,

        /// Dataset name (`wikitext2`, `c4`, `c4_wiki`, `ptb`) or file path.
        #[arg(long, default_value = "c4_wiki")]
        dataset: String,

        /// Explicit local `.txt`, `.jsonl`, or `.jsonl.gz` training corpus file.
        #[arg(long, alias = "dataset_file")]
        dataset_file: Option<PathBuf>,

        /// Output root directory for checkpoints and metrics.
        #[arg(long, alias = "save_dir", default_value = "outputs")]
        save_dir: PathBuf,

        /// Optional subdirectory name under --save-dir (defaults to timestamp).
        #[arg(long, alias = "f_name")]
        f_name: Option<String>,

        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Number of training epochs.
        #[arg(long, alias = "num_train_epochs", default_value_t = 3.0)]
        num_train_epochs: f64,

        /// Micro-batch size.
        #[arg(long, alias = "per_device_train_batch_size", default_value_t = 4)]
        batch_size: usize,

        /// Gradient accumulation steps.
        #[arg(long, alias = "gradient_accumulation_steps", default_value_t = 1)]
        gradient_accumulation_steps: usize,

        /// Warmup ratio for cosine LR schedule.
        #[arg(long, alias = "warmup_ratio", default_value_t = 0.03)]
        warmup_ratio: f64,

        /// Peak AdamW learning rate.
        #[arg(long, default_value_t = 2e-5)]
        lr: f64,

        /// AdamW beta1.
        #[arg(long, alias = "adam_beta1", default_value_t = 0.9)]
        adam_beta1: f64,

        /// AdamW beta2.
        #[arg(long, alias = "adam_beta2", default_value_t = 0.999)]
        adam_beta2: f64,

        /// Scale factor for layer-to-layer hidden-state MSE loss.
        #[arg(long, alias = "l2l_loss_scale", default_value_t = 10.0)]
        l2l_loss_scale: f64,

        /// Optional DeepSpeed reference config path (`configs/zero*.json`).
        #[arg(long, alias = "ds_config_path")]
        ds_config_path: Option<PathBuf>,

        /// Token block length for sequence chunking.
        #[arg(long, alias = "block_size", default_value_t = DEFAULT_BLOCK_SIZE)]
        block_size: usize,

        /// Optional cap on remote dataset documents downloaded per split.
        #[arg(long, alias = "max_remote_docs")]
        max_remote_docs: Option<usize>,

        /// Experiment name metadata.
        #[arg(long, alias = "exp_name", default_value = "LittleBit")]
        exp_name: String,

        /// Run name metadata.
        #[arg(long, alias = "run_name", default_value = "my_run")]
        run_name: String,

        #[command(flatten)]
        quant: QuantOverridesArgs,
    },

    /// Evaluate perplexity and/or multiple-choice zero-shot tasks (`eval.py` parity).
    Eval {
        /// Model directory or Hugging Face Hub model ID.
        #[arg(long, alias = "model_id")]
        model_id: String,

        /// Evaluate unquantized full-precision baseline instead of LittleBit model.
        #[arg(long, alias = "fp_eval", num_args = 0..=1, default_missing_value = "true", value_parser = parse_bool_flag, default_value_t = false)]
        fp_eval: bool,

        /// Execute inference directly against 1-bit packed weight buffers on CPU.
        #[arg(long, alias = "packed_inference")]
        packed_inference: bool,

        /// Comma-separated perplexity datasets (`wikitext2,c4,ptb`).
        #[arg(long, alias = "ppl_task", default_value = "")]
        ppl_task: String,

        /// Local text/JSONL file path for perplexity evaluation.
        #[arg(long, alias = "ppl_file")]
        ppl_file: Option<PathBuf>,

        /// Zero-shot task name (`boolq`, `piqa`, `hellaswag`, etc.) when using --zeroshot-file.
        #[arg(long, alias = "zeroshot_task", default_value = "")]
        zeroshot_task: String,

        /// Local JSONL file containing multiple-choice examples.
        #[arg(long, alias = "zeroshot_file")]
        zeroshot_file: Option<PathBuf>,

        /// Maximum examples to evaluate on zero-shot task.
        #[arg(long)]
        limit: Option<usize>,

        /// Context length for perplexity evaluation (defaults to model config).
        #[arg(long)]
        seqlen: Option<usize>,

        /// Emit evaluation metrics as JSON.
        #[arg(long)]
        json: bool,

        #[command(flatten)]
        quant: QuantOverridesArgs,
    },

    /// Run autoregressive text generation on a dense or LittleBit model.
    Generate {
        /// Model directory or Hugging Face Hub model ID.
        #[arg(long, alias = "model_id")]
        model_id: String,

        /// Input prompt text.
        #[arg(long)]
        prompt: String,

        /// Load unquantized dense model instead of LittleBit checkpoint.
        #[arg(long)]
        fp: bool,

        /// Execute inference directly against 1-bit packed weight buffers on CPU.
        #[arg(long)]
        packed_inference: bool,

        /// Maximum new tokens to generate.
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,

        /// Sampling temperature (`0.0` for deterministic greedy decoding).
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,

        /// Top-p nucleus threshold.
        #[arg(long, default_value_t = 1.0)]
        top_p: f64,

        /// Random seed for stochastic sampling.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        #[command(flatten)]
        quant: QuantOverridesArgs,
    },

    /// Inspect a LittleBit checkpoint without loading full dense weights into memory.
    Inspect {
        /// Checkpoint directory path.
        #[arg(long, alias = "model_id")]
        model_id: PathBuf,

        /// Output inspection report as JSON.
        #[arg(long)]
        json: bool,
    },

    /// Convert a legacy unpacked LittleBit checkpoint into 1-bit packed safetensors.
    Pack {
        /// Source unpacked checkpoint directory.
        #[arg(long)]
        input: PathBuf,

        /// Output packed checkpoint directory.
        #[arg(long)]
        output: PathBuf,

        /// Keep F32 scales instead of casting to BF16.
        #[arg(long)]
        preserve_f32: bool,

        #[command(flatten)]
        quant: QuantOverridesArgs,
    },

    /// Unpack a 1-bit packed checkpoint back to float `U`/`V` safetensors.
    Unpack {
        /// Source packed checkpoint directory.
        #[arg(long)]
        input: PathBuf,

        /// Output unpacked checkpoint directory.
        #[arg(long)]
        output: PathBuf,
    },

    /// Upload a local LittleBit checkpoint to Hugging Face Hub.
    Push {
        /// Local checkpoint directory.
        #[arg(long)]
        model_dir: PathBuf,

        /// Target Hub repository ID (`owner/repo`).
        #[arg(long)]
        repo_id: String,

        /// Create Hub repository as private.
        #[arg(long)]
        private: bool,

        /// Commit message.
        #[arg(long, default_value = "Push LittleBit quantized model to Hub")]
        commit_message: String,

        /// Optional directory containing tokenizer files to bundle.
        #[arg(long)]
        tokenizer_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Serialize)]
struct EvalReport {
    ppl: Vec<PplResult>,
    zeroshot: Vec<ZeroShotTaskResult>,
}

fn parse_bool_flag(raw: &str) -> std::result::Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "yes" | "true" | "t" | "y" | "1" => Ok(true),
        "no" | "false" | "f" | "n" | "0" => Ok(false),
        other => Err(format!("boolean value expected, got `{other}`")),
    }
}

fn resolve_device(spec: &str) -> Result<Device> {
    match spec.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" | "mps" => Device::new_metal(0).context("initializing Metal device"),
        "cuda" | "gpu" => Device::new_cuda(0).context("initializing CUDA device"),
        other => bail!("unsupported device `{other}` (expected cpu, metal, or cuda)"),
    }
}

fn resolve_dtype(spec: &str) -> Result<DType> {
    match spec.trim().to_ascii_lowercase().as_str() {
        "f32" | "float32" | "fp32" => Ok(DType::F32),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" | "fp16" => Ok(DType::F16),
        other => bail!("unsupported dtype `{other}` (expected f32, bf16, or f16)"),
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn make_load_options(
    global: &GlobalOpts,
    device: Device,
    dtype: DType,
    inference_mode: InferenceMode,
    overrides: LittleBitConfigOverrides,
) -> LoadOptions {
    LoadOptions {
        dtype,
        device,
        inference_mode,
        overrides,
        cache_dir: global.cache_dir.clone(),
        token: global.token.clone(),
        local_files_only: global.local_files_only,
    }
}

fn copy_metadata_and_tokenizer(src: &Path, dst: &Path) -> Result<()> {
    for name in [
        "config.json",
        "littlebit_config.json",
        "base_model.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    ] {
        let s = src.join(name);
        if s.is_file() {
            let d = dst.join(name);
            if s != d {
                fs::copy(&s, &d)?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let device = resolve_device(&cli.global.device)?;
    let dtype = resolve_dtype(&cli.global.dtype)?;

    match &cli.command {
        Commands::Quantize {
            model_id,
            save_dir,
            seed,
            preserve_f32,
            streaming,
            max_shard_mb,
            text_only,
            json,
            quant,
        } => {
            let lb_cfg = quant.to_config_with_defaults()?;
            let cast_policy = if *preserve_f32 {
                CastPolicy::Preserve
            } else {
                CastPolicy::ToBf16
            };

            let (summary, streaming_info) = if *streaming {
                let src_dir = resolve_model_dir(
                    model_id,
                    cli.global.cache_dir.as_deref(),
                    cli.global.token.as_deref(),
                    cli.global.local_files_only,
                )?;
                let max_shard_bytes = (*max_shard_mb).max(1) * 1024 * 1024;
                let report = stream_quantize_checkpoint(
                    &src_dir,
                    save_dir,
                    &lb_cfg,
                    &StreamingQuantizeOptions {
                        seed: *seed,
                        cast_policy,
                        max_shard_bytes,
                        device,
                        dtype,
                        text_only: *text_only,
                        base_model_id: Some(model_id.clone()),
                    },
                )?;
                (
                    report.patch_summary,
                    Some((report.shards_written, report.blocks_processed)),
                )
            } else {
                let load_opts = make_load_options(
                    &cli.global,
                    device,
                    dtype,
                    InferenceMode::Unpacked,
                    LittleBitConfigOverrides::default(),
                );
                let (mut model, src_dir) = load_dense_model(model_id, &load_opts)?;
                let summary = model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, *seed)?;

                let wrapper = LittleBitModel::new(
                    model,
                    lb_cfg,
                    Some(model_id.clone()),
                    Some(src_dir.clone()),
                );
                wrapper.save_pretrained(save_dir, Some(&src_dir), cast_policy)?;
                (summary, None)
            };

            let stats = inspect_checkpoint(save_dir)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("Quantized `{model_id}` -> `{}`", save_dir.display());
                println!("Patch summary: {summary}");
                if let Some((shards, blocks)) = streaming_info {
                    println!("Streaming: processed {blocks} blocks -> {shards} shard(s)");
                }
                println!(
                    "Packed bytes: {} | Quantized FP16 baseline: {} | Compression: {:.2}x | Avg eff_bit: {:.4}",
                    stats.quantized_layers_bytes,
                    stats.dense_fp16_bytes,
                    stats.compression_ratio_vs_fp16,
                    stats.avg_eff_bits
                );
            }
        }

        Commands::Train {
            model_id,
            data_root,
            dataset,
            dataset_file,
            save_dir,
            f_name,
            seed,
            num_train_epochs,
            batch_size,
            gradient_accumulation_steps,
            warmup_ratio,
            lr,
            adam_beta1,
            adam_beta2,
            l2l_loss_scale,
            ds_config_path,
            block_size,
            max_remote_docs,
            exp_name,
            run_name,
            quant,
        } => {
            let lb_cfg = quant.to_config_with_defaults()?;
            let run_subdir = match f_name {
                Some(name) => name.clone(),
                None => {
                    let secs = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    format!("run_{secs}")
                }
            };
            let out_dir = save_dir.join(run_subdir);
            fs::create_dir_all(&out_dir)?;

            let load_opts = make_load_options(
                &cli.global,
                device.clone(),
                dtype,
                InferenceMode::Unpacked,
                LittleBitConfigOverrides::default(),
            );
            let (teacher, src_dir) = load_dense_model(model_id, &load_opts)?;
            let (mut student, _) = load_dense_model(model_id, &load_opts)?;
            let patch_summary = student.apply_littlebit_patch(&lb_cfg, TrainMode::Train, *seed)?;
            println!("Student patch summary: {patch_summary}");

            let tokenizer = load_tokenizer(
                model_id,
                cli.global.cache_dir.as_deref(),
                cli.global.token.as_deref(),
                cli.global.local_files_only,
            )?;

            let ds_kind = match dataset_file {
                Some(path) => DatasetKind::LocalFile(path.clone()),
                None => dataset.parse::<DatasetKind>()?,
            };
            let token_ds = prepare_dataset(
                &ds_kind,
                data_root,
                &tokenizer,
                *block_size,
                *max_remote_docs,
                cli.global.cache_dir.as_deref(),
                cli.global.token.as_deref(),
            )?;

            let zero_cfg = match ds_config_path {
                Some(path) => Some(ZeroConfig::from_file(path)?),
                None => None,
            };

            let train_cfg = TrainingConfig {
                seed: *seed,
                num_train_epochs: *num_train_epochs,
                batch_size: *batch_size,
                gradient_accumulation_steps: zero_cfg
                    .as_ref()
                    .and_then(ZeroConfig::grad_accum_steps)
                    .unwrap_or(*gradient_accumulation_steps),
                warmup_ratio: *warmup_ratio,
                lr: *lr,
                adam_beta1: *adam_beta1,
                adam_beta2: *adam_beta2,
                max_grad_norm: zero_cfg
                    .as_ref()
                    .and_then(ZeroConfig::max_grad_norm)
                    .unwrap_or(1.0),
                cos_min_ratio: zero_cfg.as_ref().map_or(0.0001, ZeroConfig::cos_min_ratio),
                l2l_loss_scale: *l2l_loss_scale,
                save_dir: Some(out_dir.clone()),
                exp_name: exp_name.clone(),
                run_name: run_name.clone(),
                ..Default::default()
            };

            let mut trainer = KdTrainer::new(&mut student, train_cfg)?;
            println!("{}", trainer.parameter_stats());
            let summary = trainer.train(&student, &teacher, &token_ds.blocks, &device)?;

            KdTrainer::save_artifacts(&student, &lb_cfg, Some(model_id), &out_dir, Some(&src_dir))?;
            println!(
                "Training finished: {} steps | final_loss={:?} | saved={}",
                summary.total_steps,
                summary.final_loss,
                out_dir.display()
            );
        }

        Commands::Eval {
            model_id,
            fp_eval,
            packed_inference,
            ppl_task,
            ppl_file,
            zeroshot_task,
            zeroshot_file,
            limit,
            seqlen,
            json,
            quant,
        } => {
            let inf_mode = if *packed_inference {
                InferenceMode::Packed
            } else {
                InferenceMode::Unpacked
            };
            let model = if *fp_eval {
                let opts = make_load_options(
                    &cli.global,
                    device.clone(),
                    dtype,
                    InferenceMode::Unpacked,
                    LittleBitConfigOverrides::default(),
                );
                load_dense_model(model_id, &opts)?.0
            } else {
                let overrides = quant.to_overrides()?;
                let opts =
                    make_load_options(&cli.global, device.clone(), dtype, inf_mode, overrides);
                LittleBitModel::from_pretrained(model_id, &opts)?.model
            };

            let tokenizer = load_tokenizer(
                model_id,
                cli.global.cache_dir.as_deref(),
                cli.global.token.as_deref(),
                cli.global.local_files_only,
            )?;

            let effective_seqlen = seqlen.unwrap_or_else(|| model.config().effective_seqlen());
            let mut ppl_results = Vec::new();
            let mut zeroshot_results = Vec::new();

            if let Some(file) = ppl_file {
                let tokens = load_eval_tokens(
                    &DatasetKind::LocalFile(file.clone()),
                    &tokenizer,
                    effective_seqlen,
                    cli.global.cache_dir.as_deref(),
                    cli.global.token.as_deref(),
                )?;
                let res = evaluate_perplexity(
                    &model,
                    &tokens,
                    effective_seqlen,
                    &file.display().to_string(),
                    &device,
                )?;
                ppl_results.push(res);
            }

            for ds_name in ppl_task.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let kind = ds_name.parse::<DatasetKind>()?;
                let tokens = load_eval_tokens(
                    &kind,
                    &tokenizer,
                    effective_seqlen,
                    cli.global.cache_dir.as_deref(),
                    cli.global.token.as_deref(),
                )?;
                let res = evaluate_perplexity(&model, &tokens, effective_seqlen, ds_name, &device)?;
                ppl_results.push(res);
            }

            if let Some(zs_path) = zeroshot_file {
                let task_label = if zeroshot_task.trim().is_empty() {
                    "custom"
                } else {
                    zeroshot_task.trim()
                };
                let examples = load_multiple_choice_jsonl(zs_path)?;
                let res = evaluate_multiple_choice(
                    &model, &tokenizer, task_label, &examples, *limit, &device,
                )?;
                zeroshot_results.push(res);
            } else if !zeroshot_task.trim().is_empty() {
                bail!(
                    "zero-shot task `{}` requires --zeroshot-file <JSONL> with context/choices/gold_index records",
                    zeroshot_task.trim()
                );
            }

            let report = EvalReport {
                ppl: ppl_results,
                zeroshot: zeroshot_results,
            };
            if *json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                for r in &report.ppl {
                    println!(
                        "[{}] PPL = {:.4} (avg_nll={:.4}, samples={}, seqlen={})",
                        r.dataset, r.perplexity, r.avg_nll, r.num_samples, r.seqlen
                    );
                }
                for r in &report.zeroshot {
                    println!(
                        "[{}] acc = {:.4} | acc_norm = {:.4} (examples={})",
                        r.task, r.acc, r.acc_norm, r.num_examples
                    );
                }
            }
        }

        Commands::Generate {
            model_id,
            prompt,
            fp,
            packed_inference,
            max_tokens,
            temperature,
            top_p,
            seed,
            quant,
        } => {
            let inf_mode = if *packed_inference {
                InferenceMode::Packed
            } else {
                InferenceMode::Unpacked
            };
            let model = if *fp {
                let opts = make_load_options(
                    &cli.global,
                    device.clone(),
                    dtype,
                    InferenceMode::Unpacked,
                    LittleBitConfigOverrides::default(),
                );
                load_dense_model(model_id, &opts)?.0
            } else {
                let opts = make_load_options(
                    &cli.global,
                    device.clone(),
                    dtype,
                    inf_mode,
                    quant.to_overrides()?,
                );
                LittleBitModel::from_pretrained(model_id, &opts)?.model
            };

            let tokenizer = load_tokenizer(
                model_id,
                cli.global.cache_dir.as_deref(),
                cli.global.token.as_deref(),
                cli.global.local_files_only,
            )?;
            let enc = tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
            let prompt_ids = enc.get_ids();

            let mut eos = Vec::new();
            if let Some(spec) = &model.config().eos_token_id {
                match spec {
                    littlebit_models::TokenIdSpec::Single(id) => eos.push(*id),
                    littlebit_models::TokenIdSpec::Multiple(ids) => eos.extend(ids),
                }
            }

            let out_ids = model.generate(
                prompt_ids,
                &GenerateConfig {
                    max_new_tokens: *max_tokens,
                    temperature: *temperature,
                    top_p: *top_p,
                    seed: *seed,
                    eos_token_ids: eos,
                },
                &device,
            )?;
            let text = tokenizer
                .decode(&out_ids, true)
                .map_err(|e| anyhow::anyhow!("decoding failed: {e}"))?;
            println!("{text}");
        }

        Commands::Inspect { model_id, json } => {
            let stats = inspect_checkpoint(model_id)?;
            if *json {
                println!("{}", serde_json::to_string_pretty(&stats)?);
            } else {
                println!("Checkpoint: {}", model_id.display());
                println!(
                    "Format: {} | Tensors: {} | Quantized Layers: {}",
                    if stats.is_packed {
                        "1-bit packed"
                    } else {
                        "unpacked/legacy"
                    },
                    stats.tensor_count,
                    stats.quantized_layer_count
                );
                println!(
                    "QuantFunc: {} | EffBit Target: {:?} | Residual: {} | Use ITQ: {}",
                    stats.config.quant_func,
                    stats.config.eff_bit,
                    stats.config.residual,
                    stats.config.use_itq
                );
                println!(
                    "Quantized Bytes: {} | FP16 Equivalent: {} | Ratio: {:.2}x | Avg EffBit: {:.4}",
                    stats.quantized_layers_bytes,
                    stats.dense_fp16_bytes,
                    stats.compression_ratio_vs_fp16,
                    stats.avg_eff_bits
                );
                for layer in &stats.layers {
                    println!(
                        "  {:<50} {:>5}x{:<5} s={:<4} res={:<5} bpw={:.4} bytes={}",
                        layer.name,
                        layer.out_features,
                        layer.in_features,
                        layer.split_dim,
                        layer.has_residual,
                        layer.eff_bit_actual,
                        layer.serialized_bytes
                    );
                }
            }
        }

        Commands::Pack {
            input,
            output,
            preserve_f32,
            quant,
        } => {
            let opts = make_load_options(
                &cli.global,
                device,
                dtype,
                InferenceMode::Unpacked,
                quant.to_overrides()?,
            );
            let model = LittleBitModel::from_pretrained(
                input.to_str().context("non-UTF-8 input path")?,
                &opts,
            )?;
            let cast_policy = if *preserve_f32 {
                CastPolicy::Preserve
            } else {
                CastPolicy::ToBf16
            };
            model.save_pretrained(output, Some(input), cast_policy)?;
            println!("Packed checkpoint saved to {}", output.display());
        }

        Commands::Unpack { input, output } => {
            let loaded = load_and_unpack_state_dict(input, dtype, &Device::Cpu)?;
            fs::create_dir_all(output)?;
            copy_metadata_and_tokenizer(input, output)?;
            candle_core::safetensors::save(&loaded.tensors, output.join("model.safetensors"))?;
            println!(
                "Unpacked {} tensors ({:?}) to {}",
                loaded.tensors.len(),
                loaded.format,
                output.display()
            );
        }

        Commands::Push {
            model_dir,
            repo_id,
            private,
            commit_message,
            tokenizer_dir,
        } => {
            let opts = make_load_options(
                &cli.global,
                device,
                dtype,
                InferenceMode::Unpacked,
                LittleBitConfigOverrides::default(),
            );
            let model = LittleBitModel::from_pretrained(
                model_dir.to_str().context("non-UTF-8 model_dir path")?,
                &opts,
            )?;
            let url = model.push_to_hub(
                repo_id,
                &PushOptions {
                    commit_message: commit_message.clone(),
                    private: *private,
                    token: cli.global.token.clone(),
                    local_dir: Some(model_dir.clone()),
                    tokenizer_dir: tokenizer_dir.clone(),
                },
            )?;
            println!("Pushed to {url}");
        }
    }

    Ok(())
}
