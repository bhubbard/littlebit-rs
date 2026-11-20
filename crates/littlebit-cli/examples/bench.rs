//! Pure-Rust benchmark suite for `littlebit-rs`.
//!
//! Run with:
//! ```bash
//! cargo run --release -p littlebit-cli --example bench
//! cargo run --release -p littlebit-cli --features accelerate --example bench
//! ```

use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use clap::Parser;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use tempfile::TempDir;

use littlebit_core::{
    inspect_checkpoint, pack_i8_slice, reconstruct_branch_matrix, BiasMode, BitBudget, CastPolicy,
    InferenceMode, LayerShape, LittleBitConfig, LittleBitLinear, QuantFunc, TrainMode, WeightState,
};
use littlebit_models::{CausalLm, GenerateConfig, LittleBitModel, LoadOptions, ModelConfig};

#[derive(Debug, Parser)]
#[command(
    name = "littlebit-bench",
    about = "Pure-Rust benchmark suite for littlebit-rs"
)]
struct BenchArgs {
    /// Warmup iterations before each timed benchmark.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Timed iterations per benchmark case.
    #[arg(long, default_value_t = 15)]
    iters: usize,

    /// Optional backend label stored in JSON output.
    #[arg(long)]
    label: Option<String>,

    /// Optional output path for structured JSON benchmark results.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct HostInfo {
    backend_label: String,
    os: &'static str,
    arch: &'static str,
    cpu_cores: usize,
    accelerate_enabled: bool,
    warmup: usize,
    iters: usize,
}

#[derive(Debug, Serialize)]
struct PackerBenchRow {
    shape: String,
    rows: usize,
    cols: usize,
    weights: usize,
    packed_bytes: usize,
    fp16_bytes: usize,
    compression_vs_fp16: f64,
    pack_median_ms: f64,
    pack_p95_ms: f64,
    pack_mweights_per_s: f64,
    unpack_median_ms: f64,
    unpack_p95_ms: f64,
    unpack_mweights_per_s: f64,
}

#[derive(Debug, Serialize)]
struct ProjectionBenchRow {
    name: String,
    out_features: usize,
    in_features: usize,
    eff_bit_target: f64,
    residual: bool,
    split_dim: usize,
    eff_bit_actual: f64,
    fp16_bytes: usize,
    packed_bytes: usize,
    compression_vs_fp16: f64,
    dense_b1_median_ms: f64,
    dense_b1_p95_ms: f64,
    unpacked_b1_median_ms: f64,
    unpacked_b1_p95_ms: f64,
    packed_b1_median_ms: f64,
    packed_b1_p95_ms: f64,
    dense_b16_median_ms: f64,
    dense_b16_p95_ms: f64,
    unpacked_b16_median_ms: f64,
    unpacked_b16_p95_ms: f64,
    packed_b16_median_ms: f64,
    packed_b16_p95_ms: f64,
}

#[derive(Debug, Serialize)]
struct InitBenchRow {
    shape: String,
    out_features: usize,
    in_features: usize,
    eff_bit_target: f64,
    residual: bool,
    use_itq: bool,
    itq_n_iter: usize,
    split_dim: usize,
    init_ms: f64,
    rel_frobenius_error: f64,
}

#[derive(Debug, Serialize)]
struct ModelGenBenchRow {
    model: String,
    prompt_tokens: usize,
    new_tokens: usize,
    avg_eff_bits: f64,
    dense_fp16_bytes: usize,
    quantized_layers_bytes: usize,
    total_serialized_bytes: usize,
    compression_ratio_vs_fp16: f64,
    dense_gen_median_ms: f64,
    dense_tok_per_s: f64,
    unpacked_gen_median_ms: f64,
    unpacked_tok_per_s: f64,
    packed_gen_median_ms: f64,
    packed_tok_per_s: f64,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    host: HostInfo,
    packer: Vec<PackerBenchRow>,
    projections: Vec<ProjectionBenchRow>,
    init: Vec<InitBenchRow>,
    generation: Vec<ModelGenBenchRow>,
}

fn time_median_p95_ms<T, F>(warmup: usize, iters: usize, mut f: F) -> Result<(f64, f64)>
where
    F: FnMut() -> Result<T>,
{
    if iters == 0 {
        bail!("--iters must be > 0");
    }
    for _ in 0..warmup {
        black_box(f()?);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        black_box(f()?);
        samples.push(t0.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    let n = samples.len();
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    };
    let p95_idx = ((n as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    Ok((median, samples[p95_idx]))
}

fn seeded_normal_vec(len: usize, std_dev: f32, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let u1: f32 = rng.random_range(1e-7f32..1.0f32);
        let u2: f32 = rng.random_range(0.0f32..std::f32::consts::TAU);
        let r = (-2.0 * u1.ln()).sqrt() * std_dev;
        out.push(r * u2.cos());
        if out.len() < len {
            out.push(r * u2.sin());
        }
    }
    out
}

fn seeded_sign_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len)
        .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
        .collect()
}

fn bench_packer(warmup: usize, iters: usize) -> Result<Vec<PackerBenchRow>> {
    let shapes = [(1024usize, 4096usize), (4096, 4096), (4096, 11008)];
    let mut rows_out = Vec::with_capacity(shapes.len());

    for (idx, (rows, cols)) in shapes.into_iter().enumerate() {
        let mut rng = ChaCha8Rng::seed_from_u64(100 + idx as u64);
        let values: Vec<i8> = (0..rows * cols)
            .map(|_| if rng.random_bool(0.5) { 1 } else { -1 })
            .collect();

        let sample_packed = pack_i8_slice(&values, rows, cols)?;
        let packed_bytes = sample_packed.memory_bytes();
        let fp16_bytes = rows * cols * 2;
        let weights = rows * cols;

        let (pack_med, pack_p95) =
            time_median_p95_ms(warmup, iters, || Ok(pack_i8_slice(&values, rows, cols)?))?;
        let (unpack_med, unpack_p95) =
            time_median_p95_ms(warmup, iters, || Ok(sample_packed.unpack_to_i8()))?;

        rows_out.push(PackerBenchRow {
            shape: format!("{rows}x{cols}"),
            rows,
            cols,
            weights,
            packed_bytes,
            fp16_bytes,
            compression_vs_fp16: fp16_bytes as f64 / packed_bytes as f64,
            pack_median_ms: pack_med,
            pack_p95_ms: pack_p95,
            pack_mweights_per_s: (weights as f64 / 1e6) / (pack_med / 1_000.0),
            unpack_median_ms: unpack_med,
            unpack_p95_ms: unpack_p95,
            unpack_mweights_per_s: (weights as f64 / 1e6) / (unpack_med / 1_000.0),
        });
    }

    Ok(rows_out)
}

fn build_synthetic_prebinarized_layer(
    in_features: usize,
    out_features: usize,
    cfg: &LittleBitConfig,
    seed: u64,
    dev: &Device,
) -> Result<LittleBitLinear> {
    let budget = BitBudget::resolve(in_features, out_features, cfg, 1.0);
    let s = budget.split_dim;
    let mut state = HashMap::new();

    for (branch_idx, suffix) in ["", "_R"]
        .into_iter()
        .take(if cfg.residual { 2 } else { 1 })
        .enumerate()
    {
        let b_seed = seed.wrapping_add((branch_idx as u64 + 1) * 97);
        state.insert(
            format!("proj.U{suffix}"),
            Tensor::from_vec(
                seeded_sign_vec(out_features * s, b_seed),
                (out_features, s),
                dev,
            )?,
        );
        state.insert(
            format!("proj.V{suffix}"),
            Tensor::from_vec(
                seeded_sign_vec(s * in_features, b_seed + 1),
                (s, in_features),
                dev,
            )?,
        );
        state.insert(
            format!("proj.u1{suffix}"),
            Tensor::full(0.02f32, (1, out_features), dev)?,
        );
        state.insert(
            format!("proj.u2{suffix}"),
            Tensor::full(1.0f32, (1, s), dev)?,
        );
        state.insert(
            format!("proj.v1{suffix}"),
            Tensor::full(1.0f32, (1, s), dev)?,
        );
        state.insert(
            format!("proj.v2{suffix}"),
            Tensor::full(0.02f32, (1, in_features), dev)?,
        );
    }

    Ok(LittleBitLinear::from_state_dict(
        "proj",
        &state,
        LayerShape::new(in_features, out_features, BiasMode::NoBias),
        cfg,
        1.0,
        WeightState::PreBinarized,
        DType::F32,
        dev,
    )?)
}

fn bench_projections(warmup: usize, iters: usize) -> Result<Vec<ProjectionBenchRow>> {
    let dev = Device::Cpu;
    let specs = [
        (
            "Llama-7B Attn 4096x4096 @ 0.1 bpw (single)",
            4096usize,
            4096usize,
            0.1f64,
            false,
        ),
        (
            "Llama-7B Attn 4096x4096 @ 0.5 bpw (residual)",
            4096,
            4096,
            0.5,
            true,
        ),
        (
            "Llama-7B Attn 4096x4096 @ 1.0 bpw (residual)",
            4096,
            4096,
            1.0,
            true,
        ),
        (
            "Llama-7B MLP 4096x11008 @ 0.5 bpw (residual)",
            4096,
            11008,
            0.5,
            true,
        ),
    ];

    let mut out = Vec::with_capacity(specs.len());
    for (idx, (name, in_f, out_f, eff_bit, residual)) in specs.into_iter().enumerate() {
        let cfg = LittleBitConfig {
            quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
            eff_bit: Some(eff_bit),
            residual,
            use_itq: false,
            min_split_dim: 8,
            ..Default::default()
        };

        let layer = build_synthetic_prebinarized_layer(in_f, out_f, &cfg, 200 + idx as u64, &dev)?;
        let packed = layer.to_packed()?;
        let dense_w_t = Tensor::from_vec(
            seeded_normal_vec(in_f * out_f, 0.02, 300 + idx as u64),
            (in_f, out_f),
            &dev,
        )?;

        let x_b1 = Tensor::from_vec(
            seeded_normal_vec(in_f, 1.0, 400 + idx as u64),
            (1, in_f),
            &dev,
        )?;
        let x_b16 = Tensor::from_vec(
            seeded_normal_vec(16 * in_f, 1.0, 500 + idx as u64),
            (16, in_f),
            &dev,
        )?;

        let (dense_b1_med, dense_b1_p95) =
            time_median_p95_ms(warmup, iters, || Ok(x_b1.matmul(&dense_w_t)?))?;
        let (unpacked_b1_med, unpacked_b1_p95) =
            time_median_p95_ms(warmup, iters, || Ok(layer.forward(&x_b1)?))?;
        let (packed_b1_med, packed_b1_p95) =
            time_median_p95_ms(warmup, iters, || Ok(packed.forward(&x_b1)?))?;

        let (dense_b16_med, dense_b16_p95) =
            time_median_p95_ms(warmup, iters, || Ok(x_b16.matmul(&dense_w_t)?))?;
        let (unpacked_b16_med, unpacked_b16_p95) =
            time_median_p95_ms(warmup, iters, || Ok(layer.forward(&x_b16)?))?;
        let (packed_b16_med, packed_b16_p95) =
            time_median_p95_ms(warmup, iters, || Ok(packed.forward(&x_b16)?))?;

        let fp16_bytes = in_f * out_f * 2;
        let packed_bytes = packed.memory_bytes();
        let budget = layer.budget();

        out.push(ProjectionBenchRow {
            name: name.to_string(),
            out_features: out_f,
            in_features: in_f,
            eff_bit_target: eff_bit,
            residual,
            split_dim: budget.split_dim,
            eff_bit_actual: budget.eff_bit_actual,
            fp16_bytes,
            packed_bytes,
            compression_vs_fp16: fp16_bytes as f64 / packed_bytes as f64,
            dense_b1_median_ms: dense_b1_med,
            dense_b1_p95_ms: dense_b1_p95,
            unpacked_b1_median_ms: unpacked_b1_med,
            unpacked_b1_p95_ms: unpacked_b1_p95,
            packed_b1_median_ms: packed_b1_med,
            packed_b1_p95_ms: packed_b1_p95,
            dense_b16_median_ms: dense_b16_med,
            dense_b16_p95_ms: dense_b16_p95,
            unpacked_b16_median_ms: unpacked_b16_med,
            unpacked_b16_p95_ms: unpacked_b16_p95,
            packed_b16_median_ms: packed_b16_med,
            packed_b16_p95_ms: packed_b16_p95,
        });
    }

    Ok(out)
}

fn bench_init() -> Result<Vec<InitBenchRow>> {
    let dev = Device::Cpu;
    let shapes = [(1024usize, 1024usize), (2048, 2048)];
    let mut out = Vec::with_capacity(shapes.len() * 2);

    for (idx, (out_f, in_f)) in shapes.into_iter().enumerate() {
        let w = Tensor::from_vec(
            seeded_normal_vec(out_f * in_f, 0.02, 700 + idx as u64),
            (out_f, in_f),
            &dev,
        )?;
        let w_norm = f64::from(w.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();

        for (use_itq, n_iter) in [(false, 0usize), (true, 20usize)] {
            let cfg = LittleBitConfig {
                quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
                eff_bit: Some(0.5),
                residual: true,
                use_itq,
                itq_n_iter: n_iter.max(1),
                min_split_dim: 8,
                ..Default::default()
            };

            let t0 = Instant::now();
            let layer =
                LittleBitLinear::from_dense_weight(&w, None, &cfg, 1.0, TrainMode::Eval, 11)?;
            let init_ms = t0.elapsed().as_secs_f64() * 1_000.0;

            let main = layer.main_branch();
            let mut w_hat = reconstruct_branch_matrix(
                main.u.as_tensor(),
                main.v.as_tensor(),
                main.u1.as_tensor(),
                main.u2.as_tensor(),
                main.v1.as_tensor(),
                main.v2.as_tensor(),
                cfg.quant_func,
            )?;
            if let Some(res) = layer.residual_branch() {
                let w_res = reconstruct_branch_matrix(
                    res.u.as_tensor(),
                    res.v.as_tensor(),
                    res.u1.as_tensor(),
                    res.u2.as_tensor(),
                    res.v1.as_tensor(),
                    res.v2.as_tensor(),
                    cfg.quant_func,
                )?;
                w_hat = w_hat.add(&w_res)?;
            }

            let err_norm = f64::from(w.sub(&w_hat)?.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
            out.push(InitBenchRow {
                shape: format!("{out_f}x{in_f}"),
                out_features: out_f,
                in_features: in_f,
                eff_bit_target: 0.5,
                residual: true,
                use_itq,
                itq_n_iter: if use_itq { n_iter } else { 0 },
                split_dim: layer.split_dim(),
                init_ms,
                rel_frobenius_error: if w_norm > 0.0 { err_norm / w_norm } else { 0.0 },
            });
        }
    }

    Ok(out)
}

fn bench_generation() -> Result<Vec<ModelGenBenchRow>> {
    let dev = Device::Cpu;
    let prompt: Vec<u32> = (1..=16).collect();
    let new_tokens = 32usize;
    let gen_cfg = GenerateConfig {
        max_new_tokens: new_tokens,
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
        eos_token_ids: Vec::new(),
    };

    let mut llama_cfg = ModelConfig::tiny_llama(4096, 512, 4);
    llama_cfg.intermediate_size = 1376;
    llama_cfg.num_attention_heads = 8;
    llama_cfg.num_key_value_heads = Some(4);

    let mut qwen3_cfg = ModelConfig::tiny_qwen3(4096, 512, 4);
    qwen3_cfg.intermediate_size = 1376;
    qwen3_cfg.num_attention_heads = 8;
    qwen3_cfg.num_key_value_heads = Some(4);
    qwen3_cfg.head_dim = Some(64);

    let lb_cfg = LittleBitConfig {
        quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
        eff_bit: Some(0.5),
        residual: true,
        use_itq: true,
        itq_n_iter: 15,
        min_split_dim: 8,
        ..Default::default()
    };

    let models = [
        ("Llama-4L (h=512, i=1376, v=4096)", llama_cfg),
        ("Qwen3-4L (h=512, i=1376, v=4096)", qwen3_cfg),
    ];

    let mut results = Vec::with_capacity(models.len());
    for (idx, (label, cfg)) in models.into_iter().enumerate() {
        let mut model = CausalLm::new_random(cfg, DType::F32, &dev, 900 + idx as u64)?;
        let (dense_med, _) =
            time_median_p95_ms(1, 5, || Ok(model.generate(&prompt, &gen_cfg, &dev)?))?;

        model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 950 + idx as u64)?;
        let dir = TempDir::new()?;
        LittleBitModel::new(model, lb_cfg.clone(), Some(label.to_string()), None).save_pretrained(
            dir.path(),
            None,
            CastPolicy::ToBf16,
        )?;

        let stats = inspect_checkpoint(dir.path())?;
        let dir_str = dir.path().to_string_lossy().into_owned();

        let unpacked = LittleBitModel::from_pretrained(
            &dir_str,
            &LoadOptions {
                dtype: DType::F32,
                device: dev.clone(),
                inference_mode: InferenceMode::Unpacked,
                ..Default::default()
            },
        )?;
        let (unpacked_med, _) = time_median_p95_ms(1, 5, || {
            Ok(unpacked.model.generate(&prompt, &gen_cfg, &dev)?)
        })?;

        let packed = LittleBitModel::from_pretrained(
            &dir_str,
            &LoadOptions {
                dtype: DType::F32,
                device: dev.clone(),
                inference_mode: InferenceMode::Packed,
                ..Default::default()
            },
        )?;
        let (packed_med, _) = time_median_p95_ms(1, 5, || {
            Ok(packed.model.generate(&prompt, &gen_cfg, &dev)?)
        })?;

        let tok_s = |ms: f64| -> f64 { new_tokens as f64 / (ms / 1_000.0) };
        results.push(ModelGenBenchRow {
            model: label.to_string(),
            prompt_tokens: prompt.len(),
            new_tokens,
            avg_eff_bits: stats.avg_eff_bits,
            dense_fp16_bytes: stats.dense_fp16_bytes,
            quantized_layers_bytes: stats.quantized_layers_bytes,
            total_serialized_bytes: stats.total_serialized_bytes,
            compression_ratio_vs_fp16: stats.compression_ratio_vs_fp16,
            dense_gen_median_ms: dense_med,
            dense_tok_per_s: tok_s(dense_med),
            unpacked_gen_median_ms: unpacked_med,
            unpacked_tok_per_s: tok_s(unpacked_med),
            packed_gen_median_ms: packed_med,
            packed_tok_per_s: tok_s(packed_med),
        });
    }

    Ok(results)
}

fn print_summary(report: &BenchReport) {
    println!(
        "=== littlebit-rs Benchmark ({}, {}/{}, {} cores, accelerate={}) ===",
        report.host.backend_label,
        report.host.os,
        report.host.arch,
        report.host.cpu_cores,
        report.host.accelerate_enabled
    );

    println!("\n-- Binary Packer / Unpacker Throughput --");
    for row in &report.packer {
        println!(
            "{:12} | pack={:6.2}ms ({:7.1} MW/s) | unpack={:6.2}ms ({:7.1} MW/s) | packed={:.2} MiB ({:.2}x vs FP16)",
            row.shape,
            row.pack_median_ms,
            row.pack_mweights_per_s,
            row.unpack_median_ms,
            row.unpack_mweights_per_s,
            row.packed_bytes as f64 / (1024.0 * 1024.0),
            row.compression_vs_fp16,
        );
    }

    println!("\n-- LittleBitLinear Projection Forward (b=1 & b=16) --");
    for row in &report.projections {
        println!(
            "{:46} | s={:3} actual={:.3} bpw ({:5.2}x) | b=1 dense={:6.2}ms unpacked={:6.2}ms packed={:6.2}ms | b=16 dense={:6.2}ms unpacked={:6.2}ms packed={:6.2}ms",
            row.name,
            row.split_dim,
            row.eff_bit_actual,
            row.compression_vs_fp16,
            row.dense_b1_median_ms,
            row.unpacked_b1_median_ms,
            row.packed_b1_median_ms,
            row.dense_b16_median_ms,
            row.unpacked_b16_median_ms,
            row.packed_b16_median_ms,
        );
    }

    println!("\n-- Quantization & Joint-ITQ Initialization --");
    for row in &report.init {
        println!(
            "{:10} | use_itq={:<5} (iters={:2}, s={:3}) | init={:8.2}ms | rel_frob_err={:.5}",
            row.shape,
            row.use_itq,
            row.itq_n_iter,
            row.split_dim,
            row.init_ms,
            row.rel_frobenius_error
        );
    }

    println!("\n-- 4-Layer Causal LM Storage & 32-Token KV Generation --");
    for row in &report.generation {
        println!(
            "{:32} | eff={:.3} bpw | layers={:.2} MiB ({:.2}x vs FP16) | dense={:6.1} tok/s | unpacked={:6.1} tok/s | packed={:6.1} tok/s",
            row.model,
            row.avg_eff_bits,
            row.quantized_layers_bytes as f64 / (1024.0 * 1024.0),
            row.compression_ratio_vs_fp16,
            row.dense_tok_per_s,
            row.unpacked_tok_per_s,
            row.packed_tok_per_s,
        );
    }
}

fn main() -> Result<()> {
    let args = BenchArgs::parse();
    let accelerate_enabled = cfg!(feature = "accelerate");
    let backend_label = args.label.unwrap_or_else(|| {
        if accelerate_enabled {
            "CPU + Accelerate".to_string()
        } else {
            "CPU (default)".to_string()
        }
    });
    let cpu_cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);

    let report = BenchReport {
        host: HostInfo {
            backend_label,
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            cpu_cores,
            accelerate_enabled,
            warmup: args.warmup,
            iters: args.iters,
        },
        packer: bench_packer(args.warmup, args.iters)?,
        projections: bench_projections(args.warmup, args.iters)?,
        init: bench_init()?,
        generation: bench_generation()?,
    };

    print_summary(&report);

    if let Some(path) = args.json_out {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("\nSaved JSON benchmark report to {}", path.display());
    }

    Ok(())
}
