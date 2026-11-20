//! Community audit and benchmark regression tests for mixed-bit patching, checkpoint size
//! accounting, multi-layer FP16 overflow checks, and Phi split-QKV parameter overhead.

use std::collections::BTreeMap;
use std::fs;

use candle_core::{DType, Device, Tensor};
use littlebit_core::{
    compute_eff_bits, estimate_split_dim, finalize_split_dim, inspect_checkpoint,
    load_raw_safetensors_dir, CastPolicy, InferenceMode, LittleBitConfig, QuantFunc, ResidualMode,
    TrainMode,
};
use littlebit_models::{CausalLm, HiddenStatesMode, LittleBitModel, LoadOptions, ModelConfig};
use tempfile::TempDir;

fn classify_part(name: &str) -> &'static str {
    if name.contains("embed_tokens") {
        "embeddings"
    } else if name.starts_with("lm_head") {
        "lm_head"
    } else if name.ends_with("_packed") {
        "binary factors (packed)"
    } else if name.ends_with("_shape") || name.contains("_eff_bit") || name.contains("_split_dim") {
        "bookkeeping"
    } else if [
        ".u1", ".u2", ".v1", ".v2", ".u1_R", ".u2_R", ".v1_R", ".v2_R",
    ]
    .iter()
    .any(|s| name.ends_with(s))
    {
        "scales"
    } else if name.contains("norm") {
        "norms"
    } else {
        "other"
    }
}

#[test]
fn bench_mixed_bits_28_block_qwen3_and_reload_roundtrip() -> littlebit_models::Result<()> {
    let dev = Device::Cpu;
    let mut cfg = ModelConfig::tiny_qwen3(128, 64, 28);
    cfg.intermediate_size = 128;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = Some(2);
    cfg.head_dim = Some(16);
    cfg.tie_word_embeddings = true;

    let plain_cfg = LittleBitConfig {
        quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
        eff_bit: Some(1.0),
        eff_bit_layers: None,
        residual: false,
        kv_factor: 1.0,
        min_split_dim: 8,
        use_itq: false,
        ..Default::default()
    };
    let mixed_cfg = LittleBitConfig {
        eff_bit_layers: Some("0-7,27:2.0".to_string()),
        ..plain_cfg.clone()
    };

    let mut plain_model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 0)?;
    plain_model.apply_littlebit_patch(&plain_cfg, TrainMode::Eval, 11)?;

    let mut mixed_model = CausalLm::new_random(cfg, DType::F32, &dev, 0)?;
    mixed_model.apply_littlebit_patch(&mixed_cfg, TrainMode::Eval, 11)?;

    assert!(mixed_model.parameter_stats().all_params > plain_model.parameter_stats().all_params);

    let plain_sd = plain_model.pack_state_dict()?;
    let mixed_sd = mixed_model.pack_state_dict()?;

    let mid_split = mixed_sd["model.layers.10.self_attn.q_proj.U_shape"].to_vec1::<i64>()?[1];
    for block in 0..28 {
        let target_key = format!("model.layers.{block}.self_attn.q_proj._eff_bit_target");
        let shape_key = format!("model.layers.{block}.self_attn.q_proj.U_shape");

        let plain_target = plain_sd[&target_key].to_scalar::<f32>()?;
        assert!((plain_target - 1.0).abs() < 1e-6);

        let mixed_target = mixed_sd[&target_key].to_scalar::<f32>()?;
        let plain_split = plain_sd[&shape_key].to_vec1::<i64>()?[1];
        let mixed_split = mixed_sd[&shape_key].to_vec1::<i64>()?[1];

        if block <= 7 || block == 27 {
            assert!((mixed_target - 2.0).abs() < 1e-6, "block {block}");
            assert!(
                mixed_split > mid_split,
                "block {block} split_dim {mixed_split} should exceed mid block {mid_split}"
            );
        } else {
            assert!((mixed_target - 1.0).abs() < 1e-6, "block {block}");
            assert_eq!(mixed_split, plain_split, "block {block}");
        }
    }

    let input = Tensor::from_slice(&[1u32, 7, 19, 31], (1, 4), &dev)?;
    let expected_logits = mixed_model
        .forward(&input, None, HiddenStatesMode::None)?
        .logits;

    let dir = TempDir::new().expect("tempdir");
    let wrapper = LittleBitModel::new(mixed_model, mixed_cfg, Some("tiny-qwen3".to_string()), None);
    wrapper.save_pretrained(dir.path(), None, CastPolicy::Preserve)?;

    let lb_cfg_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(
        dir.path().join("littlebit_config.json"),
    )?)?;
    assert_eq!(lb_cfg_json["eff_bit_layers"], "0-7,27:2.0");

    for mode in [InferenceMode::Unpacked, InferenceMode::Packed] {
        let reloaded = LittleBitModel::from_pretrained(
            dir.path().to_str().expect("utf8 path"),
            &LoadOptions {
                dtype: DType::F32,
                device: dev.clone(),
                inference_mode: mode,
                ..Default::default()
            },
        )?;
        assert_eq!(
            reloaded.littlebit_config.eff_bit_layers.as_deref(),
            Some("0-7,27:2.0")
        );
        let got_logits = reloaded
            .model
            .forward(&input, None, HiddenStatesMode::None)?
            .logits;
        let diff = expected_logits
            .sub(&got_logits)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "mixed-bits reload drift ({mode:?}): {diff}");
    }

    Ok(())
}

#[test]
fn bench_sizes_accounting_and_body_bpw_breakdown() -> littlebit_models::Result<()> {
    let dev = Device::Cpu;
    let cfg = ModelConfig::tiny_llama(64, 64, 2);
    let mut model = CausalLm::new_random(cfg, DType::F32, &dev, 7)?;
    let lb_cfg = LittleBitConfig {
        eff_bit: Some(0.75),
        residual: true,
        use_itq: false,
        ..Default::default()
    };
    model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 13)?;

    let dir = TempDir::new().expect("tempdir");
    LittleBitModel::new(model, lb_cfg, None, None).save_pretrained(
        dir.path(),
        None,
        CastPolicy::ToBf16,
    )?;

    let raw = load_raw_safetensors_dir(dir.path(), &dev)?;
    let mut sizes: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut compressed_weights = 0usize;

    for (name, tensor) in &raw {
        let bytes = tensor.elem_count() * tensor.dtype().size_in_bytes();
        *sizes.entry(classify_part(name)).or_insert(0) += bytes;
        if let Some(prefix) = name.strip_suffix(".U_shape") {
            let u_shape = tensor.to_vec1::<i64>()?;
            let v_shape = raw[&format!("{prefix}.V_shape")].to_vec1::<i64>()?;
            compressed_weights += (u_shape[0] as usize) * (v_shape[1] as usize);
        }
    }

    let total_bytes: usize = sizes.values().sum();
    let stats = inspect_checkpoint(dir.path())?;
    assert_eq!(total_bytes, stats.total_serialized_bytes);
    assert!(compressed_weights > 0);

    let packed_bytes = sizes.get("binary factors (packed)").copied().unwrap_or(0);
    let scale_bytes = sizes.get("scales").copied().unwrap_or(0);
    assert!(packed_bytes > 0);
    assert!(scale_bytes > 0);

    let body_bpw = ((packed_bytes + scale_bytes) * 8) as f64 / (compressed_weights as f64);
    assert!(body_bpw.is_finite() && body_bpw > 0.0);
    Ok(())
}

#[test]
fn bench_multilayer_fp16_vs_fp32_overflow_scan() -> littlebit_models::Result<()> {
    let dev = Device::Cpu;
    let mut cfg = ModelConfig::tiny_qwen3(64, 64, 2);
    cfg.intermediate_size = 128;

    let lb_cfg = LittleBitConfig {
        quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
        eff_bit: Some(1.0),
        residual: true,
        use_itq: false,
        ..Default::default()
    };

    let mut base = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 29)?;
    base.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 29)?;

    let dir = TempDir::new().expect("tempdir");
    LittleBitModel::new(base, lb_cfg, None, None).save_pretrained(
        dir.path(),
        None,
        CastPolicy::Preserve,
    )?;

    let path = dir.path().to_str().expect("utf8 path");
    let model_f32 = LittleBitModel::from_pretrained(
        path,
        &LoadOptions {
            dtype: DType::F32,
            device: dev.clone(),
            inference_mode: InferenceMode::Unpacked,
            ..Default::default()
        },
    )?;
    let model_f16 = LittleBitModel::from_pretrained(
        path,
        &LoadOptions {
            dtype: DType::F16,
            device: dev.clone(),
            inference_mode: InferenceMode::Unpacked,
            ..Default::default()
        },
    )?;

    let input = Tensor::from_slice(&[1u32, 4, 9, 17], (1, 4), &dev)?;
    let out32 = model_f32
        .model
        .forward(&input, None, HiddenStatesMode::All)?;
    let out16 = model_f16
        .model
        .forward(&input, None, HiddenStatesMode::All)?;

    assert_eq!(out32.hidden_states.len(), out16.hidden_states.len());
    for (idx, h) in out16.hidden_states.iter().enumerate() {
        let vals = h.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "non-finite hidden state at layer {idx}"
        );
    }

    let l32 = out32.logits.flatten_all()?.to_vec1::<f32>()?;
    let l16 = out16
        .logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(l16.iter().all(|v| v.is_finite()));
    let max_diff = l32
        .iter()
        .zip(l16.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 5e-2, "F16 vs F32 max logit diff: {max_diff}");
    Ok(())
}

#[test]
fn audit_phi_split_has_more_scale_elements_than_fused() {
    for eff in [1.0f64, 0.55, 0.3, 0.1] {
        for mode in [ResidualMode::Single, ResidualMode::Dual] {
            let paths = match mode {
                ResidualMode::Single => 1usize,
                ResidualMode::Dual => 2usize,
            };
            let split_for = |a: usize, b: usize| -> usize {
                finalize_split_dim(estimate_split_dim(a, b, Some(eff), mode), 1024, 8)
            };
            let scale_elems = |a: usize, b: usize, s: usize| -> usize { paths * (a + b + 2 * s) };

            let s_fused = split_for(5120, 7680);
            let fused_scales = scale_elems(5120, 7680, s_fused);

            let s_q = split_for(5120, 5120);
            let s_k = split_for(5120, 1280);
            let s_v = split_for(5120, 1280);
            let split_scales = scale_elems(5120, 5120, s_q)
                + scale_elems(5120, 1280, s_k)
                + scale_elems(5120, 1280, s_v);

            assert!(
                split_scales > fused_scales,
                "eff={eff}, mode={mode:?}: split_scales ({split_scales}) <= fused_scales ({fused_scales})"
            );
            assert!(compute_eff_bits(5120, 7680, s_fused, mode).is_finite());
        }
    }
}
