//! Upstream parity tests for PhiQKVSplitAttention, patching exclusion, and legacy checkpoints.

use std::collections::HashMap;
use std::fs;

use candle_core::{DType, Device, Tensor};
use littlebit_core::{
    CastPolicy, InferenceMode, LittleBitConfig, LittleBitConfigOverrides, QuantFunc, TrainMode,
};
use littlebit_models::{
    CausalLm, HiddenStatesMode, LittleBitModel, LoadOptions, ModelConfig, PhiAttentionProjection,
};
use tempfile::TempDir;

#[test]
fn phi_qkv_split_attention_is_mathematically_exact_with_and_without_bias(
) -> littlebit_models::Result<()> {
    let dev = Device::Cpu;
    let input = Tensor::from_slice(&[1u32, 4, 9, 17], (1, 4), &dev)?;

    for attention_bias in [false, true] {
        let mut cfg = ModelConfig::tiny_phi3(64, 32, 2);
        cfg.attention_bias = attention_bias;

        let mut model = CausalLm::new_random(cfg, DType::F32, &dev, 31)?;
        let fused_logits = model.forward(&input, None, HiddenStatesMode::None)?.logits;

        let CausalLm::Phi(ref mut phi) = model else {
            panic!("expected Phi variant");
        };
        assert!(matches!(
            phi.layer_qkv_projection(0),
            Some(PhiAttentionProjection::Fused { .. })
        ));
        phi.split_qkv()?;
        assert!(matches!(
            phi.layer_qkv_projection(0),
            Some(PhiAttentionProjection::Split { .. })
        ));

        let split_logits = model.forward(&input, None, HiddenStatesMode::None)?.logits;
        let diff = fused_logits
            .sub(&split_logits)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "PhiQKVSplitAttention mismatch (bias={attention_bias}): {diff}"
        );
    }
    Ok(())
}

#[test]
fn phi_patch_scales_kv_only_and_excludes_lm_head() -> littlebit_models::Result<()> {
    let dev = Device::Cpu;
    let cfg = ModelConfig::tiny_phi3(64, 128, 1);
    let mut model = CausalLm::new_random(cfg, DType::F32, &dev, 17)?;

    let lb_cfg = LittleBitConfig {
        quant_func: QuantFunc::SteBinary,
        eff_bit: Some(1.0),
        kv_factor: 2.0,
        min_split_dim: 8,
        residual: false,
        use_itq: false,
        ..Default::default()
    };

    let summary = model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 5)?;
    // Phi layer has 6 patched projections (q, k, v, o, gate_up, down) and 1 skipped (lm_head).
    assert_eq!(summary.patched_layers, 6);
    assert_eq!(summary.skipped_layers, 1);

    let packed = model.pack_state_dict()?;
    assert!(packed.contains_key("lm_head.weight"));
    assert!(!packed.contains_key("lm_head.U_packed"));

    // Confirm k_proj and v_proj received higher split_dim than q_proj due to kv_factor=2.0.
    let q_shape = packed["model.layers.0.self_attn.q_proj.U_shape"].to_vec1::<i64>()?;
    let k_shape = packed["model.layers.0.self_attn.k_proj.U_shape"].to_vec1::<i64>()?;
    let v_shape = packed["model.layers.0.self_attn.v_proj.U_shape"].to_vec1::<i64>()?;
    assert!(
        k_shape[1] > q_shape[1],
        "expected k_proj split_dim {} > q_proj split_dim {}",
        k_shape[1],
        q_shape[1]
    );
    assert_eq!(k_shape[1], v_shape[1]);
    Ok(())
}

#[test]
fn legacy_unpacked_checkpoint_and_tied_embeddings_loading() -> littlebit_models::Result<()> {
    let dev = Device::Cpu;
    let mut cfg = ModelConfig::tiny_llama(48, 32, 1);
    cfg.tie_word_embeddings = true;

    let lb_cfg = LittleBitConfig {
        quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
        eff_bit: None,
        split_dim: 8,
        residual: true,
        use_itq: true,
        itq_n_iter: 10,
        ..Default::default()
    };

    let mut model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 9)?;
    model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 21)?;

    let input = Tensor::from_slice(&[2u32, 5, 9], (1, 3), &dev)?;
    let ref_logits = model.forward(&input, None, HiddenStatesMode::None)?.logits;

    // Save packed checkpoint with tokenizer.json present, check generated Hub files.
    let packed_dir = TempDir::new().expect("tempdir");
    let tok_src = TempDir::new().expect("tok tempdir");
    fs::write(tok_src.path().join("tokenizer.json"), "{}")?;

    let wrapper = LittleBitModel::new(
        model,
        lb_cfg.clone(),
        Some("meta-llama/Llama-2-7b-hf".to_string()),
        None,
    );
    wrapper.save_pretrained(
        packed_dir.path(),
        Some(tok_src.path()),
        CastPolicy::Preserve,
    )?;
    assert!(packed_dir.path().join("littlebit_config.json").is_file());
    assert!(packed_dir.path().join("base_model.json").is_file());
    assert!(packed_dir
        .path()
        .join("model.safetensors.index.json")
        .is_file());
    assert!(packed_dir.path().join("tokenizer.json").is_file());

    let saved_cfg_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(packed_dir.path().join("config.json"))?)?;
    assert_eq!(
        saved_cfg_json["auto_map"]["AutoModelForCausalLM"],
        "quantization.hub.LittleBitModel"
    );

    // Build a legacy unpacked checkpoint without littlebit_config.json and without lm_head.weight.
    let legacy_dir = TempDir::new().expect("legacy tempdir");
    let unpacked = littlebit_core::load_and_unpack_state_dict(packed_dir.path(), DType::F32, &dev)?;
    let filtered: HashMap<String, Tensor> = unpacked
        .tensors
        .into_iter()
        .filter(|(k, _)| k != "lm_head.weight")
        .collect();
    candle_core::safetensors::save(&filtered, legacy_dir.path().join("model.safetensors"))?;
    fs::write(
        legacy_dir.path().join("config.json"),
        serde_json::to_string_pretty(&saved_cfg_json)?,
    )?;

    let reloaded = LittleBitModel::from_pretrained(
        legacy_dir.path().to_str().expect("utf8 path"),
        &LoadOptions {
            dtype: DType::F32,
            device: dev.clone(),
            inference_mode: InferenceMode::Unpacked,
            overrides: LittleBitConfigOverrides::default(),
            ..Default::default()
        },
    )?;
    assert_eq!(reloaded.littlebit_config.split_dim, 8);
    assert!(reloaded.littlebit_config.residual);
    assert!(reloaded.littlebit_config.use_itq);

    let legacy_logits = reloaded
        .model
        .forward(&input, None, HiddenStatesMode::None)?
        .logits;
    let diff = ref_logits
        .sub(&legacy_logits)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-5, "legacy unpacked tied-weight drift: {diff}");
    Ok(())
}
