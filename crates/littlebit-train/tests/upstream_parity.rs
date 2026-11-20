//! Upstream parity tests for KD loss scaling, parameter freezing, data caching, and eval.

use std::fs::{self, File};
use std::io::Write;

use candle_core::{DType, Device, Tensor};
use flate2::write::GzEncoder;
use flate2::Compression;
use littlebit_core::{LittleBitConfig, TrainMode};
use littlebit_models::{CausalLm, CausalLmOutput, ModelConfig};
use littlebit_train::{
    compute_kd_loss, evaluate_multiple_choice, l2l_mse_loss, load_local_texts, prepare_dataset,
    DatasetKind, MultipleChoiceExample, Result,
};
use tempfile::TempDir;
use tokenizers::Tokenizer;

fn tiny_tokenizer() -> Tokenizer {
    let json = serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": {
                "<unk>": 0,
                "alpha": 1,
                "beta": 2,
                "gamma": 3,
                "delta": 4,
                "epsilon": 5
            },
            "unk_token": "<unk>"
        },
        "pre_tokenizer": { "type": "Whitespace" }
    });
    Tokenizer::from_bytes(json.to_string().as_bytes()).expect("tiny tokenizer")
}

#[test]
fn kd_loss_l2l_scale_zero_and_nonzero_parity() -> Result<()> {
    let dev = Device::Cpu;
    let logits_s = Tensor::from_slice(&[0.2f32, 0.8, -0.5, 1.1], (1, 2, 2), &dev)?;
    let logits_t = Tensor::from_slice(&[1.0f32, 0.0, 0.3, 0.7], (1, 2, 2), &dev)?;

    let s_h0 = Tensor::zeros((1, 2, 4), DType::F32, &dev)?;
    let t_h0 = Tensor::ones((1, 2, 4), DType::F32, &dev)?;
    let s_h1 = Tensor::full(0.5f32, (1, 2, 4), &dev)?;
    let t_h1 = Tensor::zeros((1, 2, 4), DType::F32, &dev)?;
    let s_h2 = Tensor::full(1.5f32, (1, 2, 4), &dev)?;
    let t_h2 = Tensor::ones((1, 2, 4), DType::F32, &dev)?;

    let student_out = CausalLmOutput {
        logits: logits_s,
        hidden_states: vec![s_h0, s_h1.clone(), s_h2.clone()],
    };
    let teacher_out = CausalLmOutput {
        logits: logits_t,
        hidden_states: vec![t_h0, t_h1.clone(), t_h2.clone()],
    };

    // Scale 0.0 omits L2L term completely.
    let zero_scale = compute_kd_loss(&student_out, &teacher_out, 0.0)?;
    assert_eq!(zero_scale.l2l_loss, 0.0);

    // Scale 10.0 averages MSE per layer (only hidden_states[1..]), sums across layers, times 10.0.
    // Layer 1 diff = 0.5 -> MSE = 0.25; Layer 2 diff = 0.5 -> MSE = 0.25; sum = 0.5; * 10 = 5.0.
    let scaled = compute_kd_loss(&student_out, &teacher_out, 10.0)?;
    assert!((scaled.l2l_loss - 5.0).abs() < 1e-5);
    assert!((scaled.kd_loss - zero_scale.kd_loss).abs() < 1e-6);

    let direct_l2l = l2l_mse_loss(&[s_h1, s_h2], &[t_h1, t_h2], 10.0)?.to_scalar::<f32>()?;
    assert!((f64::from(direct_l2l) - 5.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn parameter_stats_display_and_freeze_parity() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = ModelConfig::tiny_llama(32, 32, 1);
    let mut student = CausalLm::new_random(cfg, DType::F32, &dev, 4)?;
    let lb_cfg = LittleBitConfig {
        eff_bit: None,
        split_dim: 8,
        ..Default::default()
    };
    student.apply_littlebit_patch(&lb_cfg, TrainMode::Train, 1)?;
    student.prepare_for_training(&["lm_head", "embed"])?;

    let stats = student.parameter_stats();
    assert!(stats.trainable_params > 0);
    assert!(stats.trainable_params < stats.all_params);
    let formatted = stats.to_string();
    assert!(formatted.contains("trainable params:"));
    assert!(formatted.contains("all params:"));
    assert!(formatted.contains("trainable%:"));
    Ok(())
}

#[test]
fn datautils_local_txt_jsonl_and_gzip_with_cache_reuse() -> Result<()> {
    let tmp = TempDir::new().expect("tempdir");
    let tok = tiny_tokenizer();

    let txt_path = tmp.path().join("corpus.txt");
    fs::write(&txt_path, "alpha beta gamma delta epsilon alpha beta gamma")?;
    assert_eq!(load_local_texts(&txt_path)?.len(), 1);

    let jsonl_path = tmp.path().join("corpus.jsonl");
    fs::write(
        &jsonl_path,
        "{\"text\":\"alpha beta gamma\"}\n{\"sentence\":\"delta epsilon alpha\"}\n",
    )?;
    assert_eq!(load_local_texts(&jsonl_path)?.len(), 2);

    let gz_path = tmp.path().join("corpus.jsonl.gz");
    {
        let f = File::create(&gz_path)?;
        let mut enc = GzEncoder::new(f, Compression::default());
        writeln!(enc, "{{\"text\":\"alpha beta gamma delta\"}}")?;
        writeln!(enc, "{{\"content\":\"epsilon alpha beta gamma\"}}")?;
        enc.finish()?;
    }
    let gz_texts = load_local_texts(&gz_path)?;
    assert_eq!(gz_texts.len(), 2);

    let data_root = tmp.path().join("cache_root");
    let kind = DatasetKind::LocalFile(gz_path);
    let ds1 = prepare_dataset(&kind, &data_root, &tok, 4, None, None, None)?;
    assert_eq!(ds1.blocks.len(), 2);
    let cache_dir = ds1.cache_dir.clone().expect("cache dir recorded");
    assert!(cache_dir.join("blocks.bin").is_file());
    assert!(cache_dir.join("tokenizer_info").is_file());

    // Delete source file to prove the second prepare_dataset call serves strictly from disk cache.
    fs::remove_file(match &kind {
        DatasetKind::LocalFile(p) => p,
        _ => unreachable!(),
    })?;
    let ds2 = prepare_dataset(&kind, &data_root, &tok, 4, None, None, None)?;
    assert_eq!(ds2.blocks, ds1.blocks);
    Ok(())
}

#[test]
fn eval_seqlen_fallback_and_zeroshot_limit() -> Result<()> {
    let mut cfg = ModelConfig::tiny_llama(32, 16, 1);
    cfg.max_position_embeddings = 1024;
    cfg.n_ctx = Some(512);
    assert_eq!(cfg.effective_seqlen(), 512);

    cfg.n_ctx = None;
    assert_eq!(cfg.effective_seqlen(), 1024);

    cfg.max_position_embeddings = 0;
    assert_eq!(cfg.effective_seqlen(), 2048);

    let dev = Device::Cpu;
    let tok = tiny_tokenizer();
    let model = CausalLm::new_random(ModelConfig::tiny_llama(16, 16, 1), DType::F32, &dev, 3)?;
    let examples = vec![
        MultipleChoiceExample {
            context: "alpha".to_string(),
            choices: vec![" beta".to_string(), " gamma".to_string()],
            gold_index: 0,
        },
        MultipleChoiceExample {
            context: "delta".to_string(),
            choices: vec![" epsilon".to_string(), " alpha".to_string()],
            gold_index: 1,
        },
    ];
    let limited = evaluate_multiple_choice(&model, &tok, "limit_check", &examples, Some(1), &dev)?;
    assert_eq!(limited.num_examples, 1);
    Ok(())
}
