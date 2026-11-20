//! End-to-end integration tests for the `littlebit` CLI.

use std::fs;
use std::path::Path;
use std::process::Command;

use candle_core::{DType, Device};
use littlebit_core::{CastPolicy, LittleBitConfig, TrainMode};
use littlebit_models::{CausalLm, LittleBitModel, ModelConfig};
use tempfile::TempDir;

fn write_wordlevel_tokenizer(dir: &Path, vocab_size: usize) {
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".to_string(), serde_json::json!(0));
    for i in 1..vocab_size {
        vocab.insert(format!("tok{i}"), serde_json::json!(i));
    }
    let tok = serde_json::json!({
        "version": "1.0",
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        },
        "pre_tokenizer": { "type": "Whitespace" }
    });
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tok).expect("serialize tokenizer"),
    )
    .expect("write tokenizer.json");
}

fn create_dense_llama_fixture(dir: &Path, vocab_size: usize) {
    fs::create_dir_all(dir).expect("mkdir dense fixture");
    let cfg = ModelConfig::tiny_llama(vocab_size, 32, 2);
    let model = CausalLm::new_random(cfg.clone(), DType::F32, &Device::Cpu, 123).expect("random");
    let state = model.pack_state_dict().expect("pack dense state");
    candle_core::safetensors::save(&state, dir.join("model.safetensors")).expect("save tensors");
    fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&cfg).expect("serialize config"),
    )
    .expect("write config.json");
    write_wordlevel_tokenizer(dir, vocab_size);
}

#[test]
fn cli_end_to_end_quantize_inspect_unpack_pack_eval_train_generate() {
    let bin = env!("CARGO_BIN_EXE_littlebit");
    let tmp = TempDir::new().expect("tempdir");
    let dense_dir = tmp.path().join("dense");
    let quant_dir = tmp.path().join("quantized");
    let unpacked_dir = tmp.path().join("unpacked");
    let repacked_dir = tmp.path().join("repacked");
    let train_out_dir = tmp.path().join("train_out");

    create_dense_llama_fixture(&dense_dir, 48);

    // 1. Quantize with LittleBit-2 Joint-ITQ + residual.
    let status = Command::new(bin)
        .args([
            "quantize",
            "--model-id",
            dense_dir.to_str().expect("path"),
            "--save-dir",
            quant_dir.to_str().expect("path"),
            "--quant-func",
            "SmoothSign",
            "--eff-bit",
            "0.75",
            "--residual",
            "True",
            "--use-itq",
            "True",
            "--itq-n-iter",
            "8",
            "--preserve-f32",
        ])
        .status()
        .expect("run littlebit quantize");
    assert!(status.success(), "quantize failed");
    assert!(quant_dir.join("littlebit_config.json").is_file());
    assert!(quant_dir.join("model.safetensors").is_file());
    assert!(quant_dir.join("model.safetensors.index.json").is_file());

    // 2. Inspect as JSON.
    let inspect_out = Command::new(bin)
        .args([
            "inspect",
            "--model-id",
            quant_dir.to_str().expect("path"),
            "--json",
        ])
        .output()
        .expect("run littlebit inspect");
    assert!(inspect_out.status.success(), "inspect failed");
    let inspect_json: serde_json::Value =
        serde_json::from_slice(&inspect_out.stdout).expect("parse inspect json");
    assert_eq!(inspect_json["is_packed"], serde_json::json!(true));
    assert_eq!(inspect_json["quantized_layer_count"], serde_json::json!(14));

    // 3. Unpack and re-pack.
    let unpack_status = Command::new(bin)
        .args([
            "unpack",
            "--input",
            quant_dir.to_str().expect("path"),
            "--output",
            unpacked_dir.to_str().expect("path"),
        ])
        .status()
        .expect("run unpack");
    assert!(unpack_status.success(), "unpack failed");
    assert!(unpacked_dir.join("model.safetensors").is_file());

    let pack_status = Command::new(bin)
        .args([
            "pack",
            "--input",
            unpacked_dir.to_str().expect("path"),
            "--output",
            repacked_dir.to_str().expect("path"),
            "--preserve-f32",
        ])
        .status()
        .expect("run pack");
    assert!(pack_status.success(), "pack failed");

    // 4. Eval PPL + Zero-shot (both standard and --packed-inference).
    let corpus_file = tmp.path().join("corpus.txt");
    let corpus_line = (1..40)
        .map(|i| format!("tok{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&corpus_file, format!("{corpus_line} {corpus_line}")).expect("write corpus");

    let zs_file = tmp.path().join("zs.jsonl");
    fs::write(
        &zs_file,
        "{\"context\":\"tok1 tok2\",\"choices\":[\" tok3 tok4\",\" tok5 tok6\"],\"gold_index\":0}\n",
    )
    .expect("write zs.jsonl");

    let eval_out = Command::new(bin)
        .args([
            "eval",
            "--model-id",
            repacked_dir.to_str().expect("path"),
            "--packed-inference",
            "--ppl-file",
            corpus_file.to_str().expect("path"),
            "--seqlen",
            "8",
            "--zeroshot-task",
            "custom_mc",
            "--zeroshot-file",
            zs_file.to_str().expect("path"),
            "--json",
        ])
        .output()
        .expect("run eval");
    assert!(
        eval_out.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_json: serde_json::Value =
        serde_json::from_slice(&eval_out.stdout).expect("parse eval json");
    assert!(eval_json["ppl"][0]["perplexity"].as_f64().is_some());
    assert_eq!(eval_json["zeroshot"][0]["task"], "custom_mc");

    // 4b. Baseline full-precision --fp-eval True check matching upstream eval.py.
    let fp_eval_out = Command::new(bin)
        .args([
            "eval",
            "--model-id",
            dense_dir.to_str().expect("path"),
            "--fp-eval",
            "True",
            "--ppl-file",
            corpus_file.to_str().expect("path"),
            "--seqlen",
            "8",
            "--json",
        ])
        .output()
        .expect("run fp-eval");
    assert!(
        fp_eval_out.status.success(),
        "fp-eval failed: {}",
        String::from_utf8_lossy(&fp_eval_out.stderr)
    );
    let fp_eval_json: serde_json::Value =
        serde_json::from_slice(&fp_eval_out.stdout).expect("parse fp-eval json");
    assert!(fp_eval_json["ppl"][0]["perplexity"].as_f64().is_some());

    // 5. Train QAT + KD using local corpus and verify metrics.jsonl + saved checkpoint.
    let train_status = Command::new(bin)
        .args([
            "train",
            "--model-id",
            dense_dir.to_str().expect("path"),
            "--dataset-file",
            corpus_file.to_str().expect("path"),
            "--data-root",
            tmp.path().join("cache").to_str().expect("path"),
            "--save-dir",
            train_out_dir.to_str().expect("path"),
            "--f-name",
            "smoke",
            "--block-size",
            "8",
            "--num-train-epochs",
            "2",
            "--batch-size",
            "2",
            "--lr",
            "1e-3",
            "--quant-func",
            "SmoothSign",
            "--eff-bit",
            "0.75",
            "--use-itq",
            "True",
            "--itq-n-iter",
            "5",
        ])
        .status()
        .expect("run train");
    assert!(train_status.success(), "train failed");
    assert!(train_out_dir.join("smoke").join("metrics.jsonl").is_file());
    assert!(train_out_dir
        .join("smoke")
        .join("model.safetensors")
        .is_file());

    // 6. Generate text from the trained LittleBit checkpoint.
    let gen_out = Command::new(bin)
        .args([
            "generate",
            "--model-id",
            train_out_dir.join("smoke").to_str().expect("path"),
            "--packed-inference",
            "--prompt",
            "tok1 tok2 tok3",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("run generate");
    assert!(
        gen_out.status.success(),
        "generate failed: {}",
        String::from_utf8_lossy(&gen_out.stderr)
    );
    let text = String::from_utf8_lossy(&gen_out.stdout);
    assert!(!text.trim().is_empty());

    // Touch library save helper to keep unused warning clean in test crate.
    let _ = (
        LittleBitModel::new,
        LittleBitConfig::default(),
        TrainMode::Eval,
        CastPolicy::ToBf16,
    );
}
