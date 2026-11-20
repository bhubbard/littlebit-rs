//! Perplexity and zero-shot multiple-choice evaluation (`eval.py` parity).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use littlebit_models::{CausalLm, HiddenStatesMode};

use crate::error::{Result, TrainError};

/// Perplexity measurement result for one dataset split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PplResult {
    /// Dataset name.
    pub dataset: String,
    /// Exponentiated average negative log-likelihood (`exp(total_nll / tokens)`).
    pub perplexity: f64,
    /// Mean negative log-likelihood per predicted token.
    pub avg_nll: f64,
    /// Number of full `seqlen` windows evaluated.
    pub num_samples: usize,
    /// Evaluation context window length.
    pub seqlen: usize,
}

/// Evaluate autoregressive perplexity on a flat token stream matching `eval.py`.
///
/// # Errors
/// Returns an error if `seqlen < 2`, `token_ids.len() < seqlen`, or model forward passes fail.
pub fn evaluate_perplexity(
    model: &CausalLm,
    token_ids: &[u32],
    seqlen: usize,
    dataset_name: &str,
    device: &Device,
) -> Result<PplResult> {
    if seqlen < 2 {
        return Err(TrainError::InvalidArgument(format!(
            "seqlen must be >= 2 for perplexity evaluation, got {seqlen}"
        )));
    }
    let nsamples = token_ids.len() / seqlen;
    if nsamples == 0 {
        return Err(TrainError::InvalidArgument(format!(
            "not enough tokens ({}) for seqlen {seqlen} on `{dataset_name}`",
            token_ids.len()
        )));
    }

    let mut total_nll = 0.0_f64;
    let pred_tokens_per_sample = seqlen - 1;

    for i in 0..nsamples {
        let chunk = &token_ids[i * seqlen..(i + 1) * seqlen];
        let batch = Tensor::from_slice(chunk, (1, seqlen), device)?;
        let out = model.forward(&batch, None, HiddenStatesMode::None)?;
        let vocab_size = out.logits.dims()[2];

        let shift_logits = out
            .logits
            .i((0, ..pred_tokens_per_sample, ..))?
            .reshape((pred_tokens_per_sample, vocab_size))?
            .to_dtype(DType::F32)?;
        let shift_labels = Tensor::from_slice(&chunk[1..], pred_tokens_per_sample, device)?;
        let loss = candle_nn::loss::cross_entropy(&shift_logits, &shift_labels)?
            .to_scalar::<f32>()? as f64;
        total_nll += loss * (pred_tokens_per_sample as f64);
    }

    let total_tokens = (nsamples * pred_tokens_per_sample) as f64;
    let avg_nll = total_nll / total_tokens;
    let perplexity = avg_nll.exp();

    Ok(PplResult {
        dataset: dataset_name.to_string(),
        perplexity,
        avg_nll,
        num_samples: nsamples,
        seqlen,
    })
}

/// One multiple-choice zero-shot evaluation example (`boolq`, `piqa`, `hellaswag`, `arc`, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleChoiceExample {
    /// Prompt or conditioning context string.
    pub context: String,
    /// Candidate continuation strings.
    pub choices: Vec<String>,
    /// 0-based index of the correct choice in `choices`.
    pub gold_index: usize,
}

/// Aggregated accuracy metrics for a zero-shot multiple-choice benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotTaskResult {
    /// Benchmark task name.
    pub task: String,
    /// Number of evaluated examples.
    pub num_examples: usize,
    /// Unnormalized log-likelihood accuracy (`acc`).
    pub acc: f64,
    /// Byte-length-normalized log-likelihood accuracy (`acc_norm`).
    pub acc_norm: f64,
}

/// Load multiple-choice examples from a JSONL file.
///
/// Accepts records formatted as `{"context": "...", "choices": ["...", ...], "gold_index": 0}`.
///
/// # Errors
/// Returns an error if the file cannot be read or contains invalid JSON lines.
pub fn load_multiple_choice_jsonl(path: &Path) -> Result<Vec<MultipleChoiceExample>> {
    let file = File::open(path)?;
    let mut examples = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let ex: MultipleChoiceExample = serde_json::from_str(trimmed)?;
        examples.push(ex);
    }
    Ok(examples)
}

/// Evaluate conditional log-likelihood accuracy across multiple-choice examples,
/// matching EleutherAI `lm-eval` scoring (`acc` and byte-normalized `acc_norm`).
///
/// # Errors
/// Returns an error if `examples` is empty, tokenization fails, or model forward fails.
pub fn evaluate_multiple_choice(
    model: &CausalLm,
    tokenizer: &Tokenizer,
    task_name: &str,
    examples: &[MultipleChoiceExample],
    limit: Option<usize>,
    device: &Device,
) -> Result<ZeroShotTaskResult> {
    let slice = match limit {
        Some(n) if n > 0 && n < examples.len() => &examples[..n],
        _ => examples,
    };
    if slice.is_empty() {
        return Err(TrainError::InvalidArgument(format!(
            "no examples provided for zero-shot task `{task_name}`"
        )));
    }

    let mut correct_raw = 0usize;
    let mut correct_norm = 0usize;

    for ex in slice {
        if ex.choices.is_empty() || ex.gold_index >= ex.choices.len() {
            return Err(TrainError::InvalidArgument(format!(
                "invalid multiple-choice example in task `{task_name}`"
            )));
        }

        let ctx_ids = if ex.context.is_empty() {
            Vec::new()
        } else {
            tokenizer
                .encode(ex.context.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("encode failed: {err}")))?
                .get_ids()
                .to_vec()
        };

        let mut best_raw_idx = 0usize;
        let mut best_raw_score = f64::NEG_INFINITY;
        let mut best_norm_idx = 0usize;
        let mut best_norm_score = f64::NEG_INFINITY;

        for (c_idx, choice) in ex.choices.iter().enumerate() {
            let full_text = if ex.context.is_empty() {
                choice.clone()
            } else {
                format!("{}{choice}", ex.context)
            };
            let full_ids = tokenizer
                .encode(full_text.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("encode failed: {err}")))?
                .get_ids()
                .to_vec();

            if full_ids.len() < 2 {
                continue;
            }

            let input_t = Tensor::from_slice(&full_ids, (1, full_ids.len()), device)?;
            let logits = model
                .forward(&input_t, None, HiddenStatesMode::None)?
                .logits
                .get(0)?
                .to_dtype(DType::F32)?;
            let log_probs = candle_nn::ops::log_softmax(&logits, D::Minus1)?;

            let start_pos = ctx_ids.len().max(1).min(full_ids.len() - 1);
            let mut sum_ll = 0.0_f64;
            for (pos, &tok) in full_ids.iter().enumerate().skip(start_pos) {
                let lp = log_probs
                    .get(pos - 1)?
                    .get(tok as usize)?
                    .to_scalar::<f32>()? as f64;
                sum_ll += lp;
            }
            let norm_ll = sum_ll / (choice.len().max(1) as f64);

            if sum_ll > best_raw_score {
                best_raw_score = sum_ll;
                best_raw_idx = c_idx;
            }
            if norm_ll > best_norm_score {
                best_norm_score = norm_ll;
                best_norm_idx = c_idx;
            }
        }

        if best_raw_idx == ex.gold_index {
            correct_raw += 1;
        }
        if best_norm_idx == ex.gold_index {
            correct_norm += 1;
        }
    }

    let n = slice.len() as f64;
    Ok(ZeroShotTaskResult {
        task: task_name.to_string(),
        num_examples: slice.len(),
        acc: (correct_raw as f64) / n,
        acc_norm: (correct_norm as f64) / n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use littlebit_models::ModelConfig;
    use std::collections::HashMap;

    fn test_tokenizer() -> Tokenizer {
        let json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "the": 1,
                    "cat": 2,
                    "sat": 3,
                    "dog": 4,
                    "flew": 5,
                    "on": 6,
                    "mat": 7
                },
                "unk_token": "<unk>"
            },
            "pre_tokenizer": { "type": "Whitespace" }
        });
        Tokenizer::from_bytes(json.to_string().as_bytes()).expect("test tokenizer")
    }

    #[test]
    fn perplexity_on_zero_weights_equals_vocab_size() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::tiny_llama(32, 16, 1);
        // Zero lm_head weights produce identically 0 logits -> uniform softmax 1/32 -> PPL = 32.0.
        let mut tensors = HashMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((32, 16), DType::F32, &dev)?,
        );
        tensors.insert(
            "model.layers.0.input_layernorm.weight".to_string(),
            Tensor::ones(16, DType::F32, &dev)?,
        );
        tensors.insert(
            "model.layers.0.post_attention_layernorm.weight".to_string(),
            Tensor::ones(16, DType::F32, &dev)?,
        );
        for name in ["self_attn.q_proj", "self_attn.o_proj"] {
            tensors.insert(
                format!("model.layers.0.{name}.weight"),
                Tensor::zeros((16, 16), DType::F32, &dev)?,
            );
        }
        for name in ["self_attn.k_proj", "self_attn.v_proj"] {
            tensors.insert(
                format!("model.layers.0.{name}.weight"),
                Tensor::zeros((8, 16), DType::F32, &dev)?,
            );
        }
        tensors.insert(
            "model.layers.0.mlp.gate_proj.weight".to_string(),
            Tensor::zeros((32, 16), DType::F32, &dev)?,
        );
        tensors.insert(
            "model.layers.0.mlp.up_proj.weight".to_string(),
            Tensor::zeros((32, 16), DType::F32, &dev)?,
        );
        tensors.insert(
            "model.layers.0.mlp.down_proj.weight".to_string(),
            Tensor::zeros((16, 32), DType::F32, &dev)?,
        );
        tensors.insert(
            "model.norm.weight".to_string(),
            Tensor::ones(16, DType::F32, &dev)?,
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            Tensor::zeros((32, 16), DType::F32, &dev)?,
        );

        let model = CausalLm::from_tensors(
            cfg,
            &tensors,
            None,
            littlebit_core::WeightState::Raw,
            littlebit_core::InferenceMode::Unpacked,
            DType::F32,
            &dev,
        )?;

        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let ppl = evaluate_perplexity(&model, &tokens, 6, "uniform_test", &dev)?;
        assert_eq!(ppl.num_samples, 2);
        assert!(
            (ppl.perplexity - 32.0).abs() < 1e-3,
            "expected uniform PPL=32.0, got {}",
            ppl.perplexity
        );
        Ok(())
    }

    #[test]
    fn multiple_choice_scores_candidate_continuations() -> Result<()> {
        let dev = Device::Cpu;
        let tok = test_tokenizer();
        let cfg = ModelConfig::tiny_llama(16, 16, 1);
        let model = CausalLm::new_random(cfg, DType::F32, &dev, 7)?;
        let examples = vec![MultipleChoiceExample {
            context: "the cat".to_string(),
            choices: vec![" sat on mat".to_string(), " flew".to_string()],
            gold_index: 0,
        }];
        let res = evaluate_multiple_choice(&model, &tok, "synthetic", &examples, None, &dev)?;
        assert_eq!(res.num_examples, 1);
        assert!(res.acc == 0.0 || res.acc == 1.0);
        assert!(res.acc_norm == 0.0 || res.acc_norm == 1.0);
        Ok(())
    }
}
