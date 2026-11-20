# littlebit-rs

Pure-Rust port of **[SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)** — implementing **LittleBit** (NeurIPS 2025) and **LittleBit-2 Joint-ITQ** (ICML 2026) for sub-1-bit LLM compression, Quantization-Aware Knowledge Distillation (QAT), bit-packed safetensors persistence, and compact 1-bit inference.

**Documentation & Rustdoc API:** [https://bhubbard.github.io/littlebit-rs/](https://bhubbard.github.io/littlebit-rs/)

---

## Overview

**LittleBit** compresses large language models into the sub-1-bit regime (down to `0.1` bits per weight) by factorizing each dense projection matrix $W \in \mathbb{R}^{b \times a}$ into low-rank latent factors $U \in \mathbb{R}^{b \times s}, V \in \mathbb{R}^{s \times a}$, binarizing those factors (`STEBinary` or `SmoothSign`), and restoring magnitude via rank-1 scale vectors $(u_1, u_2, v_1, v_2)$ (plus an optional dual residual factorization branch $(U_R, V_R, \dots)$).

**LittleBit-2** aligns the SVD-derived latent factors with the binary hypercube via **Internal Latent Rotation with Joint Iterative Quantization (Joint-ITQ)** (`--use-itq`), minimizing $\| \text{sign}(X R) - X R \|_F^2$ over $X = [U; V^\top]$ with zero extra inference overhead.

---

## Workspace Architecture

| Crate | Purpose |
| :--- | :--- |
| `littlebit-core` | `STEBinary` & `SmoothSign` autograd ops, LSB-first 32-bit `binary_packer` / `binary_unpacker`, compact `PackedBinaryMatrix` matmul, `faer`-backed thin & randomized low-rank SVD, LittleBit-2 Joint-ITQ rotation, `LittleBitLinear` / `PackedLittleBitLinear`, safetensors & header-only checkpoint inspection |
| `littlebit-models` | Differentiable decoders for OPT, Llama/Llama 2/3, Phi-3/4 (`PhiQKVSplitAttention`), Qwen2.5/QwQ, Gemma 2/3, and Qwen3; dynamic `apply_littlebit_patch` (`kv_factor`, `lm_head` exclusion); KV-cached generation; Hugging Face Hub `from_pretrained` & `push_to_hub` |
| `littlebit-train` | Knowledge Distillation trainer (`KdTrainer`: KL-divergence logit loss + scaled layer-to-layer hidden-state MSE), cosine-warmup scheduler, token-block dataset caching (SHA-256[:7]), perplexity & zero-shot multiple-choice evaluation, DeepSpeed `configs/zero*.json` parser |
| `littlebit-cli` | Unified `littlebit` binary: `quantize`, `train`, `eval`, `generate`, `inspect`, `pack`, `unpack`, and `push` |

---

## Supported Model Families

- **Llama / Llama 2 / Llama 3** (`model_type = "llama"`)
- **Qwen2 / Qwen2.5 / QwQ** (`model_type = "qwen2"`)
- **Qwen3** (`model_type = "qwen3"`, with per-head `q_norm` / `k_norm`)
- **Phi-3 / Phi-4** (`model_type = "phi3"`, splitting fused `qkv_proj` via `PhiQKVSplitAttention` before applying `kv_factor`)
- **Gemma 2 / Gemma 3** (`model_type = "gemma2"`, `"gemma3"`, `"gemma3_text"`)
- **Meta OPT** (`model_type = "opt"`)

---

## Building

Requires Rust `1.87+`:

```bash
# Default CPU build (uses faer multi-threaded BLAS/SVD + rayon)
cargo build --release

# Apple Silicon Accelerate framework
cargo build --release --features accelerate

# Apple Silicon Metal GPU
cargo build --release --features metal

# NVIDIA CUDA GPU
cargo build --release --features cuda
```

---

## CLI Usage

The CLI supports both kebab-case (`--use-itq`) and upstream underscored (`--use_itq True`) flags.

### 1. Post-Training Quantization (`quantize`)

```bash
cargo run --release -p littlebit-cli -- quantize \
    --model-id meta-llama/Llama-2-7b-hf \
    --save-dir ./outputs/Llama-2-7b-LittleBit-2 \
    --quant-func SmoothSign \
    --eff-bit 1.0 \
    --eff-bit-layers "0-3:2.0" \
    --residual True \
    --kv-factor 1.0 \
    --min-split-dim 8 \
    --use-itq True \
    --itq-n-iter 50
```

### 2. QAT + Knowledge Distillation (`train`)

```bash
cargo run --release -p littlebit-cli -- train \
    --model-id meta-llama/Llama-2-7b-hf \
    --dataset c4_wiki \
    --save-dir ./outputs/Llama-2-7b-LittleBit-2 \
    --num-train-epochs 5.0 \
    --per-device-train-batch-size 4 \
    --lr 4e-5 \
    --warmup-ratio 0.02 \
    --quant-func SmoothSign \
    --residual True \
    --eff-bit 1.0 \
    --kv-factor 1.0 \
    --min-split-dim 8 \
    --l2l-loss-scale 10.0 \
    --use-itq True
```

You can also train directly from a local `.txt`, `.jsonl`, or `.jsonl.gz` corpus with `--dataset-file <PATH>` and optionally read scheduler/clipping settings from `configs/zero3.json` via `--ds-config-path`.

### 3. Evaluation (`eval`)

```bash
# Perplexity evaluation using memory-compact 1-bit packed inference
cargo run --release -p littlebit-cli -- eval \
    --model-id ./outputs/Llama-2-7b-LittleBit-2 \
    --packed-inference \
    --seqlen 2048 \
    --ppl-task wikitext2,c4

# Zero-shot multiple-choice log-likelihood evaluation from JSONL
cargo run --release -p littlebit-cli -- eval \
    --model-id ./outputs/Llama-2-7b-LittleBit-2 \
    --zeroshot-task hellaswag \
    --zeroshot-file ./data/hellaswag_val.jsonl
```

### 4. Inspection, Packing, Generation & Hub Upload

```bash
# Inspect checkpoint layers, effective bits, and FP16 compression ratio (header-only)
littlebit inspect --model-id ./outputs/Llama-2-7b-LittleBit-2

# Autoregressive generation (with 1-bit packed weights)
littlebit generate --model-id ./outputs/Llama-2-7b-LittleBit-2 \
    --packed-inference \
    --prompt "Sub-1-bit LLM compression works by" \
    --max-tokens 64

# Convert between legacy unpacked and 1-bit packed safetensors
littlebit pack --input ./legacy_ckpt --output ./packed_ckpt
littlebit unpack --input ./packed_ckpt --output ./unpacked_ckpt

# Push quantized checkpoint + model card to Hugging Face Hub
littlebit push --model-dir ./outputs/Llama-2-7b-LittleBit-2 \
    --repo-id username/llama-2-7b-littlebit-1bpw
```

---

## Upstream Parity Notes

- **Safetensors Bit-Packing Parity**: Uses the exact upstream LSB-first `i32` encoding (`+1 -> bit 0`, `-1 -> bit 1`, `+1` padding) and `*_packed` / `*_shape` tensor keys so checkpoints round-trip cleanly with Python `SamsungLabs/LittleBit`.
- **Config Loading Priority**: Resolves parameters in upstream order: explicit CLI overrides > `littlebit_config.json` > `config.json` fallback > defaults.
- **Single-Node Training**: `littlebit train` executes single-process QAT + KD using Candle autograd (`rms_norm_slow`, `layer_norm_slow`, `rope_slow`, and custom `STEBinary` / `SmoothSign` backward ops) and parses `configs/zero*.json` for scheduler/clipping/gradient-accumulation hyperparameters without multi-node MPI/DeepSpeed dependency.
- **Zero-Shot Evaluation**: Implements native conditional log-likelihood and byte-normalized `acc_norm` scoring over JSONL multiple-choice benchmarks (`context`, `choices`, `gold_index`) without shelling out to Python `lm-eval`.
- **Mixed-Bit Overrides & Verification**: Supports per-block `--eff-bit-layers` bit-budget overrides and includes a 59-test Rust verification suite covering upstream issue regressions (Issue #21 bit unpacking and Issue #24 FP16 matmul overflow guard) alongside community audit and benchmark checks.

---

## Citations

```bibtex
@inproceedings{lee2026littlebit2,
  title={LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment},
  author={Lee, Banseok and Kim, Youngmin},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026}
}

@inproceedings{lee2025littlebit,
  title={LittleBit: Ultra Low-Bit Quantization via Latent Factorization},
  author={Lee, Banseok and Kim, Dongkyu and You, Youngcheon and Kim, Youngmin},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

Licensed under [CC BY-NC 4.0](LICENSE), matching upstream `SamsungLabs/LittleBit`. See [NOTICE](NOTICE) for third-party attributions (`PhiQKVSplitAttention` Apache-2.0 and `datautils` MIT).
