# littlebit-rs

Shrink large open-source AI models down to less than 1 bit per weight so they fit in a fraction of the RAM and run offline on an everyday laptop — packaged as a single pure-Rust binary with zero Python or PyTorch dependencies. Based on Samsung Research's **[LittleBit](https://github.com/SamsungLabs/LittleBit)** (NeurIPS 2025) and **LittleBit-2** (ICML 2026).

**Documentation & Rustdoc API:** [https://code.brandonhubbard.com/littlebit-rs/](https://code.brandonhubbard.com/littlebit-rs/) &middot; **Benchmarks:** [BENCHMARKS.md](BENCHMARKS.md)

---

## Overview

Standard 16-bit open models are huge: a 7B model needs ~14 GB of memory, and 26B–31B models need 50–60+ GB before you can even load them. **littlebit-rs** compresses those models into tiny 1-bit patterns paired with lightweight scale factors (down to **0.1–1.0 bits per weight**), and executes directly on compact packed weights (`i32` bitfields) instead of inflating everything back to full precision in memory.

- **Fits Big Models on Normal Hardware:** Cut resident weight memory dramatically (`26×–30×` lower resident RAM in our packed benchmarks vs. stock PyTorch unpacking) and start up in ~10 ms.
- **Better Accuracy at Extreme Compression (`--use-itq`):** Optionally enables LittleBit-2's rotation step during compression to better preserve model quality with zero runtime overhead, plus optional Knowledge Distillation fine-tuning (`train`).
- **Stream Giant Checkpoints Without High RAM (`--streaming`):** Quantize massive checkpoints (such as Gemma 4 26B MoE or 31B Dense) one transformer block or expert at a time directly from disk into sharded safetensors—without needing a 64 GB+ cloud machine.
- **Compatible with Python Checkpoints:** Reads and writes the exact same packed `.safetensors` format as `SamsungLabs/LittleBit`.

<details>
<summary>Technical details: How LittleBit &amp; LittleBit-2 work under the hood</summary>

Each dense projection matrix $W \in \mathbb{R}^{b \times a}$ is factorized into low-rank latent factors $U \in \mathbb{R}^{b \times s}$ and $V \in \mathbb{R}^{s \times a}$, binarized (`STEBinary` or `SmoothSign`), and rescaled using rank-1 scale vectors $(u_1, u_2, v_1, v_2)$ (plus an optional dual residual branch $(U_R, V_R, \dots)$). LittleBit-2 **Joint-ITQ** aligns the SVD factors to the binary hypercube by minimizing $\| \text{sign}(X R) - X R \|_F^2$ over $X = [U; V^\top]$ with no extra runtime parameters.

</details>

---

## Example Use Cases

- **Run large open models on an everyday laptop:** Compress 7B–31B models (Llama 3, Qwen3, Phi-4, Gemma 4) to fit comfortably alongside normal desktop apps on CPU or Apple Silicon (`--features accelerate`).
- **Private local classification, triage & routing:** Score, tag, or route emails, support tickets, or legal intake forms completely offline with `$0` API costs—especially effective for single-pass next-token logit scoring (choosing between discrete labels `"A"` / `"B"` / `"C"` without multi-token generation drift).
- **Offline micro-agents & speculative draft models:** Run short, structured extraction or agent tasks locally, or pair a fast 0.5–1.0 bpw compressed checkpoint as a draft model in a local serving stack.
- **Compress huge Hugging Face checkpoints without renting a big GPU box:** Use `--streaming --text-only` to quantize multi-shard 26B–31B models block-by-block on an 18 GB laptop and emit sharded output files ready for Hub upload or local execution.
- **Embed LLM inference cleanly in Rust applications:** Drop `littlebit-core` / `littlebit-models` directly into Rust CLIs, local voice pipelines, desktop apps, or proxies without bundling Python, conda, or PyTorch.

---

## Workspace Architecture

| Crate | Purpose |
| :--- | :--- |
| `littlebit-core` | `STEBinary` & `SmoothSign` autograd ops, LSB-first 32-bit `binary_packer` / `binary_unpacker`, compact `PackedBinaryMatrix` matmul, `faer`-backed thin & randomized low-rank SVD, LittleBit-2 Joint-ITQ rotation, `LittleBitLinear` / `PackedLittleBitLinear`, streaming `SafetensorsReader` / `ShardedSafetensorsWriter`, and header-only checkpoint inspection |
| `littlebit-models` | Differentiable decoders for OPT, Llama/Llama 2/3, Phi-3/4 (`PhiQKVSplitAttention`), Qwen2.5/QwQ, Qwen3, Gemma 2/3, and Gemma 4 Dense & MoE (p-RoPE, `k_eq_v`, sparse expert dispatch); streaming layer-by-layer quantizer; dynamic `apply_littlebit_patch` (`kv_factor`, `lm_head`/`router` exclusion); KV-cached generation; Hugging Face Hub `from_pretrained` & `push_to_hub` |
| `littlebit-train` | Knowledge Distillation trainer (`KdTrainer`: KL-divergence logit loss + scaled layer-to-layer hidden-state MSE), cosine-warmup scheduler, token-block dataset caching (SHA-256[:7]), perplexity & zero-shot multiple-choice evaluation, DeepSpeed `configs/zero*.json` parser |
| `littlebit-cli` | Unified `littlebit` binary: `quantize` (with `--streaming`), `train`, `eval`, `generate`, `inspect`, `pack`, `unpack`, and `push` |

---

## Supported Model Families

- **Llama / Llama 2 / Llama 3** (`model_type = "llama"`)
- **Qwen2 / Qwen2.5 / QwQ** (`model_type = "qwen2"`)
- **Qwen3** (`model_type = "qwen3"`, with per-head `q_norm` / `k_norm`)
- **Phi-3 / Phi-4** (`model_type = "phi3"`, splitting fused `qkv_proj` via `PhiQKVSplitAttention` before applying `kv_factor`)
- **Gemma 2 / Gemma 3** (`model_type = "gemma2"`, `"gemma3"`, `"gemma3_text"`)
- **Gemma 4 Dense & MoE** (`model_type = "gemma4"`, `"gemma4_text"` — hybrid sliding/full attention, proportional RoPE, `attention_k_eq_v`, shared MLP + unquantized top-k router with 1-bit packed experts; 26B-A4B / 31B text backbones)
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

For larger checkpoints on a memory-constrained laptop (e.g., Gemma 4 26B MoE or 31B Dense), stream one block at a time directly from safetensors shards and optionally drop vision/audio towers with `--text-only`:

```bash
cargo run --release --features accelerate -p littlebit-cli -- quantize \
    --model-id google/gemma-4-26B-A4B-it \
    --save-dir ./outputs/Gemma-4-26B-LittleBit-2 \
    --streaming \
    --text-only \
    --max-shard-mb 2048 \
    --eff-bit 1.5 \
    --residual True \
    --use-itq True \
    --itq-n-iter 15
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
- **Mixed-Bit Overrides & Verification**: Supports per-block `--eff-bit-layers` bit-budget overrides and includes a 69-test Rust verification suite covering upstream issue regressions (Issue #21 bit unpacking and Issue #24 FP16 matmul overflow guard) alongside Gemma 4 Dense/MoE, streaming sharded quantization, and community audit checks.

---

## Benchmarks

Run the pure-Rust release benchmark suite (`pack_i8_slice`, `LittleBitLinear` vs. `PackedLittleBitLinear`, Joint-ITQ initialization, and 4-layer KV-cached generation):

```bash
cargo run --release -p littlebit-cli --example bench
cargo run --release -p littlebit-cli --features accelerate --example bench
```

See [BENCHMARKS.md](BENCHMARKS.md) and the [GitHub Pages benchmark tables](https://code.brandonhubbard.com/littlebit-rs/#benchmarks) for same-machine **Apple M3 Pro** head-to-head results comparing `littlebit-rs` against upstream `SamsungLabs/LittleBit` (`42d658b`, Python 3.15.2 / PyTorch 2.15.0): `41.2×` smaller deployment footprint (`18.59 MiB` vs. `766.79 MiB`), `373×`–`393×` faster cold startup (`9.80 ms` vs. `3,854.33 ms`), `4.1×`–`17.6×` faster binary unpacking, and `26.7×`–`30.3×` lower resident 1-bit weight RAM (plus `51.6`–`58.1 tok/s` packed 4-layer KV-cached decode).

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
