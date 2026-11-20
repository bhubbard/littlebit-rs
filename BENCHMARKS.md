# Benchmarks

Measured pure-Rust release performance, compression ratios, and reconstruction error for **`littlebit-rs`** on synthetic seeded tensors (no external model downloads).

---

## Test Environment

| Property | Value |
| :--- | :--- |
| **Machine** | Apple M3 Pro (`arm64` / `aarch64`, 11 CPU cores, 18 GiB RAM, macOS) |
| **Rust Toolchain** | `rustc 1.100.0 (0ef1bc386 2026-11-09)` / `cargo 1.100.0 (3acc79c0c 2026-11-05)` |
| **Build Profile** | `--release` (`lto = "thin"`, `codegen-units = 1`, `strip = "symbols"`) |
| **Iteration Settings** | `warmup = 3`, `iters = 15` (microbenchmarks); `warmup = 1`, `iters = 5` (`generate`) |
| **Backends Tested** | `CPU (default)` (`--features ""`) vs. `CPU + Accelerate` (`--features accelerate`) |

### Reproduce Locally

```bash
# Default CPU (faer + rayon)
cargo run --release -p littlebit-cli --example bench

# macOS Apple Accelerate BLAS
cargo run --release -p littlebit-cli --features accelerate --example bench
```

See [examples/bench.rs](crates/littlebit-cli/examples/bench.rs) for the complete benchmark harness.

---

## 1. Binary Packer & Unpacker Throughput (`pack_i8_slice` / `unpack_to_i8`)

LSB-first 32-bit packing of `{-1, +1}` weight matrices (`i8` → `i32` words):

| Matrix Shape | Weights | Packed Size | Compression vs FP16 | Pack Median / p95 (CPU) | Pack Speed (CPU) | Unpack Median / p95 (CPU) | Unpack Speed (CPU) | Pack / Unpack (Accelerate) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024 × 4096` | 4,194,304 | 0.50 MiB (524,288 B) | `16.00×` | `18.03 ms` / `18.78 ms` | `232.7 MW/s` | `2.69 ms` / `2.97 ms` | `1,559.1 MW/s` | `18.34 ms` / `2.71 ms` |
| `4096 × 4096` | 16,777,216 | 2.00 MiB (2,097,152 B) | `16.00×` | `72.87 ms` / `87.20 ms` | `230.2 MW/s` | `11.18 ms` / `11.59 ms` | `1,500.6 MW/s` | `73.04 ms` / `11.16 ms` |
| `4096 × 11008` | 45,088,768 | 5.38 MiB (5,636,096 B) | `16.00×` | `196.00 ms` / `201.16 ms` | `230.0 MW/s` | `29.82 ms` / `34.86 ms` | `1,512.1 MW/s` | `196.82 ms` / `30.24 ms` |

---

## 2. `LittleBitLinear` Projection Microbenchmarks

Comparison of `Dense F32` (`x @ W^T`), `Unpacked LittleBitLinear` (`F32` decomposed matmuls), and `PackedLittleBitLinear` (direct 1-bit `PackedBinaryMatrix::matmul_transposed_f32` without unpacking weights).

### Single-Token Decode (`batch = 1`)

| Layer Configuration | `split_dim` | Actual `bpw` | Packed Size (vs FP16) | Dense F32 (Default / Accel) | Unpacked LB (Default / Accel) | Packed 1-Bit (Default / Accel) | Packed vs Unpacked (Accel) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-7B Attn 4096×4096 @ 0.1 bpw` (`single`) | `184` | `0.098` | `222.2 KiB` (`147.50×`) | `1.62 ms` / `1.57 ms` | `4.55 ms` / `4.56 ms` | **`0.39 ms`** / **`0.27 ms`** | **`17.1×` faster** |
| `Llama-7B Attn 4096×4096 @ 0.5 bpw` (`residual`) | `488` | `0.493` | `1.05 MiB` (`30.47×`) | `1.71 ms` / `1.16 ms` | `25.97 ms` / `24.85 ms` | **`1.40 ms`** / **`1.30 ms`** | **`19.1×` faster** |
| `Llama-7B Attn 4096×4096 @ 1.0 bpw` (`residual`) | `1000` | `0.994` | `2.06 MiB` (`15.52×`) | `1.62 ms` / `1.58 ms` | `55.41 ms` / `52.01 ms` | **`2.41 ms`** / **`2.51 ms`** | **`20.7×` faster** |
| `Llama-7B MLP 4096×11008 @ 0.5 bpw` (`residual`) | `728` | `0.499` | `2.77 MiB` (`31.00×`) | `5.42 ms` / `4.29 ms` | `68.40 ms` / `68.04 ms` | **`2.77 ms`** / **`2.77 ms`** | **`24.6×` faster** |

### Batched Prefill (`batch = 16`)

| Layer Configuration | Dense F32 Median / p95 (Default) | Dense F32 Median / p95 (Accel) | Unpacked LB Median (Default / Accel) | Packed 1-Bit Median (Default / Accel) |
| :--- | ---: | ---: | ---: | ---: |
| `Llama-7B Attn 4096×4096 @ 0.1 bpw` (`single`) | `10.84 ms` / `13.97 ms` | **`4.22 ms`** / `4.39 ms` | `5.61 ms` / `4.59 ms` | `5.03 ms` / `4.74 ms` |
| `Llama-7B Attn 4096×4096 @ 0.5 bpw` (`residual`) | `10.34 ms` / `13.42 ms` | **`4.42 ms`** / `4.68 ms` | `34.91 ms` / `26.46 ms` | `26.33 ms` / `22.18 ms` |
| `Llama-7B Attn 4096×4096 @ 1.0 bpw` (`residual`) | `11.30 ms` / `15.09 ms` | **`4.44 ms`** / `5.16 ms` | `66.04 ms` / `51.94 ms` | `49.28 ms` / `35.71 ms` |
| `Llama-7B MLP 4096×11008 @ 0.5 bpw` (`residual`) | `16.83 ms` / `18.24 ms` | **`5.65 ms`** / `6.18 ms` | `75.06 ms` / `70.09 ms` | `45.65 ms` / `47.52 ms` |

---

## 3. Quantization & LittleBit-2 Joint-ITQ Initialization

Zero-shot factorization of seeded standard-normal weight matrices `W` (`std = 0.02`, `eff_bit = 0.5`, `residual = true`), comparing SVD-only initialization (LittleBit v1) against 20-iteration Joint-ITQ latent rotation (LittleBit-2):

| Matrix Shape | `split_dim` | Method | Init Time (Default / Accel) | Relative Frobenius Error `‖W − Ŵ‖_F / ‖W‖_F` | Error Reduction |
| :--- | ---: | :--- | ---: | ---: | ---: |
| `1024 × 1024` | `104` | SVD-only (`use_itq = false`) | `92.46 ms` / `95.51 ms` | `0.88363` | — |
| `1024 × 1024` | `104` | **Joint-ITQ** (`20` iters) | `201.67 ms` / `222.82 ms` | **`0.86192`** | **`-0.02171`** |
| `2048 × 2048` | `232` | SVD-only (`use_itq = false`) | `454.61 ms` / `517.75 ms` | `0.87422` | — |
| `2048 × 2048` | `232` | **Joint-ITQ** (`20` iters) | `949.41 ms` / `983.07 ms` | **`0.84829`** | **`-0.02593`** |

---

## 4. End-to-End 4-Layer Causal LM Storage & KV-Cached Generation

Compact 4-layer synthetic transformers (`hidden_size = 512`, `intermediate_size = 1376`, `vocab_size = 4096`, `num_attention_heads = 8`, `num_key_value_heads = 4`), quantized at `eff_bit = 0.5`, `residual = true`, `use_itq = true` (`15` iters), saved with `CastPolicy::ToBf16`, generating **32 new tokens** from a **16-token prompt** (greedy, KV-cached):

| Model | Avg `eff_bit` | Quantized Layers Size | Dense FP16 Layers | Compression (Layers) | Total Checkpoint (`BF16` embed/head) | Dense `F32` (Default / Accel) | Unpacked LB (Default / Accel) | Packed 1-Bit (Default / Accel) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-4L (h=512, i=1376, v=4096)` | `0.469` | `0.76 MiB` (794,816 B) | `22.12 MiB` (23,199,744 B) | **`29.19×`** | `8.77 MiB` (9,192,640 B) | `9.2` / `23.9 tok/s` (`3,473.4` / `1,338.7 ms`) | `46.4` / `51.3 tok/s` (`690.0` / `623.6 ms`) | `43.1` / **`69.4 tok/s`** (`741.8` / `461.3 ms`) |
| `Qwen3-4L (h=512, i=1376, v=4096)` | `0.469` | `0.76 MiB` (794,816 B) | `22.12 MiB` (23,199,744 B) | **`29.19×`** | `8.77 MiB` (9,193,664 B) | `21.4` / `23.7 tok/s` (`1,497.1` / `1,348.8 ms`) | `46.3` / `50.6 tok/s` (`691.0` / `632.8 ms`) | `58.9` / **`67.0 tok/s`** (`542.9` / `478.0 ms`) |

---

## Key Observations

1. **1-Bit Packed Decode Advantage (`b = 1`):** On memory-bound single-token decode, `PackedLittleBitLinear` delivers `0.27 ms` at `0.1 bpw` (`4096×4096`, `5.9×` faster than Accelerate dense `F32` and `17.1×` faster than unpacked LittleBit) and `2.77 ms` on `4096×11008` MLP (`1.55×` faster than Accelerate dense `F32` and `24.6×` faster than unpacked), while cutting layer storage by **`15.5×`–`147.5×`** vs. FP16.
2. **End-to-End KV-Cached Generation:** Packed 1-bit inference reaches **`69.4 tok/s`** on `Llama-4L` (`2.90×` faster than Accelerate dense `F32` at `23.9 tok/s`) and **`67.0 tok/s`** on `Qwen3-4L` (`2.82×` faster than Accelerate dense `F32` at `23.7 tok/s`), compressing linear projection weights by **`29.19×`**.
3. **Batched Prefill (`b = 16`):** Apple Accelerate's tiled SGEMM outperforms per-row CPU packed bit-scanning when `batch = 16` (`4.22–5.65 ms` dense vs. `22.18–47.52 ms` packed at `0.5–1.0 bpw`), because dense GEMM amortizes weight loads across the batch whereas `PackedBinaryMatrix::matmul_transposed_f32` currently evaluates one batch row at a time.
4. **Joint-ITQ Initialization Tradeoff:** Enabling `use_itq = true` (`20` iterations) takes ~`2.1×–2.2×` longer during offline quantization (`201.7 ms` vs. `92.5 ms` at `1024×1024`; `949.4 ms` vs. `454.6 ms` at `2048×2048`) while consistently reducing post-initialization relative Frobenius reconstruction error (`-0.0217` and `-0.0259`) with zero runtime overhead.
