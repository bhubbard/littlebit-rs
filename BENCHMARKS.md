# Benchmarks

Measured pure-Rust release performance, resident memory footprint, compression ratios, and reconstruction error for **`littlebit-rs`**, including a direct same-machine head-to-head comparison against unmodified upstream **[SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)** (Python / PyTorch).

> [!NOTE]
> All benchmarks use deterministic synthetic seeded tensors (no external model downloads) executed back-to-back on the same host. Where upstream Python uses zero-copy strided float views or BLAS-accelerated LAPACK, those numbers are reported side by side alongside Rust's wins.

---

## Test Environment

| Property | `littlebit-rs` (Rust) | Original `SamsungLabs/LittleBit` (Python / PyTorch) |
| :--- | :--- | :--- |
| **Host Machine** | Apple M3 Pro (`arm64` / `aarch64`, 11 CPU cores, 18 GiB RAM, macOS) | Same machine |
| **Version / Commit** | Workspace `0.1.0` (`rustc 1.100.0` / `cargo 1.100.0`) | Upstream commit `42d658b` (`Python 3.15.2`, `torch 2.15.0`) |
| **Build / Threads** | `--release` (`lto = "thin"`, `codegen-units = 1`, `strip = "symbols"`); `11` threads (`rayon` + `faer`) | CPU default (`torch.get_num_threads() = 5`) |
| **Backends Tested** | `CPU (default)` and `CPU + Accelerate` (`--features accelerate`) | Stock PyTorch CPU (`torch.no_grad()`) |
| **Timings** | `warmup = 3`, `iters = 15` (microbenchmarks); `warmup = 1`, `iters = 5` (`generate`) | `warmup = 3`, `iters = 15` (`time.perf_counter()`) |

### Reproduce Rust Suite Locally

```bash
# Default CPU (faer + rayon)
cargo run --release -p littlebit-cli --example bench

# macOS Apple Accelerate BLAS
cargo run --release -p littlebit-cli --features accelerate --example bench
```

See [examples/bench.rs](crates/littlebit-cli/examples/bench.rs) for the complete Rust benchmark harness.

---

## Head-to-Head: `littlebit-rs` (Rust) vs. Original `SamsungLabs/LittleBit` (Python / PyTorch)

### 1. Deployment Footprint, Cold Startup & Runtime Capabilities

| Metric / Capability | `littlebit-rs` (Rust) | Original `SamsungLabs/LittleBit` (`42d658b`) | Comparison |
| :--- | ---: | ---: | :--- |
| **Install / Binary Size** | **`18.59 MiB`** (`19,492,544 B`, Accel) / `18.67 MiB` (default) | `766.79 MiB` (`804,040,704 B`, `.venv` w/ `torch` only) | **`41.2×` smaller** standalone binary |
| **Cold Startup (5-run median)** | **`9.80 ms`** (default) / **`10.32 ms`** (Accel `littlebit --version`) | `3,854.33 ms` (`python -c "import torch"`) | **`373×`–`393×` faster** cold start |
| **1-Bit Packed Runtime Matmul** | Built-in (`PackedLittleBitLinear`) | None (always unpacks `*_packed` to float RAM) | **1-bit resident inference** in Rust |
| **FP16 Overflow Guard (Issue #24)** | Included (`pow2_at_least` + CPU `F32` accumulation) | Open issue (not merged upstream) | Prevents NaN on `> 65,504` sums |
| **Per-Block Mixed Bits (`--eff-bit-layers`)** | Built-in (`littlebit_config.json` + `config.json`) | Requires third-party patch | Native per-block rank scaling |
| **Verification Suite** | **59 Rust tests** (0 `unsafe`, 0 `.py` files) | `0` unit tests in upstream repo | Full regression coverage |

### 2. Binary Packer & Unpacker Speed

LSB-first `i8` → `i32` (`16.00×` smaller than `FP16`):

| Matrix Shape | Weights | Packed Size | Rust Unpack (Default / Accel) | PyTorch Unpack (`42d658b`) | Unpack Speedup (Rust Accel vs. PyTorch) | Rust Pack (Default / Accel) | PyTorch Pack (`42d658b`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024 × 4096` | 4,194,304 | 0.50 MiB | `3.46 ms` / **`3.01 ms`** (`1,394.3 MW/s`) | `14.47 ms` (`289.8 MW/s`) | **`4.81×` faster** | **`18.23 ms`** / **`18.23 ms`** (`230.1 MW/s`) | `36.15 ms` (`116.0 MW/s`) |
| `4096 × 4096` | 16,777,216 | 2.00 MiB | `14.45 ms` / **`11.59 ms`** (`1,447.1 MW/s`) | `203.77 ms` (`82.3 MW/s`) | **`17.58×` faster** | `73.70 ms` / `72.17 ms` (`232.5 MW/s`) | **`58.01 ms`** (`289.2 MW/s`) |
| `4096 × 11008` | 45,088,768 | 5.38 MiB | `41.71 ms` / **`35.61 ms`** (`1,266.2 MW/s`) | `147.12 ms` (`306.5 MW/s`) | **`4.13×` faster** | `199.43 ms` / `196.81 ms` (`229.1 MW/s`) | **`43.68 ms`** (`1,032.2 MW/s`) |

### 3. Resident Weight RAM & Projection Latency (`batch = 1` and `batch = 16`)

Because upstream Python unpacks `*_packed` tensors into dense floats in RAM at load time (`F32` on CPU; `BF16` on GPU), Rust's `PackedLittleBitLinear` holds **`26.7×`–`30.3×` less resident parameter RAM** than PyTorch `F32` (`13.3×`–`15.2×` less than `BF16`) while keeping single-token decode latency within `0.35–4.74 ms`:

| Layer Configuration | `split_dim` (`bpw`) | Rust 1-Bit Resident RAM | PyTorch Resident RAM (`F32` / `BF16`) | Memory Savings vs. PyTorch (`F32` / `BF16`) | `b=1` Rust Packed (Default / Accel) | `b=1` PyTorch `LittleBitLinear` | `b=1` Rust Unpacked (Accel) | `b=16` Rust Packed (Accel) vs. PyTorch `LittleBitLinear` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-7B Attn 4096×4096 @ 0.1` (`single`) | `184` (`0.098`) | **`222.2 KiB`** (227,488 B) | `5.78 MiB` / `2.89 MiB` | **`26.65×`** / `13.33×` | `0.47 ms` / **`0.35 ms`** | **`0.06 ms`** | `4.69 ms` | `7.02 ms` vs. **`1.69 ms`** |
| `Llama-7B Attn 4096×4096 @ 0.5` (`residual`) | `488` (`0.493`) | **`1.05 MiB`** (1,101,248 B) | `30.57 MiB` / `15.28 MiB` | **`29.11×`** / `14.55×` | `1.70 ms` / **`1.42 ms`** | **`0.88 ms`** | `25.67 ms` | `25.75 ms` vs. **`5.68 ms`** |
| `Llama-7B Attn 4096×4096 @ 1.0` (`residual`) | `1000` (`0.994`) | **`2.06 MiB`** (2,162,112 B) | `62.58 MiB` / `31.29 MiB` | **`30.35×`** / `15.17×` | `3.04 ms` / **`2.83 ms`** | **`2.06 ms`** | `54.69 ms` | `45.70 ms` vs. **`9.47 ms`** |
| `Llama-7B MLP 4096×11008 @ 0.5` (`residual`) | `728` (`0.499`) | **`2.77 MiB`** (2,909,248 B) | `84.02 MiB` / `42.01 MiB` | **`30.28×`** / `15.14×` | **`4.11 ms`** / `4.74 ms` | **`3.36 ms`** | `73.65 ms` | `56.82 ms` vs. **`12.51 ms`** |

*(Dense `F32` baseline at `b=1` / `b=16` for `4096×4096` and `4096×11008`: Rust Accel `1.09–3.64 ms` / `4.09–5.90 ms`; PyTorch `1.40–6.05 ms` / `7.23–21.93 ms`.)*

### 4. SVD & LittleBit-2 Joint-ITQ Initialization

Zero-shot factorization of identical seeded normal weights (`eff_bit = 0.5`, `residual = true`):

| Matrix Shape | `split_dim` | Method | Rust (`faer`) Init (Default / Accel) | PyTorch Init (`42d658b`) | Rust Rel. Frobenius Error | PyTorch Rel. Frobenius Error |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| `1024 × 1024` | `104` | SVD-only (`use_itq = false`) | `111.24 ms` / `107.70 ms` | **`86.51 ms`** | **`0.88363`** | `0.88563` |
| `1024 × 1024` | `104` | **Joint-ITQ** (`20` iters) | `275.57 ms` / `226.42 ms` | **`126.62 ms`** | **`0.86192`** | `0.86314` |
| `2048 × 2048` | `232` | SVD-only (`use_itq = false`) | `523.81 ms` / `557.67 ms` | **`293.77 ms`** | **`0.87422`** | `0.87471` |
| `2048 × 2048` | `232` | **Joint-ITQ** (`20` iters) | `1,362.42 ms` / `1,093.93 ms` | **`700.80 ms`** | **`0.84829`** | `0.84890` |

---

## Detailed `littlebit-rs` Measurements (`CPU Default` vs. `CPU + Accelerate`)

### 1. Binary Packer & Unpacker Throughput

| Matrix Shape | Weights | Packed Size | Compression vs FP16 | Pack Median / p95 (Default) | Pack Median / p95 (Accel) | Unpack Median / p95 (Default) | Unpack Median / p95 (Accel) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024 × 4096` | 4,194,304 | 0.50 MiB (524,288 B) | `16.00×` | `18.23 ms` / `19.16 ms` | `18.23 ms` / `21.41 ms` | `3.46 ms` / `4.08 ms` | `3.01 ms` / `3.97 ms` |
| `4096 × 4096` | 16,777,216 | 2.00 MiB (2,097,152 B) | `16.00×` | `73.70 ms` / `76.01 ms` | `72.17 ms` / `74.66 ms` | `14.45 ms` / `17.22 ms` | `11.59 ms` / `15.44 ms` |
| `4096 × 11008` | 45,088,768 | 5.38 MiB (5,636,096 B) | `16.00×` | `199.43 ms` / `232.76 ms` | `196.81 ms` / `202.70 ms` | `41.71 ms` / `182.51 ms` | `35.61 ms` / `39.01 ms` |

### 2. Projection Decode (`batch = 1`) & Batched Prefill (`batch = 16`)

| Layer Configuration | `split_dim` | Actual `bpw` | Packed Size (vs FP16) | `b=1` Dense / Unpacked / Packed (Default) | `b=1` Dense / Unpacked / Packed (Accel) | `b=16` Dense / Unpacked / Packed (Default) | `b=16` Dense / Unpacked / Packed (Accel) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-7B Attn 4096×4096 @ 0.1` (`single`) | `184` | `0.098` | `222.2 KiB` (`147.50×`) | `2.47` / `5.11` / **`0.47 ms`** | `1.66` / `4.69` / **`0.35 ms`** | `17.75` / `7.19` / `8.18 ms` | **`4.31`** / `4.96` / `7.02 ms` |
| `Llama-7B Attn 4096×4096 @ 0.5` (`residual`) | `488` | `0.493` | `1.05 MiB` (`30.47×`) | `1.98` / `29.48` / **`1.70 ms`** | `1.09` / `25.67` / **`1.42 ms`** | `13.47` / `37.38` / `25.14 ms` | **`4.09`** / `27.09` / `25.75 ms` |
| `Llama-7B Attn 4096×4096 @ 1.0` (`residual`) | `1000` | `0.994` | `2.06 MiB` (`15.52×`) | `1.60` / `57.08` / **`3.04 ms`** | `1.36` / `54.69` / **`2.83 ms`** | `12.13` / `71.48` / `41.27 ms` | **`4.40`** / `55.27` / `45.70 ms` |
| `Llama-7B MLP 4096×11008 @ 0.5` (`residual`) | `728` | `0.499` | `2.77 MiB` (`31.00×`) | `5.47` / `72.08` / **`4.11 ms`** | `3.64` / `73.65` / **`4.74 ms`** | `22.68` / `83.64` / `63.18 ms` | **`5.90`** / `71.44` / `56.82 ms` |

### 3. End-to-End 4-Layer Causal LM Storage & 32-Token KV Generation

Synthetic 4-layer transformers (`hidden_size = 512`, `intermediate_size = 1376`, `vocab_size = 4096`, `num_attention_heads = 8`, `num_key_value_heads = 4`), quantized at `eff_bit = 0.5`, `residual = true`, `use_itq = true` (`15` iters), saved with `CastPolicy::ToBf16`, generating **32 new tokens** from a **16-token prompt**:

| Model | Avg `eff_bit` | Quantized Layers Size | Dense FP16 Layers | Compression (Layers) | Total Checkpoint (`BF16` embed/head) | Dense `F32` (Default / Accel) | Unpacked LB (Default / Accel) | Packed 1-Bit (Default / Accel) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-4L (h=512, i=1376, v=4096)` | `0.469` | `0.76 MiB` (794,816 B) | `22.12 MiB` (23,199,744 B) | **`29.19×`** | `8.77 MiB` (9,192,640 B) | `22.6` / `23.3 tok/s` (`1,417.1` / `1,375.8 ms`) | `46.6` / `48.3 tok/s` (`686.3` / `663.2 ms`) | **`58.1`** / `51.6 tok/s` (`550.3` / `620.3 ms`) |
| `Qwen3-4L (h=512, i=1376, v=4096)` | `0.469` | `0.76 MiB` (794,816 B) | `22.12 MiB` (23,199,744 B) | **`29.19×`** | `8.77 MiB` (9,193,664 B) | `22.8` / `21.8 tok/s` (`1,405.2` / `1,468.6 ms`) | `47.0` / `50.2 tok/s` (`680.8` / `637.7 ms`) | `57.7` / **`58.0 tok/s`** (`554.6` / `551.5 ms`) |

---

## Key Observations

1. **`41.2×` Smaller Footprint & `373×`–`393×` Faster Cold Startup:** The stripped native `littlebit` binary (`18.59 MiB`) starts in `9.80–10.32 ms`, compared to `766.79 MiB` and `3,854.33 ms` just to launch Python 3.15 and `import torch` (`2.15.0`) before loading `transformers` or `deepspeed`.
2. **`4.1×`–`17.6×` Faster Bit-Unpacking & `26.7×`–`30.3×` Lower Resident Weight RAM:** Rust unpacks `i32` checkpoints up to `1,447.1 MW/s` (`11.59 ms` vs. `203.77 ms` in PyTorch at `4096×4096`), and `PackedLittleBitLinear` executes 1-bit inference directly without inflating `222 KiB–2.77 MiB` packed weights back into `5.8–84.0 MiB` of float tensors.
3. **Where Upstream PyTorch Currently Leads:**
   - **Batched float projection matmuls (`b=1` & `b=16`):** Upstream Python (`0.06–3.36 ms` at `b=1`; `1.69–12.51 ms` at `b=16`) passes non-contiguous transpose views (`Xq.t()`) directly to Apple BLAS without copying, whereas Candle's unpacked `LittleBitLinear` currently calls `.t()?.contiguous()?` on each forward pass (`4.69–73.65 ms`) and `PackedBinaryMatrix::matmul_transposed_f32` processes one batch row at a time (`7.02–56.82 ms` at `b=16`).
   - **Large dense packing & BLAS-accelerated SVD:** Vectorized PyTorch `binary_packer` runs `4.5×` faster at `4096×11008` (`43.68 ms` vs. `196.81 ms`), and `torch.svd_lowrank` + `linalg.svd` initializes `1024²` / `2048²` matrices `1.2×`–`1.9×` faster while Rust `faer` matches or slightly betters final relative Frobenius error (`0.86192` vs. `0.86314`, and `0.84829` vs. `0.84890`).
4. **End-to-End 4-Layer Generation:** On CPU + Accelerate, Rust's `PackedLittleBitLinear` still reaches **`51.6–58.0 tok/s`** (up to `58.1 tok/s` on default CPU), **`2.2×`–`2.7×` faster** than the dense `F32` baseline (`21.8–23.3 tok/s`) with **`29.19×`** layer compression.
