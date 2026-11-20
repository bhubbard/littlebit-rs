//! Deterministic audit, storage accounting, FP16 overflow, and mixed-bit unit tests
//! ported from `timurista/littlebit-deployment-audit` and `christopherrobertbrooks-tech/littlebit-bench`.

use candle_core::{DType, Device, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

use littlebit_core::{
    binary_packer, binary_unpacker, compute_eff_bits, estimate_split_dim, finalize_split_dim,
    inspect_checkpoint, load_and_unpack_state_dict, pack_i8_slice, parse_eff_bit_layers,
    pow2_at_least, save_packed_safetensors, unpack_i32_slice, BiasMode, BitBudget, CastPolicy,
    LayerShape, LittleBitConfig, LittleBitError, LittleBitLinear, QuantFunc, ResidualMode,
    TrainMode, WeightState, FP16_MAX_SAFE,
};

const WIDTHS: [usize; 7] = [1, 7, 8, 31, 32, 33, 64];

fn byte_tiled_row(width: usize, byte: u8) -> Vec<i8> {
    (0..width)
        .map(|j| if ((byte >> (j % 8)) & 1) != 0 { -1 } else { 1 })
        .collect()
}

fn one_hot_row(width: usize, col: usize) -> Vec<i8> {
    (0..width).map(|j| if j == col { -1 } else { 1 }).collect()
}

/// Independent little-endian byte-table decoder with zero word-level bit shifts
/// (mirrors `timurista/src/audit.py::reference_unpack_bytes`).
fn reference_unpack_bytes(words: &[i32], rows: usize, cols: usize) -> Vec<i8> {
    let words_per_row = cols.div_ceil(32);
    assert_eq!(words.len(), rows * words_per_row);

    let mut bit_table = [[0u8; 8]; 256];
    for (byte, entry) in bit_table.iter_mut().enumerate() {
        for (bit_idx, slot) in entry.iter_mut().enumerate() {
            *slot = u8::from((byte & (1 << bit_idx)) != 0);
        }
    }

    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let mut row_bits = Vec::with_capacity(words_per_row * 32);
        for &w in &words[row * words_per_row..(row + 1) * words_per_row] {
            for b in w.to_le_bytes() {
                row_bits.extend_from_slice(&bit_table[usize::from(b)]);
            }
        }
        for &bit in &row_bits[..cols] {
            out.push(if bit == 0 { 1 } else { -1 });
        }
    }
    out
}

fn signed_multirow_matrix(width: usize) -> Vec<Vec<i8>> {
    let mut rows = vec![
        vec![-1i8; width],
        vec![1i8; width],
        byte_tiled_row(width, 0x80),
        byte_tiled_row(width, 0xFF),
        byte_tiled_row(width, 0x5A),
        (0..width)
            .map(|j| if j % 32 == 31 { -1 } else { 1 })
            .collect(),
        (0..width)
            .map(|j| if j % 2 == 0 { -1 } else { 1 })
            .collect(),
    ];
    for col in [0usize, 1, 30, 31, 32] {
        if col < width {
            rows.push(one_hot_row(width, col));
        }
    }
    rows
}

fn sign_example_rows(width: usize) -> Vec<Vec<i8>> {
    vec![
        vec![1i8; width],
        vec![-1i8; width],
        (0..width)
            .map(|j| if j % 2 == 0 { -1 } else { 1 })
            .collect(),
        (0..width)
            .map(|j| if j % 2 == 0 { 1 } else { -1 })
            .collect(),
        (0..width)
            .map(|j| if j % 32 == 0 { -1 } else { 1 })
            .collect(),
        (0..width)
            .map(|j| if j % 32 == 31 { -1 } else { 1 })
            .collect(),
    ]
}

fn advertised_bits(a: usize, b: usize, s: usize, mode: ResidualMode) -> usize {
    match mode {
        ResidualMode::Single => s * (a + b + 16) + 16 * (a + b),
        ResidualMode::Dual => s * 2 * (a + b + 16) + 32 * (a + b),
    }
}

#[test]
fn audit_known_words_padding_and_first_counterexample() -> littlebit_core::Result<()> {
    let expected_words = [
        (1usize, 1usize),
        (7, 1),
        (8, 1),
        (31, 1),
        (32, 1),
        (33, 2),
        (64, 2),
        (65, 3),
    ];
    for (cols, want_words) in expected_words {
        assert_eq!(cols.div_ceil(32), want_words, "cols={cols}");
    }

    assert_eq!(pack_i8_slice(&[1], 1, 1)?.words(), &[0i32]);
    assert_eq!(pack_i8_slice(&[-1], 1, 1)?.words(), &[1i32]);
    assert_eq!(pack_i8_slice(&[1, -1], 1, 2)?.words(), &[2i32]);
    assert_eq!(pack_i8_slice(&[-1; 32], 1, 32)?.words(), &[-1i32]);

    let mut only_bit31 = vec![1i8; 32];
    only_bit31[31] = -1;
    assert_eq!(pack_i8_slice(&only_bit31, 1, 32)?.words(), &[i32::MIN]);

    assert_eq!(pack_i8_slice(&[-1; 33], 1, 33)?.words(), &[-1i32, 1i32]);
    assert_eq!(pack_i8_slice(&[1; 64], 1, 64)?.words(), &[0i32, 0i32]);

    // Width 7 padded with +1 must equal explicit +1 columns up to 32.
    let w7 = [-1i8, 1, -1, 1, 1, -1, -1];
    let mut w7_padded = w7.to_vec();
    w7_padded.resize(32, 1);
    assert_eq!(
        pack_i8_slice(&w7, 1, 7)?.words(),
        pack_i8_slice(&w7_padded, 1, 32)?.words()
    );

    // First upstream counterexample from evidence/first-upstream-counterexample.json.
    let dev = Device::Cpu;
    let cex = [1i8, -1, 1, -1, -1, 1, 1, -1];
    let packed_cex = pack_i8_slice(&cex, 1, 8)?;
    assert_eq!(packed_cex.words(), &[154i32]);
    assert_eq!(packed_cex.unpack_to_i8(), cex);

    let (packed_t, _) = packed_cex.to_tensors(&dev)?;
    let decoded_f32 = binary_unpacker(&packed_t, (1, 8), DType::F32, &dev)?.to_vec2::<f32>()?;
    let expected_f32: Vec<f32> = cex.iter().map(|&v| f32::from(v)).collect();
    assert_eq!(decoded_f32, vec![expected_f32]);

    // binarize_sign_no_grad maps 0.0 -> +1 like upstream (val >= 0 => +1).
    let zero_and_neg = Tensor::from_slice(&[0.0f32, -1.0], (1, 2), &dev)?;
    let one_and_neg = Tensor::from_slice(&[1.0f32, -1.0], (1, 2), &dev)?;
    let (pz, _) = binary_packer(&zero_and_neg)?;
    let (po, _) = binary_packer(&one_and_neg)?;
    assert_eq!(pz.to_vec2::<i32>()?, po.to_vec2::<i32>()?);
    assert!(matches!(
        pack_i8_slice(&[0, -1], 1, 2),
        Err(LittleBitError::InvalidBinaryValue { .. })
    ));

    Ok(())
}

#[test]
fn audit_exhaustive_2010_cases_with_independent_byte_decoder() -> littlebit_core::Result<()> {
    let dev = Device::Cpu;
    let mut total_cases = 0usize;

    for width in WIDTHS {
        let mut width_rows: Vec<Vec<i8>> = Vec::with_capacity(256 + width + 6);
        for byte in 0..=u8::MAX {
            width_rows.push(byte_tiled_row(width, byte));
        }
        for col in 0..width {
            width_rows.push(one_hot_row(width, col));
        }
        width_rows.extend(sign_example_rows(width));

        let n_cases = width_rows.len();
        total_cases += n_cases;

        // Per-case check with pack_i8_slice, unpack_to_i8, and independent byte-table decoder.
        for (idx, row) in width_rows.iter().enumerate() {
            let packed = pack_i8_slice(row, 1, width)?;
            assert_eq!(
                packed.unpack_to_i8(),
                *row,
                "shift unpack mismatch at width={width}, case={idx}"
            );
            assert_eq!(
                reference_unpack_bytes(packed.words(), 1, width),
                *row,
                "byte-table unpack mismatch at width={width}, case={idx}"
            );
        }

        // Batch check through Tensor-based binary_packer -> binary_unpacker.
        let flat_f32: Vec<f32> = width_rows.iter().flatten().map(|&v| f32::from(v)).collect();
        let flat_i8: Vec<i8> = width_rows.iter().flatten().copied().collect();
        let mat_t = Tensor::from_vec(flat_f32, (n_cases, width), &dev)?;
        let (packed_t, shape_t) = binary_packer(&mat_t)?;
        assert_eq!(
            shape_t.to_vec1::<i64>()?,
            vec![n_cases as i64, width as i64]
        );

        let unpacked_t = binary_unpacker(&packed_t, (n_cases, width), DType::F32, &dev)?;
        let decoded: Vec<i8> = unpacked_t
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|v| v as i8)
            .collect();
        assert_eq!(
            decoded, flat_i8,
            "batch round-trip mismatch for width={width}"
        );

        // 256 byte-tiled matrix packing matches row-wise packing word-for-word.
        let mut rowwise_words = Vec::new();
        for row in &width_rows[..256] {
            rowwise_words.extend_from_slice(pack_i8_slice(row, 1, width)?.words());
        }
        let byte_matrix_words = pack_i8_slice(
            &width_rows[..256]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            256,
            width,
        )?
        .words()
        .to_vec();
        assert_eq!(byte_matrix_words, rowwise_words, "width={width}");
    }

    assert_eq!(total_cases, 2010);
    Ok(())
}

#[test]
fn audit_signed_multirow_31_32_33_64_and_sign_bit_rows() -> littlebit_core::Result<()> {
    let dev = Device::Cpu;
    for width in [31usize, 32, 33, 64] {
        let rows = signed_multirow_matrix(width);
        let n_rows = rows.len();
        let flat: Vec<i8> = rows.iter().flatten().copied().collect();
        let packed = pack_i8_slice(&flat, n_rows, width)?;
        let neg_count = packed.words().iter().filter(|&&w| w < 0).count();
        if width >= 32 {
            assert!(neg_count > 0, "width={width} expected negative i32 words");
        } else {
            assert_eq!(neg_count, 0, "width={width} must have no negative words");
        }

        assert_eq!(packed.unpack_to_i8(), flat);
        assert_eq!(reference_unpack_bytes(packed.words(), n_rows, width), flat);

        let flat_f32: Vec<f32> = flat.iter().map(|&v| f32::from(v)).collect();
        let tensor_in = Tensor::from_vec(flat_f32, (n_rows, width), &dev)?;
        let (packed_t, _) = binary_packer(&tensor_in)?;
        let unpacked = binary_unpacker(&packed_t, (n_rows, width), DType::F32, &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let unpacked_i8: Vec<i8> = unpacked.into_iter().map(|v| v as i8).collect();
        assert_eq!(unpacked_i8, flat);
    }

    // Column 31 set in every word: most negative i32 (i32::MIN).
    for width in [32usize, 33, 64] {
        let row: Vec<i8> = (0..width)
            .map(|j| if j % 32 == 31 { -1 } else { 1 })
            .collect();
        let packed = pack_i8_slice(&row, 1, width)?;
        assert_eq!(packed.words()[0], i32::MIN, "width={width}");
        assert_eq!(unpack_i32_slice(packed.words(), 1, width)?, row);
        assert_eq!(reference_unpack_bytes(packed.words(), 1, width), row);
    }

    Ok(())
}

#[test]
fn audit_patched_layer_and_residual_roundtrip_bitwise_equal() -> littlebit_core::Result<()> {
    let dev = Device::Cpu;
    let configs = [
        (96usize, 160usize, 1.0f64, false, 0u64),
        (96, 160, 1.0, true, 0),
        (256, 256, 1.0, true, 1),
        (512, 1024, 0.1, true, 2),
        (1024, 512, 1.0, true, 3),
    ];

    for (in_f, out_f, eff_bit, residual, seed) in configs {
        let mut rng = ChaCha8Rng::seed_from_u64(1000 + seed);
        let w_data: Vec<f32> = (0..out_f * in_f)
            .map(|_| rng.random_range(-0.5f32..0.5f32))
            .collect();
        let w = Tensor::from_vec(w_data, (out_f, in_f), &dev)?;

        let cfg = LittleBitConfig {
            quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
            eff_bit: Some(eff_bit),
            residual,
            use_itq: false,
            min_split_dim: 8,
            ..Default::default()
        };

        let layer =
            LittleBitLinear::from_dense_weight(&w, None, &cfg, 1.0, TrainMode::Train, seed)?;
        let x_data: Vec<f32> = (0..4 * in_f)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        let x = Tensor::from_vec(x_data, (4, in_f), &dev)?;
        let y_mem = layer.forward(&x)?;

        let state = layer.pack_state_dict("m")?;
        let mut packed_keys = vec!["m.U_packed", "m.V_packed"];
        if residual {
            assert!(state.contains_key("m.U_R_packed"));
            assert!(state.contains_key("m.V_R_packed"));
            packed_keys.extend(["m.U_R_packed", "m.V_R_packed"]);
        }
        let mut neg_words = 0usize;
        for key in packed_keys {
            neg_words += state[key]
                .flatten_all()?
                .to_vec1::<i32>()?
                .iter()
                .filter(|&&v| v < 0)
                .count();
        }
        assert!(
            neg_words > 0,
            "expected negative i32 words across packed factors for ({in_f}, {out_f}, residual={residual})"
        );

        let dir = TempDir::new().expect("tempdir");
        save_packed_safetensors(dir.path(), &state, CastPolicy::Preserve)?;
        let loaded = load_and_unpack_state_dict(dir.path(), DType::F32, &dev)?;

        let shape = LayerShape::new(in_f, out_f, BiasMode::NoBias);
        let reloaded = LittleBitLinear::from_state_dict(
            "m",
            &loaded.tensors,
            shape,
            &cfg,
            1.0,
            WeightState::PreBinarized,
            DType::F32,
            &dev,
        )?;

        // Check 100% sign agreement across U/V and U_R/V_R.
        let orig_u = cfg
            .quant_func
            .apply(layer.main_branch().u.as_tensor())?
            .to_vec2::<f32>()?;
        let got_u = reloaded.main_branch().u.as_tensor().to_vec2::<f32>()?;
        assert_eq!(orig_u, got_u, "U sign mismatch on ({in_f}, {out_f})");

        let orig_v = cfg
            .quant_func
            .apply(layer.main_branch().v.as_tensor())?
            .to_vec2::<f32>()?;
        let got_v = reloaded.main_branch().v.as_tensor().to_vec2::<f32>()?;
        assert_eq!(orig_v, got_v, "V sign mismatch on ({in_f}, {out_f})");

        if residual {
            let orig_res = layer.residual_branch().expect("orig residual");
            let got_res = reloaded.residual_branch().expect("reloaded residual");
            assert_eq!(
                cfg.quant_func
                    .apply(orig_res.u.as_tensor())?
                    .to_vec2::<f32>()?,
                got_res.u.as_tensor().to_vec2::<f32>()?
            );
            assert_eq!(
                cfg.quant_func
                    .apply(orig_res.v.as_tensor())?
                    .to_vec2::<f32>()?,
                got_res.v.as_tensor().to_vec2::<f32>()?
            );
        }

        // Exact F32 equality between in-memory training layer and packed-then-reloaded layer.
        let y_reloaded = reloaded.forward(&x)?;
        let y_mem_vec = y_mem.flatten_all()?.to_vec1::<f32>()?;
        let y_rel_vec = y_reloaded.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            y_mem_vec, y_rel_vec,
            "expected bitwise equal F32 outputs for ({in_f}, {out_f}, residual={residual})"
        );

        let packed_layer = layer.to_packed()?;
        let y_packed = packed_layer.forward(&x)?;
        let diff_packed = y_mem.sub(&y_packed)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff_packed < 1e-4,
            "packed forward drift {diff_packed} for ({in_f}, {out_f})"
        );
    }

    Ok(())
}

#[test]
fn audit_rank_and_storage_accounting_formulas() -> littlebit_core::Result<()> {
    // min_split_floor counterexample: (64, 64, 0.1, Single) estimates < 0, floors to 8, actual bpw = 0.78125.
    let est_neg = estimate_split_dim(64, 64, Some(0.1), ResidualMode::Single).expect("estimate");
    assert!(est_neg < 0.0);
    let s_floor = finalize_split_dim(Some(est_neg), 1024, 8);
    assert_eq!(s_floor, 8);
    assert!((compute_eff_bits(64, 64, s_floor, ResidualMode::Single) - 0.78125).abs() < 1e-12);

    // Llama-2 7B reference shapes.
    let s_llama_01 = finalize_split_dim(
        estimate_split_dim(4096, 4096, Some(0.1), ResidualMode::Single),
        1024,
        8,
    );
    assert_eq!(s_llama_01, 184);

    let s_llama_10_res = finalize_split_dim(
        estimate_split_dim(4096, 4096, Some(1.0), ResidualMode::Dual),
        1024,
        8,
    );
    assert_eq!(s_llama_10_res, 1000);

    // Sweep illustrative layer shapes x targets x residual modes.
    let illustrative_layers = [
        (64usize, 64usize),
        (100, 36),
        (4096, 4096),
        (4096, 11008),
        (11008, 4096),
        (5120, 7680),
        (5120, 5120),
        (5120, 1280),
    ];
    for (a, b) in illustrative_layers {
        for eff in [1.0, 0.55, 0.3, 0.1] {
            for mode in [ResidualMode::Single, ResidualMode::Dual] {
                let est = estimate_split_dim(a, b, Some(eff), mode);
                let s = finalize_split_dim(est, 1024, 8);
                assert_eq!(s % 8, 0);
                assert!(s >= 8);
                let adv = advertised_bits(a, b, s, mode) as f64 / ((a * b) as f64);
                let got = compute_eff_bits(a, b, s, mode);
                assert!((adv - got).abs() < 1e-12);
            }
        }
    }

    // Row-padding bit formulas matching audit.py::test_row_padding.
    let pad_bits = |a: usize, b: usize, s: usize| -> (usize, usize) {
        let u_pad = b * (s.div_ceil(32) * 32 - s);
        let v_pad = s * (a.div_ceil(32) * 32 - a);
        (u_pad, v_pad)
    };
    assert_eq!(pad_bits(4096, 4096, 184), (4096 * 8, 0));
    assert_eq!(pad_bits(4096, 4096, 192), (0, 0));
    assert_eq!(pad_bits(100, 36, 8), (36 * 24, 8 * 28));

    // Verify real LittleBitLinear packed storage for (100, 36) at split_dim=8, residual=true.
    let dev = Device::Cpu;
    let w = Tensor::randn(0.0f32, 0.25, (36, 100), &dev)?;
    let cfg = LittleBitConfig {
        eff_bit: None,
        split_dim: 8,
        residual: true,
        use_itq: false,
        ..Default::default()
    };
    let layer = LittleBitLinear::from_dense_weight(&w, None, &cfg, 1.0, TrainMode::Eval, 11)?;
    let state = layer.pack_state_dict("x")?;
    assert_eq!(state["x.U_packed"].dims(), &[36, 1]);
    assert_eq!(state["x.V_packed"].dims(), &[8, 4]);
    let scale_elems: usize = ["u1", "u2", "v1", "v2", "u1_R", "u2_R", "v1_R", "v2_R"]
        .iter()
        .map(|s| state[&format!("x.{s}")].elem_count())
        .sum();
    assert_eq!(scale_elems, 2 * (100 + 36 + 2 * 8));

    let dir = TempDir::new().expect("tempdir");
    save_packed_safetensors(dir.path(), &state, CastPolicy::ToBf16)?;
    let stats = inspect_checkpoint(dir.path())?;
    assert_eq!(stats.quantized_layer_count, 1);
    assert_eq!(stats.layers[0].split_dim, 8);
    let _ = BitBudget::resolve(100, 36, &cfg, 1.0);
    Ok(())
}

#[test]
fn bench_fp16_overflow_dual_residual_both_stages_and_warm_start() -> littlebit_core::Result<()> {
    let dev = Device::Cpu;
    let in_features = 2048usize;
    let out_features = 32usize;
    let split_dim = 64usize;

    let cfg = LittleBitConfig {
        quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
        eff_bit: None,
        split_dim,
        residual: true,
        use_itq: false,
        ..Default::default()
    };

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut state = std::collections::HashMap::new();
    for suffix in ["", "_R"] {
        let u_vals: Vec<f32> = (0..out_features * split_dim)
            .map(|idx| {
                let col = idx % split_dim;
                if col < split_dim / 2 || rng.random_bool(0.5) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        let v_vals: Vec<f32> = (0..split_dim * in_features)
            .map(|idx| {
                let row = idx / in_features;
                if row < split_dim / 2 || rng.random_bool(0.5) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        state.insert(
            format!("layer.U{suffix}"),
            Tensor::from_vec(u_vals, (out_features, split_dim), &dev)?,
        );
        state.insert(
            format!("layer.V{suffix}"),
            Tensor::from_vec(v_vals, (split_dim, in_features), &dev)?,
        );
        state.insert(
            format!("layer.u1{suffix}"),
            Tensor::full(0.001f32, (1, out_features), &dev)?,
        );
        state.insert(
            format!("layer.u2{suffix}"),
            Tensor::full(0.02f32, (1, split_dim), &dev)?,
        );
        state.insert(
            format!("layer.v1{suffix}"),
            Tensor::full(1.0f32, (1, split_dim), &dev)?,
        );
        state.insert(
            format!("layer.v2{suffix}"),
            Tensor::full(2.0f32, (1, in_features), &dev)?,
        );
    }

    let shape = LayerShape::new(in_features, out_features, BiasMode::NoBias);
    let layer_f32 = LittleBitLinear::from_state_dict(
        "layer",
        &state,
        shape,
        &cfg,
        1.0,
        WeightState::PreBinarized,
        DType::F32,
        &dev,
    )?;
    let layer_f16 = LittleBitLinear::from_state_dict(
        "layer",
        &state,
        shape,
        &cfg,
        1.0,
        WeightState::PreBinarized,
        DType::F16,
        &dev,
    )?;

    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| 30.0 + ((i % 5) as f32) * 0.25)
        .collect();
    let x_f32 = Tensor::from_vec(x_vals, (2, in_features), &dev)?;
    let x_f16 = x_f32.to_dtype(DType::F16)?;

    // Verify both stage-1 and stage-2 scaling factors s1, s2 > 1.0 are triggered under FP16 limits.
    let s1 = pow2_at_least(60.0 * (in_features as f64) / FP16_MAX_SAFE);
    let s2 =
        pow2_at_least((60.0 * (in_features as f64)) * 0.02 * (split_dim as f64) / FP16_MAX_SAFE);
    assert!(s1 >= 2.0, "expected s1 >= 2, got {s1}");
    assert!(s2 >= 2.0, "expected s2 >= 2, got {s2}");

    let out_f32 = layer_f32.forward(&x_f32)?.to_vec2::<f32>()?;
    let out_f16 = layer_f16
        .forward(&x_f16)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    for (r, (row32, row16)) in out_f32.iter().zip(out_f16.iter()).enumerate() {
        for (c, (&v32, &v16)) in row32.iter().zip(row16.iter()).enumerate() {
            assert!(v16.is_finite(), "FP16 NaN/Inf at ({r}, {c}): {v16}");
            let rel_err = (v32 - v16).abs() / v32.abs().max(1.0);
            assert!(rel_err < 1e-2, "({r}, {c}) f32={v32}, f16={v16}");
        }
    }

    // Save packed checkpoint, reload PreBinarized, then warm-start with warm_mag=0.03.
    let dir = TempDir::new().expect("tempdir");
    let packed_sd = layer_f32.pack_state_dict("layer")?;
    save_packed_safetensors(dir.path(), &packed_sd, CastPolicy::Preserve)?;
    let reloaded_sd = load_and_unpack_state_dict(dir.path(), DType::F32, &dev)?;
    let mut pre_bin = LittleBitLinear::from_state_dict(
        "layer",
        &reloaded_sd.tensors,
        shape,
        &cfg,
        1.0,
        WeightState::PreBinarized,
        DType::F32,
        &dev,
    )?;

    let y_pre = pre_bin.forward(&x_f32)?;
    assert!(matches!(
        pre_bin.warm_start_from_binarized(0.0),
        Err(LittleBitError::InvalidConfig(_))
    ));
    pre_bin.warm_start_from_binarized(0.03)?;
    assert!(!pre_bin.is_binarized());

    let y_warm = pre_bin.forward(&x_f32)?;
    let warm_diff = y_pre.sub(&y_warm)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(warm_diff < 1e-5, "step-0 warm start drift: {warm_diff}");

    let loss = y_warm.sqr()?.mean_all()?;
    let grads = loss.backward()?;
    let vars = pre_bin.trainable_vars();
    assert_eq!(vars.len(), 12);
    for (idx, var) in vars.iter().enumerate() {
        let g = grads.get(var.as_tensor()).expect("warm-start grad present");
        let norm = g.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            norm.is_finite() && norm > 0.0,
            "var[{idx}] grad must be finite and non-zero (got {norm})"
        );
    }

    Ok(())
}

#[test]
fn bench_parse_eff_bit_layers_spec_cases() -> littlebit_core::Result<()> {
    assert_eq!(parse_eff_bit_layers(None)?, Vec::new());
    assert_eq!(parse_eff_bit_layers(Some(""))?, Vec::new());
    assert_eq!(parse_eff_bit_layers(Some(" ; "))?, Vec::new());
    assert_eq!(
        parse_eff_bit_layers(Some("0-7,27:2.0"))?,
        vec![(vec![0, 1, 2, 3, 4, 5, 6, 7, 27], 2.0)]
    );
    assert_eq!(
        parse_eff_bit_layers(Some("1:1.5; 3-4:0.5"))?,
        vec![(vec![1], 1.5), (vec![3, 4], 0.5)]
    );
    assert_eq!(
        parse_eff_bit_layers(Some("3,1-2,2:1.25"))?,
        vec![(vec![1, 2, 3], 1.25)]
    );

    for bad in ["0-7", "5-2:1.0", "0:0.0", "0:-1.0", "abc:1.0", ":1.0"] {
        assert!(
            matches!(
                parse_eff_bit_layers(Some(bad)),
                Err(LittleBitError::InvalidConfig(_))
            ),
            "expected InvalidConfig for `{bad}`"
        );
    }
    Ok(())
}
