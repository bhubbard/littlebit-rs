//! Upstream parity and GitHub issue regression tests ported from `SamsungLabs/LittleBit`.

use candle_core::{DType, Device, Tensor, Var};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use littlebit_core::{
    binary_packer, binary_unpacker, compute_eff_bits, estimate_split_dim, finalize_split_dim,
    pack_i8_slice, pow2_at_least, smooth_sign, ste_binary, BiasMode, BitBudget, LayerShape,
    LittleBitConfig, LittleBitError, LittleBitLinear, QuantFunc, ResidualMode, TrainMode,
    WeightState, FP16_MAX_SAFE,
};

#[test]
fn issue_21_binary_unpacker_round_trip_matrix_sweep() -> littlebit_core::Result<()> {
    let dev = Device::Cpu;
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    // Exact repro shapes from SamsungLabs/LittleBit Issue #21.
    for (rows, cols) in [
        (8usize, 31usize),
        (8, 32),
        (8, 33),
        (64, 1024),
        (1024, 4096),
    ] {
        let vals: Vec<i8> = (0..rows * cols)
            .map(|_| if rng.random_bool(0.5) { 1 } else { -1 })
            .collect();
        let packed = pack_i8_slice(&vals, rows, cols)?;
        let (packed_t, shape_t) = packed.to_tensors(&dev)?;
        let dims = shape_t.to_vec1::<i64>()?;
        assert_eq!(dims, vec![rows as i64, cols as i64]);

        let unpacked_t = binary_unpacker(&packed_t, (rows, cols), DType::F32, &dev)?;
        let unpacked_vals = unpacked_t.flatten_all()?.to_vec1::<f32>()?;
        let matches = unpacked_vals
            .iter()
            .zip(vals.iter())
            .filter(|(&got, &want)| (got - f32::from(want)).abs() == 0.0)
            .count();
        assert_eq!(
            matches,
            vals.len(),
            "shape ({rows}, {cols}) had flipped signs after pack->unpack"
        );
    }

    // All -1 row across 64 columns: every index 0..64 must decode to -1 (not just 0, 32).
    let neg_row = vec![-1i8; 64];
    let (packed_neg, _) = pack_i8_slice(&neg_row, 1, 64)?.to_tensors(&dev)?;
    let decoded_neg = binary_unpacker(&packed_neg, (1, 64), DType::F32, &dev)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(decoded_neg.iter().all(|&v| v == -1.0));

    // Negative signed i32 word (bit 31 set) round-trips identically.
    let mut bit31_only = vec![1i8; 32];
    bit31_only[31] = -1;
    let packed_b31 = pack_i8_slice(&bit31_only, 1, 32)?;
    assert_eq!(packed_b31.words(), &[i32::MIN]);
    assert_eq!(packed_b31.unpack_to_i8(), bit31_only);

    // Error handling checks matching upstream TypeError / ValueError cases.
    let wrong_dtype = Tensor::zeros((2, 2), DType::F32, &dev)?;
    assert!(matches!(
        binary_unpacker(&wrong_dtype, (2, 64), DType::F32, &dev),
        Err(LittleBitError::UnsupportedDtype(_))
    ));
    let wrong_rank = Tensor::zeros(4, DType::I32, &dev)?;
    assert!(matches!(
        binary_unpacker(&wrong_rank, (1, 128), DType::F32, &dev),
        Err(LittleBitError::InvalidShape { .. })
    ));
    let wrong_shape = Tensor::zeros((2, 3), DType::I32, &dev)?;
    assert!(matches!(
        binary_unpacker(&wrong_shape, (2, 32), DType::F32, &dev),
        Err(LittleBitError::InvalidShape { .. })
    ));

    Ok(())
}

#[test]
fn issue_24_fp16_forward_avoids_overflow_and_matches_fp32() -> littlebit_core::Result<()> {
    // Reproduces SamsungLabs/LittleBit Issue #24:
    // Intermediate binary matmul sum over in_features=2048 of ~50.0 terms reaches ~102,400 > 65,504
    // (exceeding FP16 max), while v1*u2 shrinks intermediate magnitudes back into FP16 range.
    assert_eq!(pow2_at_least(0.5), 1.0);
    assert_eq!(pow2_at_least(1.0), 1.0);
    assert_eq!(pow2_at_least(1.01), 2.0);
    assert_eq!(pow2_at_least(53.0 * 2048.0 / FP16_MAX_SAFE), 2.0);

    let dev = Device::Cpu;
    let in_features = 2048usize;
    let out_features = 32usize;
    let split_dim = 64usize;

    let cfg = LittleBitConfig {
        quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
        eff_bit: None,
        split_dim,
        residual: false,
        use_itq: false,
        ..Default::default()
    };

    let mut rng = ChaCha8Rng::seed_from_u64(24);
    let mut state = std::collections::HashMap::new();
    let u_vals: Vec<f32> = (0..out_features * split_dim)
        .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
        .collect();
    // Make first half of V rows predominantly +1 so intermediate sum literally exceeds 65,504.
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
        "layer.U".to_string(),
        Tensor::from_vec(u_vals, (out_features, split_dim), &dev)?,
    );
    state.insert(
        "layer.V".to_string(),
        Tensor::from_vec(v_vals, (split_dim, in_features), &dev)?,
    );
    state.insert(
        "layer.u1".to_string(),
        Tensor::full(1.0f32, (1, out_features), &dev)?,
    );
    state.insert(
        "layer.u2".to_string(),
        Tensor::full(0.05f32, (1, split_dim), &dev)?,
    );
    state.insert(
        "layer.v1".to_string(),
        Tensor::full(0.01f32, (1, split_dim), &dev)?,
    );
    state.insert(
        "layer.v2".to_string(),
        Tensor::full(2.0f32, (1, in_features), &dev)?,
    );

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

    // Construct positive-biased activations so x * v2 sum is > 100,000 before shrinkage.
    let x_vals: Vec<f32> = (0..2 * in_features)
        .map(|i| 25.0 + ((i % 7) as f32) * 0.25)
        .collect();
    let x_f32 = Tensor::from_vec(x_vals, (2, in_features), &dev)?;
    let x_f16 = x_f32.to_dtype(DType::F16)?;

    let out_f32 = layer_f32.forward(&x_f32)?.to_vec2::<f32>()?;
    let out_f16 = layer_f16
        .forward(&x_f16)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;

    for (r, (row32, row16)) in out_f32.iter().zip(out_f16.iter()).enumerate() {
        for (c, (&v32, &v16)) in row32.iter().zip(row16.iter()).enumerate() {
            assert!(v16.is_finite(), "FP16 overflow/NaN at ({r}, {c}): {v16}");
            let rel_err = (v32 - v16).abs() / v32.abs().max(1.0);
            assert!(
                rel_err < 1e-2,
                "FP16 drift at ({r}, {c}): f32={v32}, f16={v16}, rel={rel_err}"
            );
        }
    }

    // Also verify F16 trainable backward stays finite through the power-of-two rescaled path.
    let x_var = Var::from_tensor(&x_f16)?;
    let w_tiny = Tensor::randn(0.0f32, 0.05, (16, in_features), &dev)?.to_dtype(DType::F16)?;
    let q_f16 = LittleBitLinear::from_dense_weight(&w_tiny, None, &cfg, 1.0, TrainMode::Train, 5)?;
    let y = q_f16.forward(x_var.as_tensor())?;
    let loss = y.to_dtype(DType::F32)?.sqr()?.mean_all()?;
    let grads = loss.backward()?;
    for var in q_f16.trainable_vars() {
        let g = grads
            .get(var.as_tensor())
            .expect("gradient present in F16 QAT")
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(g.iter().all(|v| v.is_finite()));
    }

    Ok(())
}

#[test]
fn binary_quant_funcs_dtypes_strided_and_boundaries() -> littlebit_core::Result<()> {
    let dev = Device::Cpu;
    let base = Tensor::from_slice(&[-2.0f32, -0.5, -0.0, 0.0, 0.5, 2.0], (2, 3), &dev)?;
    // Non-contiguous transposed input exercises strided CustomOp1 path.
    let strided = base.t()?;
    assert!(!strided.is_contiguous());

    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let x = strided.to_dtype(dtype)?;
        let ys = ste_binary(&x)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let ym = smooth_sign(&x, 100.0)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert_eq!(ys, vec![vec![-1.0, 1.0], vec![-1.0, 1.0], vec![1.0, 1.0]]);
        assert_eq!(ym, ys);
    }

    // SmoothSign at large |x| = 10.0 saturates tanh(100 * 10) -> derivative ~0 with no NaN.
    let x_large = Var::from_slice(&[-10.0f32, 0.0, 10.0], 3, &dev)?;
    let loss = smooth_sign(x_large.as_tensor(), 100.0)?.sum_all()?;
    let grads = loss.backward()?;
    let g = grads
        .get(x_large.as_tensor())
        .expect("smooth_sign grad")
        .to_vec1::<f32>()?;
    assert!(g.iter().all(|v| v.is_finite()));
    assert_eq!(g[0], 0.0);
    assert!((g[1] - 100.0).abs() < 1e-4);
    assert_eq!(g[2], 0.0);

    Ok(())
}

#[test]
fn littlebit_budget_extra_repr_state_dict_and_binarize_parity() -> littlebit_core::Result<()> {
    let dev = Device::Cpu;

    // Zero dimensions handled gracefully like upstream.
    assert_eq!(
        estimate_split_dim(0, 128, Some(1.0), ResidualMode::Single),
        None
    );
    assert_eq!(
        compute_eff_bits(0, 128, 8, ResidualMode::Single),
        f64::INFINITY
    );
    assert_eq!(finalize_split_dim(Some(15.9), 1024, 8), 8);

    // LittleBit v1 (use_itq = false) vs LittleBit-2 (use_itq = true) with residual = true.
    let w = Tensor::randn(0.0f32, 0.5, (48, 64), &dev)?;
    let bias = Tensor::ones(48, DType::F32, &dev)?;
    for use_itq in [false, true] {
        let cfg = LittleBitConfig {
            quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
            eff_bit: Some(0.5),
            residual: true,
            use_itq,
            itq_n_iter: 12,
            ..Default::default()
        };
        let mut layer =
            LittleBitLinear::from_dense_weight(&w, Some(&bias), &cfg, 1.0, TrainMode::Eval, 7)?;

        let repr = layer.extra_repr();
        for token in [
            "in_features=64",
            "out_features=48",
            "bias=true",
            "split_dim=",
            "eff_bit_target=0.5000",
            "eff_bit_actual=",
            "residual=true",
            "total_bit_usage=",
        ] {
            assert!(repr.contains(token), "extra_repr missing `{token}`: {repr}");
        }

        let sd = layer.pack_state_dict("model.layers.0.mlp.down_proj")?;
        for expected_key in [
            "model.layers.0.mlp.down_proj.U_packed",
            "model.layers.0.mlp.down_proj.U_shape",
            "model.layers.0.mlp.down_proj.V_packed",
            "model.layers.0.mlp.down_proj.V_shape",
            "model.layers.0.mlp.down_proj.U_R_packed",
            "model.layers.0.mlp.down_proj.V_R_packed",
            "model.layers.0.mlp.down_proj._eff_bit_target",
            "model.layers.0.mlp.down_proj._split_dim_final",
            "model.layers.0.mlp.down_proj._eff_bit_actual",
            "model.layers.0.mlp.down_proj.bias",
        ] {
            assert!(sd.contains_key(expected_key), "missing {expected_key}");
        }
        assert!(!sd.contains_key("model.layers.0.mlp.down_proj.U"));
        assert!(!sd.contains_key("model.layers.0.mlp.down_proj.V"));

        let x = Tensor::randn(0.0f32, 1.0, (2, 5, 64), &dev)?;
        let y_before = layer.forward(&x)?;
        assert!(!layer.is_binarized());
        layer.binarize_in_place()?;
        assert!(layer.is_binarized());
        let y_after = layer.forward(&x)?;
        let diff = y_before
            .sub(&y_after)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "binarize_in_place changed output by {diff}");
    }

    let _ = (BitBudget::resolve, binary_packer);
    Ok(())
}
