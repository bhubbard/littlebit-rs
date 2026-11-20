//! Core quantization primitives for LittleBit (NeurIPS 2025) and LittleBit-2 (ICML 2026).

pub mod binary;
pub mod config;
pub mod error;
pub mod itq;
pub mod linear;
pub mod packer;
pub mod safetensors_io;
pub mod svd;

pub use binary::{binarize_sign_no_grad, smooth_sign, ste_binary};
pub use config::{
    compute_eff_bits, estimate_split_dim, finalize_split_dim, parse_eff_bit_layers, BiasMode,
    BitBudget, CastPolicy, InferenceMode, InitMode, ItqMode, LayerShape, LittleBitConfig,
    LittleBitConfigOverrides, QuantFunc, ResidualMode, TrainMode, WeightState,
    SMOOTH_SIGN_DEFAULT_ALPHA,
};
pub use error::{LittleBitError, Result};
pub use itq::{apply_joint_itq, compute_itq_rotation, random_orthogonal_matrix};
pub use linear::{
    pow2_at_least, reconstruct_branch_matrix, FactorBranch, LittleBitLinear, PackedLittleBitLinear,
    Param, FP16_MAX_SAFE,
};
pub use packer::{
    binary_packer, binary_unpacker, pack_i8_slice, unpack_i32_slice, PackedBinaryMatrix,
};
pub use safetensors_io::{
    inspect_checkpoint, load_and_unpack_state_dict, load_raw_safetensors_dir,
    save_packed_safetensors, CheckpointFormat, CheckpointStats, LayerInspection, LoadedStateDict,
};
pub use svd::{exact_thin_svd, rank_one_decompose, svd_lowrank, ThinSvd, DEFAULT_LOWRANK_NITER};
