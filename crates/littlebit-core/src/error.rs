//! Typed errors for `littlebit-core`.

use thiserror::Error;

/// Result alias for `littlebit-core` operations.
pub type Result<T> = std::result::Result<T, LittleBitError>;

/// Errors produced by LittleBit quantization, SVD/ITQ decomposition, packing, or I/O.
#[derive(Debug, Error)]
pub enum LittleBitError {
    /// Underlying Candle tensor operation failed.
    #[error("tensor operation failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// Safetensors serialization or deserialization failed.
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    /// Filesystem I/O failure.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON configuration or index parsing failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid quantization configuration or hyperparameter.
    #[error("invalid quantization config: {0}")]
    InvalidConfig(String),

    /// Tensor shape did not match expected dimensions.
    #[error("invalid tensor shape: expected {expected}, got {actual}")]
    InvalidShape {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        actual: String,
    },

    /// Binary tensor contained a value other than `-1` or `+1`.
    #[error("invalid binary value {value} at ({row}, {col}); expected -1 or +1")]
    InvalidBinaryValue {
        /// Row index.
        row: usize,
        /// Column index.
        col: usize,
        /// Observed value.
        value: i8,
    },

    /// Linear algebra SVD computation failed.
    #[error("svd decomposition failed: {0}")]
    SvdFailed(String),

    /// Required checkpoint tensor was missing from the state dictionary.
    #[error("missing tensor `{0}` in state dict")]
    MissingTensor(String),

    /// Tensor dtype is not supported by the operation.
    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDtype(String),

    /// Model directory contains no recognized checkpoint weights.
    #[error("no model weights found in `{0}`")]
    CheckpointNotFound(String),
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<LittleBitError>();
};
