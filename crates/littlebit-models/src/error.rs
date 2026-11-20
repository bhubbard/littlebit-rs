//! Typed errors for `littlebit-models`.

use thiserror::Error;

/// Result alias for `littlebit-models` operations.
pub type Result<T> = std::result::Result<T, ModelError>;

/// Errors produced during model configuration parsing, architecture construction,
/// LittleBit patching, generation, or Hugging Face Hub I/O.
#[derive(Debug, Error)]
pub enum ModelError {
    /// Error propagated from `littlebit-core`.
    #[error(transparent)]
    Core(#[from] littlebit_core::LittleBitError),

    /// Underlying Candle tensor operation failed.
    #[error("tensor operation failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// Filesystem I/O failure.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON config parsing or serialization failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Hugging Face Hub client error.
    #[error("huggingface hub error: {0}")]
    Hub(#[from] hf_hub::HFError),

    /// Unsupported model architecture or invalid `config.json`.
    #[error("unsupported or invalid model config: {0}")]
    UnsupportedArchitecture(String),

    /// Missing required tensor weight in checkpoint.
    #[error("missing model tensor `{0}`")]
    MissingWeight(String),

    /// Invalid generation or patching argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<ModelError>();
};
