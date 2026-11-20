//! Typed errors for `littlebit-train`.

use thiserror::Error;

/// Result alias for `littlebit-train` operations.
pub type Result<T> = std::result::Result<T, TrainError>;

/// Errors produced during QAT knowledge distillation, data preparation, or evaluation.
#[derive(Debug, Error)]
pub enum TrainError {
    /// Error from `littlebit-core`.
    #[error(transparent)]
    Core(#[from] littlebit_core::LittleBitError),

    /// Error from `littlebit-models`.
    #[error(transparent)]
    Model(#[from] littlebit_models::ModelError),

    /// Underlying Candle tensor error.
    #[error("tensor operation failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// Filesystem I/O failure.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parse or serialization failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Hugging Face Hub error.
    #[error("huggingface hub error: {0}")]
    Hub(#[from] hf_hub::HFError),

    /// Tokenizer error.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// HTTP dataset fetch error.
    #[error("http fetch error: {0}")]
    Http(String),

    /// Invalid training or evaluation argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<TrainError>();
};
