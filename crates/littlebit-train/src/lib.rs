//! QAT Knowledge Distillation trainer, dataset tokenizer/cache, and perplexity/zero-shot evaluator.

pub mod data;
pub mod error;
pub mod eval;
pub mod loss;
pub mod scheduler;
pub mod trainer;
pub mod zero_config;

pub use data::{
    group_token_blocks, load_eval_tokens, load_local_texts, load_tokenizer, prepare_dataset,
    tokenizer_cache_key, DatasetKind, TokenBlockDataset, DEFAULT_BLOCK_SIZE,
};
pub use error::{Result, TrainError};
pub use eval::{
    evaluate_multiple_choice, evaluate_perplexity, load_multiple_choice_jsonl,
    MultipleChoiceExample, PplResult, ZeroShotTaskResult,
};
pub use loss::{compute_kd_loss, kl_divergence_batchmean, l2l_mse_loss, KdLossBreakdown};
pub use scheduler::CosineWarmupScheduler;
pub use trainer::{KdTrainer, StepMetrics, TrainingConfig, TrainingSummary};
pub use zero_config::ZeroConfig;
