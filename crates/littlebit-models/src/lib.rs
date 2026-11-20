//! LLM architectures, dynamic LittleBit patching, KV cache, and Hugging Face Hub integration.

pub mod config;
pub mod error;
pub mod gemma;
pub mod hub;
pub mod layer;
pub mod llama;
pub mod model;
pub mod opt;
pub mod patch;
pub mod phi;
pub mod phi_attn;

pub use config::{ModelArchitecture, ModelConfig, TokenIdSpec};
pub use error::{ModelError, Result};
pub use gemma::GemmaModel;
pub use hub::{load_dense_model, resolve_model_dir, LittleBitModel, LoadOptions, PushOptions};
pub use layer::{
    apply_activation, causal_mask, repeat_kv, Embedding, KvCache, LayerNorm, LinearLayer,
    LinearLoadSpec, NormKind, RmsNorm, RotaryEmbedding,
};
pub use llama::LlamaModel;
pub use model::{CausalLm, CausalLmOutput, GenerateConfig, HiddenStatesMode, ParameterStats};
pub use opt::OptModel;
pub use patch::{derive_layer_seed, patch_linear_layer, PatchPolicy, PatchSummary};
pub use phi::PhiModel;
pub use phi_attn::PhiAttentionProjection;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use littlebit_core::{CastPolicy, InferenceMode, LittleBitConfig, QuantFunc, TrainMode};
    use tempfile::TempDir;

    #[test]
    fn all_six_families_forward_patch_save_and_reload_packed() -> Result<()> {
        let dev = Device::Cpu;
        let configs = [
            ModelConfig::tiny_llama(128, 64, 2),
            ModelConfig::tiny_qwen2(128, 64, 2),
            ModelConfig::tiny_qwen3(128, 64, 2),
            ModelConfig::tiny_phi3(128, 64, 2),
            ModelConfig::tiny_gemma2(128, 64, 2),
            ModelConfig::tiny_gemma3(128, 64, 2),
            ModelConfig::tiny_opt(128, 64, 2),
        ];

        let lb_cfg = LittleBitConfig {
            quant_func: QuantFunc::SmoothSign { alpha: 100.0 },
            eff_bit: Some(0.75),
            residual: true,
            kv_factor: 1.5,
            min_split_dim: 8,
            use_itq: true,
            itq_n_iter: 10,
            ..Default::default()
        };

        let input = Tensor::from_slice(&[1u32, 5, 19, 42, 7, 11], (2, 3), &dev)?;

        for cfg in configs {
            let arch = cfg.detect_architecture()?;
            let mut model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 77)?;

            let dense_out = model.forward(&input, None, HiddenStatesMode::All)?;
            assert_eq!(dense_out.logits.dims(), &[2, 3, cfg.vocab_size]);
            assert_eq!(dense_out.hidden_states.len(), cfg.num_hidden_layers + 1);

            let summary = model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 101)?;
            assert!(summary.patched_layers > 0, "arch {arch:?} patched 0 layers");

            if let CausalLm::Phi(ref phi) = model {
                assert!(
                    matches!(
                        phi.layer_qkv_projection(0),
                        Some(PhiAttentionProjection::Split { .. })
                    ),
                    "Phi must split qkv_proj during patching"
                );
            }

            let out_unpacked = model.forward(&input, None, HiddenStatesMode::None)?;

            let dir = TempDir::new().expect("tempdir");
            let wrapper = LittleBitModel::new(
                model,
                lb_cfg.clone(),
                Some(format!("test/{:?}", arch)),
                None,
            );
            wrapper.save_pretrained(dir.path(), None, CastPolicy::Preserve)?;

            let readme = wrapper.generate_readme("owner/test-littlebit");
            assert!(readme.contains("LittleBit-2"));
            assert!(readme.contains(&cfg.model_type));

            let reloaded_unpacked = LittleBitModel::from_pretrained(
                dir.path().to_str().expect("utf8 path"),
                &LoadOptions {
                    inference_mode: InferenceMode::Unpacked,
                    ..Default::default()
                },
            )?;
            let out_reloaded =
                reloaded_unpacked
                    .model
                    .forward(&input, None, HiddenStatesMode::None)?;
            let drift_unpacked = out_unpacked
                .logits
                .sub(&out_reloaded.logits)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                drift_unpacked < 1e-4,
                "{arch:?} unpacked reload drift: {drift_unpacked}"
            );

            let reloaded_packed = LittleBitModel::from_pretrained(
                dir.path().to_str().expect("utf8 path"),
                &LoadOptions {
                    inference_mode: InferenceMode::Packed,
                    ..Default::default()
                },
            )?;
            let out_packed = reloaded_packed
                .model
                .forward(&input, None, HiddenStatesMode::None)?;
            let drift_packed = out_unpacked
                .logits
                .sub(&out_packed.logits)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                drift_packed < 1e-4,
                "{arch:?} packed reload drift: {drift_packed}"
            );
        }
        Ok(())
    }

    #[test]
    fn kv_cache_decoding_matches_full_prefill() -> Result<()> {
        let dev = Device::Cpu;
        let configs = [
            ModelConfig::tiny_llama(96, 64, 2),
            ModelConfig::tiny_qwen3(96, 64, 2),
            ModelConfig::tiny_phi3(96, 64, 2),
            ModelConfig::tiny_gemma3(96, 64, 2),
            ModelConfig::tiny_opt(96, 64, 2),
        ];
        let prompt = [3u32, 12, 7, 25, 14];

        for cfg in configs {
            let arch = cfg.detect_architecture()?;
            let mut model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 55)?;
            let lb_cfg = LittleBitConfig {
                eff_bit: Some(1.0),
                use_itq: true,
                itq_n_iter: 5,
                ..Default::default()
            };
            model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 13)?;

            let full_t = Tensor::from_slice(&prompt, (1, prompt.len()), &dev)?;
            let full_logits = model
                .forward(&full_t, None, HiddenStatesMode::None)?
                .logits
                .get(0)?
                .get(prompt.len() - 1)?;

            let mut cache = KvCache::new(cfg.num_hidden_layers);
            let mut step_logits = None;
            for &tok in &prompt {
                let tok_t = Tensor::from_slice(&[tok], (1, 1), &dev)?;
                let out = model.forward(&tok_t, Some(&mut cache), HiddenStatesMode::None)?;
                step_logits = Some(out.logits.get(0)?.get(0)?);
            }
            let cached_last = step_logits.expect("last token logits");
            let diff = full_logits
                .sub(&cached_last)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(diff < 1e-4, "{arch:?} KV cache drift: {diff}");
        }
        Ok(())
    }

    #[test]
    fn generate_produces_requested_length() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::tiny_llama(64, 32, 2);
        let model = CausalLm::new_random(cfg, DType::F32, &dev, 9)?;
        let out = model.generate(
            &[1, 2, 3],
            &GenerateConfig {
                max_new_tokens: 5,
                temperature: 0.0,
                ..Default::default()
            },
            &dev,
        )?;
        assert_eq!(out.len(), 8);
        Ok(())
    }
}
