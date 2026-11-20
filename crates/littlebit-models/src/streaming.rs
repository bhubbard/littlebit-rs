//! Streaming layer-by-layer quantization for large checkpoints under constrained RAM.
//!
//! Reads one transformer block (or one MoE expert slice) at a time directly from `.safetensors`
//! shards via [`SafetensorsReader`], runs SVD + Joint-ITQ factorization and `i32` binary packing,
//! streams the packed outputs to sharded safetensors via [`ShardedSafetensorsWriter`], and
//! immediately drops dense weights between blocks.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use regex::Regex;
use tracing::info;

use littlebit_core::{
    CastPolicy, LittleBitConfig, LittleBitLinear, SafetensorsReader, ShardedSafetensorsWriter,
    TrainMode,
};

use crate::config::{ModelArchitecture, ModelConfig};
use crate::error::{ModelError, Result};
use crate::hub::write_quantized_metadata;
use crate::patch::{derive_layer_seed, PatchPolicy, PatchSummary};

const DEFAULT_MAX_SHARD_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Configuration options for [`stream_quantize_checkpoint`].
#[derive(Debug, Clone)]
pub struct StreamingQuantizeOptions {
    /// Deterministic RNG seed for randomized SVD and Joint-ITQ (`42` default).
    pub seed: u64,
    /// Serialization cast policy for scale/norm tensors (`ToBf16` or `Preserve`).
    pub cast_policy: CastPolicy,
    /// Maximum byte size per output shard before flushing to disk (`2 GiB` default).
    pub max_shard_bytes: usize,
    /// Compute device used during SVD/ITQ decomposition (`Device::Cpu` default).
    pub device: Device,
    /// Floating-point compute dtype during factorization (`DType::F32` default).
    pub dtype: DType,
    /// Skip multimodal vision/audio tower weights when quantizing multimodal checkpoints.
    pub text_only: bool,
    /// Optional base model ID recorded in `base_model.json`.
    pub base_model_id: Option<String>,
}

impl Default for StreamingQuantizeOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            cast_policy: CastPolicy::ToBf16,
            max_shard_bytes: DEFAULT_MAX_SHARD_BYTES,
            device: Device::Cpu,
            dtype: DType::F32,
            text_only: false,
            base_model_id: None,
        }
    }
}

/// Summary report returned by [`stream_quantize_checkpoint`].
#[derive(Debug, Clone)]
pub struct StreamingQuantizeReport {
    /// Quantization layer & bit-budget statistics across all processed blocks.
    pub patch_summary: PatchSummary,
    /// Number of `.safetensors` output shards written.
    pub shards_written: usize,
    /// Number of decoder transformer blocks processed.
    pub blocks_processed: usize,
    /// Number of non-quantized passthrough tensors copied directly (embeddings, norms, router, head).
    pub passthrough_tensors: usize,
}

fn is_multimodal_tensor(name: &str) -> bool {
    [
        "vision_tower",
        "audio_tower",
        "multi_modal_projector",
        "embed_vision",
        "embed_audio",
    ]
    .iter()
    .any(|pat| name.contains(pat))
}

fn quantize_and_write_linear(
    module_name: &str,
    weight: &Tensor,
    bias: Option<&Tensor>,
    lb_cfg: &LittleBitConfig,
    policy: &PatchPolicy,
    seed: u64,
    writer: &mut ShardedSafetensorsWriter,
    summary: &mut PatchSummary,
) -> Result<()> {
    let effective_cfg = lb_cfg.effective_for_module(module_name)?;
    let ratio = policy.ratio_factor(module_name, &effective_cfg);
    let layer_seed = derive_layer_seed(seed, module_name);
    let lb = LittleBitLinear::from_dense_weight(
        weight,
        bias,
        &effective_cfg,
        ratio,
        TrainMode::Eval,
        layer_seed,
    )?;
    summary.record_budget(lb.budget());
    let packed_sd = lb.pack_state_dict(module_name)?;
    for (k, v) in packed_sd {
        writer.add_tensor(k, v)?;
    }
    Ok(())
}

/// Quantize a dense local checkpoint in streaming layer-by-layer fashion without loading the
/// full model weights into RAM.
///
/// Handles single-file and multi-shard input checkpoints across all supported architectures,
/// including Phi-3/4 `qkv_proj` splitting, OPT `project_in`/`project_out`, and Gemma 4
/// fused 3D MoE expert tensors (`gate_up_proj` / `down_proj`, loaded one expert slice at a time).
///
/// # Errors
/// Returns an error if configs are invalid, required weights are malformed, or disk I/O fails.
#[allow(clippy::too_many_lines)]
pub fn stream_quantize_checkpoint(
    src_dir: &Path,
    dst_dir: &Path,
    lb_cfg: &LittleBitConfig,
    opts: &StreamingQuantizeOptions,
) -> Result<StreamingQuantizeReport> {
    lb_cfg.validate()?;
    let model_cfg = ModelConfig::from_file(&src_dir.join("config.json"))?;
    let arch = model_cfg.detect_architecture()?;
    let mut reader = SafetensorsReader::open(src_dir)?;
    let mut writer =
        ShardedSafetensorsWriter::new(dst_dir, opts.max_shard_bytes, opts.cast_policy)?;
    let policy = PatchPolicy::new();
    let mut summary = PatchSummary::default();
    let mut passthrough_tensors = 0usize;

    let layer_re = Regex::new(r"(?:^|\.)layers\.(\d+)\.").map_err(|err| {
        ModelError::InvalidArgument(format!("failed to compile layer index regex: {err}"))
    })?;

    let all_names: Vec<String> = reader.tensor_names().map(str::to_string).collect();
    let mut blocks: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut global_keys: Vec<String> = Vec::new();

    for name in all_names {
        let multimodal = is_multimodal_tensor(&name);
        if opts.text_only && multimodal {
            continue;
        }
        if !multimodal {
            if let Some(caps) = layer_re.captures(&name) {
                if let Ok(idx) = caps[1].parse::<usize>() {
                    blocks.entry(idx).or_default().push(name);
                    continue;
                }
            }
        }
        global_keys.push(name);
    }

    // 1) Process non-layer (global) tensors.
    let global_set: HashSet<&str> = global_keys.iter().map(String::as_str).collect();
    let mut global_consumed: HashSet<String> = HashSet::new();

    for key in &global_keys {
        if global_consumed.contains(key) {
            continue;
        }
        if arch == ModelArchitecture::Opt
            && (key == "model.decoder.project_in.weight"
                || key == "model.decoder.project_out.weight")
        {
            let module_prefix = key.trim_end_matches(".weight");
            let bias_key = format!("{module_prefix}.bias");
            let w = reader
                .load_tensor(key, &opts.device)?
                .to_dtype(opts.dtype)?;
            let b = if global_set.contains(bias_key.as_str()) {
                Some(
                    reader
                        .load_tensor(&bias_key, &opts.device)?
                        .to_dtype(opts.dtype)?,
                )
            } else {
                None
            };
            quantize_and_write_linear(
                module_prefix,
                &w,
                b.as_ref(),
                lb_cfg,
                &policy,
                opts.seed,
                &mut writer,
                &mut summary,
            )?;
            global_consumed.insert(key.clone());
            if b.is_some() {
                global_consumed.insert(bias_key);
            }
            writer.flush_if_needed()?;
            continue;
        }

        if let Some(prefix) = key.strip_suffix(".weight") {
            if policy.is_excluded(prefix) {
                summary.record_skipped();
            }
        }
        let tensor = reader.load_tensor(key, &Device::Cpu)?;
        writer.add_tensor(key.clone(), tensor)?;
        passthrough_tensors += 1;
        global_consumed.insert(key.clone());
        writer.flush_if_needed()?;
    }

    // 2) Process decoder transformer blocks one at a time.
    let mut blocks_processed = 0usize;
    let total_blocks = blocks.len();

    for (block_idx, keys) in blocks {
        let key_set: HashSet<&str> = keys.iter().map(String::as_str).collect();
        let mut consumed: HashSet<String> = HashSet::new();

        // 2a) Phi-3 / Phi-4 fused `qkv_proj` split.
        if arch == ModelArchitecture::Phi3 {
            for key in &keys {
                if let Some(module_prefix) = key.strip_suffix(".weight") {
                    if let Some(attn_prefix) = module_prefix.strip_suffix(".qkv_proj") {
                        let bias_key = format!("{module_prefix}.bias");
                        let w = reader
                            .load_tensor(key, &opts.device)?
                            .to_dtype(opts.dtype)?;
                        let b = if key_set.contains(bias_key.as_str()) {
                            Some(
                                reader
                                    .load_tensor(&bias_key, &opts.device)?
                                    .to_dtype(opts.dtype)?,
                            )
                        } else {
                            None
                        };

                        let head_dim = model_cfg.effective_head_dim();
                        let q_dim = model_cfg.num_attention_heads * head_dim;
                        let kv_dim = model_cfg.effective_kv_heads() * head_dim;

                        let qw = w.i(0..q_dim)?.contiguous()?;
                        let kw = w.i(q_dim..q_dim + kv_dim)?.contiguous()?;
                        let vw = w.i(q_dim + kv_dim..q_dim + 2 * kv_dim)?.contiguous()?;

                        let (qb, kb, vb) = if let Some(ref bt) = b {
                            (
                                Some(bt.i(0..q_dim)?.contiguous()?),
                                Some(bt.i(q_dim..q_dim + kv_dim)?.contiguous()?),
                                Some(bt.i(q_dim + kv_dim..q_dim + 2 * kv_dim)?.contiguous()?),
                            )
                        } else {
                            (None, None, None)
                        };
                        drop(w);
                        drop(b);

                        for (suffix, slice_w, slice_b) in
                            [("q_proj", qw, qb), ("k_proj", kw, kb), ("v_proj", vw, vb)]
                        {
                            let split_name = format!("{attn_prefix}.{suffix}");
                            quantize_and_write_linear(
                                &split_name,
                                &slice_w,
                                slice_b.as_ref(),
                                lb_cfg,
                                &policy,
                                opts.seed,
                                &mut writer,
                                &mut summary,
                            )?;
                            drop(slice_w);
                            drop(slice_b);
                            writer.flush_if_needed()?;
                        }

                        consumed.insert(key.clone());
                        if key_set.contains(bias_key.as_str()) {
                            consumed.insert(bias_key);
                        }
                    }
                }
            }
        }

        // 2b) Gemma 4 fused 3D MoE experts (`experts.gate_up_proj`, `experts.down_proj`).
        for key in &keys {
            if consumed.contains(key) {
                continue;
            }
            let is_fused_expert =
                key.ends_with(".experts.gate_up_proj") || key.ends_with(".experts.down_proj");
            if !is_fused_expert {
                continue;
            }
            let Some(loc) = reader.tensor_info(key) else {
                continue;
            };
            if loc.shape.len() != 3 {
                continue;
            }
            let num_experts = loc.shape[0];
            let Some((experts_prefix, proj_name)) = key.rsplit_once('.') else {
                continue;
            };
            let experts_prefix = experts_prefix.to_string();
            let proj_name = proj_name.to_string();

            for e in 0..num_experts {
                let w_2d = reader
                    .load_tensor_slice_dim0(key, e, &opts.device)?
                    .to_dtype(opts.dtype)?;
                let mod_name = format!("{experts_prefix}.{e}.{proj_name}");
                quantize_and_write_linear(
                    &mod_name,
                    &w_2d,
                    None,
                    lb_cfg,
                    &policy,
                    opts.seed,
                    &mut writer,
                    &mut summary,
                )?;
                drop(w_2d);
                writer.flush_if_needed()?;
            }
            consumed.insert(key.clone());
        }

        // 2c) Standard 2D linear weights (plus excluded router projections).
        for key in &keys {
            if consumed.contains(key) {
                continue;
            }
            let Some(module_prefix) = key.strip_suffix(".weight") else {
                continue;
            };
            let Some(loc) = reader.tensor_info(key) else {
                continue;
            };
            if loc.shape.len() != 2 {
                continue;
            }

            let bias_key = format!("{module_prefix}.bias");
            if policy.is_excluded(module_prefix) {
                summary.record_skipped();
                let w = reader.load_tensor(key, &Device::Cpu)?;
                writer.add_tensor(key.clone(), w)?;
                passthrough_tensors += 1;
                consumed.insert(key.clone());
                if key_set.contains(bias_key.as_str()) {
                    let b = reader.load_tensor(&bias_key, &Device::Cpu)?;
                    writer.add_tensor(bias_key.clone(), b)?;
                    passthrough_tensors += 1;
                    consumed.insert(bias_key);
                }
                writer.flush_if_needed()?;
                continue;
            }

            let w = reader
                .load_tensor(key, &opts.device)?
                .to_dtype(opts.dtype)?;
            let b = if key_set.contains(bias_key.as_str()) {
                Some(
                    reader
                        .load_tensor(&bias_key, &opts.device)?
                        .to_dtype(opts.dtype)?,
                )
            } else {
                None
            };
            quantize_and_write_linear(
                module_prefix,
                &w,
                b.as_ref(),
                lb_cfg,
                &policy,
                opts.seed,
                &mut writer,
                &mut summary,
            )?;
            drop(w);
            drop(b);
            consumed.insert(key.clone());
            if key_set.contains(bias_key.as_str()) {
                consumed.insert(bias_key);
            }
            writer.flush_if_needed()?;
        }

        // 2d) Remaining unconsumed non-linear tensors inside the block (norms, scales, buffers).
        for key in &keys {
            if consumed.contains(key) {
                continue;
            }
            let tensor = reader.load_tensor(key, &Device::Cpu)?;
            writer.add_tensor(key.clone(), tensor)?;
            passthrough_tensors += 1;
            writer.flush_if_needed()?;
        }

        blocks_processed += 1;
        info!(
            block = block_idx + 1,
            total = total_blocks,
            patched = summary.patched_layers,
            avg_eff_bits = summary.avg_eff_bits,
            "stream-quantized transformer block"
        );
    }

    let shards_written = writer.finish()?;
    write_quantized_metadata(
        dst_dir,
        &model_cfg,
        lb_cfg,
        opts.base_model_id.as_deref(),
        Some(src_dir),
    )?;

    Ok(StreamingQuantizeReport {
        patch_summary: summary,
        shards_written,
        blocks_processed,
        passthrough_tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::{LittleBitModel, LoadOptions};
    use crate::model::{CausalLm, HiddenStatesMode};
    use littlebit_core::{inspect_checkpoint, InferenceMode, QuantFunc};
    use std::collections::HashMap;
    use std::fs;
    use tempfile::TempDir;

    fn save_dense_checkpoint(
        dir: &Path,
        cfg: &ModelConfig,
        state_dict: &HashMap<String, Tensor>,
    ) -> Result<()> {
        fs::create_dir_all(dir)?;
        fs::write(dir.join("config.json"), serde_json::to_string_pretty(cfg)?)?;
        candle_core::safetensors::save(state_dict, dir.join("model.safetensors"))?;
        Ok(())
    }

    #[test]
    fn streaming_quantize_gemma4_dense_matches_in_memory_and_reloads_packed() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::tiny_gemma4_dense(96, 64, 2);
        let src_dir = TempDir::new().expect("src");
        let dst_dir = TempDir::new().expect("dst");

        let mut ref_model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 51)?;
        let dense_sd = ref_model.pack_state_dict()?;
        save_dense_checkpoint(src_dir.path(), &cfg, &dense_sd)?;

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

        let ref_summary = ref_model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 101)?;
        let report = stream_quantize_checkpoint(
            src_dir.path(),
            dst_dir.path(),
            &lb_cfg,
            &StreamingQuantizeOptions {
                seed: 101,
                cast_policy: CastPolicy::Preserve,
                max_shard_bytes: 16 * 1024,
                ..Default::default()
            },
        )?;

        assert_eq!(report.blocks_processed, cfg.num_hidden_layers);
        assert_eq!(
            report.patch_summary.patched_layers,
            ref_summary.patched_layers
        );
        assert!(report.shards_written >= 2, "expected sharded output");

        let input = Tensor::from_slice(&[1u32, 4, 7, 15], (1, 4), &dev)?;
        let ref_logits = ref_model
            .forward(&input, None, HiddenStatesMode::None)?
            .logits;

        for mode in [InferenceMode::Unpacked, InferenceMode::Packed] {
            let reloaded = LittleBitModel::from_pretrained(
                dst_dir.path().to_str().expect("utf8"),
                &LoadOptions {
                    inference_mode: mode,
                    ..Default::default()
                },
            )?;
            let got_logits = reloaded
                .model
                .forward(&input, None, HiddenStatesMode::None)?
                .logits;
            let diff = ref_logits
                .sub(&got_logits)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(
                diff < 1e-4,
                "streaming vs in-memory Gemma 4 Dense drift ({mode:?}): {diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn streaming_quantize_gemma4_moe_from_fused_3d_experts_multi_shard() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::tiny_gemma4_moe(80, 32, 2, 6, 2);
        let src_dir = TempDir::new().expect("src");
        let dst_dir = TempDir::new().expect("dst");

        let mut ref_model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 88)?;
        let mut hf_sd = ref_model.pack_state_dict()?;

        // Convert per-expert keys to Hugging Face fused 3D `experts.gate_up_proj` / `experts.down_proj`.
        for l in 0..cfg.num_hidden_layers {
            let mut gu_slices = Vec::new();
            let mut dn_slices = Vec::new();
            for e in 0..6 {
                gu_slices.push(
                    hf_sd
                        .remove(&format!("model.layers.{l}.experts.{e}.gate_up_proj.weight"))
                        .expect("gu"),
                );
                dn_slices.push(
                    hf_sd
                        .remove(&format!("model.layers.{l}.experts.{e}.down_proj.weight"))
                        .expect("dn"),
                );
            }
            hf_sd.insert(
                format!("model.layers.{l}.experts.gate_up_proj"),
                Tensor::stack(&gu_slices, 0)?,
            );
            hf_sd.insert(
                format!("model.layers.{l}.experts.down_proj"),
                Tensor::stack(&dn_slices, 0)?,
            );
        }
        save_dense_checkpoint(src_dir.path(), &cfg, &hf_sd)?;

        let lb_cfg = LittleBitConfig {
            eff_bit: Some(0.8),
            residual: false,
            use_itq: true,
            itq_n_iter: 6,
            ..Default::default()
        };
        let ref_summary = ref_model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 55)?;

        let report = stream_quantize_checkpoint(
            src_dir.path(),
            dst_dir.path(),
            &lb_cfg,
            &StreamingQuantizeOptions {
                seed: 55,
                cast_policy: CastPolicy::Preserve,
                max_shard_bytes: 8 * 1024,
                ..Default::default()
            },
        )?;
        assert_eq!(
            report.patch_summary.patched_layers,
            ref_summary.patched_layers
        );
        assert_eq!(
            report.patch_summary.skipped_layers,
            ref_summary.skipped_layers
        );
        assert!(report.shards_written >= 2);

        let stats = inspect_checkpoint(dst_dir.path())?;
        assert!(stats.is_packed);
        assert_eq!(stats.quantized_layer_count, ref_summary.patched_layers);

        let loaded = LittleBitModel::from_pretrained(
            dst_dir.path().to_str().expect("utf8"),
            &LoadOptions {
                inference_mode: InferenceMode::Packed,
                ..Default::default()
            },
        )?;
        if let CausalLm::Gemma4(ref g4) = loaded.model {
            let moe = g4.layer_moe(0).expect("moe 0");
            assert!(matches!(
                moe.router.proj,
                crate::layer::LinearLayer::Dense { .. }
            ));
            assert!(matches!(
                moe.experts.experts[0].gate_up_proj,
                crate::layer::LinearLayer::Packed(_)
            ));
        } else {
            panic!("expected Gemma4");
        }

        let input = Tensor::from_slice(&[2u32, 9, 13], (1, 3), &dev)?;
        let ref_logits = ref_model
            .forward(&input, None, HiddenStatesMode::None)?
            .logits;
        let got_logits = loaded
            .model
            .forward(&input, None, HiddenStatesMode::None)?
            .logits;
        let diff = ref_logits
            .sub(&got_logits)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "streaming MoE vs in-memory drift: {diff}");
        Ok(())
    }

    #[test]
    fn streaming_quantize_phi3_and_text_only_stripping() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = ModelConfig::tiny_phi3(64, 32, 2);
        let src_dir = TempDir::new().expect("src");
        let dst_dir = TempDir::new().expect("dst");

        let mut ref_model = CausalLm::new_random(cfg.clone(), DType::F32, &dev, 17)?;
        let mut raw_sd = ref_model.pack_state_dict()?;
        raw_sd.insert(
            "vision_tower.layers.0.proj.weight".to_string(),
            Tensor::ones((16, 16), DType::F32, &dev)?,
        );
        save_dense_checkpoint(src_dir.path(), &cfg, &raw_sd)?;

        let lb_cfg = LittleBitConfig {
            eff_bit: Some(1.0),
            use_itq: false,
            ..Default::default()
        };
        let ref_summary = ref_model.apply_littlebit_patch(&lb_cfg, TrainMode::Eval, 29)?;

        let report = stream_quantize_checkpoint(
            src_dir.path(),
            dst_dir.path(),
            &lb_cfg,
            &StreamingQuantizeOptions {
                seed: 29,
                cast_policy: CastPolicy::Preserve,
                text_only: true,
                ..Default::default()
            },
        )?;
        assert_eq!(report.blocks_processed, cfg.num_hidden_layers);
        assert_eq!(
            report.patch_summary.patched_layers,
            ref_summary.patched_layers
        );

        let raw_out = littlebit_core::load_raw_safetensors_dir(dst_dir.path(), &dev)?;
        assert!(!raw_out.keys().any(|k| k.contains("vision_tower")));

        let reloaded = LittleBitModel::from_pretrained(
            dst_dir.path().to_str().expect("utf8"),
            &LoadOptions {
                inference_mode: InferenceMode::Packed,
                ..Default::default()
            },
        )?;
        let input = Tensor::from_slice(&[1u32, 5, 8], (1, 3), &dev)?;
        let diff = ref_model
            .forward(&input, None, HiddenStatesMode::None)?
            .logits
            .sub(
                &reloaded
                    .model
                    .forward(&input, None, HiddenStatesMode::None)?
                    .logits,
            )?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-4, "streaming Phi-3 drift: {diff}");
        Ok(())
    }
}
