//! Safetensors checkpoint loading, binary unpacking, packed saving, and header inspection.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use regex::Regex;
use safetensors::tensor::TensorInfo;
use serde::{Deserialize, Serialize};

use crate::config::{
    compute_eff_bits, CastPolicy, LittleBitConfig, LittleBitConfigOverrides, ResidualMode,
};
use crate::error::{LittleBitError, Result};
use crate::packer::binary_unpacker;

/// Detected checkpoint storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckpointFormat {
    /// Binary factors are stored bit-packed as `*_packed` (`I32`) and `*_shape` (`I64`).
    Packed,
    /// Binary factors are stored unpacked as float `U` / `V` tensors (legacy format).
    LegacyUnpacked,
}

/// Unpacked checkpoint tensors alongside detected format metadata.
#[derive(Debug)]
pub struct LoadedStateDict {
    /// Tensors loaded onto the requested device, with `*_packed`/`*_shape` unpacked to `U`/`V`.
    pub tensors: HashMap<String, Tensor>,
    /// Original format detected on disk (`Packed` vs. `LegacyUnpacked`).
    pub format: CheckpointFormat,
}

#[derive(Debug, Deserialize, Serialize)]
struct SafetensorsIndex {
    #[serde(default)]
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct IndexMetadata {
    #[serde(default)]
    total_size: usize,
}

fn resolve_safetensor_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let raw = fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&raw)?;
        let shards: BTreeSet<String> = index.weight_map.into_values().collect();
        if shards.is_empty() {
            return Err(LittleBitError::CheckpointNotFound(
                dir.display().to_string(),
            ));
        }
        return Ok(shards.into_iter().map(|s| dir.join(s)).collect());
    }

    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    Err(LittleBitError::CheckpointNotFound(
        dir.display().to_string(),
    ))
}

/// Load all raw safetensors tensors from a checkpoint directory (single file or sharded index).
///
/// # Errors
/// Returns [`LittleBitError::CheckpointNotFound`] when neither `model.safetensors` nor
/// `model.safetensors.index.json` exists, or I/O/safetensors decoding errors.
pub fn load_raw_safetensors_dir(dir: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let shard_paths = resolve_safetensor_files(dir)?;
    let mut state_dict = HashMap::new();
    for path in shard_paths {
        let loaded = candle_core::safetensors::load(&path, device)?;
        state_dict.extend(loaded);
    }
    Ok(state_dict)
}

/// Load a checkpoint directory and automatically unpack `*_packed`/`*_shape` tensors into
/// `target_dtype` float tensors matching upstream `_load_and_process_state_dict`.
///
/// # Errors
/// Returns an error if files are missing, corrupted, or packed shapes are malformed.
pub fn load_and_unpack_state_dict(
    dir: &Path,
    target_dtype: DType,
    device: &Device,
) -> Result<LoadedStateDict> {
    let raw = load_raw_safetensors_dir(dir, &Device::Cpu)?;
    let has_packed = raw.keys().any(|k| k.ends_with("_packed"));

    if !has_packed {
        let mut moved = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            let t = if v.device().same_device(device) {
                v
            } else {
                v.to_device(device)?
            };
            moved.insert(k, t);
        }
        return Ok(LoadedStateDict {
            tensors: moved,
            format: CheckpointFormat::LegacyUnpacked,
        });
    }

    let re = Regex::new(r"^(.*)\.([^.]+?)_(packed|shape)$").map_err(|err| {
        LittleBitError::InvalidConfig(format!("failed to compile packed tensor regex: {err}"))
    })?;

    let mut packed_groups: HashMap<String, HashMap<String, Tensor>> = HashMap::new();
    let mut final_dict = HashMap::new();

    for (key, value) in raw {
        if let Some(caps) = re.captures(&key) {
            let prefix = caps[1].to_string();
            let param_name = caps[2].to_string();
            let suffix_type = caps[3].to_string();
            packed_groups
                .entry(prefix)
                .or_default()
                .insert(format!("{param_name}_{suffix_type}"), value);
        } else {
            let t = if value.device().same_device(device) {
                value
            } else {
                value.to_device(device)?
            };
            final_dict.insert(key, t);
        }
    }

    for (prefix, components) in packed_groups {
        let param_names: Vec<String> = components
            .keys()
            .filter_map(|k| k.strip_suffix("_packed").map(str::to_string))
            .collect();

        for name in param_names {
            let packed_t = components
                .get(&format!("{name}_packed"))
                .ok_or_else(|| LittleBitError::MissingTensor(format!("{prefix}.{name}_packed")))?;
            let shape_t = components
                .get(&format!("{name}_shape"))
                .ok_or_else(|| LittleBitError::MissingTensor(format!("{prefix}.{name}_shape")))?;
            let shape_vec = shape_t
                .to_dtype(DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?;
            if shape_vec.len() != 2 || shape_vec[0] < 0 || shape_vec[1] < 0 {
                return Err(LittleBitError::InvalidShape {
                    expected: "2 non-negative shape elements".to_string(),
                    actual: format!("{shape_vec:?}"),
                });
            }
            let shape = (shape_vec[0] as usize, shape_vec[1] as usize);
            let unpacked = binary_unpacker(packed_t, shape, target_dtype, device)?;
            final_dict.insert(format!("{prefix}.{name}"), unpacked);
        }
    }

    Ok(LoadedStateDict {
        tensors: final_dict,
        format: CheckpointFormat::Packed,
    })
}

/// Save a state dictionary to `dir/model.safetensors` and write `dir/model.safetensors.index.json`.
///
/// Matches upstream `main.py` / `hub.py` checkpoint layout.
///
/// # Errors
/// Returns an error if creating `dir`, converting tensors, or writing safetensors/index fails.
pub fn save_packed_safetensors(
    dir: &Path,
    state_dict: &HashMap<String, Tensor>,
    cast_policy: CastPolicy,
) -> Result<()> {
    fs::create_dir_all(dir)?;

    let mut prepared: HashMap<String, Tensor> = HashMap::with_capacity(state_dict.len());
    let mut weight_map = BTreeMap::new();
    let mut total_size = 0usize;

    for (key, tensor) in state_dict {
        let is_metadata_tensor = key.contains("packed")
            || key.contains("shape")
            || key.ends_with("._eff_bit_target")
            || key.ends_with("._split_dim_final")
            || key.ends_with("._eff_bit_actual");

        let cpu_t = if tensor.device().is_cpu() {
            tensor.clone()
        } else {
            tensor.to_device(&Device::Cpu)?
        };

        let final_t = match cast_policy {
            CastPolicy::ToBf16 if !is_metadata_tensor && cpu_t.dtype() == DType::F32 => {
                cpu_t.to_dtype(DType::BF16)?
            }
            _ => cpu_t,
        };

        total_size += final_t.elem_count() * final_t.dtype().size_in_bytes();
        weight_map.insert(key.clone(), "model.safetensors".to_string());
        prepared.insert(key.clone(), final_t);
    }

    let safetensors_path = dir.join("model.safetensors");
    candle_core::safetensors::save(&prepared, &safetensors_path)?;

    let index = SafetensorsIndex {
        metadata: IndexMetadata { total_size },
        weight_map,
    };
    let index_json = serde_json::to_string_pretty(&index)?;
    fs::write(dir.join("model.safetensors.index.json"), index_json)?;
    Ok(())
}

/// Per-layer metadata extracted from a LittleBit checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInspection {
    /// Module path prefix (e.g., `model.layers.0.self_attn.q_proj`).
    pub name: String,
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// Latent rank `split_dim`.
    pub split_dim: usize,
    /// Whether this layer contains a residual factorization branch (`U_R`, `V_R`).
    pub has_residual: bool,
    /// Actual effective bits per weight.
    pub eff_bit_actual: f64,
    /// Total serialized bytes for this layer in the checkpoint.
    pub serialized_bytes: usize,
    /// Equivalent dense FP16 weight bytes (`out_features * in_features * 2`).
    pub dense_fp16_bytes: usize,
}

/// Summary statistics for a LittleBit checkpoint directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointStats {
    /// Resolved quantization config.
    pub config: LittleBitConfig,
    /// Whether any tensor uses `*_packed` binary packing.
    pub is_packed: bool,
    /// Number of tensors in checkpoint.
    pub tensor_count: usize,
    /// Number of factorized `LittleBitLinear` layers found.
    pub quantized_layer_count: usize,
    /// Total serialized tensor payload bytes across all shards.
    pub total_serialized_bytes: usize,
    /// Total serialized bytes across all quantized layers.
    pub quantized_layers_bytes: usize,
    /// Equivalent dense FP16 bytes across quantized layers.
    pub dense_fp16_bytes: usize,
    /// Weighted average effective bits per weight across all quantized layers.
    pub avg_eff_bits: f64,
    /// Compression ratio (`dense_fp16_bytes / quantized_layers_bytes`).
    pub compression_ratio_vs_fp16: f64,
    /// Sorted per-layer inspection entries.
    pub layers: Vec<LayerInspection>,
}

fn read_header_and_slice(
    file: &mut File,
    data_start: u64,
    offsets: (usize, usize),
) -> Result<Vec<u8>> {
    let len = offsets.1.saturating_sub(offsets.0);
    let mut buf = vec![0u8; len];
    file.seek(SeekFrom::Start(data_start + offsets.0 as u64))?;
    file.read_exact(&mut buf)?;
    Ok(buf)
}

fn decode_shape_tensor_bytes(bytes: &[u8], dtype: safetensors::Dtype) -> Option<(usize, usize)> {
    match dtype {
        safetensors::Dtype::I64 if bytes.len() == 16 => {
            let r = i64::from_le_bytes(bytes[0..8].try_into().ok()?);
            let c = i64::from_le_bytes(bytes[8..16].try_into().ok()?);
            if r >= 0 && c >= 0 {
                Some((r as usize, c as usize))
            } else {
                None
            }
        }
        safetensors::Dtype::I32 if bytes.len() == 8 => {
            let r = i32::from_le_bytes(bytes[0..4].try_into().ok()?);
            let c = i32::from_le_bytes(bytes[4..8].try_into().ok()?);
            if r >= 0 && c >= 0 {
                Some((r as usize, c as usize))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsHeader {
    #[serde(default, rename = "__metadata__")]
    _metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

/// Inspect a local LittleBit checkpoint without loading full weight tensors into memory.
///
/// # Errors
/// Returns an error if the directory has no safetensors weights or headers are corrupted.
pub fn inspect_checkpoint(dir: &Path) -> Result<CheckpointStats> {
    let config = LittleBitConfig::load_with_priority(dir, &LittleBitConfigOverrides::default())?;
    let shard_paths = resolve_safetensor_files(dir)?;

    let mut tensor_shapes: HashMap<String, (Vec<usize>, usize)> = HashMap::new();
    let mut decoded_shapes: HashMap<String, (usize, usize)> = HashMap::new();
    let mut is_packed = false;
    let mut total_serialized_bytes = 0usize;

    for path in shard_paths {
        let mut file = File::open(&path)?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        if header_len > 100_000_000 {
            return Err(LittleBitError::InvalidConfig(
                "safetensors header exceeds 100 MB limit".to_string(),
            ));
        }

        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header: SafetensorsHeader = serde_json::from_slice(&header_bytes)?;
        let data_start = 8 + header_len as u64;

        for (name, info) in header.tensors {
            let byte_len = info.data_offsets.1.saturating_sub(info.data_offsets.0);
            total_serialized_bytes += byte_len;
            if name.ends_with("_packed") {
                is_packed = true;
            }
            if name.ends_with(".U_shape") || name.ends_with(".V_shape") {
                let raw = read_header_and_slice(&mut file, data_start, info.data_offsets)?;
                if let Some(dim) = decode_shape_tensor_bytes(&raw, info.dtype) {
                    decoded_shapes.insert(name.clone(), dim);
                }
            }
            tensor_shapes.insert(name, (info.shape, byte_len));
        }
    }

    let mut layer_prefixes: BTreeSet<String> = BTreeSet::new();
    for key in tensor_shapes.keys() {
        if let Some(prefix) = key
            .strip_suffix(".U_packed")
            .or_else(|| key.strip_suffix(".U"))
        {
            layer_prefixes.insert(prefix.to_string());
        }
    }

    let mut layers = Vec::with_capacity(layer_prefixes.len());
    let mut quantized_layers_bytes = 0usize;
    let mut dense_fp16_bytes = 0usize;
    let mut total_bits = 0.0_f64;
    let mut total_weights = 0usize;

    for prefix in layer_prefixes {
        let u_shape = decoded_shapes
            .get(&format!("{prefix}.U_shape"))
            .copied()
            .or_else(|| {
                let (dims, _) = tensor_shapes.get(&format!("{prefix}.U"))?;
                (dims.len() == 2).then_some((dims[0], dims[1]))
            });
        let v_shape = decoded_shapes
            .get(&format!("{prefix}.V_shape"))
            .copied()
            .or_else(|| {
                let (dims, _) = tensor_shapes.get(&format!("{prefix}.V"))?;
                (dims.len() == 2).then_some((dims[0], dims[1]))
            });

        let (Some((out_features, split_dim)), Some((_, in_features))) = (u_shape, v_shape) else {
            continue;
        };

        let has_residual = tensor_shapes.contains_key(&format!("{prefix}.U_R_packed"))
            || tensor_shapes.contains_key(&format!("{prefix}.U_R"));
        let residual_mode = ResidualMode::from(has_residual);
        let eff_bit_actual = compute_eff_bits(in_features, out_features, split_dim, residual_mode);

        let prefix_dot = format!("{prefix}.");
        let layer_bytes: usize = tensor_shapes
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix_dot))
            .map(|(_, (_, bytes))| *bytes)
            .sum();

        let fp16_bytes = out_features * in_features * 2;
        quantized_layers_bytes += layer_bytes;
        dense_fp16_bytes += fp16_bytes;
        total_bits += eff_bit_actual * (out_features * in_features) as f64;
        total_weights += out_features * in_features;

        layers.push(LayerInspection {
            name: prefix,
            in_features,
            out_features,
            split_dim,
            has_residual,
            eff_bit_actual,
            serialized_bytes: layer_bytes,
            dense_fp16_bytes: fp16_bytes,
        });
    }

    let avg_eff_bits = if total_weights > 0 {
        total_bits / total_weights as f64
    } else {
        0.0
    };
    let compression_ratio_vs_fp16 = if quantized_layers_bytes > 0 {
        dense_fp16_bytes as f64 / quantized_layers_bytes as f64
    } else {
        0.0
    };

    Ok(CheckpointStats {
        config,
        is_packed,
        tensor_count: tensor_shapes.len(),
        quantized_layer_count: layers.len(),
        total_serialized_bytes,
        quantized_layers_bytes,
        dense_fp16_bytes,
        avg_eff_bits,
        compression_ratio_vs_fp16,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainMode;
    use crate::linear::LittleBitLinear;
    use tempfile::TempDir;

    #[test]
    fn packed_save_unpack_load_and_inspect_round_trip() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let dev = Device::Cpu;
        let w = Tensor::randn(0.0f32, 0.5, (64, 96), &dev)?;
        let cfg = LittleBitConfig {
            eff_bit: Some(0.5),
            residual: true,
            use_itq: true,
            itq_n_iter: 15,
            ..Default::default()
        };
        let layer = LittleBitLinear::from_dense_weight(&w, None, &cfg, 1.0, TrainMode::Eval, 5)?;
        let state = layer.pack_state_dict("model.layers.0.self_attn.q_proj")?;

        fs::write(
            dir.path().join("littlebit_config.json"),
            serde_json::to_string_pretty(&cfg)?,
        )?;
        save_packed_safetensors(dir.path(), &state, CastPolicy::ToBf16)?;

        let stats = inspect_checkpoint(dir.path())?;
        assert!(stats.is_packed);
        assert_eq!(stats.quantized_layer_count, 1);
        assert_eq!(stats.layers[0].in_features, 96);
        assert_eq!(stats.layers[0].out_features, 64);
        assert!(stats.layers[0].has_residual);

        let loaded = load_and_unpack_state_dict(dir.path(), DType::F32, &dev)?;
        assert_eq!(loaded.format, CheckpointFormat::Packed);
        assert!(loaded
            .tensors
            .contains_key("model.layers.0.self_attn.q_proj.U"));
        assert!(loaded
            .tensors
            .contains_key("model.layers.0.self_attn.q_proj.V_R"));
        assert_eq!(
            loaded
                .tensors
                .get("model.layers.0.self_attn.q_proj.U")
                .expect("U present")
                .dtype(),
            DType::F32
        );
        assert_eq!(
            loaded
                .tensors
                .get("model.layers.0.self_attn.q_proj.u1")
                .expect("u1 present")
                .dtype(),
            DType::BF16
        );
        Ok(())
    }

    #[test]
    fn loads_sharded_safetensors_via_index_json() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let dev = Device::Cpu;

        let mut s1 = HashMap::new();
        s1.insert(
            "a.weight".to_string(),
            Tensor::ones((4, 4), DType::F32, &dev)?,
        );
        candle_core::safetensors::save(&s1, dir.path().join("shard-00001.safetensors"))?;

        let mut s2 = HashMap::new();
        s2.insert(
            "b.weight".to_string(),
            Tensor::zeros((2, 3), DType::F32, &dev)?,
        );
        candle_core::safetensors::save(&s2, dir.path().join("shard-00002.safetensors"))?;

        let index = serde_json::json!({
            "metadata": { "total_size": 88 },
            "weight_map": {
                "a.weight": "shard-00001.safetensors",
                "b.weight": "shard-00002.safetensors"
            }
        });
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index)?,
        )?;

        let loaded = load_and_unpack_state_dict(dir.path(), DType::F32, &dev)?;
        assert_eq!(loaded.format, CheckpointFormat::LegacyUnpacked);
        assert_eq!(loaded.tensors.len(), 2);
        assert_eq!(loaded.tensors["a.weight"].dims(), &[4, 4]);
        assert_eq!(loaded.tensors["b.weight"].dims(), &[2, 3]);
        Ok(())
    }
}
