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

fn prepare_tensor_for_save(
    key: &str,
    tensor: &Tensor,
    cast_policy: CastPolicy,
) -> Result<(Tensor, usize)> {
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
    let bytes = final_t.elem_count() * final_t.dtype().size_in_bytes();
    Ok((final_t, bytes))
}

fn safetensors_dtype_to_candle(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::I16 => Ok(DType::I16),
        safetensors::Dtype::I32 => Ok(DType::I32),
        safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F64 => Ok(DType::F64),
        other => Err(LittleBitError::InvalidConfig(format!(
            "unsupported safetensors dtype {other:?} in streaming reader"
        ))),
    }
}

fn read_safetensors_header(file: &mut File) -> Result<(SafetensorsHeader, u64)> {
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
    Ok((header, data_start))
}

/// Location metadata for a single tensor inside a safetensors shard on disk.
#[derive(Debug, Clone)]
pub struct TensorLocation {
    /// Path to the `.safetensors` file holding this tensor.
    pub shard_path: PathBuf,
    /// Byte offset where the binary payload section starts (`8 + header_len`).
    pub data_start: u64,
    /// Safetensors element data type.
    pub dtype: safetensors::Dtype,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Byte range `(start, end)` relative to `data_start`.
    pub data_offsets: (usize, usize),
}

/// Safe streaming safetensors reader that parses headers once and seeks to load
/// individual tensors or 3D dimension-0 slices on demand using zero `unsafe` code.
#[derive(Debug)]
pub struct SafetensorsReader {
    locations: BTreeMap<String, TensorLocation>,
    open_files: HashMap<PathBuf, File>,
}

impl SafetensorsReader {
    /// Open a local single-file or sharded safetensors checkpoint directory.
    ///
    /// # Errors
    /// Returns [`LittleBitError::CheckpointNotFound`] if no safetensors checkpoint exists in `dir`,
    /// or I/O/header parsing errors.
    pub fn open(dir: &Path) -> Result<Self> {
        let shard_paths = resolve_safetensor_files(dir)?;
        let mut locations = BTreeMap::new();
        let mut open_files = HashMap::with_capacity(shard_paths.len());

        for path in shard_paths {
            let mut file = File::open(&path)?;
            let (header, data_start) = read_safetensors_header(&mut file)?;
            for (name, info) in header.tensors {
                locations.insert(
                    name,
                    TensorLocation {
                        shard_path: path.clone(),
                        data_start,
                        dtype: info.dtype,
                        shape: info.shape,
                        data_offsets: info.data_offsets,
                    },
                );
            }
            open_files.insert(path, file);
        }

        Ok(Self {
            locations,
            open_files,
        })
    }

    /// Iterate over all tensor keys across all shards in deterministic lexicographical order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.locations.keys().map(String::as_str)
    }

    /// Return true if `name` is present in any shard.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.locations.contains_key(name)
    }

    /// Return location & shape metadata for `name` without reading tensor bytes from disk.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorLocation> {
        self.locations.get(name)
    }

    /// Total number of tensors indexed across all shards.
    #[must_use]
    pub fn len(&self) -> usize {
        self.locations.len()
    }

    /// Return true if zero tensors were indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    fn read_slice(
        &mut self,
        shard_path: &Path,
        abs_offset: u64,
        byte_len: usize,
    ) -> Result<Vec<u8>> {
        if !self.open_files.contains_key(shard_path) {
            let file = File::open(shard_path)?;
            self.open_files.insert(shard_path.to_path_buf(), file);
        }
        let file = self
            .open_files
            .get_mut(shard_path)
            .ok_or_else(|| LittleBitError::CheckpointNotFound(shard_path.display().to_string()))?;
        let mut buf = vec![0u8; byte_len];
        file.seek(SeekFrom::Start(abs_offset))?;
        file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Read and materialize a single tensor `name` onto `device`.
    ///
    /// # Errors
    /// Returns [`LittleBitError::MissingTensor`] if absent or I/O/decoding errors.
    pub fn load_tensor(&mut self, name: &str, device: &Device) -> Result<Tensor> {
        let loc = self
            .locations
            .get(name)
            .cloned()
            .ok_or_else(|| LittleBitError::MissingTensor(name.to_string()))?;
        let candle_dtype = safetensors_dtype_to_candle(loc.dtype)?;
        let elem_count: usize = loc.shape.iter().product();
        let expected_bytes = elem_count * candle_dtype.size_in_bytes();
        let actual_bytes = loc.data_offsets.1.saturating_sub(loc.data_offsets.0);
        if actual_bytes != expected_bytes {
            return Err(LittleBitError::InvalidShape {
                expected: format!("{expected_bytes} payload bytes for `{name}`"),
                actual: format!("{actual_bytes} payload bytes"),
            });
        }
        let buf = self.read_slice(
            &loc.shard_path,
            loc.data_start + loc.data_offsets.0 as u64,
            actual_bytes,
        )?;
        Ok(Tensor::from_raw_buffer(
            &buf,
            candle_dtype,
            &loc.shape,
            device,
        )?)
    }

    /// Stream only `tensor[index]` along dimension 0 without reading the rest of the tensor
    /// into memory (enabling single-expert loading from fused 3D MoE weights).
    ///
    /// # Errors
    /// Returns an error if `name` is missing, rank `< 2`, `index` is out of bounds, or reading fails.
    pub fn load_tensor_slice_dim0(
        &mut self,
        name: &str,
        index: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let loc = self
            .locations
            .get(name)
            .cloned()
            .ok_or_else(|| LittleBitError::MissingTensor(name.to_string()))?;
        if loc.shape.len() < 2 {
            return Err(LittleBitError::InvalidShape {
                expected: "tensor with rank >= 2".to_string(),
                actual: format!("{:?}", loc.shape),
            });
        }
        if index >= loc.shape[0] {
            return Err(LittleBitError::InvalidShape {
                expected: format!("dim0 index < {}", loc.shape[0]),
                actual: format!("{index}"),
            });
        }
        let candle_dtype = safetensors_dtype_to_candle(loc.dtype)?;
        let slice_shape = &loc.shape[1..];
        let slice_elems: usize = slice_shape.iter().product();
        let slice_bytes = slice_elems * candle_dtype.size_in_bytes();
        let start_rel = loc.data_offsets.0 + index * slice_bytes;
        if start_rel + slice_bytes > loc.data_offsets.1 {
            return Err(LittleBitError::InvalidShape {
                expected: format!("slice ending at byte {}", start_rel + slice_bytes),
                actual: format!("tensor payload ends at byte {}", loc.data_offsets.1),
            });
        }
        let buf = self.read_slice(
            &loc.shard_path,
            loc.data_start + start_rel as u64,
            slice_bytes,
        )?;
        Ok(Tensor::from_raw_buffer(
            &buf,
            candle_dtype,
            slice_shape,
            device,
        )?)
    }
}

/// Incremental sharded safetensors writer that flushes completed shards to disk
/// and frees their in-memory buffers as quantization progresses.
#[derive(Debug)]
pub struct ShardedSafetensorsWriter {
    output_dir: PathBuf,
    max_shard_bytes: usize,
    cast_policy: CastPolicy,
    pending: HashMap<String, Tensor>,
    pending_bytes: usize,
    completed_shards: Vec<(PathBuf, Vec<String>)>,
    total_size: usize,
}

impl ShardedSafetensorsWriter {
    /// Initialize a streaming sharded safetensors writer under `output_dir`.
    ///
    /// # Errors
    /// Returns [`LittleBitError::InvalidConfig`] if `max_shard_bytes == 0` or creating `output_dir` fails.
    pub fn new(
        output_dir: impl AsRef<Path>,
        max_shard_bytes: usize,
        cast_policy: CastPolicy,
    ) -> Result<Self> {
        if max_shard_bytes == 0 {
            return Err(LittleBitError::InvalidConfig(
                "max_shard_bytes must be >= 1".to_string(),
            ));
        }
        let dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self {
            output_dir: dir,
            max_shard_bytes,
            cast_policy,
            pending: HashMap::new(),
            pending_bytes: 0,
            completed_shards: Vec::new(),
            total_size: 0,
        })
    }

    /// Stage a tensor for writing, applying [`CastPolicy`] and recording its serialized byte size.
    ///
    /// # Errors
    /// Returns an error if moving or casting the tensor fails.
    pub fn add_tensor(&mut self, key: String, tensor: Tensor) -> Result<()> {
        let (final_t, bytes) = prepare_tensor_for_save(&key, &tensor, self.cast_policy)?;
        if let Some(prev) = self.pending.insert(key, final_t) {
            let prev_bytes = prev.elem_count() * prev.dtype().size_in_bytes();
            self.pending_bytes = self.pending_bytes.saturating_sub(prev_bytes);
            self.total_size = self.total_size.saturating_sub(prev_bytes);
        }
        self.pending_bytes += bytes;
        self.total_size += bytes;
        Ok(())
    }

    /// Flush the current pending shard to disk if `pending_bytes >= max_shard_bytes`.
    ///
    /// # Errors
    /// Returns an error if writing the shard fails.
    pub fn flush_if_needed(&mut self) -> Result<()> {
        if self.pending_bytes >= self.max_shard_bytes {
            self.flush_shard()?;
        }
        Ok(())
    }

    /// Force-flush any buffered tensors into a numbered temporary shard and clear memory.
    ///
    /// # Errors
    /// Returns an error if writing the shard file fails.
    pub fn flush_shard(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let shard_idx = self.completed_shards.len();
        let tmp_path = self
            .output_dir
            .join(format!("._tmp_shard_{shard_idx:05}.safetensors"));
        let keys: Vec<String> = self.pending.keys().cloned().collect();
        candle_core::safetensors::save(&self.pending, &tmp_path)?;
        self.pending.clear();
        self.pending_bytes = 0;
        self.completed_shards.push((tmp_path, keys));
        Ok(())
    }

    /// Finalize shard filenames (`model.safetensors` or `model-00001-of-NNNNN.safetensors`),
    /// write `model.safetensors.index.json`, and return the number of output shards written.
    ///
    /// # Errors
    /// Returns an error if flushing, renaming, or writing `model.safetensors.index.json` fails.
    pub fn finish(mut self) -> Result<usize> {
        self.flush_shard()?;
        let n = self.completed_shards.len();
        let mut weight_map = BTreeMap::new();

        let shards_written = if n == 0 {
            let empty: HashMap<String, Tensor> = HashMap::new();
            candle_core::safetensors::save(&empty, self.output_dir.join("model.safetensors"))?;
            1
        } else if n == 1 {
            let (tmp_path, keys) = self.completed_shards.remove(0);
            fs::rename(&tmp_path, self.output_dir.join("model.safetensors"))?;
            for k in keys {
                weight_map.insert(k, "model.safetensors".to_string());
            }
            1
        } else {
            for (idx, (tmp_path, keys)) in self.completed_shards.into_iter().enumerate() {
                let shard_name = format!("model-{:05}-of-{:05}.safetensors", idx + 1, n);
                fs::rename(&tmp_path, self.output_dir.join(&shard_name))?;
                for k in keys {
                    weight_map.insert(k, shard_name.clone());
                }
            }
            n
        };

        let index = SafetensorsIndex {
            metadata: IndexMetadata {
                total_size: self.total_size,
            },
            weight_map,
        };
        let index_json = serde_json::to_string_pretty(&index)?;
        fs::write(
            self.output_dir.join("model.safetensors.index.json"),
            index_json,
        )?;
        Ok(shards_written)
    }
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
        let (final_t, bytes) = prepare_tensor_for_save(key, tensor, cast_policy)?;
        total_size += bytes;
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
        let (header, data_start) = read_safetensors_header(&mut file)?;

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

    #[test]
    fn streaming_reader_loads_single_sharded_and_3d_slices_exact() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let dev = Device::Cpu;

        let t3d = Tensor::randn(0.0f32, 1.0, (4, 6, 8), &dev)?.to_dtype(DType::BF16)?;
        let t_f32 = Tensor::randn(0.0f32, 0.5, (5, 7), &dev)?;
        let t_f16 = Tensor::randn(0.0f32, 0.5, (3, 4), &dev)?.to_dtype(DType::F16)?;
        let t_i32 = Tensor::from_slice(&[1i32, -2, 3, 4], (2, 2), &dev)?;
        let t_i64 = Tensor::from_slice(&[10i64, 20], (2,), &dev)?;

        let mut s1 = HashMap::new();
        s1.insert("experts.gate_up_proj".to_string(), t3d.clone());
        s1.insert("w.f32".to_string(), t_f32.clone());
        candle_core::safetensors::save(&s1, dir.path().join("model-00001-of-00002.safetensors"))?;

        let mut s2 = HashMap::new();
        s2.insert("w.f16".to_string(), t_f16.clone());
        s2.insert("idx.i32".to_string(), t_i32.clone());
        s2.insert("shape.i64".to_string(), t_i64.clone());
        candle_core::safetensors::save(&s2, dir.path().join("model-00002-of-00002.safetensors"))?;

        let index = serde_json::json!({
            "metadata": { "total_size": 0 },
            "weight_map": {
                "experts.gate_up_proj": "model-00001-of-00002.safetensors",
                "w.f32": "model-00001-of-00002.safetensors",
                "w.f16": "model-00002-of-00002.safetensors",
                "idx.i32": "model-00002-of-00002.safetensors",
                "shape.i64": "model-00002-of-00002.safetensors"
            }
        });
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&index)?,
        )?;

        let mut reader = SafetensorsReader::open(dir.path())?;
        assert_eq!(reader.len(), 5);
        assert!(reader.contains("experts.gate_up_proj"));

        let baseline = load_raw_safetensors_dir(dir.path(), &dev)?;
        for name in ["w.f32", "w.f16", "idx.i32", "shape.i64"] {
            let got = reader.load_tensor(name, &dev)?;
            let exp = &baseline[name];
            assert_eq!(got.dtype(), exp.dtype());
            assert_eq!(got.dims(), exp.dims());
            let diff = got
                .to_dtype(DType::F32)?
                .sub(&exp.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(diff, 0.0, "tensor `{name}` mismatch");
        }

        for e in 0..4 {
            let slice = reader.load_tensor_slice_dim0("experts.gate_up_proj", e, &dev)?;
            let expected = t3d.get(e)?;
            assert_eq!(slice.dims(), &[6, 8]);
            assert_eq!(slice.dtype(), DType::BF16);
            let diff = slice
                .to_dtype(DType::F32)?
                .sub(&expected.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(diff, 0.0, "3D slice {e} mismatch");
        }
        Ok(())
    }

    #[test]
    fn sharded_writer_splits_shards_and_roundtrips_with_inspect() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let dev = Device::Cpu;

        let cfg = LittleBitConfig {
            eff_bit: Some(0.5),
            residual: false,
            use_itq: true,
            itq_n_iter: 5,
            ..Default::default()
        };
        fs::write(
            dir.path().join("littlebit_config.json"),
            serde_json::to_string_pretty(&cfg)?,
        )?;

        // Small max_shard_bytes forces at least 2 output shards across two LittleBit layers.
        let mut writer = ShardedSafetensorsWriter::new(dir.path(), 512, CastPolicy::ToBf16)?;
        for idx in 0..2 {
            let w = Tensor::randn(0.0f32, 0.5, (64, 64), &dev)?;
            let layer = LittleBitLinear::from_dense_weight(
                &w,
                None,
                &cfg,
                1.0,
                TrainMode::Eval,
                idx as u64,
            )?;
            let sd = layer.pack_state_dict(&format!("model.layers.{idx}.self_attn.q_proj"))?;
            for (k, v) in sd {
                writer.add_tensor(k, v)?;
            }
            writer.flush_if_needed()?;
        }
        let shards_written = writer.finish()?;
        assert!(
            shards_written >= 2,
            "expected multiple shards, got {shards_written}"
        );
        assert!(dir.path().join("model.safetensors.index.json").is_file());
        assert!(dir
            .path()
            .join(format!("model-00001-of-{shards_written:05}.safetensors"))
            .is_file());

        let stats = inspect_checkpoint(dir.path())?;
        assert!(stats.is_packed);
        assert_eq!(stats.quantized_layer_count, 2);

        let unpacked = load_and_unpack_state_dict(dir.path(), DType::F32, &dev)?;
        assert_eq!(unpacked.format, CheckpointFormat::Packed);
        assert!(unpacked
            .tensors
            .contains_key("model.layers.0.self_attn.q_proj.U"));
        assert!(unpacked
            .tensors
            .contains_key("model.layers.1.self_attn.q_proj.U"));
        Ok(())
    }
}
