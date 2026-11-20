// -----------------------------------------------------------------------------
// Derived from utils/datautils.py in SamsungLabs/LittleBit under the MIT License:
// Copyright (c) 2024 Dongwon Jo
// Modified by Dongkyu Kim in 2026 (Samsung Electronics Co., Ltd.)
// -----------------------------------------------------------------------------

//! Dataset loading, tokenization, SHA-256 caching, and block chunking (`utils/datautils.py`).

use std::fmt;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use flate2::read::GzDecoder;
use hf_hub::{split_id, HFClient};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tracing::info;

use littlebit_models::resolve_model_dir;

use crate::error::{Result, TrainError};

/// Default sequence block length matching `utils/datautils.py`.
pub const DEFAULT_BLOCK_SIZE: usize = 2048;

/// Supported training and perplexity evaluation datasets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetKind {
    /// WikiText-2 raw (`Salesforce/wikitext` / `wikitext-2-raw-v1`).
    WikiText2,
    /// AllenAI C4 English (`allenai/c4`).
    C4,
    /// Combined C4 + WikiText-2 training corpus (`c4_wiki`).
    C4Wiki,
    /// Penn Treebank validation split (`ptb`).
    Ptb,
    /// Penn Treebank test split (`ptb_new`).
    PtbNew,
    /// AllenAI C4 contiguous slice (`c4_new`).
    C4New,
    /// Local `.txt`, `.jsonl`, or `.jsonl.gz` corpus file.
    LocalFile(PathBuf),
}

impl DatasetKind {
    /// Canonical cache directory name matching upstream `utils/datautils.py`.
    #[must_use]
    pub fn cache_name(&self) -> &str {
        match self {
            Self::WikiText2 => "wikitext2",
            Self::C4 => "c4",
            Self::C4Wiki => "c4_wiki",
            Self::Ptb => "ptb",
            Self::PtbNew => "ptb_new",
            Self::C4New => "c4_new",
            Self::LocalFile(_) => "local",
        }
    }
}

impl fmt::Display for DatasetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalFile(p) => write!(f, "{}", p.display()),
            _ => f.write_str(self.cache_name()),
        }
    }
}

impl FromStr for DatasetKind {
    type Err = TrainError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "wikitext2" => Ok(Self::WikiText2),
            "c4" => Ok(Self::C4),
            "c4_wiki" => Ok(Self::C4Wiki),
            "ptb" => Ok(Self::Ptb),
            "ptb_new" | "ptb-new" => Ok(Self::PtbNew),
            "c4_new" | "c4-new" => Ok(Self::C4New),
            _ => {
                let p = PathBuf::from(s);
                if p.exists() {
                    Ok(Self::LocalFile(p))
                } else {
                    Err(TrainError::InvalidArgument(format!(
                        "unknown dataset `{s}` (expected wikitext2, c4, c4_wiki, ptb, ptb_new, c4_new, or an existing file path)"
                    )))
                }
            }
        }
    }
}

/// Compute 7-character SHA-256 hex cache key from tokenizer config metadata,
/// matching `hashlib.sha256(hash_key.encode()).hexdigest()[:7]`.
#[must_use]
pub fn tokenizer_cache_key(tokenizer_descriptor: &str) -> String {
    let digest = Sha256::digest(tokenizer_descriptor.as_bytes());
    let full = hex::encode(digest);
    full[..7].to_string()
}

/// Concatenate tokenized streams into contiguous `block_size` chunks, dropping any short tail
/// remainder (exact match to `group_texts` in `utils/datautils.py`).
#[must_use]
pub fn group_token_blocks(streams: &[Vec<u32>], block_size: usize) -> Vec<Vec<u32>> {
    if block_size == 0 {
        return Vec::new();
    }
    let total_tokens: usize = streams.iter().map(Vec::len).sum();
    let usable = (total_tokens / block_size) * block_size;
    if usable == 0 {
        return Vec::new();
    }

    let mut flat = Vec::with_capacity(usable);
    for stream in streams {
        let remaining = usable - flat.len();
        if remaining == 0 {
            break;
        }
        let take = stream.len().min(remaining);
        flat.extend_from_slice(&stream[..take]);
    }
    flat.chunks_exact(block_size).map(Vec::from).collect()
}

/// Load a Hugging Face `tokenizer.json` from a local checkpoint directory or Hub model repo ID.
///
/// # Errors
/// Returns [`TrainError::Tokenizer`] or [`TrainError::Hub`] if `tokenizer.json` cannot be resolved.
pub fn load_tokenizer(
    model_id_or_path: &str,
    cache_dir: Option<&Path>,
    token: Option<&str>,
    local_files_only: bool,
) -> Result<Tokenizer> {
    let local_tok = Path::new(model_id_or_path).join("tokenizer.json");
    let tok_path = if local_tok.is_file() {
        local_tok
    } else if Path::new(model_id_or_path).is_dir() {
        return Err(TrainError::Tokenizer(format!(
            "directory `{model_id_or_path}` does not contain `tokenizer.json`"
        )));
    } else {
        let dir = resolve_model_dir(model_id_or_path, cache_dir, token, local_files_only)?;
        let cached = dir.join("tokenizer.json");
        if cached.is_file() {
            cached
        } else {
            let mut builder = HFClient::builder();
            if let Some(t) = token {
                builder = builder.token(t);
            }
            if let Some(c) = cache_dir {
                builder = builder.cache_dir(c);
            }
            let client = builder.build_sync()?;
            let (owner, name) = split_id(model_id_or_path);
            client
                .model(owner, name)
                .download_file()
                .filename("tokenizer.json")
                .local_files_only(local_files_only)
                .send()?
        }
    };

    Tokenizer::from_file(&tok_path)
        .map_err(|err| TrainError::Tokenizer(format!("failed to load tokenizer: {err}")))
}

#[derive(Debug, Serialize, Deserialize)]
struct DatasetCacheMeta {
    dataset: String,
    block_size: usize,
    num_blocks: usize,
    tokenizer_hash: String,
}

/// Persisted token-block dataset with fast binary cache round-trip (`blocks.bin` + `metadata.json`).
#[derive(Debug, Clone)]
pub struct TokenBlockDataset {
    /// Fixed-length token sequences `(num_blocks, block_size)`.
    pub blocks: Vec<Vec<u32>>,
    /// Block sequence length.
    pub block_size: usize,
    /// Cache directory if persisted.
    pub cache_dir: Option<PathBuf>,
}

impl TokenBlockDataset {
    /// Construct in-memory from raw token blocks.
    #[must_use]
    pub fn from_blocks(blocks: Vec<Vec<u32>>, block_size: usize) -> Self {
        Self {
            blocks,
            block_size,
            cache_dir: None,
        }
    }

    /// Save token blocks as little-endian `u32` chunks alongside `metadata.json` and `tokenizer_info`.
    ///
    /// # Errors
    /// Returns an I/O or serialization error if writing fails.
    pub fn save_to_disk(&self, dir: &Path, dataset_name: &str, tokenizer_info: &str) -> Result<()> {
        fs::create_dir_all(dir)?;
        let hash = tokenizer_cache_key(tokenizer_info);
        let meta = DatasetCacheMeta {
            dataset: dataset_name.to_string(),
            block_size: self.block_size,
            num_blocks: self.blocks.len(),
            tokenizer_hash: hash,
        };
        fs::write(
            dir.join("metadata.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;
        fs::write(dir.join("tokenizer_info"), tokenizer_info)?;

        let mut file = File::create(dir.join("blocks.bin"))?;
        for block in &self.blocks {
            for &id in block {
                file.write_all(&id.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Load previously cached token blocks from `dir`.
    ///
    /// # Errors
    /// Returns an error if metadata or `blocks.bin` is missing or truncated.
    pub fn load_from_disk(dir: &Path) -> Result<Self> {
        let meta_raw = fs::read_to_string(dir.join("metadata.json"))?;
        let meta: DatasetCacheMeta = serde_json::from_str(&meta_raw)?;
        let mut file = File::open(dir.join("blocks.bin"))?;
        let mut raw = Vec::new();
        file.read_to_end(&mut raw)?;

        let expected_bytes = meta.num_blocks * meta.block_size * std::mem::size_of::<u32>();
        if raw.len() != expected_bytes {
            return Err(TrainError::InvalidArgument(format!(
                "corrupted dataset cache `{}`: expected {expected_bytes} bytes, got {}",
                dir.display(),
                raw.len()
            )));
        }

        let mut blocks = Vec::with_capacity(meta.num_blocks);
        for chunk in raw.chunks_exact(meta.block_size * 4) {
            let mut block = Vec::with_capacity(meta.block_size);
            for bytes in chunk.chunks_exact(4) {
                block.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
            blocks.push(block);
        }

        Ok(Self {
            blocks,
            block_size: meta.block_size,
            cache_dir: Some(dir.to_path_buf()),
        })
    }
}

/// Extract text documents from a local `.txt`, `.jsonl`, or `.jsonl.gz` file.
///
/// # Errors
/// Returns an error if reading or gzip decoding fails.
pub fn load_local_texts(path: &Path) -> Result<Vec<String>> {
    let ext_gz = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|s| s.ends_with(".gz"));

    if ext_gz {
        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        read_lines_or_jsonl(BufReader::new(decoder), true)
    } else {
        let is_jsonl = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"));
        if is_jsonl {
            let file = File::open(path)?;
            read_lines_or_jsonl(BufReader::new(file), true)
        } else {
            let text = fs::read_to_string(path)?;
            Ok(vec![text])
        }
    }
}

fn read_lines_or_jsonl<R: BufRead>(reader: R, parse_json: bool) -> Result<Vec<String>> {
    let mut texts = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if parse_json {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if let Some(txt) = val
                    .get("text")
                    .or_else(|| val.get("sentence"))
                    .or_else(|| val.get("content"))
                    .and_then(serde_json::Value::as_str)
                {
                    if !txt.is_empty() {
                        texts.push(txt.to_string());
                    }
                    continue;
                }
            }
        }
        texts.push(trimmed.to_string());
    }
    Ok(texts)
}

fn tokenize_texts(tokenizer: &Tokenizer, texts: &[String]) -> Result<Vec<Vec<u32>>> {
    let mut streams = Vec::with_capacity(texts.len());
    for text in texts {
        let enc = tokenizer
            .encode(text.as_str(), false)
            .map_err(|err| TrainError::Tokenizer(format!("tokenization failed: {err}")))?;
        streams.push(enc.get_ids().to_vec());
    }
    Ok(streams)
}

fn fetch_hf_rows(dataset: &str, config: &str, split: &str, max_rows: usize) -> Result<Vec<String>> {
    let mut texts = Vec::new();
    let mut offset = 0usize;
    let page_size = 100usize;

    while texts.len() < max_rows {
        let length = page_size.min(max_rows - texts.len());
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
        );
        let body: serde_json::Value = ureq::get(&url)
            .call()
            .map_err(|err| TrainError::Http(format!("GET {url} failed: {err}")))?
            .body_mut()
            .read_json()
            .map_err(|err| TrainError::Http(format!("JSON decode from {url} failed: {err}")))?;

        let Some(rows) = body.get("rows").and_then(serde_json::Value::as_array) else {
            break;
        };
        if rows.is_empty() {
            break;
        }
        for row_entry in rows {
            if let Some(txt) = row_entry
                .get("row")
                .and_then(|r| r.get("text").or_else(|| r.get("sentence")))
                .and_then(serde_json::Value::as_str)
            {
                texts.push(txt.to_string());
            }
        }
        offset += rows.len();
        if rows.len() < length {
            break;
        }
    }
    Ok(texts)
}

fn fetch_c4_shard_texts(
    filename: &str,
    max_docs: usize,
    cache_dir: Option<&Path>,
    token: Option<&str>,
) -> Result<Vec<String>> {
    let mut builder = HFClient::builder();
    if let Some(t) = token {
        builder = builder.token(t);
    }
    if let Some(c) = cache_dir {
        builder = builder.cache_dir(c);
    }
    let client = builder.build_sync()?;
    let path = client
        .dataset("allenai", "c4")
        .download_file()
        .filename(filename)
        .send()?;

    let file = File::open(&path)?;
    let decoder = GzDecoder::new(file);
    let mut texts = Vec::new();
    for line in BufReader::new(decoder).lines() {
        let line = line?;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(t) = val.get("text").and_then(serde_json::Value::as_str) {
                texts.push(t.to_string());
                if texts.len() >= max_docs {
                    break;
                }
            }
        }
    }
    Ok(texts)
}

/// Prepare or load cached training token blocks matching `utils/datautils.py::prepare_dataset`.
///
/// # Errors
/// Returns an error if downloading, local reading, tokenization, or disk caching fails.
#[allow(clippy::too_many_arguments)]
pub fn prepare_dataset(
    kind: &DatasetKind,
    data_root: &Path,
    tokenizer: &Tokenizer,
    block_size: usize,
    max_remote_docs: Option<usize>,
    cache_dir: Option<&Path>,
    token: Option<&str>,
) -> Result<TokenBlockDataset> {
    let tok_desc = tokenizer
        .to_string(false)
        .map_err(|err| TrainError::Tokenizer(format!("serialize tokenizer failed: {err}")))?;
    let hash_value = tokenizer_cache_key(&tok_desc);
    let ds_dir = data_root.join(kind.cache_name()).join(&hash_value);

    if ds_dir.join("blocks.bin").is_file() && ds_dir.join("metadata.json").is_file() {
        if let Ok(ds) = TokenBlockDataset::load_from_disk(&ds_dir) {
            if ds.block_size == block_size {
                info!("Loaded cached dataset from {}", ds_dir.display());
                return Ok(ds);
            }
        }
    }

    let doc_limit = max_remote_docs.unwrap_or(usize::MAX);
    let texts = match kind {
        DatasetKind::LocalFile(p) => load_local_texts(p)?,
        DatasetKind::WikiText2 => {
            let rows = fetch_hf_rows(
                "Salesforce/wikitext",
                "wikitext-2-raw-v1",
                "train",
                doc_limit,
            )?;
            vec![rows.join("\n\n")]
        }
        DatasetKind::C4 => fetch_c4_shard_texts(
            "en/c4-train.00000-of-01024.json.gz",
            doc_limit,
            cache_dir,
            token,
        )?,
        DatasetKind::C4Wiki => {
            let mut c4 = fetch_c4_shard_texts(
                "en/c4-train.00000-of-01024.json.gz",
                doc_limit,
                cache_dir,
                token,
            )?;
            let wt = fetch_hf_rows(
                "Salesforce/wikitext",
                "wikitext-2-raw-v1",
                "train",
                doc_limit.min(50_000),
            )?;
            c4.push(wt.join("\n\n"));
            c4
        }
        DatasetKind::Ptb => {
            let rows = fetch_hf_rows("ptb_text_only", "penn_treebank", "validation", doc_limit)?;
            vec![rows.join("\n\n")]
        }
        DatasetKind::PtbNew => {
            let rows = fetch_hf_rows("ptb_text_only", "penn_treebank", "test", doc_limit)?;
            vec![rows.join(" ")]
        }
        DatasetKind::C4New => {
            let rows = fetch_c4_shard_texts(
                "en/c4-validation.00000-of-00008.json.gz",
                1100.min(doc_limit),
                cache_dir,
                token,
            )?;
            vec![rows.join(" ")]
        }
    };

    let streams = tokenize_texts(tokenizer, &texts)?;
    let blocks = group_token_blocks(&streams, block_size);
    let ds = TokenBlockDataset {
        blocks,
        block_size,
        cache_dir: Some(ds_dir.clone()),
    };
    ds.save_to_disk(&ds_dir, kind.cache_name(), &tok_desc)?;
    info!("Saved dataset cache to {}", ds_dir.display());
    Ok(ds)
}

/// Build a single contiguous evaluation token vector for perplexity measurement (`get_eval_loaders`).
///
/// # Errors
/// Returns an error if loading or tokenizing fails.
pub fn load_eval_tokens(
    kind: &DatasetKind,
    tokenizer: &Tokenizer,
    seqlen: usize,
    cache_dir: Option<&Path>,
    token: Option<&str>,
) -> Result<Vec<u32>> {
    match kind {
        DatasetKind::LocalFile(p) => {
            let texts = load_local_texts(p)?;
            let joined = texts.join("\n\n");
            let enc = tokenizer
                .encode(joined.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("tokenization failed: {err}")))?;
            Ok(enc.get_ids().to_vec())
        }
        DatasetKind::WikiText2 => {
            let rows = fetch_hf_rows("Salesforce/wikitext", "wikitext-2-raw-v1", "test", 10_000)?;
            let joined = rows.join("\n\n");
            let enc = tokenizer
                .encode(joined.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("tokenization failed: {err}")))?;
            Ok(enc.get_ids().to_vec())
        }
        DatasetKind::Ptb => {
            let rows = fetch_hf_rows("ptb_text_only", "penn_treebank", "validation", 10_000)?;
            let joined = rows.join("\n\n");
            let enc = tokenizer
                .encode(joined.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("tokenization failed: {err}")))?;
            Ok(enc.get_ids().to_vec())
        }
        DatasetKind::PtbNew => {
            let rows = fetch_hf_rows("ptb_text_only", "penn_treebank", "test", 10_000)?;
            let joined = rows.join(" ");
            let enc = tokenizer
                .encode(joined.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("tokenization failed: {err}")))?;
            Ok(enc.get_ids().to_vec())
        }
        DatasetKind::C4 => {
            let val_docs = fetch_c4_shard_texts(
                "en/c4-validation.00000-of-00008.json.gz",
                10_000,
                cache_dir,
                token,
            )?;
            let mut rng = ChaCha8Rng::seed_from_u64(0);
            let mut out = Vec::with_capacity(256 * seqlen);
            let mut attempts = 0usize;
            for _ in 0..256 {
                loop {
                    attempts += 1;
                    if attempts > 10_000 || val_docs.is_empty() {
                        break;
                    }
                    let doc_idx = rng.random_range(0..val_docs.len());
                    let enc = tokenizer
                        .encode(val_docs[doc_idx].as_str(), false)
                        .map_err(|err| {
                            TrainError::Tokenizer(format!("tokenization failed: {err}"))
                        })?;
                    let ids = enc.get_ids();
                    if ids.len() > seqlen {
                        let start = rng.random_range(0..(ids.len() - seqlen));
                        out.extend_from_slice(&ids[start..start + seqlen]);
                        break;
                    }
                }
            }
            Ok(out)
        }
        DatasetKind::C4New => {
            let val_docs = fetch_c4_shard_texts(
                "en/c4-validation.00000-of-00008.json.gz",
                1100,
                cache_dir,
                token,
            )?;
            let joined = val_docs.join(" ");
            let enc = tokenizer
                .encode(joined.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("tokenization failed: {err}")))?;
            let mut ids = enc.get_ids().to_vec();
            ids.truncate(256 * seqlen);
            Ok(ids)
        }
        DatasetKind::C4Wiki => {
            let rows = fetch_hf_rows("Salesforce/wikitext", "wikitext-2-raw-v1", "test", 10_000)?;
            let joined = rows.join("\n\n");
            let enc = tokenizer
                .encode(joined.as_str(), false)
                .map_err(|err| TrainError::Tokenizer(format!("tokenization failed: {err}")))?;
            Ok(enc.get_ids().to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn group_token_blocks_drops_short_remainder() {
        let streams = vec![vec![1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10]];
        let blocks = group_token_blocks(&streams, 4);
        assert_eq!(blocks, vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);
    }

    #[test]
    fn tokenizer_cache_key_is_deterministic_7_hex_chars() {
        let k1 = tokenizer_cache_key("LlamaTokenizer(vocab_size=32000)");
        let k2 = tokenizer_cache_key("LlamaTokenizer(vocab_size=32000)");
        let k3 = tokenizer_cache_key("LlamaTokenizer(vocab_size=128256)");
        assert_eq!(k1.len(), 7);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
        assert!(k1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn token_block_dataset_disk_cache_round_trip() -> Result<()> {
        let dir = TempDir::new().expect("tempdir");
        let blocks = vec![vec![10, 20, 30, 40], vec![50, 60, 70, 80]];
        let ds = TokenBlockDataset::from_blocks(blocks.clone(), 4);
        ds.save_to_disk(dir.path(), "wikitext2", "dummy_tok")?;
        assert!(dir.path().join("tokenizer_info").exists());

        let loaded = TokenBlockDataset::load_from_disk(dir.path())?;
        assert_eq!(loaded.block_size, 4);
        assert_eq!(loaded.blocks, blocks);
        Ok(())
    }
}
