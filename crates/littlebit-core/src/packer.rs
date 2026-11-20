//! LSB-first 32-bit binary matrix packing, unpacking, and compact 1-bit matmul.

use candle_core::{DType, Device, Tensor};
use rayon::prelude::*;

use crate::binary::binarize_sign_no_grad;
use crate::error::{LittleBitError, Result};

/// Row-major 1-bit packed matrix with values in `{-1, +1}`.
///
/// Bit encoding follows upstream `quantization/utils/binary_packer.py` (`lsb_first`):
/// - `+1` maps to bit `0`
/// - `-1` maps to bit `1`
/// - Rows are padded on the right to multiples of 32 with `+1` (bit `0`)
/// - Each 32-bit chunk is stored as a signed `i32` word (LSB = lowest column index)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedBinaryMatrix {
    rows: usize,
    cols: usize,
    words_per_row: usize,
    data: Vec<i32>,
}

impl PackedBinaryMatrix {
    /// Construct from pre-packed row-major 32-bit words after validating dimensions.
    ///
    /// # Errors
    /// Returns [`LittleBitError::InvalidShape`] when `data.len() != rows * cols.div_ceil(32)`.
    pub fn from_words(rows: usize, cols: usize, data: Vec<i32>) -> Result<Self> {
        let words_per_row = cols.div_ceil(32);
        let expected_len = rows.saturating_mul(words_per_row);
        if data.len() != expected_len {
            return Err(LittleBitError::InvalidShape {
                expected: format!("{rows}x{words_per_row} ({expected_len} words)"),
                actual: format!("{} words", data.len()),
            });
        }
        Ok(Self {
            rows,
            cols,
            words_per_row,
            data,
        })
    }

    /// Number of rows in the original unpacked matrix.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns in the original unpacked matrix.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of `i32` words stored per row (`cols.div_ceil(32)`).
    #[must_use]
    pub fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    /// Borrow the underlying row-major `i32` word buffer.
    #[must_use]
    pub fn words(&self) -> &[i32] {
        &self.data
    }

    /// Heap memory footprint in bytes of the packed word storage.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<i32>()
    }

    /// Unpack back to row-major `Vec<i8>` containing only `-1` and `+1`.
    #[must_use]
    pub fn unpack_to_i8(&self) -> Vec<i8> {
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            let row_words = &self.data[row * self.words_per_row..(row + 1) * self.words_per_row];
            for col in 0..self.cols {
                let word = row_words[col / 32] as u32;
                let bit = (word >> (col % 32)) & 1;
                out.push(if bit == 0 { 1 } else { -1 });
            }
        }
        out
    }

    /// Compute `Y = X @ W^T` where `X` is `(batch_rows, cols)` in row-major `f32`
    /// and `W` (`self`) is `(rows, cols)` packed at 1 bit per weight.
    ///
    /// Returns a row-major `Vec<f32>` of shape `(batch_rows, rows)` without unpacking `W`.
    ///
    /// # Errors
    /// Returns [`LittleBitError::InvalidShape`] when `x.len() != batch_rows * self.cols`.
    pub fn matmul_transposed_f32(&self, x: &[f32], batch_rows: usize) -> Result<Vec<f32>> {
        let expected_len = batch_rows.saturating_mul(self.cols);
        if x.len() != expected_len {
            return Err(LittleBitError::InvalidShape {
                expected: format!("{batch_rows}x{} ({expected_len} elements)", self.cols),
                actual: format!("{} elements", x.len()),
            });
        }
        if batch_rows == 0 || self.rows == 0 {
            return Ok(vec![0.0; batch_rows * self.rows]);
        }

        let mut output = vec![0.0f32; batch_rows * self.rows];
        let cols = self.cols;
        let words_per_row = self.words_per_row;
        let words = &self.data;

        for (b, out_row) in output.chunks_exact_mut(self.rows).enumerate() {
            let x_row = &x[b * cols..(b + 1) * cols];
            let x_total: f64 = x_row.iter().map(|&v| f64::from(v)).sum();

            let compute_row = |w_row_idx: usize| -> f32 {
                let w_words = &words[w_row_idx * words_per_row..(w_row_idx + 1) * words_per_row];
                let mut neg_sum = 0.0_f64;
                for (w_idx, &word_i32) in w_words.iter().enumerate() {
                    let mut w = word_i32 as u32;
                    let col_base = w_idx * 32;
                    let valid_cols = (cols - col_base).min(32);
                    if valid_cols < 32 {
                        w &= (1u32 << valid_cols) - 1;
                    }
                    let x_chunk = &x_row[col_base..col_base + valid_cols];
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        neg_sum += f64::from(x_chunk[bit]);
                        w &= w - 1;
                    }
                }
                (x_total - 2.0 * neg_sum) as f32
            };

            if self.rows >= 64 {
                out_row.par_iter_mut().enumerate().for_each(|(r, dst)| {
                    *dst = compute_row(r);
                });
            } else {
                for (r, dst) in out_row.iter_mut().enumerate() {
                    *dst = compute_row(r);
                }
            }
        }

        Ok(output)
    }

    /// Convert packed storage into `(packed_i32, shape_i64)` tensors matching upstream safetensors.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn to_tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let packed = Tensor::from_slice(&self.data, (self.rows, self.words_per_row), device)?;
        let shape = Tensor::from_slice(&[self.rows as i64, self.cols as i64], 2, device)?;
        Ok((packed, shape))
    }
}

/// Pack a row-major `i8` matrix containing only `-1` and `+1` into [`PackedBinaryMatrix`].
///
/// # Errors
/// Returns [`LittleBitError::InvalidShape`] when `values.len() != rows * cols`, or
/// [`LittleBitError::InvalidBinaryValue`] if any element is not `-1` or `+1`.
pub fn pack_i8_slice(values: &[i8], rows: usize, cols: usize) -> Result<PackedBinaryMatrix> {
    let expected = rows.saturating_mul(cols);
    if values.len() != expected {
        return Err(LittleBitError::InvalidShape {
            expected: format!("{rows}x{cols} ({expected} elements)"),
            actual: format!("{} elements", values.len()),
        });
    }

    let words_per_row = cols.div_ceil(32);
    let mut data = vec![0i32; rows * words_per_row];

    for row in 0..rows {
        let row_in = &values[row * cols..(row + 1) * cols];
        let row_out = &mut data[row * words_per_row..(row + 1) * words_per_row];
        for (col, &val) in row_in.iter().enumerate() {
            match val {
                1 => {}
                -1 => {
                    let word_idx = col / 32;
                    let bit_idx = col % 32;
                    row_out[word_idx] = ((row_out[word_idx] as u32) | (1u32 << bit_idx)) as i32;
                }
                other => {
                    return Err(LittleBitError::InvalidBinaryValue {
                        row,
                        col,
                        value: other,
                    });
                }
            }
        }
    }

    Ok(PackedBinaryMatrix {
        rows,
        cols,
        words_per_row,
        data,
    })
}

/// Unpack a raw slice of row-major 32-bit words back to `{-1, +1}` `Vec<i8>`.
///
/// # Errors
/// Returns [`LittleBitError::InvalidShape`] if `words.len() != rows * cols.div_ceil(32)`.
pub fn unpack_i32_slice(words: &[i32], rows: usize, cols: usize) -> Result<Vec<i8>> {
    let mat = PackedBinaryMatrix::from_words(rows, cols, words.to_vec())?;
    Ok(mat.unpack_to_i8())
}

/// Binarize a 2D Candle tensor (`>= 0 => +1`, `< 0 => -1`) and pack it into
/// `(packed_i32, shape_i64)` tensors matching upstream `binary_packer`.
///
/// # Errors
/// Returns an error if `tensor` is not 2D or cannot be converted to host values.
pub fn binary_packer(tensor: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(LittleBitError::InvalidShape {
            expected: "2D tensor".to_string(),
            actual: format!("{dims:?}"),
        });
    }
    let (rows, cols) = (dims[0], dims[1]);
    let bin = binarize_sign_no_grad(&tensor.to_dtype(DType::F32)?)?;
    let host = bin.to_device(&Device::Cpu)?.to_vec2::<f32>()?;

    let mut i8_buf = Vec::with_capacity(rows * cols);
    for row_vals in host {
        for v in row_vals {
            i8_buf.push(if v < 0.0 { -1i8 } else { 1i8 });
        }
    }

    let packed = pack_i8_slice(&i8_buf, rows, cols)?;
    packed.to_tensors(tensor.device())
}

/// Unpack a 2D `I32` packed Candle tensor back into a `{-1, +1}` tensor of `target_dtype` on `device`.
///
/// Matches upstream `binary_unpacker`.
///
/// # Errors
/// Returns an error if `packed` is not a 2D `I32` tensor matching `(rows, cols.div_ceil(32))`.
pub fn binary_unpacker(
    packed: &Tensor,
    original_shape: (usize, usize),
    target_dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if packed.dtype() != DType::I32 {
        return Err(LittleBitError::UnsupportedDtype(format!(
            "expected I32 packed tensor, got {:?}",
            packed.dtype()
        )));
    }
    let dims = packed.dims();
    let (rows, cols) = original_shape;
    let words_per_row = cols.div_ceil(32);
    if dims != [rows, words_per_row] {
        return Err(LittleBitError::InvalidShape {
            expected: format!("[{rows}, {words_per_row}]"),
            actual: format!("{dims:?}"),
        });
    }

    let words = packed
        .to_device(&Device::Cpu)?
        .reshape(rows * words_per_row)?
        .to_vec1::<i32>()?;
    let i8_vals = unpack_i32_slice(&words, rows, cols)?;
    let f32_vals: Vec<f32> = i8_vals.into_iter().map(f32::from).collect();
    let t = Tensor::from_vec(f32_vals, (rows, cols), device)?;
    if target_dtype == DType::F32 {
        Ok(t)
    } else {
        Ok(t.to_dtype(target_dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bit_patterns_match_upstream_lsb_first() -> Result<()> {
        // 32 ones -> 0i32
        let ones = vec![1i8; 32];
        assert_eq!(pack_i8_slice(&ones, 1, 32)?.words(), &[0i32]);

        // 32 minus-ones -> all bits set -> -1i32
        let neg_ones = vec![-1i8; 32];
        assert_eq!(pack_i8_slice(&neg_ones, 1, 32)?.words(), &[-1i32]);

        // Single -1 at bit 31 -> 0x8000_0000_u32 as i32 == i32::MIN
        let mut bit31 = vec![1i8; 32];
        bit31[31] = -1;
        assert_eq!(pack_i8_slice(&bit31, 1, 32)?.words(), &[i32::MIN]);

        // Alternating +1 (even idx -> bit 0) and -1 (odd idx -> bit 1) => 0xAAAA_AAAA
        let alt: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        assert_eq!(
            pack_i8_slice(&alt, 1, 32)?.words(),
            &[0xAAAA_AAAA_u32 as i32]
        );
        Ok(())
    }

    #[test]
    fn round_trip_across_edge_widths() -> Result<()> {
        for cols in [1usize, 7, 8, 31, 32, 33, 63, 64, 65, 100] {
            let rows = 5;
            let vals: Vec<i8> = (0..rows * cols)
                .map(|k| if (k * 7 + 3) % 3 == 0 { -1 } else { 1 })
                .collect();
            let packed = pack_i8_slice(&vals, rows, cols)?;
            assert_eq!(packed.unpack_to_i8(), vals, "width={cols}");
        }
        Ok(())
    }

    #[test]
    fn rejects_non_binary_values() {
        let err = pack_i8_slice(&[1, -1, 0, 1], 2, 2).expect_err("zero must fail");
        assert!(matches!(
            err,
            LittleBitError::InvalidBinaryValue {
                row: 1,
                col: 0,
                value: 0
            }
        ));
    }

    #[test]
    fn packed_matmul_matches_dense_matmul_exactly() -> Result<()> {
        let rows = 71;
        let cols = 100;
        let batch = 4;
        let w_i8: Vec<i8> = (0..rows * cols)
            .map(|k| if (k * 13 + 5) % 5 < 2 { -1 } else { 1 })
            .collect();
        let packed = pack_i8_slice(&w_i8, rows, cols)?;

        let x: Vec<f32> = (0..batch * cols)
            .map(|k| ((k % 19) as f32 - 9.0) * 0.13)
            .collect();
        let got = packed.matmul_transposed_f32(&x, batch)?;

        let dev = Device::Cpu;
        let x_t = Tensor::from_slice(&x, (batch, cols), &dev)?;
        let w_f32: Vec<f32> = w_i8.iter().map(|&v| f32::from(v)).collect();
        let w_t = Tensor::from_slice(&w_f32, (rows, cols), &dev)?;
        let want = x_t.matmul(&w_t.t()?)?.flatten_all()?.to_vec1::<f32>()?;

        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-4, "idx {i}: got {g}, want {w}");
        }
        Ok(())
    }
}
