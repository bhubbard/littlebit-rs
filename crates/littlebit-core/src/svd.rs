//! Pure-Rust thin SVD and randomized low-rank SVD backed by `faer`.

use faer::{Mat, MatRef};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{LittleBitError, Result};

/// Default subspace iteration count matching `torch.svd_lowrank(..., niter=2)`.
pub const DEFAULT_LOWRANK_NITER: usize = 2;

/// Thin singular value decomposition `A ≈ U * diag(S) * V^T` stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct ThinSvd {
    /// Left singular vectors `(rows, rank)` in row-major order.
    pub u: Vec<f32>,
    /// Non-negative singular values `(rank,)` sorted in descending order.
    pub s: Vec<f32>,
    /// Right singular vectors transposed `(rank, cols)` in row-major order.
    pub vt: Vec<f32>,
    /// Number of rows in `A`.
    pub rows: usize,
    /// Number of columns in `A`.
    pub cols: usize,
    /// Retained rank `<= min(rows, cols)`.
    pub rank: usize,
}

#[inline]
pub(crate) fn mat_from_row_major(data: &[f32], rows: usize, cols: usize) -> Result<Mat<f32>> {
    if data.len() != rows.saturating_mul(cols) {
        return Err(LittleBitError::InvalidShape {
            expected: format!("{rows}x{cols} ({} elements)", rows * cols),
            actual: format!("{} elements", data.len()),
        });
    }
    Ok(Mat::from_fn(rows, cols, |r, c| data[r * cols + c]))
}

#[inline]
pub(crate) fn mat_to_row_major(mat: MatRef<'_, f32>) -> Vec<f32> {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out.push(mat[(r, c)]);
        }
    }
    out
}

/// Compute exact thin SVD of a row-major `f32` matrix `(rows, cols)`.
///
/// # Errors
/// Returns [`LittleBitError::InvalidShape`] on dimension mismatch or
/// [`LittleBitError::SvdFailed`] if `faer` fails to converge.
pub fn exact_thin_svd(data: &[f32], rows: usize, cols: usize) -> Result<ThinSvd> {
    if rows == 0 || cols == 0 {
        return Err(LittleBitError::InvalidShape {
            expected: "non-empty matrix".to_string(),
            actual: format!("{rows}x{cols}"),
        });
    }
    let mat = mat_from_row_major(data, rows, cols)?;
    let svd = mat
        .thin_svd()
        .map_err(|err| LittleBitError::SvdFailed(format!("{err:?}")))?;

    let rank = rows.min(cols);
    let u_ref = svd.U();
    let v_ref = svd.V();
    let s_ref = svd.S().column_vector();

    let mut u = Vec::with_capacity(rows * rank);
    for r in 0..rows {
        for k in 0..rank {
            u.push(u_ref[(r, k)]);
        }
    }

    let mut s = Vec::with_capacity(rank);
    for k in 0..rank {
        s.push(s_ref[k]);
    }

    let mut vt = Vec::with_capacity(rank * cols);
    for k in 0..rank {
        for c in 0..cols {
            vt.push(v_ref[(c, k)]);
        }
    }

    Ok(ThinSvd {
        u,
        s,
        vt,
        rows,
        cols,
        rank,
    })
}

/// Compute a deterministic randomized low-rank SVD (`A ≈ U * diag(S) * V^T`),
/// matching `torch.svd_lowrank` (Halko-Martinsson-Tropp Algorithm 5.1).
///
/// Falls back to exact thin SVD when `min(rows, cols) <= 256` or when `rank + 10 >= min(rows, cols)`.
///
/// # Errors
/// Returns an error if the matrix is empty, dimensions mismatch `data.len()`, or SVD fails.
pub fn svd_lowrank(
    data: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    niter: usize,
    seed: u64,
) -> Result<ThinSvd> {
    if rows == 0 || cols == 0 || rank == 0 {
        return Err(LittleBitError::InvalidShape {
            expected: "non-zero rows, cols, and rank".to_string(),
            actual: format!("rows={rows}, cols={cols}, rank={rank}"),
        });
    }

    let min_dim = rows.min(cols);
    let target_rank = rank.min(min_dim);

    if min_dim <= 256 || target_rank + 10 >= min_dim {
        let exact = exact_thin_svd(data, rows, cols)?;
        return Ok(truncate_svd(exact, target_rank));
    }

    let sketch_dim = (target_rank + 10).min(min_dim);
    let mat = mat_from_row_major(data, rows, cols)?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let omega = Mat::from_fn(cols, sketch_dim, |_, _| -> f32 {
        StandardNormal.sample(&mut rng)
    });

    let mut y = &mat * &omega;
    let mut q = y.qr().compute_thin_Q();

    for _ in 0..niter {
        let z = mat.transpose() * &q;
        let qz = z.qr().compute_thin_Q();
        y = &mat * &qz;
        q = y.qr().compute_thin_Q();
    }

    let b = q.transpose() * &mat;
    let b_svd = b
        .thin_svd()
        .map_err(|err| LittleBitError::SvdFailed(format!("{err:?}")))?;

    let u_full = &q * b_svd.U();
    let v_ref = b_svd.V();
    let s_ref = b_svd.S().column_vector();

    let mut u = Vec::with_capacity(rows * target_rank);
    for r in 0..rows {
        for k in 0..target_rank {
            u.push(u_full[(r, k)]);
        }
    }

    let mut s = Vec::with_capacity(target_rank);
    for k in 0..target_rank {
        s.push(s_ref[k]);
    }

    let mut vt = Vec::with_capacity(target_rank * cols);
    for k in 0..target_rank {
        for c in 0..cols {
            vt.push(v_ref[(c, k)]);
        }
    }

    Ok(ThinSvd {
        u,
        s,
        vt,
        rows,
        cols,
        rank: target_rank,
    })
}

fn truncate_svd(svd: ThinSvd, target_rank: usize) -> ThinSvd {
    if target_rank >= svd.rank {
        return svd;
    }
    let mut u = Vec::with_capacity(svd.rows * target_rank);
    for r in 0..svd.rows {
        let row_start = r * svd.rank;
        u.extend_from_slice(&svd.u[row_start..row_start + target_rank]);
    }
    let s = svd.s[..target_rank].to_vec();
    let vt = svd.vt[..target_rank * svd.cols].to_vec();
    ThinSvd {
        u,
        s,
        vt,
        rows: svd.rows,
        cols: svd.cols,
        rank: target_rank,
    }
}

/// Perform rank-1 SVD factorization on `abs_data` `(rows, cols)` returning `(u_scale, v_scale)`
/// of lengths `rows` and `cols` such that `u_scale^T * v_scale ≈ abs_data`.
///
/// Matches `LittleBitLinear._rank_one_decompose`.
///
/// # Errors
/// Returns an error if `abs_data` dimensions are invalid or SVD fails.
pub fn rank_one_decompose(
    abs_data: &[f32],
    rows: usize,
    cols: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let svd = svd_lowrank(abs_data, rows, cols, 1, 4, seed)?;
    let sqrt_s0 = svd.s[0].max(0.0).sqrt();

    let mut u_comp: Vec<f32> = (0..rows).map(|r| svd.u[r * svd.rank] * sqrt_s0).collect();
    let mut v_comp: Vec<f32> = (0..cols).map(|c| svd.vt[c] * sqrt_s0).collect();

    // Canonicalize Perron-Frobenius sign so non-negative matrices yield non-negative scales.
    let u_sum: f32 = u_comp.iter().sum();
    if u_sum < 0.0 {
        for v in &mut u_comp {
            *v = -*v;
        }
        for v in &mut v_comp {
            *v = -*v;
        }
    }

    Ok((u_comp, v_comp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_thin_svd_reconstructs_rectangular_matrix() -> Result<()> {
        let rows = 6;
        let cols = 4;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 + 2) % 13) as f32 * 0.25 - 1.1)
            .collect();
        let svd = exact_thin_svd(&data, rows, cols)?;
        assert_eq!(svd.rank, 4);

        // Singular values are non-increasing and non-negative.
        for pair in svd.s.windows(2) {
            assert!(pair[0] >= pair[1] && pair[1] >= 0.0);
        }

        // Reconstruct U * diag(S) * Vt.
        let mut err_sq = 0.0f32;
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..svd.rank {
                    acc += svd.u[r * svd.rank + k] * svd.s[k] * svd.vt[k * cols + c];
                }
                let diff = data[r * cols + c] - acc;
                err_sq += diff * diff;
            }
        }
        assert!(err_sq.sqrt() < 1e-4, "reconstruction error: {err_sq}");
        Ok(())
    }

    #[test]
    fn randomized_lowrank_svd_recovers_low_rank_matrix() -> Result<()> {
        let rows = 300;
        let cols = 320;
        let true_rank = 8;

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let a: Vec<f32> = (0..rows * true_rank)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let b: Vec<f32> = (0..true_rank * cols)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let mut mat = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..true_rank {
                    acc += a[r * true_rank + k] * b[k * cols + c];
                }
                mat[r * cols + c] = acc;
            }
        }

        let svd = svd_lowrank(&mat, rows, cols, true_rank, 2, 42)?;
        let mut norm_sq = 0.0f32;
        let mut err_sq = 0.0f32;
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = 0.0f32;
                for k in 0..true_rank {
                    acc += svd.u[r * true_rank + k] * svd.s[k] * svd.vt[k * cols + c];
                }
                let target = mat[r * cols + c];
                norm_sq += target * target;
                let diff = target - acc;
                err_sq += diff * diff;
            }
        }
        let rel_err = (err_sq / norm_sq).sqrt();
        assert!(rel_err < 1e-4, "relative lowrank error too high: {rel_err}");
        Ok(())
    }

    #[test]
    fn rank_one_decompose_produces_non_negative_factors() -> Result<()> {
        let data: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
        let (u1, v1) = rank_one_decompose(&data, 3, 4, 7)?;
        assert_eq!(u1.len(), 3);
        assert_eq!(v1.len(), 4);
        assert!(u1.iter().all(|&x| x >= 0.0));
        assert!(v1.iter().all(|&x| x >= 0.0));
        Ok(())
    }
}
