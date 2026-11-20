//! LittleBit-2 Joint Iterative Quantization (Joint-ITQ) latent rotation.

use faer::Mat;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{LittleBitError, Result};
use crate::svd::{mat_from_row_major, mat_to_row_major};

/// Generate a Haar-distributed orthogonal `(dim, dim)` matrix via QR with positive `R` diagonal.
#[must_use]
pub fn random_orthogonal_matrix(dim: usize, seed: u64) -> Mat<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let gaussian = Mat::from_fn(dim, dim, |_, _| -> f32 { StandardNormal.sample(&mut rng) });
    let qr = gaussian.qr();
    let mut q = qr.compute_thin_Q();
    let r = qr.R();
    for j in 0..dim {
        let diag = r[(j, j)];
        let sign = if diag < 0.0 { -1.0 } else { 1.0 };
        if sign < 0.0 {
            for i in 0..dim {
                q[(i, j)] *= sign;
            }
        }
    }
    q
}

/// Find orthogonal rotation `R` of shape `(dim, dim)` aligning `X` (`(rows, dim)`)
/// to the binary hypercube: `min_R || sign(X @ R) - X @ R ||_F^2`.
///
/// Exact Rust port of `LittleBitLinear._compute_itq_rotation`.
///
/// # Errors
/// Returns an error if matrix dimensions are invalid or inner Procrustes SVD fails.
pub fn compute_itq_rotation(
    x_combined: &[f32],
    rows: usize,
    dim: usize,
    n_iter: usize,
    seed: u64,
) -> Result<Vec<f32>> {
    if rows == 0 || dim == 0 {
        return Err(LittleBitError::InvalidShape {
            expected: "non-empty matrix".to_string(),
            actual: format!("{rows}x{dim}"),
        });
    }
    let x = mat_from_row_major(x_combined, rows, dim)?;
    let mut r = random_orthogonal_matrix(dim, seed);

    for _ in 0..n_iter {
        let z = &x * &r;
        let b = Mat::from_fn(rows, dim, |i, j| {
            let v = z[(i, j)];
            if v > 0.0 {
                1.0f32
            } else if v < 0.0 {
                -1.0f32
            } else {
                0.0f32
            }
        });
        let m = b.transpose() * &x;
        let svd = m
            .thin_svd()
            .map_err(|err| LittleBitError::SvdFailed(format!("ITQ SVD failed: {err:?}")))?;
        // M = U_p * S * V_p^T => R = V_p * U_p^T (exact Procrustes solution).
        r = svd.V() * svd.U().transpose();
    }

    Ok(mat_to_row_major(r.as_ref()))
}

/// Jointly align SVD factors `U` `(out_features, split_dim)` and `V` `(split_dim, in_features)`
/// via `X_combined = [U; V^T]`, updating `U <- U @ R` and `V <- R^T @ V`.
///
/// # Errors
/// Returns an error if factor slice sizes do not match dimensions or Joint-ITQ fails.
pub fn apply_joint_itq(
    u: &mut [f32],
    v: &mut [f32],
    out_features: usize,
    in_features: usize,
    split_dim: usize,
    n_iter: usize,
    seed: u64,
) -> Result<()> {
    if u.len() != out_features.saturating_mul(split_dim) {
        return Err(LittleBitError::InvalidShape {
            expected: format!("{out_features}x{split_dim}"),
            actual: format!("{} elements", u.len()),
        });
    }
    if v.len() != split_dim.saturating_mul(in_features) {
        return Err(LittleBitError::InvalidShape {
            expected: format!("{split_dim}x{in_features}"),
            actual: format!("{} elements", v.len()),
        });
    }

    let total_rows = out_features + in_features;
    let mut x_combined = Vec::with_capacity(total_rows * split_dim);
    x_combined.extend_from_slice(u);
    for c in 0..in_features {
        for r in 0..split_dim {
            x_combined.push(v[r * in_features + c]);
        }
    }

    let r_buf = compute_itq_rotation(&x_combined, total_rows, split_dim, n_iter, seed)?;
    let r_mat = mat_from_row_major(&r_buf, split_dim, split_dim)?;

    let u_mat = mat_from_row_major(u, out_features, split_dim)?;
    let u_rot = &u_mat * &r_mat;
    u.copy_from_slice(&mat_to_row_major(u_rot.as_ref()));

    let v_mat = mat_from_row_major(v, split_dim, in_features)?;
    let v_rot = r_mat.transpose() * &v_mat;
    v.copy_from_slice(&mat_to_row_major(v_rot.as_ref()));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alignment_error(x: &[f32], r: &[f32], rows: usize, dim: usize) -> f32 {
        let mut total = 0.0f32;
        for i in 0..rows {
            for j in 0..dim {
                let mut z = 0.0f32;
                for k in 0..dim {
                    z += x[i * dim + k] * r[k * dim + j];
                }
                let b = if z >= 0.0 { 1.0 } else { -1.0 };
                let d = b - z;
                total += d * d;
            }
        }
        total
    }

    #[test]
    fn rotation_is_orthogonal_and_preserves_product() -> Result<()> {
        let out_f = 24;
        let in_f = 16;
        let rank = 8;

        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let u0: Vec<f32> = (0..out_f * rank)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let v0: Vec<f32> = (0..rank * in_f)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let r_buf = compute_itq_rotation(&u0, out_f, rank, 30, 5)?;
        let r_mat = mat_from_row_major(&r_buf, rank, rank)?;
        let rtr = r_mat.transpose() * &r_mat;
        for i in 0..rank {
            for j in 0..rank {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[(i, j)] - want).abs() < 1e-4, "R^T R[{i},{j}]");
            }
        }

        let mut u = u0.clone();
        let mut v = v0.clone();
        apply_joint_itq(&mut u, &mut v, out_f, in_f, rank, 50, 11)?;

        // Check U0 @ V0 == U @ V.
        let u0_mat = mat_from_row_major(&u0, out_f, rank)?;
        let v0_mat = mat_from_row_major(&v0, rank, in_f)?;
        let prod0 = &u0_mat * &v0_mat;

        let u_mat = mat_from_row_major(&u, out_f, rank)?;
        let v_mat = mat_from_row_major(&v, rank, in_f)?;
        let prod1 = &u_mat * &v_mat;

        let mut max_diff = 0.0f32;
        for i in 0..out_f {
            for j in 0..in_f {
                max_diff = max_diff.max((prod0[(i, j)] - prod1[(i, j)]).abs());
            }
        }
        assert!(max_diff < 1e-4, "product drift: {max_diff}");
        Ok(())
    }

    #[test]
    fn joint_itq_improves_binary_hypercube_alignment() -> Result<()> {
        let rows = 64;
        let dim = 16;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        // Scale down lightly so binarization error responds to rotation.
        let x: Vec<f32> = (0..rows * dim)
            .map(|_| {
                let s: f32 = StandardNormal.sample(&mut rng);
                s * 0.5
            })
            .collect();
        // Skewed columns concentrate energy on axis 0.
        let mut skewed = x;
        for i in 0..rows {
            skewed[i * dim] *= 4.0;
        }
        let mut eye = vec![0.0f32; dim * dim];
        for i in 0..dim {
            eye[i * dim + i] = 1.0;
        }
        let before = alignment_error(&skewed, &eye, rows, dim);
        let r = compute_itq_rotation(&skewed, rows, dim, 50, 7)?;
        let after = alignment_error(&skewed, &r, rows, dim);
        assert!(
            after < before,
            "ITQ should reduce hypercube alignment loss: before={before}, after={after}"
        );
        Ok(())
    }
}
