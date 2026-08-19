//! Eberhard / adjacency effect and cross-layer inhibitor diffusion via reaction–diffusion unsharp mask.
//!
//! Developer exhaustion and inhibitor release (DIR) depend on the local developed
//! dye field. Adjacency and interlayer inhibitor coupling are applied to the
//! realized image-bearing planes (particle realizations convolved with dye cloud PSF):
//!
//! ```text
//! D_i' = D_i + sum_j M_ij * (D_j − (D_j ⊛ G_σ))
//! ```
//!
//! where:
//! - M_ii = β_self (intra-layer acutance / exhaustion)
//! - M_ij = β_record for fast/slow layers of the same color record
//! - M_ij = β_cross for layers across different color records (Y <-> M <-> C)
//! - σ = developer_diffusion_length in pixels.
//!
//! Because the spatial expectation of (D_j − (D_j ⊛ G_σ)) is zero, total mean
//! density is conserved: ⟨D_i'⟩ = ⟨D_i⟩.

use crate::film::blur::gaussian_blur_separable;
use crate::film::types::DyePlanes;
use rayon::prelude::*;

/// Apply Eberhard adjacency correction to realized image dye planes (diagonal intra-layer only).
pub fn apply_adjacency(dyes: &mut DyePlanes, sigma_px: f32, beta: f32) {
    if beta.abs() < 1e-8 || sigma_px < 1e-3 || dyes.image_dye.is_empty() {
        return;
    }
    let n = dyes.image_dye.len();
    let mut matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        matrix[i][i] = beta;
    }
    apply_cross_layer_adjacency(dyes, sigma_px, &matrix);
}

/// Apply cross-layer inhibitor diffusion and adjacency correction to realized image dye planes.
///
/// For each layer `i`, the updated density is:
/// ```text
/// D_i' = D_i + sum_j M_ij * (D_j - G_sigma * D_j)
/// ```
/// where `M_ij` is the coupling matrix between layer `i` and layer `j`.
pub fn apply_cross_layer_adjacency(
    dyes: &mut DyePlanes,
    sigma_px: f32,
    coupling_matrix: &[Vec<f32>],
) {
    if sigma_px < 1e-3 || dyes.image_dye.is_empty() || coupling_matrix.is_empty() {
        return;
    }
    let is_all_zero = coupling_matrix
        .iter()
        .all(|row| row.iter().all(|&val| val.abs() < 1e-8));
    if is_all_zero {
        return;
    }
    let width = dyes.width;
    let height = dyes.height;
    let n = dyes.image_dye.len();
    assert_eq!(
        coupling_matrix.len(),
        n,
        "coupling matrix row count must match image_dye planes"
    );

    // Compute Delta_j = D_j - G_sigma * D_j for each layer j.
    let deltas: Vec<Vec<f32>> = dyes
        .image_dye
        .par_iter()
        .map(|plane| {
            let mut blurred = plane.clone();
            gaussian_blur_separable(&mut blurred, width, height, sigma_px);
            plane
                .iter()
                .zip(blurred.iter())
                .map(|(&d, &b)| d - b)
                .collect()
        })
        .collect();

    // D_i' = D_i + sum_j M_ij * Delta_j
    dyes.image_dye
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, plane)| {
            let row = &coupling_matrix[i];
            let num_pixels = plane.len();
            for p in 0..num_pixels {
                let mut sum = 0.0f32;
                for (j, &m_ij) in row.iter().enumerate() {
                    if m_ij.abs() > 1e-8 && j < deltas.len() {
                        sum += m_ij * deltas[j][p];
                    }
                }
                plane[p] += sum;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacency_step_edge() {
        let width = 64;
        let height = 32;
        let mut dyes = DyePlanes {
            width,
            height,
            image_dye: vec![vec![0.0f32; width * height]],
            mask_dye: vec![vec![0.0f32; width * height]],
        };
        // Horizontal step: left dark, right light.
        for y in 0..height {
            for x in 0..width {
                dyes.image_dye[0][y * width + x] = if x < width / 2 { 0.2 } else { 1.0 };
            }
        }
        let edge = width / 2;
        let before_light = dyes.image_dye[0][height / 2 * width + edge];
        let before_dark = dyes.image_dye[0][height / 2 * width + edge - 1];
        apply_adjacency(&mut dyes, 3.0, 0.5);
        let after_light = dyes.image_dye[0][height / 2 * width + edge];
        let after_dark = dyes.image_dye[0][height / 2 * width + edge - 1];
        // Light side near edge gains density; dark side loses.
        assert!(
            after_light > before_light,
            "light side should gain: before={before_light} after={after_light}"
        );
        assert!(
            after_dark < before_dark,
            "dark side should lose: before={before_dark} after={after_dark}"
        );
    }

    #[test]
    fn adjacency_flag_off() {
        let mut dyes = DyePlanes {
            width: 16,
            height: 16,
            image_dye: vec![(0..256).map(|i| (i % 16) as f32 / 16.0).collect()],
            mask_dye: vec![vec![0.0; 256]],
        };
        let before = dyes.image_dye[0].clone();
        apply_adjacency(&mut dyes, 2.0, 0.0);
        assert_eq!(dyes.image_dye[0], before);
    }

    #[test]
    fn adjacency_beta_zero() {
        let mut dyes = DyePlanes {
            width: 16,
            height: 16,
            image_dye: vec![vec![0.5f32; 256]],
            mask_dye: vec![vec![0.0; 256]],
        };
        let before = dyes.image_dye.clone();
        apply_adjacency(&mut dyes, 5.0, 0.0);
        assert_eq!(dyes.image_dye, before);
    }

    #[test]
    fn adjacency_classic_eberhard_formula() {
        let width = 32;
        let height = 16;
        let n = width * height;
        let mut dyes = DyePlanes {
            width,
            height,
            image_dye: vec![(0..n).map(|i| (i % width) as f32 / width as f32).collect()],
            mask_dye: vec![vec![0.0; n]],
        };
        let reference = dyes.clone();
        let sigma = 2.5f32;
        let beta = 0.35f32;
        apply_adjacency(&mut dyes, sigma, beta);

        let mut blurred = reference.image_dye[0].clone();
        crate::film::blur::gaussian_blur_separable(&mut blurred, width, height, sigma);
        for (got, (&d, &b)) in dyes.image_dye[0]
            .iter()
            .zip(reference.image_dye[0].iter().zip(blurred.iter()))
        {
            let expected = d + beta * (d - b);
            assert!(
                (got - expected).abs() < 1e-5,
                "got={got} expected={expected}"
            );
        }
    }

    #[test]
    fn cross_layer_adjacency_coupling_two_layers() {
        let width = 64;
        let height = 32;
        let n = width * height;
        // Layer 0 has step edge; Layer 1 is flat.
        let mut l0 = vec![0.0f32; n];
        let l1 = vec![0.5f32; n];
        for y in 0..height {
            for x in 0..width {
                l0[y * width + x] = if x < width / 2 { 0.2 } else { 0.8 };
            }
        }
        let mut dyes = DyePlanes {
            width,
            height,
            image_dye: vec![l0.clone(), l1.clone()],
            mask_dye: vec![vec![0.0; n]; 2],
        };

        // Coupling matrix: Layer 0 has self beta=0.4; Layer 1 gets cross-coupled beta=0.2 from Layer 0.
        let matrix = vec![vec![0.4, 0.0], vec![0.2, 0.3]];
        let sigma = 3.0f32;
        apply_cross_layer_adjacency(&mut dyes, sigma, &matrix);

        let edge = width / 2;
        // Layer 1 was initially flat at 0.5. Near the step edge of Layer 0, Layer 1 should be modulated.
        let l1_near_light = dyes.image_dye[1][height / 2 * width + edge];
        let l1_near_dark = dyes.image_dye[1][height / 2 * width + edge - 1];
        assert!(
            l1_near_light > 0.5,
            "layer 1 near layer 0's light side should receive positive boost: got {l1_near_light}"
        );
        assert!(
            l1_near_dark < 0.5,
            "layer 1 near layer 0's dark side should receive depression: got {l1_near_dark}"
        );

        // Verify mean density of Layer 1 is preserved (since <Delta_0> = 0 and <Delta_1> = 0).
        let mean_l1_after = dyes.image_dye[1].iter().sum::<f32>() / n as f32;
        assert!(
            (mean_l1_after - 0.5).abs() < 1e-5,
            "layer 1 mean density must be strictly conserved: got {mean_l1_after}"
        );
    }

    #[test]
    fn cross_layer_adjacency_mean_conservation() {
        let width = 48;
        let height = 48;
        let n = width * height;
        let num_layers = 6;
        let mut image_dye = Vec::new();
        for l in 0..num_layers {
            let mut plane = vec![0.0f32; n];
            for y in 0..height {
                for x in 0..width {
                    plane[y * width + x] = 0.1 * (l as f32 + 1.0)
                        + 0.3 * ((x * y + l * 13) % 17) as f32 / 17.0;
                }
            }
            image_dye.push(plane);
        }
        let mut dyes = DyePlanes {
            width,
            height,
            image_dye,
            mask_dye: vec![vec![0.0; n]; num_layers],
        };
        let before_means: Vec<f64> = dyes
            .image_dye
            .iter()
            .map(|plane| plane.iter().map(|&x| x as f64).sum::<f64>() / n as f64)
            .collect();

        // 6x6 coupling matrix with self, record, and cross terms.
        let mut matrix = vec![vec![0.08f32; num_layers]; num_layers];
        for i in 0..num_layers {
            matrix[i][i] = 0.5; // self
            let pair = if i % 2 == 0 { i + 1 } else { i - 1 };
            matrix[i][pair] = 0.2; // same record
        }

        apply_cross_layer_adjacency(&mut dyes, 4.0, &matrix);

        let after_means: Vec<f64> = dyes
            .image_dye
            .iter()
            .map(|plane| plane.iter().map(|&x| x as f64).sum::<f64>() / n as f64)
            .collect();

        for (i, (&before, &after)) in before_means.iter().zip(after_means.iter()).enumerate() {
            assert!(
                (before - after).abs() < 2e-3,
                "layer {i} mean must be conserved within boundary reflection precision: before={before}, after={after}"
            );
        }
    }
}
