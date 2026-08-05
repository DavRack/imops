//! Eberhard / adjacency effect via reaction–diffusion unsharp mask.
//!
//! D' = D + β * (D − (D ⊛ G_σ))
//! σ = developer_diffusion_length in pixels; β = exhaustion sensitivity.
//! Derived from reaction–diffusion, not a creative sharpening tool.

use crate::film::blur::gaussian_blur_separable;
use crate::film::types::DyePlanes;

/// Apply adjacency correction to image dye planes.
pub fn apply_adjacency(dyes: &mut DyePlanes, sigma_px: f32, beta: f32) {
    if beta.abs() < 1e-8 || sigma_px < 1e-3 {
        return;
    }
    let width = dyes.width;
    let height = dyes.height;
    for plane in dyes.image_dye.iter_mut() {
        let mut blurred = plane.clone();
        gaussian_blur_separable(&mut blurred, width, height, sigma_px);
        for (d, b) in plane.iter_mut().zip(blurred.iter()) {
            *d += beta * (*d - b);
        }
    }
}

/// Apply DIR (Development Inhibitor Releasing) coupler interlayer chemical inhibition.
///
/// Developing silver halide in emulsion layer `i` releases inhibitor `I_i(x,y)`,
/// which is blurred by σ_dir = `dir_diffusion_length / pitch`.
///
/// **Matrix layout:** `matrix[source][target]` (row = source emulsion `i` releasing inhibitor,
/// column = target emulsion `j` receiving inhibition). Target `j` receives
/// `I_total,j = Σ_i matrix[i][j] · (I_i ⊛ G_σ)` and its image dye is scaled by
/// `exp(−I_total,j)`.
pub fn apply_dir_inhibition(dyes: &mut DyePlanes, sigma_dir_px: f32, matrix: &[Vec<f32>]) {
    let num_emulsions = dyes.image_dye.len();
    if num_emulsions == 0 || matrix.is_empty() || sigma_dir_px < 1e-3 {
        return;
    }
    let width = dyes.width;
    let height = dyes.height;
    let n = width * height;

    let mut diffused_inhibitors = Vec::with_capacity(num_emulsions);
    for i in 0..num_emulsions {
        let mut inh = dyes.image_dye[i].clone();
        gaussian_blur_separable(&mut inh, width, height, sigma_dir_px);
        diffused_inhibitors.push(inh);
    }

    for j in 0..num_emulsions {
        let mut delta_inhibition = vec![0.0f32; n];
        let mut active = false;

        for i in 0..num_emulsions {
            if i < matrix.len() && j < matrix[i].len() {
                let weight = matrix[i][j]; // row i = source, col j = target
                if weight.abs() > 1e-6 {
                    active = true;
                    let inh_diffused = &diffused_inhibitors[i];
                    let inh_local = &dyes.image_dye[i];
                    for p in 0..n {
                        delta_inhibition[p] += weight * (inh_diffused[p] - inh_local[p]);
                    }
                }
            }
        }

        if active {
            let target_plane = &mut dyes.image_dye[j];
            for p in 0..n {
                let factor = (-delta_inhibition[p]).exp();
                target_plane[p] *= factor;
            }
        }
    }
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
    fn dir_matrix_is_source_row_target_column() {
        // Asymmetric coupling: source 1 → target 0 (matrix[1][0] = w).
        let n = 16 * 16;
        let mut dyes = DyePlanes {
            width: 16,
            height: 16,
            image_dye: vec![vec![1.0f32; n], vec![1.0f32; n]],
            mask_dye: vec![vec![0.0; n], vec![0.0; n]],
        };
        // Step edge in layer 1
        for y in 0..16 {
            for x in 0..16 {
                dyes.image_dye[1][y * 16 + x] = if x < 8 { 0.2 } else { 1.0 };
            }
        }
        let w = 0.5f32;
        let matrix = vec![vec![0.0, 0.0], vec![w, 0.0]]; // matrix[source 1][target 0]
        apply_dir_inhibition(&mut dyes, 2.0, &matrix);
        let got0_edge = dyes.image_dye[0][8 * 16 + 8];
        let got1 = dyes.image_dye[1][8 * 16 + 8];
        assert!(
            got0_edge != 1.0,
            "target 0 should be inhibited on edge by source 1"
        );
        assert_eq!(
            got1, 1.0,
            "target 1 must be unchanged under matrix[1][0]-only coupling"
        );
    }

    #[test]
    fn dir_flat_field_preserves_hd_curve() {
        let n = 16 * 16;
        let mut dyes = DyePlanes {
            width: 16,
            height: 16,
            image_dye: vec![vec![0.8f32; n], vec![0.5f32; n]],
            mask_dye: vec![vec![0.0; n], vec![0.0; n]],
        };
        let matrix = vec![vec![0.2, 0.4], vec![0.3, 0.1]];
        apply_dir_inhibition(&mut dyes, 2.0, &matrix);
        // Flat field should remain unchanged because diffused == local
        for p in 0..n {
            assert!((dyes.image_dye[0][p] - 0.8).abs() < 1e-5);
            assert!((dyes.image_dye[1][p] - 0.5).abs() < 1e-5);
        }
    }
}
