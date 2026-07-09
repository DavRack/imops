use rayon::prelude::*;

use crate::pixel::{R_RELATIVE_LUMINANCE, G_RELATIVE_LUMINANCE, B_RELATIVE_LUMINANCE};

/// Luma-guided chroma denoising via the guided filter (He et al. 2013).
///
/// Converts the image to a luminance–chroma difference representation
/// (`C_R = R - Y`, `C_B = B - Y`), smooths each chroma plane with a
/// guided filter using the luminance as the edge-preserving guide,
/// then reconstructs the RGB output.
///
/// `radius` controls the spatial extent of the filter (typical: 2–8).
/// `epsilon` controls edge sensitivity (typical: 0.01–0.1 for linear data).
pub fn chroma_denoise(
    rgb_data: &mut Vec<[f32; 3]>,
    width: usize,
    height: usize,
    radius: usize,
    epsilon: f32,
) {
    let luma = |p: &[f32; 3]| -> f32 {
        R_RELATIVE_LUMINANCE * p[0] + G_RELATIVE_LUMINANCE * p[1] + B_RELATIVE_LUMINANCE * p[2]
    };

    let y: Vec<f32> = rgb_data.par_iter().map(luma).collect();
    let cr: Vec<f32> = rgb_data.par_iter().zip(&y).map(|(p, &yv)| p[0] - yv).collect();
    let cb: Vec<f32> = rgb_data.par_iter().zip(&y).map(|(p, &yv)| p[2] - yv).collect();

    let cr_smooth = guided_filter(&y, &cr, width, height, radius, epsilon);
    let cb_smooth = guided_filter(&y, &cb, width, height, radius, epsilon);

    let wr = R_RELATIVE_LUMINANCE;
    let wg = G_RELATIVE_LUMINANCE;
    let wb = B_RELATIVE_LUMINANCE;

    rgb_data.par_iter_mut().enumerate().for_each(|(i, p)| {
        let yv = luma(p);
        p[0] = yv + cr_smooth[i];
        p[1] = yv - (wr / wg) * cr_smooth[i] - (wb / wg) * cb_smooth[i];
        p[2] = yv + cb_smooth[i];
    });
}

/// Guided filter (single-channel).
fn guided_filter(
    guide: &[f32],
    source: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mean_g = box_filter(guide, width, height, radius);
    let mean_p = box_filter(source, width, height, radius);

    let guide_sq: Vec<f32> = guide.par_iter().map(|&v| v * v).collect();
    let guide_p: Vec<f32> = guide.par_iter().zip(source).map(|(&g, &s)| g * s).collect();

    let mean_gg = box_filter(&guide_sq, width, height, radius);
    let mean_gp = box_filter(&guide_p, width, height, radius);

    let var_g: Vec<f32> = mean_gg
        .par_iter()
        .zip(&mean_g)
        .map(|(&gg, &mg)| gg - mg * mg)
        .collect();
    let cov_gp: Vec<f32> = mean_gp
        .par_iter()
        .zip(&mean_g)
        .zip(&mean_p)
        .map(|((&gp, &mg), &mp)| gp - mg * mp)
        .collect();

    let a: Vec<f32> = var_g
        .par_iter()
        .zip(&cov_gp)
        .map(|(&v, &c)| c / (v + epsilon))
        .collect();
    let b: Vec<f32> = mean_p
        .par_iter()
        .zip(&a)
        .zip(&mean_g)
        .map(|((&mp, &a_val), &mg)| mp - a_val * mg)
        .collect();

    let mean_a = box_filter(&a, width, height, radius);
    let mean_b = box_filter(&b, width, height, radius);

    mean_a
        .par_iter()
        .zip(&mean_b)
        .zip(guide)
        .map(|((&ma, &mb), &gv)| ma * gv + mb)
        .collect()
}

/// Separable box filter (axis-aligned sliding window) with correct border
/// handling.  Each pixel is the mean over the largest window that fits inside
/// the image bounds at that location.
fn box_filter(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut tmp = vec![0.0; data.len()];

    // Horizontal pass
    for y in 0..height {
        let row = y * width;
        let mut sum = 0.0;
        for dx in 0..=radius.min(width - 1) {
            sum += data[row + dx];
        }
        tmp[row] = sum;
        for x in 1..width {
            if x > radius {
                sum -= data[row + x - radius - 1];
            }
            if x + radius < width {
                sum += data[row + x + radius];
            }
            tmp[row + x] = sum;
        }
    }

    let mut result = vec![0.0; data.len()];

    // Vertical pass
    for x in 0..width {
        let mut sum = 0.0;
        for dy in 0..=radius.min(height - 1) {
            sum += tmp[dy * width + x];
        }
        result[x] = sum;
        for y in 1..height {
            if y > radius {
                sum -= tmp[(y - radius - 1) * width + x];
            }
            if y + radius < height {
                sum += tmp[(y + radius) * width + x];
            }
            result[y * width + x] = sum;
        }
    }

    // Normalize by the actual window size at each position
    for y in 0..height {
        let cy = (radius.min(y) + radius.min(height - 1 - y) + 1) as f32;
        for x in 0..width {
            let cx = (radius.min(x) + radius.min(width - 1 - x) + 1) as f32;
            result[y * width + x] /= cx * cy;
        }
    }

    result
}
