//! Separable Gaussian blur for halation, grain footprint, and adjacency.
//!
//! Horizontal then vertical 1D convolutions. Used wherever a Gaussian PSF is justified.

use rayon::prelude::*;

/// Blur a planar `width * height` buffer in-place with isotropic Gaussian σ (pixels).
///
/// If `sigma` is below ~1e-3 px, this is a no-op (identity).
pub fn gaussian_blur_separable(buf: &mut [f32], width: usize, height: usize, sigma: f32) {
    assert_eq!(buf.len(), width * height);
    if sigma < 1e-3 || width == 0 || height == 0 {
        return;
    }
    let kernel = make_gaussian_kernel(sigma);
    let mut tmp = vec![0.0f32; buf.len()];
    // Horizontal pass: write into tmp.
    tmp.par_chunks_mut(width)
        .zip(buf.par_chunks(width))
        .for_each(|(out_row, in_row)| {
            convolve_1d_reflect(in_row, out_row, &kernel);
        });
    // Vertical pass: write back into buf.
    let radius = kernel.len() / 2;
    buf.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            for x in 0..width {
                let mut acc = 0.0f32;
                for (k, &w) in kernel.iter().enumerate() {
                    let yy = reflect_index(y as isize + k as isize - radius as isize, height);
                    acc += tmp[yy * width + x] * w;
                }
                out_row[x] = acc;
            }
        });
}

/// Exact Gaussian kernel radius covering ~99.7% of mass using `ceil(3σ)`.
/// Returns 0 for `sigma < 1e-3`.
#[inline]
pub(crate) fn gaussian_radius(sigma: f32) -> usize {
    if sigma < 1e-3 {
        0
    } else {
        (3.0 * sigma).ceil().max(1.0) as usize
    }
}

pub(crate) fn make_gaussian_kernel(sigma: f32) -> Vec<f32> {
    // Radius ≈ 3σ covers ~99.7% of mass.
    let radius = gaussian_radius(sigma);
    let mut k = vec![0.0f32; 2 * radius + 1];
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f32;
    for (i, slot) in k.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        let v = (-x * x * inv_2s2).exp();
        *slot = v;
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// L2-norm squared (Σ w²) of the 1D separable kernel at `sigma`.
///
/// Convolving unit-variance white noise with the unit-sum kernel `H·Hᵀ`
/// scales the output variance by (Σh²)², so the output std is Σh². Dividing
/// the grain amplitude by this factor keeps the correlated noise field
/// unit-variance: the dye-cloud footprint then shapes the spectrum without
/// attenuating the per-pixel granularity (Selwyn's law at the pixel aperture).
#[inline]
pub(crate) fn gaussian_kernel_l2_sq(sigma: f32) -> f32 {
    let k = make_gaussian_kernel(sigma);
    k.iter().map(|&w| w * w).sum()
}

fn convolve_1d_reflect(input: &[f32], output: &mut [f32], kernel: &[f32]) {
    let n = input.len();
    let radius = kernel.len() / 2;
    for x in 0..n {
        let mut acc = 0.0f32;
        for (k, &w) in kernel.iter().enumerate() {
            let xx = reflect_index(x as isize + k as isize - radius as isize, n);
            acc += input[xx] * w;
        }
        output[x] = acc;
    }
}

pub(crate) fn reflect_index(i: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len as isize;
    let mut x = i;
    // Mirror reflect without including a duplicate edge sample cycle.
    loop {
        if x < 0 {
            x = -x;
        } else if x >= n {
            x = 2 * n - 2 - x;
        } else {
            return x as usize;
        }
        if n == 1 {
            return 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_sigma_zero_is_identity() {
        let width = 8;
        let height = 8;
        let mut buf: Vec<f32> = (0..width * height).map(|i| i as f32).collect();
        let original = buf.clone();
        gaussian_blur_separable(&mut buf, width, height, 0.0);
        assert_eq!(buf, original);
    }

    #[test]
    fn blur_mass_conservation() {
        let width = 64;
        let height = 64;
        let mut buf = vec![0.0f32; width * height];
        // Uniform-ish field plus a bump so edges don't dominate.
        for v in &mut buf {
            *v = 1.0;
        }
        let sum_in: f64 = buf.iter().map(|&v| v as f64).sum();
        gaussian_blur_separable(&mut buf, width, height, 2.0);
        let sum_out: f64 = buf.iter().map(|&v| v as f64).sum();
        let rel = ((sum_out - sum_in) / sum_in).abs();
        assert!(rel < 1e-3, "mass rel err {rel}");
    }

    #[test]
    fn blur_impulse_symmetry() {
        let width = 65;
        let height = 65;
        let mut buf = vec![0.0f32; width * height];
        let cx = width / 2;
        let cy = height / 2;
        buf[cy * width + cx] = 1.0;
        gaussian_blur_separable(&mut buf, width, height, 3.0);

        // Sample pairs at equal radius along axes / diagonal.
        let samples = [(5isize, 0), (0, 5), (4, 3), (3, 4), (-5, 0), (0, -5)];
        let center = buf[cy * width + cx];
        assert!(center > 0.0);
        for &(dx, dy) in &samples {
            let a = buf[(cy as isize + dy) as usize * width + (cx as isize + dx) as usize];
            let b = buf[(cy as isize - dy) as usize * width + (cx as isize - dx) as usize];
            let c = buf[(cy as isize + dx) as usize * width + (cx as isize + dy) as usize];
            assert!((a - b).abs() < 1e-5, "point symmetry fail {a} vs {b}");
            // Radial: |dx|,|dy| swap should match for isotropic kernel.
            assert!((a - c).abs() < 1e-5, "axis swap fail {a} vs {c}");
        }
    }
}
