//! Separable Gaussian blur for halation, grain footprint, and adjacency.
//!
//! Horizontal then vertical 1D convolutions. Used wherever a Gaussian PSF is justified.
//! [`exponential_blur_separable`] approximates an isotropic exponential PSF as a
//! two-Gaussian mixture (spektrafilm `fast_gaussian_filter.py`).

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
            let mut acc = vec![0.0f32; width];
            for (k, &w) in kernel.iter().enumerate() {
                let yy = reflect_index(y as isize + k as isize - radius as isize, height);
                let src = &tmp[yy * width..yy * width + width];
                for x in 0..width {
                    acc[x] += src[x] * w;
                }
            }
            out_row.copy_from_slice(&acc);
        });
}

/// Approximate the 2D isotropic exponential `exp(−r/λ)/(2πλ²)` as a two-Gaussian
/// mixture (YAGNI vs n=3). Fit from spektrafilm `fast_gaussian_filter.py`:
/// `(a, σ/λ) = (0.6235, 0.9401), (0.3765, 2.5177)` (amplitudes already sum to 1).
///
/// A component with `σ_px < 1e-3` is the Gaussian identity (same floor as
/// [`gaussian_blur_separable`]), so its amplitude stays on `Φ`.
pub fn exponential_blur_separable(buf: &mut [f32], width: usize, height: usize, lambda_px: f32) {
    assert_eq!(buf.len(), width * height);
    if lambda_px <= 0.0 || width == 0 || height == 0 {
        return;
    }
    // spektrafilm `fast_gaussian_filter.py` published 2-Gaussian fit.
    const MIX: [(f32, f32); 2] = [(0.6235, 0.9401), (0.3765, 2.5177)];

    let orig = buf.to_vec();
    // `gaussian_blur_separable` is identity for σ < 1e-3, so a skipped-narrow
    // component still contributes `amp · Φ` instead of dropping that mass.
    for (i, &(amp, sigma_over_lambda)) in MIX.iter().enumerate() {
        let sigma_px = sigma_over_lambda * lambda_px;
        let mut component = orig.clone();
        gaussian_blur_separable(&mut component, width, height, sigma_px);
        if i == 0 {
            for (dst, &src) in buf.iter_mut().zip(component.iter()) {
                *dst = amp * src;
            }
        } else {
            for (dst, &src) in buf.iter_mut().zip(component.iter()) {
                *dst += amp * src;
            }
        }
    }
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
    let left_end = radius.min(n);
    let right_start = n.saturating_sub(radius).max(left_end);
    for x in 0..left_end {
        let mut acc = 0.0f32;
        for (k, &w) in kernel.iter().enumerate() {
            let xx = reflect_index(x as isize + k as isize - radius as isize, n);
            acc += input[xx] * w;
        }
        output[x] = acc;
    }
    // Interior samples: every tap lands inside [0, n), so the reflect lookup
    // resolves to `base + k` and can be skipped without changing any value.
    for x in left_end..right_start {
        let base = x - radius;
        let mut acc = 0.0f32;
        for (k, &w) in kernel.iter().enumerate() {
            acc += input[base + k] * w;
        }
        output[x] = acc;
    }
    for x in right_start..n {
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

    #[test]
    fn exponential_blur_impulse_energy_conservation() {
        let width = 64;
        let height = 64;
        let mut buf = vec![0.0f32; width * height];
        buf[(height / 2) * width + width / 2] = 1.0;
        exponential_blur_separable(&mut buf, width, height, 4.0);
        let sum: f64 = buf.iter().map(|&v| v as f64).sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "exponential mixture should conserve impulse energy, sum={sum}"
        );
    }

    #[test]
    fn exponential_blur_partial_skip_conserves_energy() {
        // 0.9401·λ < 1e-3 ≤ 2.5177·λ → compact mix term is a Gaussian no-op
        // (identity), wide term still blurs. Mass must stay 1, not 0.3765.
        let lambda_px = 0.0008;
        assert!(0.9401 * lambda_px < 1e-3);
        assert!(2.5177 * lambda_px >= 1e-3);
        let width = 16;
        let height = 16;
        let mut buf = vec![0.0f32; width * height];
        buf[(height / 2) * width + width / 2] = 1.0;
        exponential_blur_separable(&mut buf, width, height, lambda_px);
        let sum: f64 = buf.iter().map(|&v| v as f64).sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "partial-skip mixture must conserve impulse energy, sum={sum}"
        );
    }
}
