//! Shot-noise grain on image-forming dye only.
//!
//! D_grainy = D + κ * sqrt(max(D*(D_max−D), 0)) * n
//! where n is unit-variance noise correlated by a Gaussian of σ ≈ dye-cloud radius.
//!
//! Dye-cloud correlation length: 3 µm (typical chromogenic cloud scale —
//! photographic science literature; see film-implementation.md §5.8 / §8).
//! κ is derived from ρ_grains and pixel pitch, not a free slider.
//!
//! Mask / residual colored-coupler density must NEVER receive grain modulation.

use crate::film::blur::gaussian_blur_separable;
use crate::film::constants::DYE_CLOUD_CORRELATION_UM;
use crate::film::types::DyePlanes;
use rayon::prelude::*;

/// SplitMix64 — deterministic seeded stream RNG (no `rand` crate).
/// Per-row streams use key derived from `(seed, layer, y)`.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Approximate N(0,1) via Irwin–Hall (sum of 12 uniforms).
    pub fn next_gaussian(&mut self) -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..12 {
            acc += (self.next_u64() as f32) / (u64::MAX as f32);
        }
        acc - 6.0
    }
}

/// Scale reference κ (at 1 µm pitch) to actual pixel pitch.
pub fn scale_kappa(kappa_ref: f32, pixel_pitch_um: f32) -> f32 {
    kappa_ref / pixel_pitch_um.max(1e-6)
}

/// Apply grain to image dye planes only. Mask planes are untouched.
pub fn apply_grain(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
) {
    let width = dyes.width;
    let height = dyes.height;
    let sigma_px = DYE_CLOUD_CORRELATION_UM / pixel_pitch_um.max(1e-6);

    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa = kappa_per_layer[layer_i];
        if kappa <= 0.0 || d_max <= 0.0 {
            continue;
        }

        let mut noise = vec![0.0f32; width * height];
        for y in 0..height {
            let mut rng = SplitMix64::new(
                seed
                    .wrapping_mul(0xD1B54A32D192ED03)
                    .wrapping_add((layer_i as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .wrapping_add(y as u64),
            );
            for x in 0..width {
                noise[y * width + x] = rng.next_gaussian();
            }
        }
        gaussian_blur_separable(&mut noise, width, height, sigma_px.max(1.0));
        let var: f64 = noise.iter().map(|&n| (n as f64).powi(2)).sum::<f64>() / noise.len() as f64;
        let norm = if var > 1e-12 {
            (1.0 / var.sqrt()) as f32
        } else {
            1.0
        };

        plane.par_iter_mut().zip(noise.par_iter()).for_each(|(d, &n)| {
            let dens = (*d).clamp(0.0, d_max);
            let eps_toe = 0.02 * d_max;
            let taper = (dens / (dens + eps_toe)).min(1.0);
            let sigma_d = taper * (dens * (d_max - dens)).max(0.0).sqrt();
            *d = dens + kappa * sigma_d * n * norm;
            *d = (*d).clamp(0.0, d_max * 1.05);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_dyes(d: f32, d_max: f32, w: usize, h: usize) -> DyePlanes {
        let n = w * h;
        DyePlanes {
            width: w,
            height: h,
            image_dye: vec![vec![d; n]],
            mask_dye: vec![vec![
                d_max * crate::film::constants::MASK_DENSITY_FRACTION_OF_DMAX * (1.0 - d / d_max);
                n
            ]],
        }
    }

    fn std_of(plane: &[f32]) -> f32 {
        let n = plane.len() as f64;
        let mean = plane.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = plane
            .iter()
            .map(|&x| {
                let e = x as f64 - mean;
                e * e
            })
            .sum::<f64>()
            / n;
        var.sqrt() as f32
    }

    #[test]
    fn grain_variance_mid_density() {
        let d_max = 2.0f32;
        let d = d_max / 2.0;
        let kappa = 0.15f32;
        let w = 256;
        let h = 256;
        let mut dyes = flat_dyes(d, d_max, w, h);
        let mask_before = dyes.mask_dye[0].clone();
        apply_grain(&mut dyes, &[d_max], &[kappa], 3.0, 123);
        let std = std_of(&dyes.image_dye[0]);
        let expected = kappa * (d * (d_max - d)).sqrt();
        let rel = ((std - expected) / expected).abs();
        assert!(rel < 0.15, "grain std={std} expected≈{expected} rel={rel}");
        assert_eq!(dyes.mask_dye[0], mask_before);
    }

    #[test]
    fn grain_vanishes_at_extremes() {
        let d_max = 2.0f32;
        let kappa = 0.15f32;
        let w = 128;
        let h = 128;
        let mut mid = flat_dyes(d_max / 2.0, d_max, w, h);
        apply_grain(&mut mid, &[d_max], &[kappa], 3.0, 7);
        let std_mid = std_of(&mid.image_dye[0]);

        let mut lo = flat_dyes(0.01, d_max, w, h);
        apply_grain(&mut lo, &[d_max], &[kappa], 3.0, 7);
        let std_lo = std_of(&lo.image_dye[0]);

        let mut hi = flat_dyes(d_max - 0.01, d_max, w, h);
        apply_grain(&mut hi, &[d_max], &[kappa], 3.0, 7);
        let std_hi = std_of(&hi.image_dye[0]);

        assert!(std_lo < 0.25 * std_mid, "lo={std_lo} mid={std_mid}");
        assert!(std_hi < 0.25 * std_mid, "hi={std_hi} mid={std_mid}");
    }

    #[test]
    fn grain_does_not_modulate_mask() {
        let d_max = 2.0f32;
        let mut a = flat_dyes(1.0, d_max, 64, 64);
        let mut b = flat_dyes(1.0, d_max, 64, 64);
        let mask_a = a.mask_dye[0].clone();
        apply_grain(&mut a, &[d_max], &[0.2], 3.0, 1);
        apply_grain(&mut b, &[d_max], &[0.0], 3.0, 1);
        assert_eq!(a.mask_dye[0], mask_a);
        assert_eq!(a.mask_dye[0], b.mask_dye[0]);
    }

    #[test]
    fn grain_flag_off() {
        let d_max = 2.0f32;
        let mut a = flat_dyes(1.0, d_max, 32, 32);
        let b = a.clone();
        apply_grain(&mut a, &[d_max], &[0.0], 3.0, 99);
        assert_eq!(a.image_dye, b.image_dye);
    }
}
