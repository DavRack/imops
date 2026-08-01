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
    ///
    /// Mirrors the GPU `GRAIN_NOISE` shader bit-near: WGSL has no u64, so the
    /// 64-bit state is converted to f32 as `f32(hi)·2^32 + f32(lo)` (not a
    /// correctly-rounded u64→f32 cast), then divided by 2^64.
    pub fn next_gaussian(&mut self) -> f32 {
        let mut acc = 0.0f32;
        for _ in 0..12 {
            let z = self.next_u64();
            let hi = (z >> 32) as u32;
            let lo = z as u32;
            let zf = (hi as f32) * 4294967296.0 + (lo as f32);
            acc += zf / 18446744073709551616.0;
        }
        acc - 6.0
    }
}

/// Chromogenic dye cloud granularity scale relative to metallic silver (0.15).
pub const CHROMOGENIC_DYE_GRAIN_SCALE: f32 = 0.15;

/// Minimum effective sampling pitch in µm, corresponding to a single chromogenic dye cloud size (~1.5 µm).
pub const MIN_GRAIN_PITCH_UM: f32 = 1.5;

/// Scale reference κ (at 1 µm pitch) to actual pixel pitch.
pub fn scale_kappa(kappa_ref: f32, pixel_pitch_um: f32) -> f32 {
    (kappa_ref * CHROMOGENIC_DYE_GRAIN_SCALE) / pixel_pitch_um.max(MIN_GRAIN_PITCH_UM)
}

/// Apply grain to image dye planes only. Mask planes are untouched.
pub fn apply_grain(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
) {
    let width = dyes.width;
    let height = dyes.height;

    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa = kappa_per_layer[layer_i];
        if kappa <= 0.0 || d_max <= 0.0 {
            continue;
        }

        let sublayer_scales: [f32; 2] = [1.3, 0.7];
        let n_sub = sublayer_scales.len();
        let mut sub_dyes = vec![vec![0.0f32; width * height]; n_sub];

        for (sl_idx, &sl_scale) in sublayer_scales.iter().enumerate() {
            let correlation_um = if let Some(dist) = &crystal_sizes[layer_i] {
                let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
                (mean_s / 0.7) as f32 * DYE_CLOUD_CORRELATION_UM * sl_scale
            } else {
                DYE_CLOUD_CORRELATION_UM * sl_scale
            };
            let sigma_px = correlation_um / pixel_pitch_um.max(1e-6);

            let mut noise = vec![0.0f32; width * height];
            for y in 0..height {
                let mut rng = SplitMix64::new(
                    seed
                        .wrapping_mul(0xD1B54A32D192ED03)
                        .wrapping_add((layer_i as u64).wrapping_mul(0x9E3779B97F4A7C15))
                        .wrapping_add((sl_idx as u64).wrapping_mul(0x123456789))
                        .wrapping_add(y as u64),
                );
                for x in 0..width {
                    noise[y * width + x] = rng.next_gaussian();
                }
            }
            gaussian_blur_separable(&mut noise, width, height, sigma_px.max(1.0));
            let norm = 1.0f32;

            let kappa_sub = kappa * (n_sub as f32).sqrt();
            sub_dyes[sl_idx].par_iter_mut().zip(noise.par_iter()).zip(plane.par_iter()).for_each(|((sub_d, &n), &d)| {
                use crate::film::math::div_det;
                let dens = d.clamp(0.0, d_max);
                let eps_toe = 0.05 * d_max;
                let taper = div_det(dens, dens + eps_toe).min(1.0);
                let sigma_d = taper * (dens * (d_max - dens)).max(0.0).sqrt();
                // FMA-contracted like `GRAIN_APPLY_SUB` (`dens + κ·sd·n·norm`).
                let noisy = (kappa_sub * sigma_d * n).mul_add(norm, dens);
                let knee = 0.005 * d_max;
                let d_val = if noisy >= knee {
                    noisy
                } else {
                    div_det(knee * knee, 2.0 * knee - noisy)
                };
                *sub_d = d_val.min(d_max * 1.05);
            });
        }

        // Average sublayers into target plane
        plane.par_iter_mut().enumerate().for_each(|(p, d)| {
            let mut acc = 0.0f32;
            for sl in 0..n_sub {
                acc += sub_dyes[sl][p];
            }
            *d = acc / (n_sub as f32);
        });
    }
}

/// Apply micro-structure log-normal clumping to dye plane.
pub fn add_micro_structure(_plane: &mut [f32], _pixel_pitch_um: f32, _seed: u64) {
    // Disabled: prevents double-counting grain on top of dye cloud granularity.
    return;
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
        apply_grain(&mut dyes, &[d_max], &[kappa], 3.0, 123, &[None]);
        let std = std_of(&dyes.image_dye[0]);
        // Spatial Gaussian correlation naturally attenuates white noise variance per Selwyn's Law.
        // For sigma_px ≈ 1.0, 2 sublayers attenuate variance by ~4.3x (std by ~2.08x) plus sublayer averaging (sqrt(2)/2 = 0.707).
        let expected_unblurred = kappa * (d * (d_max - d)).sqrt();
        let rel_unblurred = std / expected_unblurred;
        assert!(std > 0.01 && std < expected_unblurred, "grain std={std} expected_unblurred={expected_unblurred} rel={rel_unblurred}");
        assert_eq!(dyes.mask_dye[0], mask_before);
    }

    #[test]
    fn grain_vanishes_at_extremes() {
        let d_max = 2.0f32;
        let kappa = 0.15f32;
        let w = 128;
        let h = 128;
        let mut mid = flat_dyes(d_max / 2.0, d_max, w, h);
        apply_grain(&mut mid, &[d_max], &[kappa], 3.0, 7, &[None]);
        let std_mid = std_of(&mid.image_dye[0]);

        let mut lo = flat_dyes(0.01, d_max, w, h);
        apply_grain(&mut lo, &[d_max], &[kappa], 3.0, 7, &[None]);
        let std_lo = std_of(&lo.image_dye[0]);

        let mut hi = flat_dyes(d_max - 0.01, d_max, w, h);
        apply_grain(&mut hi, &[d_max], &[kappa], 3.0, 7, &[None]);
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
        apply_grain(&mut a, &[d_max], &[0.2], 3.0, 1, &[None]);
        apply_grain(&mut b, &[d_max], &[0.0], 3.0, 1, &[None]);
        assert_eq!(a.mask_dye[0], mask_a);
        assert_eq!(a.mask_dye[0], b.mask_dye[0]);
    }

    #[test]
    fn grain_flag_off() {
        let d_max = 2.0f32;
        let mut a = flat_dyes(1.0, d_max, 32, 32);
        let b = a.clone();
        apply_grain(&mut a, &[d_max], &[0.0], 3.0, 99, &[None]);
        assert_eq!(a.image_dye, b.image_dye);
    }
}
