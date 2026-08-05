//! Discrete shot-noise grain on image-forming dye only.
//!
//! The LUT supplies the local expected density and therefore the local
//! development probability. At a fine film pitch, the final image-forming dye
//! must then be reconstructed from the random population of developed
//! particles, not made by adding a continuous texture to the LUT result. For a
//! pixel aperture with expected crystal count `n`, the model samples a virtual
//! particle population, develops it with probability `D / D_max`, converts the
//! developed count back to optical density, and then spreads the same
//! particles through a cloud-scale PSF plus a crystal-scale microstructure
//! PSF.
//!
//! The expected variance reduces to Selwyn's law above the cloud scale and
//! saturates below it, but the realization remains discrete instead of a
//! Gaussian overlay. The particle area is derived from the stock's areal
//! crystal density (`κ_ref² = 1 / ρ`); no grain amount slider is introduced.
//!
//! Dye-cloud correlation length: 3 µm (typical chromogenic cloud scale —
//! photographic science literature; see film-implementation.md §5.8 / §8).
//!
//! Mask / residual colored-coupler density must NEVER receive grain modulation.

use crate::film::blur::{gaussian_blur_separable, gaussian_kernel_l2_sq};
use crate::film::constants::DYE_CLOUD_CORRELATION_UM;
use crate::film::types::{DyePlanes, LatentPlanes};
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

    /// Uniform variate from the same deterministic stream.
    #[inline]
    fn next_unit_f64(&mut self) -> f64 {
        // Keep the top 53 bits so the result has the precision of an f64
        // mantissa without relying on a separate RNG implementation.
        (self.next_u64() >> 11) as f64 / 9007199254740992.0
    }
}

/// Continuum sublayer scales used once the render aperture contains unresolved
/// particles. The fine-pitch path samples one physical population per plane.
pub(crate) const SUBLAYER_SCALES: [f32; 2] = [1.3, 0.7];

/// Above this mean, the Poisson count is accurately and cheaply approximated
/// by a normal variate. Below it, the exact product sampler is inexpensive.
const POISSON_NORMAL_THRESHOLD: f32 = 16.0;

/// Below this trial count, direct Bernoulli trials avoid a poor normal tail.
const BINOMIAL_NORMAL_THRESHOLD: usize = 32;

/// Per-pixel density std of the dye-cloud field at pixel pitch `p` (µm).
///
/// The developed density field is a Poisson shot-noise process of dye clouds:
/// clouds land at areal density ρ = 1/κ_ref² (from crystal packing/thickness —
/// see `FilmStock::finalize`), each with a Gaussian footprint of std
/// `correlation_um`. The variance of the density sampled with a square pixel
/// of side p is exactly
///
/// ```text
/// Var = λν² / max(p², 4π·σ_cloud²)
/// ```
///
/// (Selwyn's law for p above the cloud scale; saturation once the pixel fits
/// inside a single correlated region). Hence
///
/// ```text
/// κ(p) = κ_ref / max(p, p_sat),   p_sat = 2·√π·σ_cloud
/// ```
///
/// with κ_ref = 1/√ρ the Selwyn coefficient at the 1 µm reference pitch and
/// `correlation_um` the Gaussian cloud std (dye-cloud radius).
pub fn scale_kappa(kappa_ref: f32, pixel_pitch_um: f32, correlation_um: f32) -> f32 {
    let p_sat = 2.0 * std::f32::consts::PI.sqrt() * correlation_um.max(1e-6);
    kappa_ref / pixel_pitch_um.max(p_sat).max(1e-6)
}

/// Draw a Poisson count without adding another dependency.
fn sample_poisson(rng: &mut SplitMix64, lambda: f32) -> usize {
    if !(lambda > 0.0) {
        return 0;
    }
    if lambda < POISSON_NORMAL_THRESHOLD {
        let threshold = (-(lambda as f64)).exp();
        let mut product = 1.0f64;
        let mut count = 0usize;
        while product > threshold {
            product *= rng.next_unit_f64();
            count += 1;
        }
        return count.saturating_sub(1);
    }

    let draw = lambda + lambda.sqrt() * rng.next_gaussian();
    draw.max(0.0).round() as usize
}

/// Draw a binomial count, using direct trials only where that is cheap.
fn sample_binomial(rng: &mut SplitMix64, trials: usize, probability: f32) -> usize {
    if trials == 0 || probability <= 0.0 {
        return 0;
    }
    if probability >= 1.0 {
        return trials;
    }

    if trials < BINOMIAL_NORMAL_THRESHOLD {
        let mut count = 0usize;
        for _ in 0..trials {
            if rng.next_unit_f64() < probability as f64 {
                count += 1;
            }
        }
        return count;
    }

    // Rare-event tails are better represented by a Poisson draw than by a
    // normal approximation. The complement keeps the same accuracy near one.
    if probability < 0.05 {
        return sample_poisson(rng, trials as f32 * probability).min(trials);
    }
    if probability > 0.95 {
        return trials.saturating_sub(sample_poisson(rng, trials as f32 * (1.0 - probability)));
    }

    let mean = trials as f32 * probability;
    let variance = mean * (1.0 - probability);
    let draw = mean + variance.sqrt() * rng.next_gaussian();
    draw.round().clamp(0.0, trials as f32) as usize
}

fn mean_crystal_size_um(crystal_size: Option<&crate::film::stock::LogNormalDist>) -> f32 {
    crystal_size
        .map(|dist| (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp() as f32)
        .unwrap_or(0.7)
}

/// Resolved Gaussian footprint of one crystal population.
fn particle_cloud_sigma_um(crystal_size: Option<&crate::film::stock::LogNormalDist>) -> f32 {
    // A circular crystal of diameter s has per-axis variance (s/4)^2.
    // The larger dye-cloud extent is used below to derive the population
    // uniformity; it must not blur every individually resolved crystal.
    mean_crystal_size_um(crystal_size) * 0.25
}

/// Convert the measured circular cloud extent to an equivalent Gaussian sigma.
/// A disk of diameter D has per-axis variance (D/4)^2.
fn dye_cloud_sigma_um() -> f32 {
    DYE_CLOUD_CORRELATION_UM * 0.25
}

fn cloud_population(rho_areal: f32) -> f32 {
    let cloud_radius = DYE_CLOUD_CORRELATION_UM * 0.5;
    (rho_areal * std::f32::consts::PI * cloud_radius * cloud_radius).max(1.0)
}

/// Fraction of a cloud population that is spatially uniform.
///
/// A circular cloud of the measured diameter contains
/// `ρ · π(D/2)²` crystals in expectation. Treating the reciprocal population
/// count as the residual non-uniform fraction gives the same high-density
/// saturation behavior as a finite, highly uniform particle population, while
/// keeping the value derived from stock geometry.
fn cloud_uniformity(rho_areal: f32) -> f32 {
    let population = cloud_population(rho_areal);
    population / (population + 1.0)
}

/// Switch to discrete particles once the render aperture resolves a typical
/// crystal diameter. Above this limit the Gaussian continuum is the
/// central-limit approximation of the same population process.
pub(crate) fn particle_resolution_limit_um(
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
) -> f32 {
    crystal_sizes
        .iter()
        .filter_map(|value| value.as_ref())
        .map(|dist| mean_crystal_size_um(Some(dist)) * 2.0)
        .fold(0.0f32, f32::max)
}

/// Apply grain to image dye planes only. Mask planes are untouched.
///
/// `kappa_ref_per_layer` is the Selwyn coefficient κ_ref = 1/√ρ (grains/µm²
/// derived from the stock geometry). It supplies the expected crystal count
/// `ρ·p²` in each pixel aperture; the sampled particle field then acquires its
/// pitch-dependent variance naturally through the cloud convolution.
fn particle_field(
    probabilities: &[f32],
    width: usize,
    height: usize,
    kappa_ref: f32,
    pixel_pitch_um: f32,
    crystal_size: Option<&crate::film::stock::LogNormalDist>,
    seed: u64,
    layer_i: usize,
    fixed_crystal_sites: bool,
) -> Vec<f32> {
    let rho_areal = 1.0 / (kappa_ref * kappa_ref).max(1e-12);
    let particles_per_pixel = rho_areal * pixel_pitch_um * pixel_pitch_um;
    if particles_per_pixel <= 1e-8 {
        return probabilities.to_vec();
    }

    let crystal_sigma_px = particle_cloud_sigma_um(crystal_size) / pixel_pitch_um.max(1e-6);
    let cloud_sigma_px = dye_cloud_sigma_um() / pixel_pitch_um.max(1e-6);
    let uniformity = cloud_uniformity(rho_areal);
    let mut particles = vec![0.0f32; width * height];
    for y in 0..height {
        let mut rng = SplitMix64::new(
            seed.wrapping_mul(0xD1B54A32D192ED03)
                .wrapping_add((layer_i as u64).wrapping_mul(0x9E3779B97F4A7C15))
                .wrapping_add(y as u64),
        );
        for x in 0..width {
            let index = y * width + x;
            let probability = probabilities[index].clamp(0.0, 1.0);
            if fixed_crystal_sites {
                // Crystal locations are independent of exposure. Exposure
                // changes dye amount at those sites, not whether the film
                // contains a crystal there.
                let sites = sample_poisson(&mut rng, particles_per_pixel);
                particles[index] = sites as f32 * probability / particles_per_pixel;
            } else if probability > 0.0 {
                // The virtual population keeps the mean fraction fixed while
                // reducing residual count noise as a cloud approaches saturation.
                let saturation = (1.0 - probability * uniformity * (1.0 - 1e-6)).max(1e-6);
                let available = sample_poisson(&mut rng, particles_per_pixel / saturation);
                let developed = sample_binomial(&mut rng, available, probability);
                particles[index] = developed as f32 * saturation / particles_per_pixel;
            }
        }
    }

    // The cloud field carries the image-forming fraction. A second blur of the
    // same particles exposes crystal-scale structure without making a separate
    // texture field. Its weight follows the number of crystals in one cloud.
    let mut clouds = particles.clone();
    gaussian_blur_separable(&mut clouds, width, height, cloud_sigma_px);
    gaussian_blur_separable(&mut particles, width, height, crystal_sigma_px);
    let mut micro_clouds = particles.clone();
    gaussian_blur_separable(&mut micro_clouds, width, height, cloud_sigma_px);
    let micro_weight = (cloud_population(rho_areal) + 1.0).sqrt().recip();

    clouds
        .par_iter_mut()
        .zip(particles.par_iter())
        .zip(micro_clouds.par_iter())
        .for_each(|((cloud, &micro), &micro_cloud)| {
            *cloud = (*cloud + micro_weight * (micro - micro_cloud)).clamp(0.0, 1.05);
        });
    clouds
}

/// Reconstruct latent developable fractions from the discrete particle
/// population. This is the production grain path: reduction, DIR, adjacency,
/// and scanning all consume the particle-derived latent image.
pub fn apply_particle_grain_to_latent(
    latent: &mut LatentPlanes,
    kappa_ref_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
) {
    for (layer_i, plane) in latent.layers.iter_mut().enumerate() {
        let kappa_ref = kappa_ref_per_layer[layer_i];
        if kappa_ref <= 0.0 {
            continue;
        }
        let probabilities = plane.clone();
        *plane = particle_field(
            &probabilities,
            latent.width,
            latent.height,
            kappa_ref,
            pixel_pitch_um,
            crystal_sizes.get(layer_i).and_then(|value| value.as_ref()),
            seed,
            layer_i,
            true,
        );
    }
}

/// Gaussian continuum approximation for apertures that do not resolve
/// individual crystals. Its variance is still derived from the stock
/// population and the cloud footprint; it is not a display texture.
pub(crate) fn apply_continuum_grain(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_ref_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
) {
    let width = dyes.width;
    let height = dyes.height;
    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa_ref = kappa_ref_per_layer[layer_i];
        if kappa_ref <= 0.0 || d_max <= 0.0 {
            continue;
        }

        let mut sub_dyes = vec![vec![0.0f32; width * height]; SUBLAYER_SCALES.len()];
        for (sub_idx, &sub_scale) in SUBLAYER_SCALES.iter().enumerate() {
            let correlation_um = crystal_sizes
                .get(layer_i)
                .and_then(|value| value.as_ref())
                .map(|dist| {
                    (mean_crystal_size_um(Some(dist)) / 0.7) * dye_cloud_sigma_um() * sub_scale
                })
                .unwrap_or(dye_cloud_sigma_um() * sub_scale);
            let sigma_px = (correlation_um / pixel_pitch_um.max(1e-6)).max(1.0);
            let kappa_target = scale_kappa(kappa_ref, pixel_pitch_um, correlation_um)
                * (SUBLAYER_SCALES.len() as f32).sqrt();
            let kappa_sub = kappa_target / gaussian_kernel_l2_sq(sigma_px);

            let mut noise = vec![0.0f32; width * height];
            for y in 0..height {
                let mut rng = SplitMix64::new(
                    seed.wrapping_mul(0xD1B54A32D192ED03)
                        .wrapping_add((layer_i as u64).wrapping_mul(0x9E3779B97F4A7C15))
                        .wrapping_add((sub_idx as u64).wrapping_mul(0x123456789))
                        .wrapping_add(y as u64),
                );
                for x in 0..width {
                    noise[y * width + x] = rng.next_gaussian();
                }
            }
            gaussian_blur_separable(&mut noise, width, height, sigma_px);

            sub_dyes[sub_idx]
                .par_iter_mut()
                .zip(noise.par_iter())
                .zip(plane.par_iter())
                .for_each(|((output, &normal), &density)| {
                    let density = density.clamp(0.0, d_max);
                    let toe = 0.05 * d_max;
                    let taper = (density / (density + toe)).min(1.0);
                    let sigma_density = taper * (density * (d_max - density)).max(0.0).sqrt();
                    let noisy = density + kappa_sub * sigma_density * normal;
                    let knee = 0.005 * d_max;
                    *output = if noisy >= knee {
                        noisy
                    } else {
                        (knee * knee) / (2.0 * knee - noisy)
                    }
                    .min(d_max * 1.05);
                });
        }

        plane
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, density)| {
                *density = sub_dyes.iter().map(|sub| sub[index]).sum::<f32>()
                    / SUBLAYER_SCALES.len() as f32;
            });
    }
}

/// Apply the same particle realization to already reduced image dye planes.
/// Kept for density-level tests and callers that do not own latent planes.
pub fn apply_grain(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_ref_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
) {
    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa_ref = kappa_ref_per_layer[layer_i];
        if kappa_ref <= 0.0 || d_max <= 0.0 {
            continue;
        }
        let probabilities: Vec<f32> = plane
            .iter()
            .map(|&density| (density / d_max).clamp(0.0, 1.0))
            .collect();
        let particles = particle_field(
            &probabilities,
            dyes.width,
            dyes.height,
            kappa_ref,
            pixel_pitch_um,
            crystal_sizes.get(layer_i).and_then(|value| value.as_ref()),
            seed,
            layer_i,
            false,
        );
        plane
            .par_iter_mut()
            .zip(particles.par_iter())
            .for_each(|(density, &fraction)| {
                *density = (fraction * d_max).clamp(0.0, d_max * 1.05);
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
                d_max * crate::film::constants::MASK_DENSITY_FRACTION_OF_DMAX
                    * (1.0 - d / d_max);
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
        // Poisson particle count + binomial development, followed by the
        // unit-sum particle footprint. The count variance is
        // D_max² * saturation * f / (ρ · p²); the separable footprint scales
        // its std by the one-dimensional kernel L2 norm.
        let rho = 1.0 / (kappa * kappa);
        let particles_per_pixel = rho * 3.0 * 3.0;
        let f = d / d_max;
        let saturation = 1.0 - f * cloud_uniformity(rho);
        let sigma_px = dye_cloud_sigma_um() / 3.0;
        let kernel_l2 = crate::film::blur::gaussian_kernel_l2_sq(sigma_px);
        let expected = d_max * (saturation * f / particles_per_pixel).sqrt() * kernel_l2;
        let rel = (std - expected).abs() / expected;
        assert!(rel < 0.15, "grain std={std} expected={expected} rel={rel}");
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

    #[test]
    fn resolved_crystal_sites_persist_across_exposure() {
        let width = 128;
        let height = 128;
        let n = width * height;
        let mut bright = LatentPlanes {
            width,
            height,
            layers: vec![vec![0.8; n]],
        };
        let mut dark = LatentPlanes {
            width,
            height,
            layers: vec![vec![0.2; n]],
        };
        apply_particle_grain_to_latent(&mut bright, &[0.4], 0.25, 17, &[None]);
        apply_particle_grain_to_latent(&mut dark, &[0.4], 0.25, 17, &[None]);

        let bright_plane = &bright.layers[0];
        let dark_plane = &dark.layers[0];
        let bright_mean = bright_plane.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let dark_mean = dark_plane.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let covariance = bright_plane
            .iter()
            .zip(dark_plane)
            .map(|(&b, &d)| (b as f64 - bright_mean) * (d as f64 - dark_mean))
            .sum::<f64>()
            / n as f64;
        let bright_var = bright_plane
            .iter()
            .map(|&v| (v as f64 - bright_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let dark_var = dark_plane
            .iter()
            .map(|&v| (v as f64 - dark_mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let correlation = covariance / (bright_var * dark_var).sqrt();
        assert!(bright_mean > dark_mean, "exposure should scale dye density");
        assert!(
            correlation > 0.8,
            "crystal sites should persist, correlation={correlation}"
        );
    }
}
