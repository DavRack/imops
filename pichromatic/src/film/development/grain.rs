//! Discrete shot-noise grain on image-forming dye only.
//!
//! The LUT supplies the local expected dye density after H&D reduction. At a
//! fine film pitch, the model samples the same stock-derived particle
//! population in that reduced-density space, then spreads the resulting dye
//! clouds through a cloud-scale PSF plus a crystal-scale microstructure PSF.
//! Reduced H&D density `D = D_max·f^(1/γ)` supplies developable fraction `f`
//! for Bernoulli crystal trials; the realized dye cloud fraction is mapped back
//! linearly to optical density without re-applying the H&D toe.
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

/// Philox4×32-10 — deterministic counter-based RNG (Salmon et al. 2011,
/// "Parallel random numbers: as easy as 1, 2, 3", Random123 reference).
///
/// Pure u32 arithmetic (no 64-bit emulation), bit-compatible with the
/// official Random123 test vectors. Every pixel owns an independent stream
/// keyed by `(seed, layer, sublayer, x, y)`, so draws at one site never
/// depend on draws at neighboring sites.
///
/// NOTE: the GPU shaders (`gpu/shaders.rs`) still mirror the old SplitMix64
/// stream until the GPU milestone; they must be re-synced to this generator.
#[derive(Clone, Debug)]
pub struct Philox4x32 {
    key: [u32; 2],
    ctr: [u32; 4],
    buf: [u32; 4],
    pos: usize,
}

const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;
const PHILOX_W0: u32 = 0x9E3779B9;
const PHILOX_W1: u32 = 0xBB67AE85;

/// One Philox4×32-10 round (Random123 `_philox4xWround_tpl`).
fn philox_round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let p0 = (PHILOX_M0 as u64) * (ctr[0] as u64);
    let p1 = (PHILOX_M1 as u64) * (ctr[2] as u64);
    [
        ((p1 >> 32) as u32) ^ ctr[1] ^ key[0],
        p1 as u32,
        ((p0 >> 32) as u32) ^ ctr[3] ^ key[1],
        p0 as u32,
    ]
}

impl Philox4x32 {
    /// New stream with an explicit 64-bit counter and 2-word key.
    pub fn new(key: [u32; 2], ctr: [u32; 4]) -> Self {
        Self { key, ctr, buf: [0; 4], pos: 4 }
    }

    /// Independent per-pixel stream for `(seed, layer, sublayer)` at `(x, y)`.
    ///
    /// The key mixes the seed with layer/sublayer tags; the counter carries
    /// the pixel coordinates plus a block index so each site is independent.
    pub fn per_pixel(seed: u64, layer: u32, sublayer: u32, x: u32, y: u32) -> Self {
        let key = [
            (seed as u32) ^ layer.wrapping_mul(PHILOX_W0) ^ sublayer.wrapping_mul(PHILOX_W1),
            ((seed >> 32) as u32) ^ layer.wrapping_mul(PHILOX_W1) ^ sublayer.wrapping_mul(PHILOX_W0),
        ];
        Self::new(key, [x, y, 0, 0])
    }

    fn refill(&mut self) {
        // Pure function of (key, ctr): the 10-round evaluation keeps its own
        // Weyl key schedule, so the stream is exactly counter-based and any
        // block is independently reproducible (CPU ↔ GPU parity friendly).
        let mut c = self.ctr;
        let mut k = self.key;
        for _ in 0..10 {
            c = philox_round(c, k);
            k = [k[0].wrapping_add(PHILOX_W0), k[1].wrapping_add(PHILOX_W1)];
        }
        self.buf = c;
        self.pos = 0;
        // Advance to the next counter block for the same site.
        self.ctr[3] = self.ctr[3].wrapping_add(1);
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= 4 {
            self.refill();
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        v
    }

    /// Approximate N(0,1) via Irwin–Hall (sum of 12 uniforms).
    pub fn next_gaussian(&mut self) -> f32 {
        let mut acc = 0.0f64;
        for _ in 0..12 {
            acc += self.next_u32() as f64 / 4294967296.0;
        }
        (acc - 6.0) as f32
    }

    /// Uniform variate with f64-mantissa precision (53 bits from two words).
    #[inline]
    fn next_unit_f64(&mut self) -> f64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (((hi << 21) | (lo >> 11)) as f64) / 9007199254740992.0
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
fn sample_poisson(rng: &mut Philox4x32, lambda: f32) -> usize {
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
fn sample_binomial(rng: &mut Philox4x32, trials: usize, probability: f32) -> usize {
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
    DYE_CLOUD_CORRELATION_UM * 0.5
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

/// Positive H&D toe used for a low-density stochastic realization.
#[inline]
fn positive_density_toe(noisy: f32, d_max: f32) -> f32 {
    let knee = 0.005 * d_max;
    if noisy >= knee {
        noisy.min(d_max * 1.05)
    } else {
        ((knee * knee) / (2.0 * knee - noisy)).min(d_max * 1.05)
    }
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
/// per cloud footprint; the sampled particle field then acquires its
/// pitch-dependent variance naturally through the cloud convolution.
///
/// The population is sampled per cloud-footprint cell (side =
/// `DYE_CLOUD_CORRELATION_UM`), not per pixel. At microscope pitch a pixel
/// aperture holds far less than one crystal (λ ≈ 0.06 for fast layers at
/// 0.165 µm/px), so per-pixel Poisson sampling quantizes the realization into
/// sparse full-brightness dots and leaves voids between them. Sampling per
/// footprint (λ ≈ ρ·3µm² ≈ 20 crystals) keeps the developed fraction
/// continuous: dark regions resolve as overlapping dim clouds, not
/// salt-and-pepper.
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
    let cell_um = DYE_CLOUD_CORRELATION_UM;
    let sites_per_cell = rho_areal * cell_um * cell_um;
    if sites_per_cell <= 1e-8 {
        return probabilities.to_vec();
    }

    let cell_px = (cell_um / pixel_pitch_um.max(1e-6)).max(1.0);
    let cells_x = (width as f32 / cell_px).ceil() as usize;
    let cells_y = (height as f32 / cell_px).ceil() as usize;
    let uniformity = cloud_uniformity(rho_areal);
    let crystal_sigma_px = particle_cloud_sigma_um(crystal_size) / pixel_pitch_um.max(1e-6);
    let cloud_sigma_px = dye_cloud_sigma_um() / pixel_pitch_um.max(1e-6);

    // Cell-level realized developable fraction. Cells are independent
    // (per-cell Philox stream), so the grid builds in parallel.
    let mut grid = vec![0.0f32; cells_x * cells_y];
    grid.par_iter_mut()
        .enumerate()
        .for_each(|(cell_idx, slot)| {
            let cx = cell_idx % cells_x;
            let cy = cell_idx / cells_x;
            let x0 = (cx as f32 * cell_px) as usize;
            let y0 = (cy as f32 * cell_px) as usize;
            let x1 = (((cx as f32 + 1.0) * cell_px).ceil() as usize).min(width);
            let y1 = (((cy as f32 + 1.0) * cell_px).ceil() as usize).min(height);
            // Cell-mean developable probability from the reduced H&D field.
            let mut sum_p = 0.0f64;
            let mut cnt = 0usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum_p += probabilities[y * width + x] as f64;
                    cnt += 1;
                }
            }
            let p_cell = if cnt > 0 { (sum_p / cnt as f64) as f32 } else { 0.0 };
            let mut rng = Philox4x32::per_pixel(seed, layer_i as u32, 0, cx as u32, cy as u32);
            if fixed_crystal_sites {
                // Crystal locations are independent of exposure. Each site
                // develops stochastically; exposure sets Bernoulli(p) odds.
                let sites = sample_poisson(&mut rng, sites_per_cell);
                let developed = sample_binomial(&mut rng, sites, p_cell);
                *slot = developed as f32 / sites_per_cell;
            } else if p_cell > 0.0 {
                // The virtual population keeps the mean fraction fixed while
                // reducing residual count noise as a cloud approaches saturation.
                let saturation = (1.0 - p_cell * uniformity * (1.0 - 1e-6)).max(1e-6);
                let available = sample_poisson(&mut rng, sites_per_cell / saturation);
                let developed = sample_binomial(&mut rng, available, p_cell);
                *slot = developed as f32 * saturation / sites_per_cell;
            }
        });

    // Bilinear upscale of the cell grid to pixels: the developed fraction
    // varies continuously across cell boundaries instead of stepping in
    // blocks, matching the continuous crystal population it represents.
    let mut particles = vec![0.0f32; width * height];
    particles
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, slot)| {
            let x = index % width;
            let y = index / width;
            let gx = x as f32 / cell_px;
            let gy = y as f32 / cell_px;
            let gx0 = (gx as usize).min(cells_x - 1);
            let gy0 = (gy as usize).min(cells_y - 1);
            let gx1 = (gx0 + 1).min(cells_x - 1);
            let gy1 = (gy0 + 1).min(cells_y - 1);
            let fx = (gx - gx0 as f32).min(1.0);
            let fy = (gy - gy0 as f32).min(1.0);
            let v00 = grid[gy0 * cells_x + gx0];
            let v10 = grid[gy0 * cells_x + gx1];
            let v01 = grid[gy1 * cells_x + gx0];
            let v11 = grid[gy1 * cells_x + gx1];
            let top = v00 + (v10 - v00) * fx;
            let bottom = v01 + (v11 - v01) * fx;
            *slot = top + (bottom - top) * fy;
        });
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
                for x in 0..width {
                    noise[y * width + x] = Philox4x32::per_pixel(
                        seed,
                        layer_i as u32,
                        sub_idx as u32,
                        x as u32,
                        y as u32,
                    )
                    .next_gaussian();
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
                    *output = positive_density_toe(noisy, d_max);
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

/// Production fine-pitch grain: replace image dye planes with a particle
/// realization derived from reduced developable probabilities.
///
/// Reduced H&D density supplies only the local expected population statistics;
/// it must not survive as a scene-bearing baseline layer in the output.
pub fn apply_particle_grain_overwrite(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_ref_per_layer: &[f32],
    gamma_contrast_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    crystal_sizes: &[Option<crate::film::stock::LogNormalDist>],
    sigma_dir_px: f32,
    dir_inhibition_matrix: &[Vec<f32>],
    sigma_px: f32,
    adjacency_beta: f32,
) {
    let mut f_dev = vec![vec![0.0f32; dyes.width * dyes.height]; dyes.image_dye.len()];
    let mut skipped = vec![false; dyes.image_dye.len()];
    for (layer_i, plane) in dyes.image_dye.iter().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa_ref = kappa_ref_per_layer[layer_i];
        let gamma = gamma_contrast_per_layer[layer_i].max(1e-6);
        if kappa_ref <= 0.0 || d_max <= 0.0 {
            skipped[layer_i] = true;
            f_dev[layer_i] = plane.clone();
            continue;
        }
        let probabilities: Vec<f32> = plane
            .iter()
            .map(|&density| {
                let p_hd = (density / d_max).clamp(0.0, 1.0);
                p_hd.powf(gamma)
            })
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
            true,
        );
        f_dev[layer_i] = particles;
    }

    dyes.image_dye = f_dev;
    crate::film::development::diffusion::apply_dir_inhibition(dyes, sigma_dir_px, dir_inhibition_matrix);
    crate::film::development::diffusion::apply_adjacency(dyes, sigma_px, adjacency_beta);

    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        if skipped[layer_i] {
            continue;
        }
        let d_max = d_max_per_layer[layer_i];
        plane
            .par_iter_mut()
            .for_each(|fraction| {
                *fraction = (*fraction * d_max).clamp(0.0, d_max * 1.05);
            });
    }
}



/// Centered residual on reduced dye density — forbidden in production.
///
/// Kept only for unit tests that verify this cheat differs from overwrite.
#[cfg(test)]
pub(crate) fn apply_grain_centered_residual(
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
            .zip(probabilities.par_iter())
            .for_each(|((density, &fraction), &probability)| {
                // Test-only cheat: retain reduced dye as baseline and add a
                // centered particle residual. Production must overwrite instead.
                let noisy = *density + d_max * (fraction - probability);
                *density = positive_density_toe(noisy, d_max);
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

    /// Official Random123 known-answer tests for philox4x32-10
    /// (DEShawResearch/Random123 `tests/kat_vectors`).
    #[test]
    fn philox4x32_10_matches_random123_vectors() {
        let cases = [
            (
                [0u32, 0, 0, 0],
                [0u32, 0],
                [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8],
            ),
            (
                [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
                [0xffffffff, 0xffffffff],
                [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
            ),
            (
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0],
                [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
            ),
        ];
        for (ctr, key, expected) in cases {
            let mut rng = Philox4x32::new(key, ctr);
            let got = [rng.next_u32(), rng.next_u32(), rng.next_u32(), rng.next_u32()];
            assert_eq!(got, expected, "ctr={ctr:?} key={key:?}");
        }
    }

    /// Same counter block must be bit-identical across independent instances
    /// (counter-based reproducibility), and blocks must differ per pixel.
    #[test]
    fn philox_streams_are_per_pixel_and_reproducible() {
        let mut a = Philox4x32::per_pixel(1, 2, 0, 10, 20);
        let mut b = Philox4x32::per_pixel(1, 2, 0, 10, 20);
        assert_eq!(a.next_u32(), b.next_u32());
        assert_eq!(a.next_u32(), b.next_u32());

        let mut other = Philox4x32::per_pixel(1, 2, 0, 11, 20);
        assert_ne!(a.next_u32(), other.next_u32());
    }

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
    fn overwrite_particle_clouds_vary_at_mid_density() {
        let d_max = 2.0f32;
        let d = d_max / 2.0;
        let kappa = 0.15f32;
        let w = 256;
        let h = 256;
        let mut dyes = flat_dyes(d, d_max, w, h);
        let mask_before = dyes.mask_dye[0].clone();
        apply_particle_grain_overwrite(&mut dyes, &[d_max], &[kappa], &[1.0], 3.0, 123, &[None], 0.0, &[], 0.0, 0.0);
        let std = std_of(&dyes.image_dye[0]);
        assert!(std > 0.0, "overwrite grain should vary at mid density, std={std}");
        assert_eq!(dyes.mask_dye[0], mask_before);
    }

    #[test]
    fn grain_does_not_modulate_mask() {
        let d_max = 2.0f32;
        let mut a = flat_dyes(1.0, d_max, 64, 64);
        let mut b = flat_dyes(1.0, d_max, 64, 64);
        let mask_a = a.mask_dye[0].clone();
        apply_particle_grain_overwrite(&mut a, &[d_max], &[0.2], &[1.0], 3.0, 1, &[None], 0.0, &[], 0.0, 0.0);
        apply_particle_grain_overwrite(&mut b, &[d_max], &[0.0], &[1.0], 3.0, 1, &[None], 0.0, &[], 0.0, 0.0);
        assert_eq!(a.mask_dye[0], mask_a);
        assert_eq!(a.mask_dye[0], b.mask_dye[0]);
    }

    #[test]
    fn grain_flag_off() {
        let d_max = 2.0f32;
        let mut a = flat_dyes(1.0, d_max, 32, 32);
        let b = a.clone();
        apply_particle_grain_overwrite(&mut a, &[d_max], &[0.0], &[1.0], 3.0, 99, &[None], 0.0, &[], 0.0, 0.0);
        assert_eq!(a.image_dye, b.image_dye);
    }

    #[test]
    fn fixed_site_low_probability_has_spatial_variation() {
        let probability = 0.05f32;
        let w = 256;
        let h = 256;
        let n = w * h;
        let mut latent = LatentPlanes {
            width: w,
            height: h,
            layers: vec![vec![probability; n]],
        };
        apply_particle_grain_to_latent(&mut latent, &[0.18], 0.25, 17, &[None]);
        let std = std_of(&latent.layers[0]);
        assert!(
            std > 0.005,
            "low-probability fixed-site grain must retain shadow structure, std={std}"
        );
    }

    #[test]
    fn fixed_site_mean_tracks_probability() {
        let w = 256;
        let h = 256;
        let n = w * h;
        let kappa = 0.18f32;
        let pitch = 0.25f32;
        let seed = 23u64;
        for &probability in &[0.05f32, 0.2f32, 0.5f32, 0.8f32] {
            let mut latent = LatentPlanes {
                width: w,
                height: h,
                layers: vec![vec![probability; n]],
            };
            apply_particle_grain_to_latent(&mut latent, &[kappa], pitch, seed, &[None]);
            let mean = latent.layers[0].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            assert!(
                (mean - probability as f64).abs() < 0.03,
                "mean={mean} should track probability={probability}"
            );
        }
    }

    fn pearson_corr(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mean_b = b.iter().map(|&v| v as f64).sum::<f64>() / n;
        let cov = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - mean_a) * (y as f64 - mean_b))
            .sum::<f64>()
            / n;
        let var_a = a
            .iter()
            .map(|&v| (v as f64 - mean_a).powi(2))
            .sum::<f64>()
            / n;
        let var_b = b
            .iter()
            .map(|&v| (v as f64 - mean_b).powi(2))
            .sum::<f64>()
            / n;
        cov / (var_a * var_b).sqrt()
    }

    #[test]
    fn resolved_crystal_sites_persist_across_exposure() {
        let width = 128;
        let height = 128;
        let n = width * height;
        let seed = 17u64;
        let kappa = 0.4f32;
        let pitch = 0.25f32;
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
        let mut bright_other_seed = LatentPlanes {
            width,
            height,
            layers: vec![vec![0.8; n]],
        };
        apply_particle_grain_to_latent(&mut bright, &[kappa], pitch, seed, &[None]);
        apply_particle_grain_to_latent(&mut dark, &[kappa], pitch, seed, &[None]);
        apply_particle_grain_to_latent(&mut bright_other_seed, &[kappa], pitch, seed ^ 0xBEEF, &[None]);

        let bright_plane = &bright.layers[0];
        let dark_plane = &dark.layers[0];
        let bright_mean = bright_plane.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let dark_mean = dark_plane.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let cross_exposure = pearson_corr(bright_plane, dark_plane);
        let different_sites = pearson_corr(bright_plane, &bright_other_seed.layers[0]);
        assert!(bright_mean > dark_mean, "exposure should scale dye density");
        assert!(
            cross_exposure > different_sites + 0.15,
            "same-seed cross-exposure ({cross_exposure}) should exceed different-site baseline ({different_sites})"
        );
        assert!(
            cross_exposure > 0.35,
            "Bernoulli development should retain site-driven structure, correlation={cross_exposure}"
        );
    }

    #[test]
    fn low_fraction_overwrite_retains_spatial_variation() {
        use crate::film::StockId;

        let stock = StockId::Portra400.load().unwrap();
        let pitch = 0.5f32 * 1000.0 / 3024.0; // ≈0.165 µm, Milestone3 skirt pitch
        let w = 256usize;
        let h = 256usize;
        let n = w * h;
        let layers = stock.emulsion_layers().count();
        let d_max: Vec<f32> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.coupler.as_ref().unwrap().d_max)
            .collect();
        let kappas: Vec<f32> = stock
            .emulsion_layers()
            .map(|(idx, _)| stock.grain_kappa[idx].unwrap_or(0.0))
            .collect();
        let crystal_sizes: Vec<_> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.crystal_size.clone())
            .collect();
        let gammas: Vec<f32> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.gamma_contrast)
            .collect();

        for &f in &[0.001f32, 0.005, 0.01, 0.02, 0.05, 0.1] {
            let mut dyes = DyePlanes {
                width: w,
                height: h,
                image_dye: d_max
                    .iter()
                    .zip(gammas.iter())
                    .map(|(&d, &gamma)| {
                        vec![d * f.powf(1.0 / gamma.max(1e-6)); n]
                    })
                    .collect(),
                mask_dye: vec![vec![0.0; n]; layers],
            };
            apply_particle_grain_overwrite(
                &mut dyes,
                &d_max,
                &kappas,
                &gammas,
                pitch,
                42,
                &crystal_sizes, 0.0, &[], 0.0, 0.0);

            for (i, plane) in dyes.image_dye.iter().enumerate() {
                let std = std_of(plane);
                assert!(
                    std > 1e-4,
                    "layer {i} at f={f} must retain shadow grain structure, std={std}"
                );
            }
        }
    }

    #[test]
    fn overwrite_recovers_developable_fraction_at_low_f() {
        let d_max = 2.0f32;
        let gamma = 0.54f32;
        let developable_f = 0.05f32;
        let density = d_max * developable_f.powf(1.0 / gamma);
        let p_hd = density / d_max;
        assert!(
            developable_f > p_hd * 2.0,
            "γ<1 shadows: recovered f={developable_f} must exceed p_hd={p_hd}"
        );

        let w = 256;
        let h = 256;
        let mut recovered = flat_dyes(density, d_max, w, h);
        let mut p_hd_only = flat_dyes(density, d_max, w, h);
        apply_particle_grain_overwrite(
            &mut recovered,
            &[d_max],
            &[0.18],
            &[gamma],
            0.25,
            17,
            &[None], 0.0, &[], 0.0, 0.0);
        apply_particle_grain_overwrite(
            &mut p_hd_only,
            &[d_max],
            &[0.18],
            &[1.0],
            0.25,
            17,
            &[None], 0.0, &[], 0.0, 0.0);

        let std_recovered = std_of(&recovered.image_dye[0]);
        let std_p_hd = std_of(&p_hd_only.image_dye[0]);
        let nonzero_recovered = recovered
            .image_dye[0]
            .iter()
            .filter(|&&v| v > 1e-6)
            .count();
        let nonzero_p_hd = p_hd_only
            .image_dye[0]
            .iter()
            .filter(|&&v| v > 1e-6)
            .count();
        assert!(
            std_recovered > std_p_hd + 1e-4 || nonzero_recovered > nonzero_p_hd + 10,
            "γ recovery should yield more shadow structure: std_rec={std_recovered} std_p_hd={std_p_hd} nonzero_rec={nonzero_recovered} nonzero_p_hd={nonzero_p_hd}"
        );
    }

    #[test]
    fn dark_particle_overwrite_keeps_continuous_dye_clouds() {
        let d_max = 1.9f32;
        let gamma = 0.34f32;
        let dark_fraction = 0.2f32;
        let baseline = d_max * dark_fraction.powf(1.0 / gamma);
        let mut dyes = flat_dyes(baseline, d_max, 256, 256);

        apply_particle_grain_overwrite(
            &mut dyes,
            &[d_max],
            &[0.18],
            &[gamma],
            0.25,
            17,
            &[None], 0.0, &[], 0.0, 0.0);

        let plane = &dyes.image_dye[0];
        let min = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let max = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let nonzero = plane.iter().filter(|&&v| v > 1e-6).count();
        assert!(max > 0.0, "particle clouds must form at least some dye");
        assert!(nonzero > 0, "some pixels must develop dye");
        assert!(max > min, "particle clouds should vary optically");
        assert!(
            plane.iter().any(|&value| value > min + 1e-6 && value < max - 1e-6),
            "cloud density should vary continuously, not only by occupancy"
        );
    }

}
