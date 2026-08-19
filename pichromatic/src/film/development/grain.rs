//! Discrete shot-noise grain on image-forming dye only.
//!
//! The LUT supplies the local expected dye density after H&D reduction. At any
//! render width the model samples the same stock-derived particle population
//! in that reduced-density space, then spreads the resulting dye clouds
//! through a cloud-scale PSF. Reduced
//! H&D density supplies the Bernoulli probability directly
//! (`p = D_expected / D_max`). The realized developed fraction is the sole
//! image carrier and contributes additive optical density
//! (`D_out = D_max·f_realized`). No smooth expected-density field survives.
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

use crate::film::blur::gaussian_blur_separable;
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
        Self {
            key,
            ctr,
            buf: [0; 4],
            pos: 4,
        }
    }

    /// Independent per-pixel stream for `(seed, layer, sublayer)` at `(x, y)`.
    ///
    /// The key mixes the seed with layer/sublayer tags; the counter carries
    /// the pixel coordinates plus a block index so each site is independent.
    pub fn per_pixel(seed: u64, layer: u32, sublayer: u32, x: u32, y: u32) -> Self {
        let key = [
            (seed as u32) ^ layer.wrapping_mul(PHILOX_W0) ^ sublayer.wrapping_mul(PHILOX_W1),
            ((seed >> 32) as u32)
                ^ layer.wrapping_mul(PHILOX_W1)
                ^ sublayer.wrapping_mul(PHILOX_W0),
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

/// Convert the measured circular cloud extent to an equivalent Gaussian sigma.
/// A disk of diameter D has per-axis variance (D/4)^2.
fn dye_cloud_sigma_um() -> f32 {
    DYE_CLOUD_CORRELATION_UM * 0.25
}

/// Sample fixed crystal sites in each output-pixel aperture, develop each site
/// by Bernoulli probability, then spread their normalized dye mass once by the
/// effective dye cloud + pixel aperture PSF. The sparse impulses are the physical
/// population; no regular grid, box-averaged probability, or bilinear scene
/// reconstruction is used.
///
/// In physics, crystal sites are distributed continuously across the pixel aperture
/// area ([-p/2, p/2]^2), and dye clouds have continuous Gaussian extent σ_cloud.
/// The effective PSF integrated over a square pixel aperture has combined variance
/// σ_eff^2 = σ_cloud^2 + σ_aperture^2 where σ_aperture^2 = p^2 / 12.
/// In pixel units: σ_eff_px = sqrt((σ_cloud / p)^2 + 1/12).
fn particle_field(
    probabilities: &[f32],
    width: usize,
    height: usize,
    kappa_ref: f32,
    pixel_pitch_um: f32,
    seed: u64,
    record_i: u32,
    sublayer_i: u32,
) -> Vec<f32> {
    assert_eq!(probabilities.len(), width * height);
    assert!(
        kappa_ref.is_finite() && kappa_ref > 0.0,
        "invalid grain kappa"
    );
    assert!(
        pixel_pitch_um.is_finite() && pixel_pitch_um > 0.0,
        "invalid pixel pitch"
    );
    let rho_areal = 1.0 / (kappa_ref * kappa_ref).max(1e-12);
    let expected_sites = rho_areal * pixel_pitch_um * pixel_pitch_um;
    assert!(
        expected_sites.is_finite() && expected_sites > 0.0,
        "invalid expected crystal population"
    );

    let mut particles = vec![0.0f32; width * height];
    particles
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, slot)| {
            let x = index % width;
            let y = index / width;
            let probability = probabilities[index].clamp(0.0, 1.0);
            let mut rng = Philox4x32::per_pixel(seed, record_i, sublayer_i, x as u32, y as u32);
            // Independent Poisson counts in disjoint pixel apertures are
            // distributionally the exact restriction of a homogeneous physical
            // Poisson point process. Depositing each aperture's count at its
            // pixel center is the finite-aperture approximation: it omits the
            // sites' subpixel coordinates, not their count statistics.
            // Counts and thresholds remain deterministic for this seed/layer/pixel.
            let sites = sample_poisson(&mut rng, expected_sites);
            let developed = sample_binomial(&mut rng, sites, probability);
            *slot = developed as f32 / expected_sites;
        });
    let cloud_sigma_px = dye_cloud_sigma_um() / pixel_pitch_um;
    let eff_sigma_px = (cloud_sigma_px * cloud_sigma_px + 1.0 / 12.0).sqrt();
    gaussian_blur_separable(&mut particles, width, height, eff_sigma_px);
    particles
}

/// Reconstruct latent developable fractions from the discrete particle
/// population. This is a diagnostic/latent-side grain path; production develop
/// uses `apply_particle_grain_overwrite`. DIR is off.
pub fn apply_particle_grain_to_latent(
    latent: &mut LatentPlanes,
    kappa_ref_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
) {
    for (layer_i, plane) in latent.layers.iter_mut().enumerate() {
        let kappa_ref = kappa_ref_per_layer[layer_i];
        assert!(
            kappa_ref.is_finite() && kappa_ref > 0.0,
            "invalid grain kappa"
        );
        let probabilities = plane.clone();
        *plane = particle_field(
            &probabilities,
            latent.width,
            latent.height,
            kappa_ref,
            pixel_pitch_um,
            seed,
            layer_i as u32,
            0,
        );
    }
}

/// Production grain at every width: replace image dye planes with a particle
/// realization derived from reduced developable probabilities.
///
/// Reduced H&D density supplies only the local expected population statistics;
/// it must not survive as a scene-bearing baseline layer in the output.
///
/// The reduced density sets only `p = D_expected / d_max`. Developed sites
/// contribute additive dye density, `D = d_max·f_realized`; γ therefore shapes
/// the expected probability through `reduce` but cannot amplify microscopic
/// fluctuations after realization.
pub fn apply_particle_grain_overwrite(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_ref_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
) {
    apply_particle_grain_overwrite_with_sublayers(
        dyes,
        d_max_per_layer,
        kappa_ref_per_layer,
        pixel_pitch_um,
        seed,
        None,
    );
}

/// Variant of `apply_particle_grain_overwrite` that accepts explicit (record, sublayer)
/// indices for physical multi-speed layer coupling.
pub fn apply_particle_grain_overwrite_with_sublayers(
    dyes: &mut DyePlanes,
    d_max_per_layer: &[f32],
    kappa_ref_per_layer: &[f32],
    pixel_pitch_um: f32,
    seed: u64,
    layer_sublayers: Option<&[(u32, u32)]>,
) {
    assert_eq!(dyes.image_dye.len(), d_max_per_layer.len());
    assert_eq!(dyes.image_dye.len(), kappa_ref_per_layer.len());
    let mut f_dev = vec![vec![0.0f32; dyes.width * dyes.height]; dyes.image_dye.len()];
    for (layer_i, plane) in dyes.image_dye.iter().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        let kappa_ref = kappa_ref_per_layer[layer_i];
        assert!(d_max.is_finite() && d_max > 0.0, "invalid dye d_max");
        assert!(
            kappa_ref.is_finite() && kappa_ref > 0.0,
            "invalid grain kappa"
        );
        let (rec_i, sub_i) = match layer_sublayers {
            Some(indices) if layer_i < indices.len() => indices[layer_i],
            _ => (layer_i as u32, 0),
        };
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
            seed,
            rec_i,
            sub_i,
        );
        f_dev[layer_i] = particles;
    }

    dyes.image_dye = f_dev;

    for (layer_i, plane) in dyes.image_dye.iter_mut().enumerate() {
        let d_max = d_max_per_layer[layer_i];
        plane.par_iter_mut().for_each(|fraction| *fraction *= d_max);
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

    #[test]
    fn circular_cloud_diameter_maps_to_gaussian_sigma() {
        assert_eq!(dye_cloud_sigma_um(), DYE_CLOUD_CORRELATION_UM * 0.25);
    }

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
            let got = [
                rng.next_u32(),
                rng.next_u32(),
                rng.next_u32(),
                rng.next_u32(),
            ];
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
        apply_particle_grain_overwrite(&mut dyes, &[d_max], &[kappa], 3.0, 123);
        let std = std_of(&dyes.image_dye[0]);
        assert!(
            std > 0.0,
            "overwrite grain should vary at mid density, std={std}"
        );
        assert_eq!(dyes.mask_dye[0], mask_before);
    }

    #[test]
    fn grain_does_not_modulate_mask() {
        let d_max = 2.0f32;
        let mut a = flat_dyes(1.0, d_max, 64, 64);
        let mask_a = a.mask_dye[0].clone();
        apply_particle_grain_overwrite(&mut a, &[d_max], &[0.2], 3.0, 1);
        assert_eq!(a.mask_dye[0], mask_a);
    }

    #[test]
    #[should_panic(expected = "invalid grain kappa")]
    fn invalid_population_cannot_preserve_smooth_density() {
        let d_max = 2.0f32;
        let mut a = flat_dyes(1.0, d_max, 32, 32);
        apply_particle_grain_overwrite(&mut a, &[d_max], &[0.0], 3.0, 99);
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
        apply_particle_grain_to_latent(&mut latent, &[0.18], 0.25, 17);
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
            apply_particle_grain_to_latent(&mut latent, &[kappa], pitch, seed);
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
        let var_a = a.iter().map(|&v| (v as f64 - mean_a).powi(2)).sum::<f64>() / n;
        let var_b = b.iter().map(|&v| (v as f64 - mean_b).powi(2)).sum::<f64>() / n;
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
        apply_particle_grain_to_latent(&mut bright, &[kappa], pitch, seed);
        apply_particle_grain_to_latent(&mut dark, &[kappa], pitch, seed);
        apply_particle_grain_to_latent(&mut bright_other_seed, &[kappa], pitch, seed ^ 0xBEEF);

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
        let pitch = 0.5f32 * 1000.0 / 4032.0; // ≈0.124 µm, pre-rotation DNG width
        let w = 128usize;
        let h = 128usize;
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
        let gammas: Vec<f32> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.gamma_contrast)
            .collect();

        for &f in &[0.001f32, 0.005, 0.01, 0.02, 0.05, 0.1] {
            let expected: Vec<f64> = d_max
                .iter()
                .zip(gammas.iter())
                .map(|(&d, &gamma)| (d * f.powf(1.0 / gamma.max(1e-6))) as f64)
                .collect();
            let mut sums = vec![0.0f64; layers];
            let mut has_variation = vec![false; layers];
            for seed in 0..32 {
                let mut dyes = DyePlanes {
                    width: w,
                    height: h,
                    image_dye: expected
                        .iter()
                        .map(|&density| vec![density as f32; n])
                        .collect(),
                    mask_dye: vec![vec![0.0; n]; layers],
                };
                apply_particle_grain_overwrite(&mut dyes, &d_max, &kappas, pitch, seed);
                for (i, plane) in dyes.image_dye.iter().enumerate() {
                    sums[i] += plane.iter().map(|&value| value as f64).sum::<f64>();
                    has_variation[i] |= std_of(plane) > 0.0;
                }
            }

            for i in 0..layers {
                // Reduced H&D density is the expected additive dye density.
                // Ensemble realization must preserve that mean without a
                // second γ power; one tiny crop can contain only a few events.
                let mean = sums[i] / (32 * n) as f64;
                if f >= 0.02 {
                    assert!(
                        mean > 0.6 * expected[i] && mean < 1.6 * expected[i],
                        "layer {i} at f={f}: output mean {mean:.5} must track H&D {:.5}",
                        expected[i]
                    );
                    assert!(has_variation[i], "layer {i} at f={f} must vary spatially");
                } else {
                    assert!(
                        (mean - expected[i]).abs() < 1e-3,
                        "layer {i} at f={f}: output mean {mean:.5} deviates from fog-floor H&D {:.5}",
                        expected[i]
                    );
                }
            }
        }
    }

    #[test]
    fn overwrite_probability_is_expected_density_fraction() {
        let d_max = 2.0f32;
        let density = d_max * 0.05;
        let w = 256usize;
        let h = 256usize;
        let mut dyes = flat_dyes(density, d_max, w, h);
        apply_particle_grain_overwrite(&mut dyes, &[d_max], &[0.18], 0.25, 17);
        let mean = dyes.image_dye[0]
            .iter()
            .map(|&value| value as f64)
            .sum::<f64>()
            / (w * h) as f64;
        assert!(
            (mean - density as f64).abs() < 0.02,
            "realized mean {mean} must track expected density {density}"
        );
    }

    #[test]
    fn dark_particle_overwrite_keeps_continuous_dye_clouds() {
        let d_max = 1.9f32;
        let gamma = 0.34f32;
        let dark_fraction = 0.2f32;
        let baseline = d_max * dark_fraction.powf(1.0 / gamma);
        let mut dyes = flat_dyes(baseline, d_max, 256, 256);

        apply_particle_grain_overwrite(&mut dyes, &[d_max], &[0.18], 0.25, 17);

        let plane = &dyes.image_dye[0];
        let min = plane.iter().copied().fold(f32::INFINITY, f32::min);
        let max = plane.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let nonzero = plane.iter().filter(|&&v| v > 1e-6).count();
        assert!(max > 0.0, "particle clouds must form at least some dye");
        assert!(nonzero > 0, "some pixels must develop dye");
        assert!(max > min, "particle clouds should vary optically");
        assert!(
            plane
                .iter()
                .any(|&value| value > min + 1e-6 && value < max - 1e-6),
            "cloud density should vary continuously, not only by occupancy"
        );
    }
}
