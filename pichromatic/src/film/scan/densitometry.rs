//! Densitometric scan: stacked dyes → transmittance → ACEScg encoding.
//!
//! T(λ) = 10^(−Σ D_layer_eff(λ)) with base-10 optical density.
//! Channel via CIE CMFs → XYZ → ACEScg. Scanner gain is normalized against the
//! base-only reference; PositiveLinear separately measures processed Dmin,
//! including chemical fog, through the complete CPU path.
//!
//! All per-pixel math runs in f32 (WGSL has no core f64), mirroring the GPU
//! `SCAN` / `GRAIN_SCAN_ROI` shaders.

use crate::film::blur::gaussian_blur_separable;
use crate::film::constants::{SCANNER_OPTICAL_SIGMA_UM, SCANNER_SENSOR_SIGMA_PX};
use crate::film::development::develop;
use crate::film::error::FilmError;
use crate::film::exposure::expose_with_pitch_and_shutter;
use crate::film::exposure::upsample::spectrum_to_acescg_rgb_f32;
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::DyePlanes;
use crate::pixel::{ImageBuffer, Pixel, MIDDLE_GRAY};
use rayon::prelude::*;

/// log2(10), shared with the GPU `SCAN` shader (`exp2(-d·LOG2_10)`).
const LOG2_10: f32 = 3.3219280948873623;

const CALIBRATION_TILE_EDGE: usize = 256;
const CALIBRATION_MAX_TILES: usize = 128;
/// Numerical output-error budget: one linear 9-bit code value across the
/// complete bounded PositiveLinear transfer.
const CALIBRATION_OUTPUT_TOLERANCE: f32 = 1.0 / 512.0;
const CALIBRATION_STABLE_ROUNDS: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct ScannerCalibration {
    pub dmin: [f32; 3],
    pub mid: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct CalibrationResult {
    calibration: ScannerCalibration,
    converged: bool,
    #[cfg(test)]
    exponents: [f32; 3],
    #[cfg(test)]
    tiles: usize,
    #[cfg(test)]
    last_deltas: [f32; CALIBRATION_STABLE_ROUNDS],
    #[cfg(test)]
    delta_history: [f32; 8],
    #[cfg(test)]
    delta_count: usize,
}

/// Effective spectral density at one pixel: Σ_layers (D_image * ε_image + D_mask * ε_mask).
pub(crate) fn density_spectrum(stock: &FilmStock, dyes: &DyePlanes, pixel: usize) -> [f32; 16] {
    let mut d = [0.0f32; 16];
    let mut emulsion_i = 0usize;
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().unwrap();
        let di = dyes.image_dye[emulsion_i][pixel];
        let dm = dyes.mask_dye[emulsion_i][pixel];
        for lambda in 0..16 {
            d[lambda] += di * coupler.epsilon.samples[lambda] as f32;
            if let Some(ref mask_eps) = coupler.mask_epsilon {
                d[lambda] += dm * mask_eps.samples[lambda] as f32;
            }
        }
        emulsion_i += 1;
    }
    d
}

pub(crate) fn transmittance_from_density(d: &[f32; 16]) -> [f32; 16] {
    let mut t = [0.0f32; 16];
    for i in 0..16 {
        // 10^-d via exp2, mirroring the `SCAN` shader.
        t[i] = (-d[i] * LOG2_10).exp2();
    }
    t
}

/// Unnormalized scanner spectrum for one pixel.
fn scan_pixel_spectrum(stock: &FilmStock, dyes: &DyePlanes, pixel: usize) -> [f32; 16] {
    let dens = density_spectrum(stock, dyes, pixel);
    let mut spectrum = transmittance_from_density(&dens);
    for (value, &light) in spectrum.iter_mut().zip(stock.scanner_light.samples.iter()) {
        *value *= light as f32;
    }
    spectrum
}

/// Compute scanner optical lens and detector pixel aperture Gaussian PSF standard deviation in pixels.
///
/// Models incoherent optical intensity integration over the scanner's lens OTF and
/// detector sampling floor (OLPF, pixel aperture fill factor, and optical reconstruction):
/// sigma_scanner_px = sqrt((sigma_opt_um / pitch)^2 + sigma_sensor_px^2)
pub fn scanner_aperture_sigma_px(pixel_pitch_um: f32) -> f32 {
    let opt_px = SCANNER_OPTICAL_SIGMA_UM / pixel_pitch_um.max(1e-6);
    (opt_px * opt_px + SCANNER_SENSOR_SIGMA_PX * SCANNER_SENSOR_SIGMA_PX).sqrt()
}

/// Convolve linear scanned RGB with the scanner optical and pixel aperture PSF.
///
/// Models incoherent optical intensity integration over the scanner's lens OTF and
/// detector pixel aperture. Since ACEScg is linear in transmitted radiant flux,
/// this spatial convolution is physically exact. Energy/mean flux is strictly conserved.
pub fn apply_scanner_aperture_mtf(
    raw: &mut [Pixel],
    width: usize,
    height: usize,
    sigma_px: f32,
) {
    if sigma_px < 1e-3 || width <= 1 || height <= 1 {
        return;
    }
    let n = width * height;
    assert_eq!(raw.len(), n);

    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for px in raw.iter() {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }

    let mut channels = [r, g, b];
    channels.par_iter_mut().for_each(|plane| {
        gaussian_blur_separable(plane, width, height, sigma_px);
    });

    for (p, px) in raw.iter_mut().enumerate() {
        px[0] = channels[0][p];
        px[1] = channels[1][p];
        px[2] = channels[2][p];
    }
}

/// Scan dye planes to interleaved ACEScg RGB with base-reference scanner gain
/// and scanner optical / pixel aperture MTF at the given pixel pitch.
pub fn scan_to_acescg(
    stock: &FilmStock,
    dyes: &DyePlanes,
    pixel_pitch_um: f32,
) -> ImageBuffer {
    let n = dyes.width * dyes.height;

    // First pass: compute raw ACEScg from T(λ) * I_s(λ).
    let mut raw: Vec<Pixel> = (0..n)
        .into_par_iter()
        .map(|p| {
            let t = scan_pixel_spectrum(stock, dyes, p);
            spectrum_to_acescg_rgb_f32(&t)
        })
        .collect();

    // Scanner gain reference: zero image dye, undeveloped mask at its maximum.
    let dmin_rgb = dmin_reference_acescg(stock);
    let peak = dmin_rgb[0].max(dmin_rgb[1]).max(dmin_rgb[2]).max(1e-12);
    let scale = 1.0 / peak;
    raw.par_iter_mut().for_each(|px| {
        *px = px.map(|c| c * scale);
    });

    // Scanner optical / pixel aperture MTF: incoherent optical intensity integration.
    let sigma_px = scanner_aperture_sigma_px(pixel_pitch_um);
    apply_scanner_aperture_mtf(
        &mut raw,
        dyes.width,
        dyes.height,
        sigma_px,
    );

    raw
}

/// Base-only ACEScg before scanner peak-normalization (raw densitometric units).
pub fn dmin_reference_acescg(stock: &FilmStock) -> [f32; 3] {
    // Synthesize a 1-pixel DyePlanes at Dmin: image=0, mask=max residual.
    let mut image_dye = Vec::new();
    let mut mask_dye = Vec::new();
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().unwrap();
        image_dye.push(vec![0.0f32]);
        let mask = if coupler.mask_epsilon.is_some() {
            use crate::film::constants::MASK_DENSITY_FRACTION_OF_DMAX;
            coupler.d_max * MASK_DENSITY_FRACTION_OF_DMAX
        } else {
            0.0
        };
        mask_dye.push(vec![mask]);
    }
    let dyes = DyePlanes {
        width: 1,
        height: 1,
        image_dye,
        mask_dye,
    };
    let t = scan_pixel_spectrum(stock, &dyes, 0);
    spectrum_to_acescg_rgb_f32(&t)
}

/// Scan-normalized base-only RGB (peak ≈ 1).
///
/// Retained for scanner normalization and the intentionally desynchronized GPU.
/// CPU PositiveLinear uses [`scanner_calibration_acescg`] instead.
pub fn normalized_dmin_acescg(stock: &FilmStock) -> [f32; 3] {
    let d = dmin_reference_acescg(stock);
    let peak = d[0].max(d[1]).max(d[2]).max(1e-12);
    [d[0] / peak, d[1] / peak, d[2] / peak]
}

/// Joint processed-Dmin and neutral-mid calibration from deterministic,
/// independent equal-area samples at the render's actual film pitch.
///
/// Dmin and mid always use the same tile geometry, seeds, and stopping rule.
/// Tile count doubles until two successive area doublings change the complete
/// bounded `1 - clamp(scan / Dmin)^a` transfer by at most the numerical output
/// budget, or returns [`FilmError::ScannerCalibrationDidNotConverge`] at the
/// bounded 128-tile cap (8,388,608 pixels per reference).
pub fn scanner_calibration_acescg(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
) -> Result<ScannerCalibration, FilmError> {
    calibration_with_limit(
        stock,
        pixel_pitch_um,
        shutter_seconds,
        CALIBRATION_MAX_TILES,
    )
}

/// Processed unexposed-film scan retained as a convenience for CPU probes.
pub fn processed_dmin_acescg(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
) -> Result<[f32; 3], FilmError> {
    Ok(scanner_calibration_acescg(stock, pixel_pitch_um, shutter_seconds)?.dmin)
}

fn calibration_with_limit(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
    max_tiles: usize,
) -> Result<ScannerCalibration, FilmError> {
    let result = calibration_attempt(stock, pixel_pitch_um, shutter_seconds, max_tiles);
    if result.converged {
        Ok(result.calibration)
    } else {
        Err(FilmError::ScannerCalibrationDidNotConverge)
    }
}

fn calibration_attempt(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
    max_tiles: usize,
) -> CalibrationResult {
    let tile_edge = if pixel_pitch_um < 1.0 {
        512
    } else {
        CALIBRATION_TILE_EDGE
    };
    let n = tile_edge * tile_edge;
    let dmin_rgb = vec![[0.0; 3]; n];
    let mid_value = crate::film::relative_to_absolute_y(MIDDLE_GRAY, stock.box_iso.0);
    let mid_rgb = vec![[mid_value; 3]; n];
    let dmin_latent = expose_with_pitch_and_shutter(
        &dmin_rgb,
        tile_edge,
        tile_edge,
        stock,
        pixel_pitch_um,
        shutter_seconds,
    );
    let mid_latent = expose_with_pitch_and_shutter(
        &mid_rgb,
        tile_edge,
        tile_edge,
        stock,
        pixel_pitch_um,
        shutter_seconds,
    );

    let mut sums = [[0.0f64; 3]; 2];
    let mut tiles = 0usize;
    let mut target_tiles = 1usize;
    let mut previous_calibration = None;
    let mut stable_rounds = 0usize;
    let mut last_deltas = [f32::INFINITY; CALIBRATION_STABLE_ROUNDS];
    #[cfg(test)]
    let mut delta_history = [f32::NAN; 8];
    #[cfg(test)]
    let mut delta_count = 0usize;
    let mut calibration = ScannerCalibration {
        dmin: [0.0; 3],
        mid: [0.0; 3],
    };
    #[cfg(test)]
    let mut exponents = [0.0; 3];

    while tiles < max_tiles {
        target_tiles = target_tiles.min(max_tiles);
        while tiles < target_tiles {
            for (reference_index, latent) in [&dmin_latent, &mid_latent].into_iter().enumerate() {
                let reference = &mut sums[reference_index];
                let dyes = develop(stock, latent, tiles as u64, pixel_pitch_um);
                for px in scan_to_acescg(stock, &dyes, pixel_pitch_um) {
                    for c in 0..3 {
                        reference[c] += px[c] as f64;
                    }
                }
            }
            tiles += 1;
        }

        let samples = (tiles * n) as f64;
        calibration = ScannerCalibration {
            dmin: sums[0].map(|v| (v / samples) as f32),
            mid: sums[1].map(|v| (v / samples) as f32),
        };
        #[cfg(test)]
        {
            exponents = super::invert::invert_constants(calibration.mid, calibration.dmin).exponent;
        }

        if let Some(previous) = previous_calibration {
            let delta = max_transfer_delta(previous, calibration);
            #[cfg(test)]
            {
                delta_history[delta_count] = delta;
                delta_count += 1;
            }
            last_deltas.rotate_left(1);
            last_deltas[CALIBRATION_STABLE_ROUNDS - 1] = delta;
            stable_rounds = if delta <= CALIBRATION_OUTPUT_TOLERANCE {
                stable_rounds + 1
            } else {
                0
            };
            if stable_rounds == CALIBRATION_STABLE_ROUNDS {
                return CalibrationResult {
                    calibration,
                    converged: true,
                    #[cfg(test)]
                    exponents,
                    #[cfg(test)]
                    tiles,
                    #[cfg(test)]
                    last_deltas,
                    #[cfg(test)]
                    delta_history,
                    #[cfg(test)]
                    delta_count,
                };
            }
        }
        previous_calibration = Some(calibration);
        target_tiles = (target_tiles * 2).min(max_tiles);
    }

    CalibrationResult {
        calibration,
        converged: false,
        #[cfg(test)]
        exponents,
        #[cfg(test)]
        tiles,
        #[cfg(test)]
        last_deltas,
        #[cfg(test)]
        delta_history,
        #[cfg(test)]
        delta_count,
    }
}

fn max_transfer_delta(old: ScannerCalibration, new: ScannerCalibration) -> f32 {
    let old_constants = super::invert::invert_constants(old.mid, old.dmin);
    let new_constants = super::invert::invert_constants(new.mid, new.dmin);
    (0..3)
        .map(|c| {
            max_channel_transfer_delta(
                old_constants.dmin[c],
                old_constants.exponent[c],
                new_constants.dmin[c],
                new_constants.exponent[c],
                old_constants.eps,
            )
        })
        .fold(0.0, f32::max)
}

fn max_channel_transfer_delta(d1: f32, a1: f32, d2: f32, a2: f32, eps: f32) -> f32 {
    let transfer = |scan: f64, dmin: f32, exponent: f32| {
        1.0 - (scan / dmin as f64)
            .clamp(eps as f64, 1.0)
            .powf(exponent as f64)
    };
    let mut candidates = vec![
        0.0,
        eps as f64 * d1 as f64,
        eps as f64 * d2 as f64,
        d1 as f64,
        d2 as f64,
    ];

    // Between both clamp knees the difference is two power laws. Its only
    // possible interior extremum is the derivative root below; all other
    // pieces are constant or monotonic, so their endpoints above are enough.
    if (a1 - a2).abs() > f32::EPSILON {
        let ln_scan =
            ((a2 / a1) as f64).ln() + a1 as f64 * (d1 as f64).ln() - a2 as f64 * (d2 as f64).ln();
        let scan = (ln_scan / (a1 - a2) as f64).exp();
        let active_lo = eps as f64 * d1.max(d2) as f64;
        let active_hi = d1.min(d2) as f64;
        if scan.is_finite() && scan >= active_lo && scan <= active_hi {
            candidates.push(scan);
        }
    }

    candidates
        .into_iter()
        .map(|scan| (transfer(scan, d1, a1) - transfer(scan, d2, a2)).abs() as f32)
        .fold(0.0, f32::max)
}

#[cfg(test)]
mod calibration_tests {
    use super::*;
    use crate::film::scan::invert::invert_constants;
    use crate::film::stock::StockId;

    fn single_patch_calibration(
        stock: &FilmStock,
        pitch_um: f32,
        shutter_seconds: f32,
        edge: usize,
    ) -> ScannerCalibration {
        let n = edge * edge;
        let mid_value = crate::film::relative_to_absolute_y(MIDDLE_GRAY, stock.box_iso.0);
        let scan_mean = |value: f32| {
            let rgb = vec![[value; 3]; n];
            let latent =
                expose_with_pitch_and_shutter(&rgb, edge, edge, stock, pitch_um, shutter_seconds);
            let dyes = develop(stock, &latent, 0, pitch_um);
            super::super::invert::mean_rgb(&scan_to_acescg(stock, &dyes, pitch_um))
        };
        ScannerCalibration {
            dmin: scan_mean(0.0),
            mid: scan_mean(mid_value),
        }
    }

    #[test]
    fn half_mm_pre_rotation_calibration_converges_beyond_single_32px_patch() {
        let mut stock = StockId::Portra400.load().unwrap();
        stock.antihalation.reflectance = crate::film::spectrum::SpectralCurve::constant(0.0);
        let pitch_um = 500.0 / 4032.0;
        let shutter = 1.0 / 121.0;
        let old = single_patch_calibration(&stock, pitch_um, shutter, 32);
        let old_exponents = invert_constants(old.mid, old.dmin).exponent;
        let result = calibration_attempt(&stock, pitch_um, shutter, CALIBRATION_MAX_TILES);

        eprintln!(
            "0.5mm scanner calibration: old32 exp={old_exponents:?} dmin={:?}, final exp={:?} dmin={:?}, tiles={}, deltas_by_doubling={:?}",
            old.dmin,
            result.exponents,
            result.calibration.dmin,
            result.tiles,
            &result.delta_history[..result.delta_count]
        );
        assert!(
            max_transfer_delta(old, result.calibration) > CALIBRATION_OUTPUT_TOLERANCE,
            "the old 5.3um-wide N32 estimate must fail the output convergence gate"
        );
        assert!(result.converged, "calibration hit its bounded tile cap");
        assert!(
            result
                .last_deltas
                .iter()
                .all(|&delta| delta <= CALIBRATION_OUTPUT_TOLERANCE),
            "two additional area doublings must independently satisfy the transfer-output tolerance: {:?}",
            result.last_deltas
        );
        assert!(
            result.tiles * CALIBRATION_TILE_EDGE * CALIBRATION_TILE_EDGE > 32 * 32,
            "microscope calibration must sample more area than the old N32 patch"
        );
    }

    #[test]
    fn calibration_cap_returns_error() {
        let stock = StockId::Portra400.load().unwrap();
        let error =
            calibration_with_limit(&stock, 500.0 / 3024.0, 1.0 / stock.box_iso.0, 1).unwrap_err();
        assert_eq!(error, FilmError::ScannerCalibrationDidNotConverge);
    }

    #[test]
    fn convergence_metric_includes_dmin_shift() {
        let old = ScannerCalibration {
            dmin: [1.0; 3],
            mid: [0.5; 3],
        };
        let shifted = ScannerCalibration {
            dmin: [0.9; 3],
            mid: [0.45; 3],
        };
        assert_eq!(
            invert_constants(old.mid, old.dmin).exponent,
            invert_constants(shifted.mid, shifted.dmin).exponent,
            "fixture keeps exponents identical"
        );
        assert!(
            max_transfer_delta(old, shifted) > CALIBRATION_OUTPUT_TOLERANCE,
            "absolute scanner-domain convergence must detect a Dmin-only shift"
        );
    }

    #[test]
    fn scanner_aperture_sampling_floor_matches_expected_limits() {
        // At coarse pitch / 35mm film (~11.9 um pitch), the optical blur in pixels is small
        // and the detector sampling floor (~0.65 px) dominates.
        let sigma_35mm = scanner_aperture_sigma_px(11.9);
        assert!(
            (sigma_35mm - 0.671).abs() < 0.01,
            "35mm scanner sigma {sigma_35mm} should be around 0.67 px"
        );

        // At infinite pitch, sigma approaches the detector sampling floor SCANNER_SENSOR_SIGMA_PX.
        let sigma_inf = scanner_aperture_sigma_px(1e6);
        assert!(
            (sigma_inf - SCANNER_SENSOR_SIGMA_PX).abs() < 1e-4,
            "infinite pitch sigma {sigma_inf} must approach SCANNER_SENSOR_SIGMA_PX {SCANNER_SENSOR_SIGMA_PX}"
        );

        // At 1.0 um microscope pitch, lens optical OTF (~2.0 um) dominates.
        let sigma_1um = scanner_aperture_sigma_px(1.0);
        assert!(
            (sigma_1um - 2.103).abs() < 0.01,
            "1um pitch scanner sigma {sigma_1um} should be around 2.10 px"
        );
    }
}

