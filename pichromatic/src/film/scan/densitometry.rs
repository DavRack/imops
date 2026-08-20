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
use crate::film::error::FilmError;
use crate::film::exposure::upsample::spectrum_to_acescg_rgb_f32;
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::DyePlanes;
use crate::pixel::{ImageBuffer, Pixel};
use rayon::prelude::*;

/// log2(10), shared with the GPU `SCAN` shader (`exp2(-d·LOG2_10)`).
const LOG2_10: f32 = 3.3219280948873623;

#[derive(Clone, Copy, Debug)]
pub struct ScannerCalibration {
    pub dmin: [f32; 3],
    pub mid: [f32; 3],
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

/// Joint processed-Dmin and neutral-mid calibration computed analytically
/// from the exact emulsion expectation E[D] (using `reduce` and `scan_to_acescg` on a 1x1 patch).
pub fn scanner_calibration_acescg(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
) -> Result<ScannerCalibration, FilmError> {
    let mid_value = crate::film::relative_to_absolute_y(crate::pixel::MIDDLE_GRAY, stock.box_iso.0);
    let dmin_rgb = vec![[0.0; 3]; 1];
    let mid_rgb = vec![[mid_value; 3]; 1];
    let dmin_latent = crate::film::exposure::expose_with_pitch_and_shutter(
        &dmin_rgb, 1, 1, stock, pixel_pitch_um, shutter_seconds,
    );
    let mid_latent = crate::film::exposure::expose_with_pitch_and_shutter(
        &mid_rgb, 1, 1, stock, pixel_pitch_um, shutter_seconds,
    );
    let dmin_dyes = crate::film::development::reduction::reduce(stock, &dmin_latent);
    let mid_dyes = crate::film::development::reduction::reduce(stock, &mid_latent);
    let dmin_scan = scan_to_acescg(stock, &dmin_dyes, pixel_pitch_um)[0];
    let mid_scan = scan_to_acescg(stock, &mid_dyes, pixel_pitch_um)[0];
    Ok(ScannerCalibration {
        dmin: dmin_scan,
        mid: mid_scan,
    })
}

/// Processed unexposed-film scan retained as a convenience for CPU probes.
pub fn processed_dmin_acescg(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
) -> Result<[f32; 3], FilmError> {
    Ok(scanner_calibration_acescg(stock, pixel_pitch_um, shutter_seconds)?.dmin)
}

#[cfg(test)]
mod calibration_tests {
    use super::*;
    use crate::film::scan::invert::invert_negative;
    use crate::film::stock::StockId;

    #[test]
    fn analytical_scanner_calibration_is_finite_and_ordered() {
        let stock = StockId::Portra400.load().unwrap();
        let pitch_um = 500.0 / 4032.0;
        let shutter = 1.0 / 121.0;
        let cal = scanner_calibration_acescg(&stock, pitch_um, shutter).unwrap();
        assert!(cal.dmin.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(cal.mid.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(cal.dmin[0] > cal.mid[0]);
        assert!(cal.dmin[1] > cal.mid[1]);
        assert!(cal.dmin[2] > cal.mid[2]);

        let mut positive = vec![cal.mid];
        invert_negative(&mut positive, cal.mid, cal.dmin);
        for value in positive[0] {
            assert!(value.is_finite());
            assert!((value - crate::pixel::MIDDLE_GRAY).abs() < 1e-3);
        }
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

