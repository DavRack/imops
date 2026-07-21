//! Exposure stage: ACEScg image → latent developable-fraction planes.

pub mod absorption;
pub mod capture;
pub mod halation;
pub mod radiance;
pub mod upsample;

pub use capture::DevelopableFractionLut;

use crate::film::constants::ABSORPTION_SIGMA_SCALE_PER_UM;
use crate::film::exposure::absorption::{absorb_stack, integrated_absorbed};
use crate::film::exposure::upsample::upsample_acescg;
use crate::film::spectrum::WavelengthGrid;
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::LatentPlanes;
use crate::pixel::{ImageBuffer, Pixel};
use rayon::prelude::*;

/// Expose `rgb` (absolute-luminance ACEScg) through `stock`.
///
/// Always applies local gelatin scatter (`psf_local_um`) and support-reflection
/// halation with cross-layer bleed from the stock antihalation model.
pub fn expose(
    rgb: &ImageBuffer,
    width: usize,
    height: usize,
    stock: &FilmStock,
) -> LatentPlanes {
    expose_with_pitch(rgb, width, height, stock, 10.0)
}

/// Like [`expose`] but with explicit pixel pitch (µm) for PSF scaling.
pub fn expose_with_pitch(
    rgb: &ImageBuffer,
    width: usize,
    height: usize,
    stock: &FilmStock,
    pixel_pitch_um: f32,
) -> LatentPlanes {
    expose_with_pitch_and_shutter(rgb, width, height, stock, pixel_pitch_um, 1.0)
}

/// Like [`expose_with_pitch`] but with explicit shutter time (s) for reciprocity law failure.
pub fn expose_with_pitch_and_shutter(
    rgb: &ImageBuffer,
    width: usize,
    height: usize,
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
) -> LatentPlanes {
    use crate::film::exposure::radiance::reciprocity_factor;

    let n = width * height;
    assert_eq!(rgb.len(), n);

    let grid = WavelengthGrid::mvp();
    let emulsion_count = stock
        .layers
        .iter()
        .filter(|l| l.kind == LayerKind::Emulsion)
        .count();

    let mut absorbed_planes: Vec<Vec<f32>> = (0..emulsion_count).map(|_| vec![0.0f32; n]).collect();
    let sigma_scale = ABSORPTION_SIGMA_SCALE_PER_UM;

    let per_pixel: Vec<Vec<f32>> = rgb
        .par_iter()
        .map(|px| {
            let spectrum = pixel_fluence_spectrum(px, &grid);
            let (layers, _) = absorb_stack(&stock.layers, &spectrum, sigma_scale);
            let mut emulsion_abs = Vec::with_capacity(emulsion_count);
            for la in &layers {
                if la.produces_latent {
                    emulsion_abs.push(integrated_absorbed(&la.absorbed) as f32);
                }
            }
            emulsion_abs
        })
        .collect();

    for (p, abs_list) in per_pixel.iter().enumerate() {
        for (e, &a) in abs_list.iter().enumerate() {
            absorbed_planes[e][p] = a;
        }
    }

    crate::film::exposure::halation::apply_spatial_exposure_effects(
        &mut absorbed_planes,
        width,
        height,
        stock,
        pixel_pitch_um,
    );

    let mut fraction_planes = Vec::with_capacity(emulsion_count);
    let mut emulsion_i = 0usize;
    for (layer_idx, layer) in stock.layers.iter().enumerate() {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let lut = stock.capture_luts[layer_idx]
            .as_ref()
            .expect("emulsion has capture LUT");
        let eta = reciprocity_factor(shutter_seconds as f64, layer.reciprocity_p as f64) as f32;
        let mut frac = vec![0.0f32; n];
        frac.par_iter_mut()
            .zip(absorbed_planes[emulsion_i].par_iter())
            .for_each(|(f, &phi)| {
                *f = lut.sample((phi * eta) as f64) as f32;
            });
        fraction_planes.push(frac);
        emulsion_i += 1;
    }

    LatentPlanes {
        width,
        height,
        layers: fraction_planes,
    }
}

fn pixel_fluence_spectrum(px: &Pixel, grid: &WavelengthGrid) -> [f64; 16] {
    // Upsample already CIE-matches ACEScg including magnitude. Convert to relative
    // photon fluence only — do **not** re-scale by mean(R,G,B) or by luminance.
    // Luminance scaling under-exposes saturated blues (low Y, high shortwave energy
    // the blue emulsion must see). Absolute mid-gray calibration is restored via
    // `ABSORPTION_SIGMA_SCALE_PER_UM` (tuned for this path).
    let spectrum = upsample_acescg(*px);
    let mut out = [0.0f64; 16];
    for (i, &lambda) in grid.wavelengths_nm.iter().enumerate() {
        let e_rel = 550.0 / lambda;
        out[i] = spectrum[i] / e_rel;
    }
    out
}

#[cfg(test)]
mod fluence_scale_tests {
    use super::*;
    use crate::pixel::PixelOps;

    #[test]
    fn saturated_blue_keeps_shortwave_energy() {
        let grid = WavelengthGrid::mvp();
        let blue = [0.0f32, 0.0, 1.0];
        let gray = [1.0f32, 1.0, 1.0];
        let s_b = pixel_fluence_spectrum(&blue, &grid);
        let s_g = pixel_fluence_spectrum(&gray, &grid);
        // Shortwave bins (≤450 nm): blue stimulus must deposit far more than gray.
        let short_b: f64 = s_b.iter().take(3).sum();
        let short_g: f64 = s_g.iter().take(3).sum();
        assert!(
            short_b > short_g,
            "blue shortwave={short_b} should exceed gray={short_g}"
        );
        // Must not collapse to near-zero just because CIE Y is small.
        let e_b: f64 = s_b.iter().sum();
        assert!(e_b > 0.05 * s_g.iter().sum::<f64>(), "blue total energy too small: {e_b}");
        let _ = blue.luminance();
    }

    #[test]
    fn neutral_spectrum_scales_with_rgb() {
        let grid = WavelengthGrid::mvp();
        let a = pixel_fluence_spectrum(&[0.1, 0.1, 0.1], &grid);
        let b = pixel_fluence_spectrum(&[0.2, 0.2, 0.2], &grid);
        let ea: f64 = a.iter().sum();
        let eb: f64 = b.iter().sum();
        assert!((eb / ea.max(1e-30) - 2.0).abs() < 0.05, "ratio={}", eb / ea);
    }
}
