//! Densitometric scan: stacked dyes → transmittance → ACEScg encoding.
//!
//! T(λ) = 10^(−Σ D_layer_eff(λ)) with base-10 optical density.
//! Channel via CIE CMFs → XYZ → ACEScg. Normalize so unexposed (Dmin) peaks near 1.0.
//!
//! All per-pixel math runs in f32 (WGSL has no core f64), mirroring the GPU
//! `SCAN` / `GRAIN_SCAN_ROI` shaders.

use crate::film::exposure::upsample::spectrum_to_acescg_rgb_f32;
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::DyePlanes;
use crate::pixel::{ImageBuffer, Pixel};
use rayon::prelude::*;

/// log2(10), shared with the GPU `SCAN` shader (`exp2(-d·LOG2_10)`).
const LOG2_10: f32 = 3.3219280948873623;

/// Effective spectral density at one pixel: Σ_layers (D_image * ε_image + D_mask * ε_mask).
pub(crate) fn density_spectrum(
    stock: &FilmStock,
    dyes: &DyePlanes,
    pixel: usize,
) -> [f32; 16] {
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

/// Scan dye planes to interleaved ACEScg RGB, Dmin-normalized so peak unexposed ≈ 1.0.
pub fn scan_to_acescg(stock: &FilmStock, dyes: &DyePlanes) -> ImageBuffer {
    let n = dyes.width * dyes.height;

    // First pass: compute raw ACEScg from T(λ) * I_s(λ).
    let mut raw: Vec<Pixel> = (0..n)
        .into_par_iter()
        .map(|p| {
            let t = scan_pixel_spectrum(stock, dyes, p);
            spectrum_to_acescg_rgb_f32(&t)
        })
        .collect();

    // Dmin reference: zero image dye, mask at undeveloped (f=0) max mask if present.
    let dmin_rgb = dmin_reference_acescg(stock);
    let peak = dmin_rgb[0].max(dmin_rgb[1]).max(dmin_rgb[2]).max(1e-12);
    let scale = 1.0 / peak;
    raw.par_iter_mut().for_each(|px| {
        *px = px.map(|c| c * scale);
    });
    raw
}

/// Film-base ACEScg before scan peak-normalization (raw densitometric units).
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


/// Scan-normalized film-base RGB (same encoding as [`scan_to_acescg`]; peak ≈ 1).
pub fn normalized_dmin_acescg(stock: &FilmStock) -> [f32; 3] {
    let d = dmin_reference_acescg(stock);
    let peak = d[0].max(d[1]).max(d[2]).max(1e-12);
    [d[0] / peak, d[1] / peak, d[2] / peak]
}
