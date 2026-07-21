//! Densitometric scan: stacked dyes → transmittance → ACEScg encoding.
//!
//! T(λ) = 10^(−Σ D_layer_eff(λ)) with base-10 optical density.
//! Channel via CIE CMFs → XYZ → ACEScg. Normalize so unexposed (Dmin) peaks near 1.0.

use crate::film::exposure::upsample::{spectrum_to_acescg_rgb, CIE1931_XBAR, CIE1931_YBAR, CIE1931_ZBAR};
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::DyePlanes;
use crate::pixel::{ImageBuffer, Pixel};
use rayon::prelude::*;

/// Effective spectral density at one pixel: Σ_layers (D_image * ε_image + D_mask * ε_mask).
fn density_spectrum(
    stock: &FilmStock,
    dyes: &DyePlanes,
    pixel: usize,
) -> [f64; 16] {
    let mut d = [0.0f64; 16];
    let mut emulsion_i = 0usize;
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().unwrap();
        let di = dyes.image_dye[emulsion_i][pixel] as f64;
        let dm = dyes.mask_dye[emulsion_i][pixel] as f64;
        for lambda in 0..16 {
            d[lambda] += di * coupler.epsilon.samples[lambda];
            if let Some(ref mask_eps) = coupler.mask_epsilon {
                d[lambda] += dm * mask_eps.samples[lambda];
            }
        }
        emulsion_i += 1;
    }
    d
}

fn transmittance_from_density(d: &[f64; 16]) -> [f64; 16] {
    let mut t = [0.0f64; 16];
    for i in 0..16 {
        t[i] = 10f64.powf(-d[i]);
    }
    t
}

/// Scan dye planes to interleaved ACEScg RGB, Dmin-normalized so peak unexposed ≈ 1.0.
pub fn scan_to_acescg(stock: &FilmStock, dyes: &DyePlanes) -> ImageBuffer {
    let n = dyes.width * dyes.height;
    let illuminant = &stock.scanner_light.samples;

    // First pass: compute raw ACEScg from T(λ) * I_s(λ).
    let mut raw: Vec<Pixel> = (0..n)
        .into_par_iter()
        .map(|p| {
            let dens = density_spectrum(stock, dyes, p);
            let mut t = transmittance_from_density(&dens);
            for i in 0..16 {
                t[i] *= illuminant[i];
            }
            let rgb = spectrum_to_acescg_rgb(&t);
            [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]
        })
        .collect();

    // Dmin reference: zero image dye, mask at undeveloped (f=0) max mask if present.
    let dmin_rgb = dmin_reference_acescg(stock);
    let peak = dmin_rgb[0].max(dmin_rgb[1]).max(dmin_rgb[2]).max(1e-12);
    let scale = 1.0 / peak;
    raw.par_iter_mut().for_each(|px| {
        *px = px.map(|c| (c as f64 * scale) as f32);
    });
    raw
}

/// Film-base ACEScg before scan peak-normalization (raw densitometric units).
pub fn dmin_reference_acescg(stock: &FilmStock) -> [f64; 3] {
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
    let dens = density_spectrum(stock, &dyes, 0);
    let mut t = transmittance_from_density(&dens);
    for i in 0..16 {
        t[i] *= stock.scanner_light.samples[i];
    }
    spectrum_to_acescg_rgb(&t)
}

/// Scan-normalized film-base RGB (same encoding as [`scan_to_acescg`]; peak ≈ 1).
pub fn normalized_dmin_acescg(stock: &FilmStock) -> [f32; 3] {
    let d = dmin_reference_acescg(stock);
    let peak = d[0].max(d[1]).max(d[2]).max(1e-12);
    [
        (d[0] / peak) as f32,
        (d[1] / peak) as f32,
        (d[2] / peak) as f32,
    ]
}

/// Keep unused CMF imports available for future scanner spectral work.
#[allow(dead_code)]
fn _cmf_refs() -> (&'static [f64; 16], &'static [f64; 16], &'static [f64; 16]) {
    (&CIE1931_XBAR, &CIE1931_YBAR, &CIE1931_ZBAR)
}
