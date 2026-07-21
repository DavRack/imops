//! Latent → dye density reduction (no spatial effects).

use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::{DyePlanes, LatentPlanes};
use rayon::prelude::*;

/// Convert developable fraction planes to image/mask dye optical densities.
///
/// D_image = D_max * f^(1/γ_eff)
/// Mask density (if coupler has mask_epsilon): proportional to (1 − f), no grain later.
pub fn reduce(stock: &FilmStock, latent: &LatentPlanes) -> DyePlanes {
    let n = latent.width * latent.height;
    let mut image_dye = Vec::new();
    let mut mask_dye = Vec::new();

    let mut latent_idx = 0usize;
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer
            .coupler
            .as_ref()
            .expect("emulsion layer has coupler for development");
        let f_plane = &latent.layers[latent_idx];
        let d_max = coupler.d_max;
        let gamma = layer.gamma_contrast.max(1e-6);
        let inv_gamma = 1.0 / gamma;

        let mut d_img = vec![0.0f32; n];
        let is_reversal = layer.is_reversal;
        d_img.par_iter_mut().zip(f_plane.par_iter()).for_each(|(d, &f)| {
            let eff_f = if is_reversal {
                1.0 - f.clamp(0.0, 1.0)
            } else {
                f.clamp(0.0, 1.0)
            };
            *d = d_max * eff_f.powf(inv_gamma);
        });
        image_dye.push(d_img);

        if coupler.mask_epsilon.is_some() {
            let mut d_mask = vec![0.0f32; n];
            // Residual colored coupler ∝ undeveloped fraction.
            use crate::film::constants::MASK_DENSITY_FRACTION_OF_DMAX;
            let mask_scale = d_max * MASK_DENSITY_FRACTION_OF_DMAX;
            d_mask
                .par_iter_mut()
                .zip(f_plane.par_iter())
                .for_each(|(d, &f)| {
                    *d = mask_scale * (1.0 - f.clamp(0.0, 1.0));
                });
            mask_dye.push(d_mask);
        } else {
            mask_dye.push(vec![0.0f32; n]);
        }

        latent_idx += 1;
    }

    DyePlanes {
        width: latent.width,
        height: latent.height,
        image_dye,
        mask_dye,
    }
}
