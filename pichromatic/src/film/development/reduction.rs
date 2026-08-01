//! Latent → dye density reduction (no spatial effects).

use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::{DyePlanes, LatentPlanes};
use rayon::prelude::*;

/// Convert developable fraction planes to image/mask dye optical densities.
///
/// `D_image = D_max * f_eff^(1/γ_eff)` with `f_eff = f` (negative) or `1−f` (reversal).
///
/// Coloured film base stays constant while image dye develops. This keeps every
/// developed negative at or below the synthetic Dmin scan reference.
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
        d_img
            .par_iter_mut()
            .zip(f_plane.par_iter())
            .for_each(|(d, &f)| {
                let eff_f = if is_reversal {
                    1.0 - f.clamp(0.0, 1.0)
                } else {
                    f.clamp(0.0, 1.0)
                };
                // Deterministic pow (film::math), mirroring `REDUCE`.
                *d = if eff_f > 0.0 {
                    d_max * crate::film::math::pow_det(eff_f, inv_gamma)
                } else {
                    0.0
                };
            });

        if coupler.mask_epsilon.is_some() {
            use crate::film::constants::MASK_DENSITY_FRACTION_OF_DMAX;
            let mask_scale = d_max * MASK_DENSITY_FRACTION_OF_DMAX;
            let mut d_mask = vec![0.0f32; n];
            d_mask.par_iter_mut().for_each(|m| *m = mask_scale);
            mask_dye.push(d_mask);
        } else {
            mask_dye.push(vec![0.0f32; n]);
        }

        image_dye.push(d_img);
        latent_idx += 1;
    }

    DyePlanes {
        width: latent.width,
        height: latent.height,
        image_dye,
        mask_dye,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::{constants::MASK_DENSITY_FRACTION_OF_DMAX, StockId};

    #[test]
    fn coloured_base_is_not_cleared_by_development() {
        let stock = StockId::Portra400.load().unwrap();
        let emulsion_count = stock
            .layers
            .iter()
            .filter(|l| l.kind == LayerKind::Emulsion)
            .count();
        let latent = LatentPlanes {
            width: 2,
            height: 1,
            layers: vec![vec![0.0, 1.0]; emulsion_count],
        };
        let dyes = reduce(&stock, &latent);
        for (layer, mask) in stock
            .layers
            .iter()
            .filter(|l| l.kind == LayerKind::Emulsion)
            .zip(&dyes.mask_dye)
        {
            let expected = layer
                .coupler
                .as_ref()
                .unwrap()
                .mask_epsilon
                .as_ref()
                .map_or(0.0, |_| {
                    layer.coupler.as_ref().unwrap().d_max * MASK_DENSITY_FRACTION_OF_DMAX
                });
            assert_eq!(mask, &vec![expected; 2]);
        }
    }
}
