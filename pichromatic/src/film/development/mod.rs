//! Development stage orchestration.

pub mod diffusion;
pub mod grain;
pub mod reduction;

use crate::film::constants::CHROMOGENIC_DYE_GRAIN_SCALE;
use crate::film::development::diffusion::{apply_adjacency, apply_dir_inhibition};
use crate::film::development::grain::{apply_grain, scale_kappa};
use crate::film::development::reduction::reduce;
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::{DyePlanes, LatentPlanes};

/// Develop latent planes to dye densities.
///
/// Always applies DIR chemical inhibition, adjacency (Eberhard) and grain —
/// all are physical consequences of the stock. Zero stock params reduce to identity.
pub fn develop(
    stock: &FilmStock,
    latent: &LatentPlanes,
    seed: u64,
    pixel_pitch_um: f32,
) -> DyePlanes {
    let mut dyes = reduce(stock, latent);

    let sigma_dir_px = stock.dir_diffusion_length.0 / pixel_pitch_um.max(1e-6);
    apply_dir_inhibition(&mut dyes, sigma_dir_px, &stock.dir_inhibition_matrix);

    let sigma_px = stock.developer_diffusion_length.0 / pixel_pitch_um.max(1e-6);
    apply_adjacency(&mut dyes, sigma_px, stock.adjacency_beta);

    let mut d_max = Vec::new();
    let mut kappas = Vec::new();
    for (layer_idx, layer) in stock.layers.iter().enumerate() {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().unwrap();
        d_max.push(coupler.d_max);
        let kappa_ref = stock.grain_kappa[layer_idx].unwrap_or(0.0);
        // κ(pitch) = κ_1µm / pitch, then chromogenic dye-cloud scale (not silver-count).
        let kappa = scale_kappa(kappa_ref, pixel_pitch_um) * CHROMOGENIC_DYE_GRAIN_SCALE;
        kappas.push(kappa);
    }
    apply_grain(&mut dyes, &d_max, &kappas, pixel_pitch_um, seed);

    dyes
}
