//! Development stage orchestration.

pub mod diffusion;
pub mod grain;
pub mod reduction;

use crate::film::development::diffusion::{apply_adjacency, apply_dir_inhibition};
use crate::film::development::grain::{
    apply_continuum_grain, apply_grain, particle_resolution_limit_um,
};
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
    let mut kappas = Vec::new();
    let mut crystal_sizes = Vec::new();
    for (layer_idx, layer) in stock.layers.iter().enumerate() {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        kappas.push(stock.grain_kappa[layer_idx].unwrap_or(0.0));
        crystal_sizes.push(layer.crystal_size.clone());
    }

    // Resolve individual particles only when the render aperture can sample a
    // typical crystal. Larger apertures use the central-limit approximation of
    // the same population process.
    let resolves_particles =
        pixel_pitch_um < particle_resolution_limit_um(&crystal_sizes).max(1e-6);
    let mut dyes = reduce(stock, latent);

    if resolves_particles {
        let d_max: Vec<f32> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.coupler.as_ref().unwrap().d_max)
            .collect();
        // Apply discrete clouds after H&D reduction. Sampling latent f and
        // then raising each sparse realization to 1/gamma biases the mean and
        // turns zero-count clouds into crushed holes.
        apply_grain(
            &mut dyes,
            &d_max,
            &kappas,
            pixel_pitch_um,
            seed,
            &crystal_sizes,
        );
    }

    let sigma_dir_px = stock.dir_diffusion_length.0 / pixel_pitch_um.max(1e-6);
    apply_dir_inhibition(&mut dyes, sigma_dir_px, &stock.dir_inhibition_matrix);

    let sigma_px = stock.developer_diffusion_length.0 / pixel_pitch_um.max(1e-6);
    apply_adjacency(&mut dyes, sigma_px, stock.adjacency_beta);

    if !resolves_particles {
        let d_max: Vec<f32> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.coupler.as_ref().unwrap().d_max)
            .collect();
        apply_continuum_grain(
            &mut dyes,
            &d_max,
            &kappas,
            pixel_pitch_um,
            seed,
            &crystal_sizes,
        );
    }

    dyes
}
