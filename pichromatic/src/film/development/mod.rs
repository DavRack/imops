//! Development stage orchestration.

pub mod diffusion;
pub mod grain;
pub mod reduction;

use crate::film::development::grain::apply_particle_grain_overwrite;
use crate::film::development::reduction::reduce;
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::{DyePlanes, LatentPlanes};

/// Develop latent planes to dye densities.
///
/// One population process at every render width: reduce (chemical fog
/// supplies a nonzero developable population at zero exposure), then particle
/// overwrite — the realized crystal population convolved with the dye-cloud
/// PSF — followed by DIR chemical inhibition and adjacency (Eberhard) on the
/// realized field. Width only changes the sampling pitch: at coarse pitch the
/// population cells collapse to single pixels whose apertures hold hundreds
/// of crystals, reproducing the central-limit continuum of the same process.
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

    let mut dyes = reduce(stock, latent);

    let d_max: Vec<f32> = stock
        .emulsion_layers()
        .map(|(_, layer)| layer.coupler.as_ref().unwrap().d_max)
        .collect();
    let gammas: Vec<f32> = stock
        .emulsion_layers()
        .map(|(_, layer)| layer.gamma_contrast)
        .collect();

    let sigma_dir_px = stock.dir_diffusion_length.0 / pixel_pitch_um.max(1e-6);
    let sigma_px = stock.developer_diffusion_length.0 / pixel_pitch_um.max(1e-6);

    apply_particle_grain_overwrite(
        &mut dyes,
        &d_max,
        &kappas,
        &gammas,
        pixel_pitch_um,
        seed,
        &crystal_sizes,
        sigma_dir_px,
        &stock.dir_inhibition_matrix,
        sigma_px,
        stock.adjacency_beta,
    );

    dyes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::blur::gaussian_blur_separable;
    use crate::film::development::diffusion::{apply_adjacency, apply_dir_inhibition};
    use crate::film::types::DyePlanes;
    use crate::film::StockId;

    fn gradient_latent(w: usize, h: usize) -> LatentPlanes {
        let n = w * h;
        let mut plane = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                plane[y * w + x] = 0.15 + 0.7 * (x as f32 / w as f32);
            }
        }
        LatentPlanes {
            width: w,
            height: h,
            layers: vec![plane],
        }
    }

    fn max_plane_diff(a: &DyePlanes, b: &DyePlanes) -> f32 {
        a.image_dye
            .iter()
            .zip(b.image_dye.iter())
            .flat_map(|(pa, pb)| pa.iter().zip(pb.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0f32, f32::max)
    }

    fn stock_kappas_and_crystals(
        stock: &crate::film::stock::FilmStock,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<Option<crate::film::stock::LogNormalDist>>,
    ) {
        let mut kappas = Vec::new();
        let mut gammas = Vec::new();
        let mut crystal_sizes = Vec::new();
        for (layer_idx, layer) in stock.layers.iter().enumerate() {
            if layer.kind != LayerKind::Emulsion {
                continue;
            }
            kappas.push(stock.grain_kappa[layer_idx].unwrap_or(0.0));
            gammas.push(layer.gamma_contrast);
            crystal_sizes.push(layer.crystal_size.clone());
        }
        (kappas, gammas, crystal_sizes)
    }

    /// Forbidden cheat: adjacency residual from reduced guide added onto particles.
    fn apply_adjacency_expected_guide_cheat(
        dyes: &mut DyePlanes,
        guide: &DyePlanes,
        sigma_px: f32,
        beta: f32,
    ) {
        if beta.abs() < 1e-8 || sigma_px < 1e-3 {
            return;
        }
        let width = dyes.width;
        let height = dyes.height;
        for (plane, guide_plane) in dyes.image_dye.iter_mut().zip(guide.image_dye.iter()) {
            let mut blurred = guide_plane.clone();
            gaussian_blur_separable(&mut blurred, width, height, sigma_px);
            for (d, (&g, &b)) in plane.iter_mut().zip(guide_plane.iter().zip(blurred.iter())) {
                *d += beta * (g - b);
            }
        }
    }

    fn fine_pitch_reference(
        stock: &crate::film::stock::FilmStock,
        latent: &LatentPlanes,
        pitch_um: f32,
        seed: u64,
    ) -> DyePlanes {
        let reduced = reduce(stock, latent);
        let (kappas, gammas, crystal_sizes) = stock_kappas_and_crystals(stock);
        let d_max: Vec<f32> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.coupler.as_ref().unwrap().d_max)
            .collect();
        let mut reference = reduced;
        let sigma_dir_px = stock.dir_diffusion_length.0 / pitch_um.max(1e-6);
        let sigma_px = stock.developer_diffusion_length.0 / pitch_um.max(1e-6);
        apply_particle_grain_overwrite(
            &mut reference,
            &d_max,
            &kappas,
            &gammas,
            pitch_um,
            seed,
            &crystal_sizes,
            sigma_dir_px,
            &stock.dir_inhibition_matrix,
            sigma_px,
            stock.adjacency_beta,
        );
        reference
    }

    fn plane_std(plane: &[f32]) -> f32 {
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
    fn develop_fine_pitch_matches_overwrite_only() {
        let stock = StockId::BwStub.load().unwrap();
        let latent = gradient_latent(64, 64);
        let pitch_um = 0.25;
        let seed = 17u64;

        let produced = develop(&stock, &latent, seed, pitch_um);
        let reference = fine_pitch_reference(&stock, &latent, pitch_um, seed);
        assert_eq!(
            produced.image_dye,
            reference.image_dye,
            "fine develop must equal particle overwrite-only path (DIR/adj skipped)"
        );

        let reduced = reduce(&stock, &latent);
        let (kappas, gammas, crystal_sizes) = stock_kappas_and_crystals(&stock);
        let d_max: Vec<f32> = stock
            .emulsion_layers()
            .map(|(_, layer)| layer.coupler.as_ref().unwrap().d_max)
            .collect();
        let sigma_dir_px = stock.dir_diffusion_length.0 / pitch_um;
        let sigma_px = stock.developer_diffusion_length.0 / pitch_um;

        // DIR/adjacency now run inside `apply_particle_grain_overwrite` on the
        // realized population, so a separate "overwrite then continuum DIR/adj"
        // pass no longer exists in production. With a smooth realized field and
        // BwStub's no-op DIR matrix the two orderings are exactly equivalent
        // (linear adjacency commutes with the d_max scaling), so the old
        // carve-voids ordering check is vacuous here. The meaningful guards
        // are the two assertions below: particles must matter (reduce_only)
        // and adjacency must not come from a smooth pre-realization guide.
        let mut reduce_only = reduced.clone();
        apply_dir_inhibition(
            &mut reduce_only,
            sigma_dir_px,
            &stock.dir_inhibition_matrix,
        );
        apply_adjacency(&mut reduce_only, sigma_px, stock.adjacency_beta);
        assert!(
            max_plane_diff(&produced, &reduce_only) > 1e-4,
            "fine develop must differ from reduce+DIR+adj without particles"
        );

        // Cheat: smooth guide
        let mut cheat_adj = reduced.clone();
        apply_dir_inhibition(
            &mut cheat_adj,
            sigma_dir_px,
            &stock.dir_inhibition_matrix,
        );
        apply_adjacency_expected_guide_cheat(
            &mut cheat_adj,
            &reduced,
            sigma_px,
            stock.adjacency_beta,
        );
        apply_particle_grain_overwrite(
            &mut cheat_adj,
            &d_max,
            &kappas,
            &gammas,
            pitch_um,
            seed,
            &crystal_sizes, 0.0, &[], 0.0, 0.0);
        assert!(
            max_plane_diff(&produced, &cheat_adj) > 1e-4,
            "production adjacency must not equal pre-realization smooth guide"
        );
    }

    #[test]
    fn develop_zero_exposure_chemical_fog_nonzero_variation() {
        use crate::film::constants::FOG_OFFSET;

        let stock = StockId::BwStub.load().unwrap();
        let w = 64usize;
        let h = 64usize;
        let n = w * h;
        let layers = stock.emulsion_layers().count();
        let latent = LatentPlanes {
            width: w,
            height: h,
            layers: vec![vec![0.0f32; n]; layers],
        };
        let pitch_um = 0.25f32;
        let seed = 99u64;

        let reduced = reduce(&stock, &latent);
        for (layer_i, (plane, (_, layer))) in reduced
            .image_dye
            .iter()
            .zip(stock.emulsion_layers())
            .enumerate()
        {
            let d_max = layer.coupler.as_ref().unwrap().d_max;
            let gamma = layer.gamma_contrast.max(1e-6);
            let f_fog = (FOG_OFFSET / d_max).clamp(0.0, 1.0);
            let expected = d_max * f_fog.powf(1.0 / gamma);
            let mean = plane.iter().sum::<f32>() / plane.len() as f32;
            assert!(mean > 0.0, "layer {layer_i}: reduce fog floor must be nonzero");
            assert!(
                (mean - expected).abs() < 1e-5,
                "layer {layer_i}: mean {mean} != expected H&D fog floor {expected}"
            );
        }

        let produced = develop(&stock, &latent, seed, pitch_um);
        for (layer_i, plane) in produced.image_dye.iter().enumerate() {
            let std = plane_std(plane);
            assert!(
                std > 0.0,
                "layer {layer_i}: fine develop must show spatial variation from fog particles"
            );
        }
    }

    #[test]
    fn develop_fine_pitch_half_plane_matches_reference() {
        let stock = StockId::Portra400.load().unwrap();
        let pitch = 0.5f32 * 1000.0 / 3024.0;
        let w = 128usize;
        let h = 64usize;
        let n = w * h;
        let layers = stock.emulsion_layers().count();
        let mut plane = vec![0.02f32; n];
        for y in 0..(h / 2) {
            for x in 0..w {
                plane[y * w + x] = 0.25;
            }
        }
        let latent = LatentPlanes {
            width: w,
            height: h,
            layers: vec![plane; layers],
        };
        let seed = 7u64;
        let produced = develop(&stock, &latent, seed, pitch);
        let reference = fine_pitch_reference(&stock, &latent, pitch, seed);
        assert_eq!(
            produced.image_dye,
            reference.image_dye,
            "half-plane fine develop must match reference"
        );
    }

    #[test]
    fn develop_coarse_pitch_matches_unified_particle_reference() {
        let stock = StockId::BwStub.load().unwrap();
        let latent = gradient_latent(64, 64);
        let pitch_um = 11.9; // 35mm coarse aperture
        let seed = 42u64;

        let produced = develop(&stock, &latent, seed, pitch_um);

        // Same population process at every pitch: develop() must equal the
        // particle-overwrite reference (reduce -> particles -> DIR/adj), and
        // its mean must track the reduced H&D field (no separate continuum
        // noise layer, no base+residual).
        let reference = fine_pitch_reference(&stock, &latent, pitch_um, seed);
        assert_eq!(produced.image_dye, reference.image_dye);

        let reduced = reduce(&stock, &latent);
        for (i, (plane, red)) in produced.image_dye.iter().zip(reduced.image_dye.iter()).enumerate() {
            let mean_p: f64 = plane.iter().map(|&v| v as f64).sum::<f64>() / plane.len() as f64;
            let mean_r: f64 = red.iter().map(|&v| v as f64).sum::<f64>() / red.len() as f64;
            let rel = (mean_p - mean_r).abs() / mean_r.max(1e-6);
            assert!(
                rel < 0.15,
                "layer {i}: coarse develop mean {mean_p:.4} deviates {:.1}% from reduced H&D {mean_r:.4}",
                rel * 100.0
            );
            assert!(plane_std(plane) > 0.0, "layer {i}: coarse develop must retain grain variation");
        }
    }

    #[test]
    fn emergence_diagnostic_no_surviving_base_layer() {
        // Plan §10: production must differ from smooth expectation — scene must not
        // survive because a reduced base dye image is still present/reinjected.
        let stock = StockId::Portra400.load().unwrap();
        let pitch = 0.25f32; // fine / particle path
        let seed = 7u64;
        let w = 64usize;
        let h = 64usize;
        let latent = LatentPlanes {
            width: w,
            height: h,
            layers: stock
                .emulsion_layers()
                .enumerate()
                .map(|(li, _)| {
                    (0..w * h)
                        .map(|i| {
                            let x = (i % w) as f32 / w as f32;
                            let y = (i / w) as f32 / h as f32;
                            (0.05 + 0.7 * x * (1.0 - y) + 0.2 * ((li as f32) * 0.1))
                                .clamp(0.01, 0.95)
                        })
                        .collect()
                })
                .collect(),
        };
        let production = develop(&stock, &latent, seed, pitch);

        // Smooth expectation path: reduce only (no continuum DIR/adj at fine pitch).
        let expected_only = reduce(&stock, &latent);

        assert!(
            max_plane_diff(&production, &expected_only) > 1e-3,
            "production must not equal smooth expectation path"
        );

        for (i, (p, e)) in production
            .image_dye
            .iter()
            .zip(expected_only.image_dye.iter())
            .enumerate()
        {
            let mut max_rel = 0.0f32;
            let mut sum_abs_e = 0.0f32;
            let mut sum_abs_diff = 0.0f32;
            for (&a, &b) in p.iter().zip(e.iter()) {
                max_rel = max_rel.max((a - b).abs());
                sum_abs_e += b.abs();
                sum_abs_diff += (a - b).abs();
            }
            let rel = sum_abs_diff / sum_abs_e.max(1e-6);
            assert!(
                rel > 0.02,
                "layer {i}: production too close to smooth expectation (rel L1={rel}); base may be surviving"
            );
            assert!(max_rel > 1e-3, "layer {i}: max diff too small ({max_rel})");
        }

        let other = develop(&stock, &latent, seed ^ 0xDEAD, pitch);
        assert!(
            max_plane_diff(&production, &other) > 1e-3,
            "different seeds must change particle realization"
        );
    }
}
