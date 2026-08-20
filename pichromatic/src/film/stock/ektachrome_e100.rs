//! Kodak Ektachrome E100 class reversal (slide) film stock definition (E-6 process).
//!
//! Features:
//! - Positive image dye formation (`is_reversal: true`)
//! - No orange mask (`mask_epsilon: None`)
//! - High punchy contrast curve (γ ≈ 1.65), ultra-fine grain, brilliant whites

use crate::film::error::FilmError;
use crate::film::spectrum::{SpectralCurve, WavelengthGrid};
use crate::film::stock::kit;
use crate::film::stock::{DyeCoupler, EmulsionLayer, FilmStock, LayerKind, LogNormalDist};
use crate::film::units::{IsoSpeed, Microns};

fn gaussian_curve(peak_nm: f64, sigma_nm: f64, amplitude: f64) -> SpectralCurve {
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let d = (l - peak_nm) / sigma_nm;
            amplitude * (-0.5 * d * d).exp()
        })
        .collect();
    SpectralCurve::new(grid, samples)
}

pub fn load() -> Result<FilmStock, FilmError> {
    let overcoat = kit::overcoat(0.0);

    // Kodak E-4000: "reciprocity: compensation not required 1/10,000 s to 10 s"; digitized peaks B=425, G=570 nm.
    let blue = EmulsionLayer {
        name: "blue_sensitive",
        depth_from_surface: Microns(1.0),
        thickness: Microns(4.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(425.0, 30.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.42_f64.ln(),
            sigma_ln: 0.25,
        }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(440.0, 35.0, 1.0),
            mask_epsilon: None, // No mask on positive slide film
            d_max: 3.2,
        }),
        gamma_contrast: 1.65,
        capture_k: 3.5,
        reciprocity_p: 1.0,
        is_reversal: true,
    };

    let yellow_filter = kit::yellow_filter(5.0, 2.0, gaussian_curve(430.0, 35.0, 1.4));

    let green = EmulsionLayer {
        name: "green_sensitive",
        depth_from_surface: Microns(7.0),
        thickness: Microns(4.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(570.0, 35.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.45_f64.ln(),
            sigma_ln: 0.25,
        }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 35.0, 1.0),
            mask_epsilon: None,
            d_max: 3.2,
        }),
        gamma_contrast: 1.65,
        capture_k: 1.8,
        reciprocity_p: 1.0,
        is_reversal: true,
    };

    let red = EmulsionLayer {
        name: "red_sensitive",
        depth_from_surface: Microns(11.0),
        thickness: Microns(5.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 40.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.50_f64.ln(),
            sigma_ln: 0.26,
        }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 40.0, 1.0),
            mask_epsilon: None,
            d_max: 3.2,
        }),
        gamma_contrast: 1.65,
        capture_k: 1.1,
        reciprocity_p: 1.0,
        is_reversal: true,
    };

    let antihalation = kit::antihalation_layer(16.0);

    let stock = FilmStock {
        name: "EktachromeE100",
        box_iso: IsoSpeed(100.0),
        layers: vec![overcoat, blue, yellow_filter, green, red, antihalation],
        antihalation: kit::backing(40.0),
        irradiation_response: None,
        developer_diffusion_length: Microns(4.0),
        inhibitor_diffusion_length: Microns(0.0),
        adjacency_beta: 0.40,
        adjacency_beta_record: 0.0,
        adjacency_beta_cross: 0.0,
        adjacency_beta_dir: 0.0,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
        tabular_grain_thickness_um: None,
    };
    stock.finalize()
}
