//! Kodak Ektachrome E100 class reversal (slide) film stock definition (E-6 process).
//!
//! Features:
//! - Positive image dye formation (`is_reversal: true`)
//! - No orange mask (`mask_epsilon: None`)
//! - High punchy contrast curve (γ ≈ 1.65), ultra-fine grain, brilliant whites

use crate::film::error::FilmError;
use crate::film::spectrum::{SpectralCurve, WavelengthGrid};
use crate::film::stock::{
    AntihalationModel, DyeCoupler, EmulsionLayer, FilmStock, LayerKind, LogNormalDist,
};
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
    let overcoat = EmulsionLayer {
        name: "overcoat",
        depth_from_surface: Microns(0.0),
        thickness: Microns(1.0),
        kind: LayerKind::Overcoat,
        spectral_sensitivity: Some(SpectralCurve::constant(0.01)),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: true,
    };

    let blue = EmulsionLayer {
        name: "blue_sensitive",
        depth_from_surface: Microns(1.0),
        thickness: Microns(4.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(440.0, 30.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.42_f64.ln(), sigma_ln: 0.25 }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(440.0, 35.0, 1.0),
            mask_epsilon: None, // No mask on positive slide film
            d_max: 3.2,
        }),
        gamma_contrast: 1.65,
        capture_k: 3.5,
        reciprocity_p: 0.93,
        is_reversal: true,
    };

    let yellow_filter = EmulsionLayer {
        name: "yellow_filter",
        depth_from_surface: Microns(5.0),
        thickness: Microns(2.0),
        kind: LayerKind::Filter,
        spectral_sensitivity: Some(gaussian_curve(430.0, 35.0, 1.4)),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: true,
    };

    let green = EmulsionLayer {
        name: "green_sensitive",
        depth_from_surface: Microns(7.0),
        thickness: Microns(4.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(545.0, 35.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.45_f64.ln(), sigma_ln: 0.25 }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 35.0, 1.0),
            mask_epsilon: None,
            d_max: 3.2,
        }),
        gamma_contrast: 1.65,
        capture_k: 1.8,
        reciprocity_p: 0.94,
        is_reversal: true,
    };

    let red = EmulsionLayer {
        name: "red_sensitive",
        depth_from_surface: Microns(11.0),
        thickness: Microns(5.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 40.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.50_f64.ln(), sigma_ln: 0.26 }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 40.0, 1.0),
            mask_epsilon: None,
            d_max: 3.2,
        }),
        gamma_contrast: 1.65,
        capture_k: 1.1,
        reciprocity_p: 0.95,
        is_reversal: true,
    };

    let antihalation = EmulsionLayer {
        name: "antihalation",
        depth_from_surface: Microns(16.0),
        thickness: Microns(2.0),
        kind: LayerKind::Antihalation,
        spectral_sensitivity: Some(gaussian_curve(650.0, 80.0, 0.3)),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: true,
    };

    let dir_matrix = vec![
        vec![0.01, 0.02, 0.01],
        vec![0.02, 0.01, 0.02],
        vec![0.01, 0.02, 0.01],
    ];

    let stock = FilmStock {
        name: "EktachromeE100",
        box_iso: IsoSpeed(100.0),
        layers: vec![overcoat, blue, yellow_filter, green, red, antihalation],
        antihalation: AntihalationModel {
            reflectance: gaussian_curve(660.0, 40.0, 0.02), // Very low reflectance
            psf_local_um: 1.5,
            psf_halation_um: 40.0,
        },
        developer_diffusion_length: Microns(4.0),
        adjacency_beta: 0.40,
        dir_diffusion_length: Microns(8.0),
        dir_inhibition_matrix: dir_matrix,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
    };
    stock.finalize()
}
