//! Fujichrome Velvia 100 Professional [RVP100] film stock definition.
//!
//! Features:
//! - Positive image dye formation (`is_reversal: true`)
//! - No orange mask (`mask_epsilon: None`)
//! - Ultra-high saturation, extremely fine grain (RMS 8.0)

use crate::film::error::FilmError;
use crate::film::spectrum::{SpectralCurve, WavelengthGrid};
use crate::film::stock::{
    AntihalationModel, DyeCoupler, EmulsionLayer, FilmStock, LayerKind, LogNormalDist,
};
use crate::film::units::{IsoSpeed, Microns};

fn gaussian_curve(peak_nm: f64, sigma_nm: f64, amplitude: f64) -> SpectralCurve {
    let grid = WavelengthGrid::mvp();
    SpectralCurve::new(
        grid.clone(),
        grid.wavelengths_nm
            .iter()
            .map(|&l| {
                let d = (l - peak_nm) / sigma_nm;
                amplitude * (-0.5 * d * d).exp()
            })
            .collect(),
    )
}

pub fn load() -> Result<FilmStock, FilmError> {
    let grid = WavelengthGrid::mvp();

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
        spectral_sensitivity: Some(SpectralCurve::new(
            grid.clone(),
            vec![
                0.019953, 0.501187, 0.707946, 1.000000, 0.562341, 0.125893, 0.017783, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.40_f64.ln(),
            sigma_ln: 0.22,
        }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: SpectralCurve::new(
                grid.clone(),
                vec![
                    0.4, 0.82, 0.98, 0.18, 0.05, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0,
                    0.0, 0.0,
                ],
            ),
            mask_epsilon: None,
            d_max: 3.70,
        }),
        gamma_contrast: 1.8,
        capture_k: 3.5,
        reciprocity_p: 0.93,
        is_reversal: true,
    };

    let yellow_filter = EmulsionLayer {
        name: "yellow_filter",
        depth_from_surface: Microns(5.0),
        thickness: Microns(2.0),
        kind: LayerKind::Filter,
        spectral_sensitivity: Some(SpectralCurve::constant(0.0)),
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
        spectral_sensitivity: Some(SpectralCurve::new(
            grid.clone(),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.044668, 0.199526, 0.562341, 0.891251, 0.630957,
                0.316228, 0.017783, 0.0, 0.0, 0.0, 0.0,
            ],
        )),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.42_f64.ln(),
            sigma_ln: 0.22,
        }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: SpectralCurve::new(
                grid.clone(),
                vec![
                    0.15, 0.15, 0.15, 0.18, 0.25, 0.4, 0.72, 0.98, 0.85, 0.2, 0.08, 0.04, 0.03,
                    0.02, 0.01, 0.0,
                ],
            ),
            mask_epsilon: None,
            d_max: 3.83,
        }),
        gamma_contrast: 1.8,
        capture_k: 1.8,
        reciprocity_p: 0.94,
        is_reversal: true,
    };

    let red = EmulsionLayer {
        name: "red_sensitive",
        depth_from_surface: Microns(11.0),
        thickness: Microns(5.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(SpectralCurve::new(
            grid.clone(),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.019953, 0.044668, 0.141254, 0.446684,
                1.000000, 0.354813, 0.070795, 0.012589,
            ],
        )),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.45_f64.ln(),
            sigma_ln: 0.23,
        }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: SpectralCurve::new(
                grid.clone(),
                vec![
                    0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.09, 0.11, 0.22, 0.5, 0.88,
                    1.0, 0.15, 0.03,
                ],
            ),
            mask_epsilon: None,
            d_max: 3.38,
        }),
        gamma_contrast: 1.8,
        capture_k: 1.1,
        reciprocity_p: 0.95,
        is_reversal: true,
    };

    let antihalation = EmulsionLayer {
        name: "antihalation",
        depth_from_surface: Microns(16.0),
        thickness: Microns(2.0),
        kind: LayerKind::Antihalation,
        spectral_sensitivity: Some(SpectralCurve::constant(0.1)),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: true,
    };

    let dir_matrix = vec![
        vec![0.02, 0.04, 0.02],
        vec![0.04, 0.02, 0.04],
        vec![0.02, 0.04, 0.02],
    ];

    let stock = FilmStock {
        name: "FujichromeVelvia100",
        box_iso: IsoSpeed(100.0),
        layers: vec![overcoat, blue, yellow_filter, green, red, antihalation],
        antihalation: AntihalationModel {
            reflectance: gaussian_curve(660.0, 50.0, 0.10),
            psf_local_um: 1.0,
            psf_halation_um: 30.0,
        },
        developer_diffusion_length: Microns(3.5),
        adjacency_beta: 0.45,
        dir_diffusion_length: Microns(7.0),
        dir_inhibition_matrix: dir_matrix,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
        tabular_grain_thickness_um: None,
    };
    stock.finalize()
}
