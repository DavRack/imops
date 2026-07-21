//! Kodak Tri-X 400 class Black & White film stock definition.
//!
//! Features:
//! - Dual fast & slow panchromatic silver emulsion sub-layers
//! - High maximum density (D_max = 3.0)
//! - Classic coarse silver grain structure
//! - Moderate-to-high process contrast (γ ≈ 0.70)

use crate::film::error::FilmError;
use crate::film::spectrum::{SpectralCurve, WavelengthGrid};
use crate::film::stock::{
    AntihalationModel, DyeCoupler, EmulsionLayer, FilmStock, LayerKind, LogNormalDist,
};
use crate::film::units::{IsoSpeed, Microns};

fn panchromatic_sensitivity() -> SpectralCurve {
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&lambda| {
            let t = (lambda - 400.0) / 300.0;
            let envelope = (std::f64::consts::PI * t).sin().max(0.0);
            0.35 + 0.65 * envelope
        })
        .collect();
    SpectralCurve::new(grid, samples)
}

pub fn load() -> Result<FilmStock, FilmError> {
    let fast_emulsion = EmulsionLayer {
        name: "trix_fast",
        depth_from_surface: Microns(0.0),
        thickness: Microns(6.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(panchromatic_sensitivity()),
        crystal_size: Some(LogNormalDist { mu_ln: 0.95_f64.ln(), sigma_ln: 0.38 }),
        silver_halide_fraction: 0.22,
        coupler: Some(DyeCoupler {
            name: "neutral_silver_fast",
            epsilon: SpectralCurve::constant(1.0),
            mask_epsilon: None,
            d_max: 1.5,
        }),
        gamma_contrast: 0.70,
        capture_k: 2.2,
        reciprocity_p: 0.85,
        is_reversal: false,
    };

    let slow_emulsion = EmulsionLayer {
        name: "trix_slow",
        depth_from_surface: Microns(6.0),
        thickness: Microns(6.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(panchromatic_sensitivity()),
        crystal_size: Some(LogNormalDist { mu_ln: 0.45_f64.ln(), sigma_ln: 0.28 }),
        silver_halide_fraction: 0.24,
        coupler: Some(DyeCoupler {
            name: "neutral_silver_slow",
            epsilon: SpectralCurve::constant(1.0),
            mask_epsilon: None,
            d_max: 1.5,
        }),
        gamma_contrast: 0.70,
        capture_k: 0.9,
        reciprocity_p: 0.89,
        is_reversal: false,
    };

    let stock = FilmStock {
        name: "TriX400",
        box_iso: IsoSpeed(400.0),
        layers: vec![fast_emulsion, slow_emulsion],
        antihalation: AntihalationModel {
            reflectance: SpectralCurve::constant(0.01),
            psf_local_um: 3.5,
            psf_halation_um: 50.0,
        },
        developer_diffusion_length: Microns(6.0),
        adjacency_beta: 0.35,
        dir_diffusion_length: Microns(12.0),
        dir_inhibition_matrix: vec![
            vec![0.03, 0.02],
            vec![0.02, 0.02],
        ],
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
    };
    stock.finalize()
}
