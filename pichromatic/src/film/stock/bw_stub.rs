//! Single-layer B&W stub stock for early checkpoints.
//!
//! Parameters (order-of-magnitude from published AgX emulsion surveys):
//! - One emulsion layer, thickness 10 µm
//! - Mean crystal diameter ~0.7 µm, σ_ln = 0.35, silver_halide_fraction 0.2
//! - Broad panchromatic sensitivity (smooth curve covering 400–700 nm)
//! - Neutral silver “dye” proxy: flat ε(λ)=1 for densitometry
//! - Box ISO 100
//! - Antihalation reflectance ~0

use crate::film::error::FilmError;
use crate::film::spectrum::{SpectralCurve, WavelengthGrid};
use crate::film::stock::{
    AntihalationModel, DyeCoupler, EmulsionLayer, FilmStock, LayerKind, LogNormalDist,
};
use crate::film::units::{IsoSpeed, Microns};

/// Smooth panchromatic sensitivity: raised cosine across the MVP grid, peak-normalized.
fn panchromatic_sensitivity() -> SpectralCurve {
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&lambda| {
            // Broad response; slight falloff at extremes (representative panchromatic).
            let t = (lambda - 400.0) / 300.0;
            let envelope = (std::f64::consts::PI * t).sin().max(0.0);
            0.3 + 0.7 * envelope
        })
        .collect();
    SpectralCurve::new(grid, samples)
}

pub fn load() -> Result<FilmStock, FilmError> {
    let sensitivity = panchromatic_sensitivity();
    let coupler = DyeCoupler {
        name: "neutral_silver_proxy",
        epsilon: SpectralCurve::constant(1.0),
        mask_epsilon: None,
        // Representative max density for a processed B&W silver image.
        d_max: 2.5,
    };

    let emulsion = EmulsionLayer {
        name: "bw_panchromatic",
        depth_from_surface: Microns(0.0),
        thickness: Microns(10.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(sensitivity),
        crystal_size: Some(LogNormalDist {
            // Mean diameter 0.7 µm → mu_ln = ln(0.7).
            mu_ln: 0.7_f64.ln(),
            sigma_ln: 0.35,
        }),
        silver_halide_fraction: 0.2,
        coupler: Some(coupler),
        // Moderate process contrast; published B&W gamma class ~0.6–0.7.
        gamma_contrast: 0.65,
        // Provisional k; refined when B&W E2E calibration lands (CP9).
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    let stock = FilmStock {
        name: "BwStub",
        box_iso: IsoSpeed(100.0),
        layers: vec![emulsion],
        antihalation: AntihalationModel {
            reflectance: SpectralCurve::constant(0.0),
            psf_local_um: 2.0,
            psf_halation_um: 40.0,
        },
        developer_diffusion_length: Microns(5.0),
        adjacency_beta: 0.3,
        dir_diffusion_length: Microns(10.0),
        dir_inhibition_matrix: vec![vec![0.0]],
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
    };
    stock.finalize()
}
