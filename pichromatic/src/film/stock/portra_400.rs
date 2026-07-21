//! Kodak Portra 400 class color-negative stock definition.
//!
//! Features:
//! - Multi-layer fast/slow emulsion sub-groups (fast/slow Blue, Green, Red)
//! - DIR coupler interlayer chemical inhibition
//! - Natural warm skin tones, fine grain, soft portrait contrast curve (γ ≈ 0.54)

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

fn orange_mask_epsilon() -> SpectralCurve {
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let blue = (-0.5 * ((l - 450.0) / 50.0).powi(2)).exp();
            let green = 0.6 * (-0.5 * ((l - 520.0) / 60.0).powi(2)).exp();
            0.32 * blue + 0.22 * green + 0.04
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
        is_reversal: false,
    };

    // --- BLUE FAST & SLOW ---
    let blue_fast = EmulsionLayer {
        name: "blue_fast",
        depth_from_surface: Microns(1.0),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(450.0, 35.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.85_f64.ln(), sigma_ln: 0.35 }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(450.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.54,
        capture_k: 4.8,
        reciprocity_p: 0.85,
        is_reversal: false,
    };
    let blue_slow = EmulsionLayer {
        name: "blue_slow",
        depth_from_surface: Microns(3.5),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(450.0, 35.0, 0.8)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.42_f64.ln(), sigma_ln: 0.28 }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(450.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.54,
        capture_k: 2.2,
        reciprocity_p: 0.89,
        is_reversal: false,
    };

    let yellow_filter = EmulsionLayer {
        name: "yellow_filter",
        depth_from_surface: Microns(6.0),
        thickness: Microns(2.0),
        kind: LayerKind::Filter,
        spectral_sensitivity: Some(gaussian_curve(430.0, 40.0, 1.3)),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    // --- GREEN FAST & SLOW ---
    let green_fast = EmulsionLayer {
        name: "green_fast",
        depth_from_surface: Microns(8.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(550.0, 40.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.90_f64.ln(), sigma_ln: 0.35 }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(550.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.54,
        capture_k: 2.4,
        reciprocity_p: 0.87,
        is_reversal: false,
    };
    let green_slow = EmulsionLayer {
        name: "green_slow",
        depth_from_surface: Microns(11.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(550.0, 40.0, 0.8)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.45_f64.ln(), sigma_ln: 0.28 }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(550.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.54,
        capture_k: 1.1,
        reciprocity_p: 0.91,
        is_reversal: false,
    };

    // --- RED FAST & SLOW ---
    let red_fast = EmulsionLayer {
        name: "red_fast",
        depth_from_surface: Microns(14.0),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 45.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.95_f64.ln(), sigma_ln: 0.35 }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.54,
        capture_k: 1.5,
        reciprocity_p: 0.89,
        is_reversal: false,
    };
    let red_slow = EmulsionLayer {
        name: "red_slow",
        depth_from_surface: Microns(17.5),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 45.0, 0.8)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.48_f64.ln(), sigma_ln: 0.28 }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.54,
        capture_k: 0.7,
        reciprocity_p: 0.93,
        is_reversal: false,
    };

    let antihalation = EmulsionLayer {
        name: "antihalation",
        depth_from_surface: Microns(21.0),
        thickness: Microns(2.0),
        kind: LayerKind::Antihalation,
        spectral_sensitivity: Some(gaussian_curve(650.0, 80.0, 0.3)),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    // 6 emulsion layers: [BF, BS, GF, GS, RF, RS]
    let dir_matrix = vec![
        vec![0.02, 0.01, 0.04, 0.02, 0.03, 0.01],
        vec![0.01, 0.01, 0.02, 0.01, 0.02, 0.01],
        vec![0.04, 0.02, 0.02, 0.01, 0.05, 0.02],
        vec![0.02, 0.01, 0.01, 0.01, 0.02, 0.01],
        vec![0.03, 0.01, 0.05, 0.02, 0.02, 0.01],
        vec![0.01, 0.01, 0.02, 0.01, 0.01, 0.01],
    ];

    let stock = FilmStock {
        name: "Portra400",
        box_iso: IsoSpeed(400.0),
        layers: vec![
            overcoat,
            blue_fast,
            blue_slow,
            yellow_filter,
            green_fast,
            green_slow,
            red_fast,
            red_slow,
            antihalation,
        ],
        antihalation: AntihalationModel {
            reflectance: gaussian_curve(680.0, 60.0, 0.06),
            psf_local_um: 3.0,
            psf_halation_um: 70.0,
        },
        developer_diffusion_length: Microns(7.0),
        adjacency_beta: 0.30,
        dir_diffusion_length: Microns(16.0),
        dir_inhibition_matrix: dir_matrix,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
    };
    stock.finalize()
}
