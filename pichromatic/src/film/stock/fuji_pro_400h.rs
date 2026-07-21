//! Fuji Pro 400H class color-negative stock definition.
//!
//! Features:
//! - Multi-layer fast/slow emulsion sub-groups
//! - 4th Color Layer (cyan-sensitive sub-layer ~490 nm for precise color discrimination)
//! - DIR coupler interlayer chemical inhibition
//! - Cool green/cyan shadow undertones, fine grain

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
            let green = 0.55 * (-0.5 * ((l - 520.0) / 60.0).powi(2)).exp();
            0.28 * blue + 0.20 * green + 0.03
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
        spectral_sensitivity: Some(gaussian_curve(445.0, 35.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.82_f64.ln(), sigma_ln: 0.33 }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(445.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 4.5,
        reciprocity_p: 0.86,
        is_reversal: false,
    };
    let blue_slow = EmulsionLayer {
        name: "blue_slow",
        depth_from_surface: Microns(3.5),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(445.0, 35.0, 0.8)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.40_f64.ln(), sigma_ln: 0.28 }),
        silver_halide_fraction: 0.19,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(445.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 2.1,
        reciprocity_p: 0.90,
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

    // --- 4th COLOR LAYER (Fuji signature cyan-sensitive ~490 nm) ---
    let fourth_layer = EmulsionLayer {
        name: "cyan_4th_layer",
        depth_from_surface: Microns(8.0),
        thickness: Microns(2.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(490.0, 30.0, 0.9)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.50_f64.ln(), sigma_ln: 0.30 }),
        silver_halide_fraction: 0.15,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 0.6),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.5,
        }),
        gamma_contrast: 0.56,
        capture_k: 2.0,
        reciprocity_p: 0.88,
        is_reversal: false,
    };

    // --- GREEN FAST & SLOW ---
    let green_fast = EmulsionLayer {
        name: "green_fast",
        depth_from_surface: Microns(10.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(545.0, 40.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.88_f64.ln(), sigma_ln: 0.34 }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 2.3,
        reciprocity_p: 0.87,
        is_reversal: false,
    };
    let green_slow = EmulsionLayer {
        name: "green_slow",
        depth_from_surface: Microns(13.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(545.0, 40.0, 0.8)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.44_f64.ln(), sigma_ln: 0.28 }),
        silver_halide_fraction: 0.19,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 1.1,
        reciprocity_p: 0.91,
        is_reversal: false,
    };

    // --- RED FAST & SLOW ---
    let red_fast = EmulsionLayer {
        name: "red_fast",
        depth_from_surface: Microns(16.0),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 45.0, 1.0)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.92_f64.ln(), sigma_ln: 0.34 }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 1.4,
        reciprocity_p: 0.89,
        is_reversal: false,
    };
    let red_slow = EmulsionLayer {
        name: "red_slow",
        depth_from_surface: Microns(19.5),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 45.0, 0.8)),
        crystal_size: Some(LogNormalDist { mu_ln: 0.46_f64.ln(), sigma_ln: 0.28 }),
        silver_halide_fraction: 0.19,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 0.7,
        reciprocity_p: 0.93,
        is_reversal: false,
    };

    let antihalation = EmulsionLayer {
        name: "antihalation",
        depth_from_surface: Microns(23.0),
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

    // 7 emulsion layers: [BF, BS, 4th, GF, GS, RF, RS]
    // DIR weights: matrix[source][target] (row = source emulsion layer, column = inhibited layer).
    // Intentionally non-symmetric — e.g. 4th→GF (0.06) ≠ GF→4th (0.05).
    let dir_matrix = vec![
        vec![0.02, 0.01, 0.03, 0.04, 0.02, 0.03, 0.01],
        vec![0.01, 0.01, 0.02, 0.02, 0.01, 0.02, 0.01],
        vec![0.03, 0.02, 0.02, 0.06, 0.03, 0.04, 0.02],
        vec![0.04, 0.02, 0.05, 0.02, 0.01, 0.05, 0.02],
        vec![0.02, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01],
        vec![0.03, 0.01, 0.04, 0.05, 0.02, 0.02, 0.01],
        vec![0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01],
    ];

    let stock = FilmStock {
        name: "FujiPro400H",
        box_iso: IsoSpeed(400.0),
        layers: vec![
            overcoat,
            blue_fast,
            blue_slow,
            yellow_filter,
            fourth_layer,
            green_fast,
            green_slow,
            red_fast,
            red_slow,
            antihalation,
        ],
        antihalation: AntihalationModel {
            reflectance: gaussian_curve(670.0, 50.0, 0.05),
            psf_local_um: 2.5,
            psf_halation_um: 65.0,
        },
        developer_diffusion_length: Microns(6.5),
        adjacency_beta: 0.32,
        dir_diffusion_length: Microns(14.0),
        dir_inhibition_matrix: dir_matrix,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
    };
    stock.finalize()
}
