//! Fuji Pro 400H class color-negative stock definition.
//!
//! Features:
//! - Multi-layer fast/slow emulsion sub-groups
//! - 4th Color Layer (cyan-sensitive sub-layer ~490 nm for precise color discrimination)
//! - DIR off (no published matrix); residual acutance is `adjacency_beta`
//! - Cool green/cyan shadow undertones, fine grain

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
    let overcoat = kit::overcoat(0.0);
    // Fuji PIB: reciprocity "1/4000–1 s no compensation; 4 s +1/2; 16 s +1"; RMS 4.0 (48 µm aperture, density +1.0 above D-min) = finest of the 400-speed negatives → finer than Portra.
    // Digitized PIB spectral peaks (spektrafilm profile): B=470, G=555, R=610 nm; 4th cyan layer 490 nm.
    // HIRF branch below 1 ms overstates loss at 1/4000 s (model structural limit).

    // --- BLUE FAST & SLOW ---
    let blue_fast = EmulsionLayer {
        name: "blue_fast",
        depth_from_surface: Microns(1.0),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(470.0, 35.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.75_f64.ln(),
            sigma_ln: 0.33,
        }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(445.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 4.5,
        reciprocity_p: 0.75,
        is_reversal: false,
    };
    let blue_slow = EmulsionLayer {
        name: "blue_slow",
        depth_from_surface: Microns(3.5),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(470.0, 35.0, 0.8)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.40_f64.ln(),
            sigma_ln: 0.28,
        }),
        silver_halide_fraction: 0.19,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(445.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 2.1,
        reciprocity_p: 0.75,
        is_reversal: false,
    };

    let yellow_filter = kit::yellow_filter(6.0, 2.0, gaussian_curve(430.0, 40.0, 1.3));

    // --- 4th COLOR LAYER (Fuji signature cyan-sensitive ~490 nm) ---
    let fourth_layer = EmulsionLayer {
        name: "cyan_4th_layer",
        depth_from_surface: Microns(8.0),
        thickness: Microns(2.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(490.0, 30.0, 0.9)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.50_f64.ln(),
            sigma_ln: 0.30,
        }),
        silver_halide_fraction: 0.15,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 0.6),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.5,
        }),
        gamma_contrast: 0.56,
        capture_k: 2.0,
        reciprocity_p: 0.75,
        is_reversal: false,
    };

    // --- GREEN FAST & SLOW ---
    let green_fast = EmulsionLayer {
        name: "green_fast",
        depth_from_surface: Microns(10.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(555.0, 40.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.80_f64.ln(),
            sigma_ln: 0.34,
        }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 2.3,
        reciprocity_p: 0.75,
        is_reversal: false,
    };
    let green_slow = EmulsionLayer {
        name: "green_slow",
        depth_from_surface: Microns(13.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(555.0, 40.0, 0.8)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.44_f64.ln(),
            sigma_ln: 0.28,
        }),
        silver_halide_fraction: 0.19,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 1.1,
        reciprocity_p: 0.75,
        is_reversal: false,
    };

    // --- RED FAST & SLOW ---
    let red_fast = EmulsionLayer {
        name: "red_fast",
        depth_from_surface: Microns(16.0),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(610.0, 45.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.84_f64.ln(),
            sigma_ln: 0.34,
        }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 1.4,
        reciprocity_p: 0.75,
        is_reversal: false,
    };
    let red_slow = EmulsionLayer {
        name: "red_slow",
        depth_from_surface: Microns(19.5),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(610.0, 45.0, 0.8)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.46_f64.ln(),
            sigma_ln: 0.28,
        }),
        silver_halide_fraction: 0.19,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.1,
        }),
        gamma_contrast: 0.56,
        capture_k: 0.7,
        reciprocity_p: 0.75,
        is_reversal: false,
    };

    let antihalation = kit::antihalation_layer(23.0);

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
        antihalation: kit::backing(65.0),
        irradiation_response: None,
        developer_diffusion_length: Microns(6.5),
        adjacency_beta: 0.32,
        adjacency_beta_record: 0.0,
        adjacency_beta_cross: 0.0,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
        tabular_grain_thickness_um: None,
    };
    stock.finalize()
}
