//! Kodak Ektar 100 class color-negative stock definition.
//!
//! High-contrast, high-saturation C-41 negative (landscape / vivid-color profile)
//! versus soft portrait stocks like Portra.
//!
//! Dye density is `D = D_max · f^(1/γ)`. Lower γ → steeper mid/highlight punch.
//! Fast (large-crystal) layers use a milder γ so the toe still densifies; slow
//! layers use a harder γ for snap above mid. Narrow dye ε for saturation;
//! DIR is off (no published matrix). Residual acutance is `adjacency_beta`.
//! Not a densitometric match to measured Ektar.

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
    // Slightly stronger mask than Portra — supports denser dye packs / saturation.
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let blue = (-0.5 * ((l - 450.0) / 50.0).powi(2)).exp();
            let green = 0.6 * (-0.5 * ((l - 520.0) / 60.0).powi(2)).exp();
            0.38 * blue + 0.26 * green + 0.05
        })
        .collect();
    SpectralCurve::new(grid, samples)
}

pub fn load() -> Result<FilmStock, FilmError> {
    let overcoat = kit::overcoat(0.0);

    // Kodak E-4040 datasheet: "no compensation required 1/10,000 s to 1 s"; digitized spectral peak B=470 nm (official curve).
    // Beyond 1 s: LIRF rolloff not modeled (single-exponent limit).
    // --- BLUE FAST & SLOW ---
    // Fast: larger crystals + milder γ → open toe. Slow: fine + hard γ → punch.
    // Ektar 100 is a fine-grain 100-speed tabular-grain emulsion; fast-layer
    // mean crystal ~0.7 µm (finer than the 400-speed Portra class by the
    // ~2-stop speed gap), so crystal bases sit at 0.8× the Portra-class values.
    let blue_fast = EmulsionLayer {
        name: "blue_fast",
        depth_from_surface: Microns(1.0),
        thickness: Microns(2.8),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(470.0, 32.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.66_f64.ln(),
            sigma_ln: 0.32,
        }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(450.0, 30.0, 1.25),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.25,
        }),
        gamma_contrast: 0.58,
        capture_k: 4.8,
        reciprocity_p: 1.0,
        is_reversal: false,
    };
    let blue_slow = EmulsionLayer {
        name: "blue_slow",
        depth_from_surface: Microns(3.8),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(470.0, 32.0, 0.85)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.30_f64.ln(),
            sigma_ln: 0.24,
        }),
        silver_halide_fraction: 0.21,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(450.0, 30.0, 1.25),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.90,
        }),
        gamma_contrast: 0.34,
        capture_k: 6.5,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    let yellow_filter = kit::yellow_filter(6.0, 2.0, gaussian_curve(430.0, 38.0, 1.35));

    // --- GREEN FAST & SLOW ---
    let green_fast = EmulsionLayer {
        name: "green_fast",
        depth_from_surface: Microns(8.0),
        thickness: Microns(3.2),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(545.0, 36.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.70_f64.ln(),
            sigma_ln: 0.32,
        }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 30.0, 1.25),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.25,
        }),
        gamma_contrast: 0.58,
        capture_k: 2.6,
        reciprocity_p: 1.0,
        is_reversal: false,
    };
    let green_slow = EmulsionLayer {
        name: "green_slow",
        depth_from_surface: Microns(11.2),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(545.0, 36.0, 0.85)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.32_f64.ln(),
            sigma_ln: 0.24,
        }),
        silver_halide_fraction: 0.21,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(545.0, 30.0, 1.25),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.90,
        }),
        gamma_contrast: 0.34,
        capture_k: 3.6,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    // --- RED FAST & SLOW ---
    let red_fast = EmulsionLayer {
        name: "red_fast",
        depth_from_surface: Microns(14.2),
        thickness: Microns(3.6),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 40.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.74_f64.ln(),
            sigma_ln: 0.32,
        }),
        silver_halide_fraction: 0.17,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 32.0, 1.25),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.25,
        }),
        gamma_contrast: 0.58,
        capture_k: 1.6,
        reciprocity_p: 1.0,
        is_reversal: false,
    };
    let red_slow = EmulsionLayer {
        name: "red_slow",
        depth_from_surface: Microns(17.8),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 40.0, 0.85)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.34_f64.ln(),
            sigma_ln: 0.24,
        }),
        silver_halide_fraction: 0.21,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 32.0, 1.25),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 1.90,
        }),
        gamma_contrast: 0.34,
        capture_k: 2.4,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    let antihalation = kit::antihalation_layer(21.5);

    let stock = FilmStock {
        name: "Ektar100",
        box_iso: IsoSpeed(100.0),
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
        antihalation: kit::backing(55.0),
        irradiation_response: None,
        developer_diffusion_length: Microns(6.0),
        inhibitor_diffusion_length: Microns(0.0),
        adjacency_beta: 0.72,
        adjacency_beta_record: 0.0,
        adjacency_beta_cross: 0.0,
        adjacency_beta_dir: 0.0,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
        tabular_grain_thickness_um: Some(kit::T_GRAIN_THICKNESS_UM),
    };
    stock.finalize()
}

#[cfg(test)]
mod tests {
    use crate::film::stock::{LayerKind, StockId};

    #[test]
    fn ektar_loads_and_is_steeper_than_portra() {
        let ektar = StockId::Ektar100.load().unwrap();
        let portra = StockId::Portra400.load().unwrap();
        assert_eq!(ektar.name, "Ektar100");
        assert!((ektar.box_iso.0 - 100.0).abs() < 1e-6);

        let e_gamma: Vec<f32> = ektar
            .layers
            .iter()
            .filter(|l| l.kind == LayerKind::Emulsion)
            .map(|l| l.gamma_contrast)
            .collect();
        let p_gamma: Vec<f32> = portra
            .layers
            .iter()
            .filter(|l| l.kind == LayerKind::Emulsion)
            .map(|l| l.gamma_contrast)
            .collect();
        assert!(
            e_gamma.iter().any(|&g| g < 0.36),
            "slow layers should be hard-γ"
        );
        assert!(
            e_gamma.iter().any(|&g| g > 0.55),
            "fast layers should be toe-open γ"
        );
        assert!(p_gamma.iter().all(|&g| (0.5..0.72).contains(&g)));
        assert!(ektar.adjacency_beta > portra.adjacency_beta);
    }
}
