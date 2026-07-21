//! Kodak Ektar 100 class color-negative stock definition.
//!
//! High-contrast, high-saturation C-41 negative (landscape / vivid-color profile)
//! versus soft portrait stocks like Portra.
//!
//! Dye density is `D = D_max · f^(1/γ)`. Lower γ → steeper mid/highlight punch.
//! Fast (large-crystal) layers use a milder γ so the toe still densifies; slow
//! layers use a harder γ for snap above mid. Strong DIR + narrow dye ε for
//! saturation. Not a densitometric match to measured Ektar.

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
    // Fast: larger crystals + milder γ → open toe. Slow: fine + hard γ → punch.
    let blue_fast = EmulsionLayer {
        name: "blue_fast",
        depth_from_surface: Microns(1.0),
        thickness: Microns(2.8),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(450.0, 32.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.82_f64.ln(),
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
        reciprocity_p: 0.86,
        is_reversal: false,
    };
    let blue_slow = EmulsionLayer {
        name: "blue_slow",
        depth_from_surface: Microns(3.8),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(450.0, 32.0, 0.85)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.38_f64.ln(),
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
        reciprocity_p: 0.90,
        is_reversal: false,
    };

    let yellow_filter = EmulsionLayer {
        name: "yellow_filter",
        depth_from_surface: Microns(6.0),
        thickness: Microns(2.0),
        kind: LayerKind::Filter,
        spectral_sensitivity: Some(gaussian_curve(430.0, 38.0, 1.35)),
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
        thickness: Microns(3.2),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(545.0, 36.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.88_f64.ln(),
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
        reciprocity_p: 0.88,
        is_reversal: false,
    };
    let green_slow = EmulsionLayer {
        name: "green_slow",
        depth_from_surface: Microns(11.2),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(545.0, 36.0, 0.85)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.40_f64.ln(),
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
        reciprocity_p: 0.92,
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
            mu_ln: 0.92_f64.ln(),
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
        reciprocity_p: 0.90,
        is_reversal: false,
    };
    let red_slow = EmulsionLayer {
        name: "red_slow",
        depth_from_surface: Microns(17.8),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 40.0, 0.85)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.42_f64.ln(),
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
        reciprocity_p: 0.94,
        is_reversal: false,
    };

    let antihalation = EmulsionLayer {
        name: "antihalation",
        depth_from_surface: Microns(21.5),
        thickness: Microns(2.0),
        kind: LayerKind::Antihalation,
        spectral_sensitivity: Some(gaussian_curve(650.0, 80.0, 0.25)),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    // 6 emulsion layers: [BF, BS, GF, GS, RF, RS]
    // Stronger off-diagonal DIR than Portra, but not so strong that slow
    // highlight layers are extinguished by fast-layer inhibitor.
    let dir_matrix = vec![
        vec![0.03, 0.02, 0.09, 0.04, 0.07, 0.03],
        vec![0.02, 0.02, 0.04, 0.02, 0.04, 0.02],
        vec![0.09, 0.04, 0.03, 0.02, 0.10, 0.04],
        vec![0.04, 0.02, 0.02, 0.02, 0.04, 0.02],
        vec![0.07, 0.03, 0.10, 0.04, 0.03, 0.02],
        vec![0.03, 0.02, 0.04, 0.02, 0.02, 0.02],
    ];

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
        antihalation: AntihalationModel {
            reflectance: gaussian_curve(680.0, 60.0, 0.05),
            psf_local_um: 2.5,
            psf_halation_um: 55.0,
        },
        developer_diffusion_length: Microns(6.0),
        adjacency_beta: 0.55,
        dir_diffusion_length: Microns(14.0),
        dir_inhibition_matrix: dir_matrix,
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
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
        assert!(e_gamma.iter().any(|&g| g < 0.36), "slow layers should be hard-γ");
        assert!(e_gamma.iter().any(|&g| g > 0.55), "fast layers should be toe-open γ");
        assert!(p_gamma.iter().all(|&g| (0.5..0.6).contains(&g)));
        assert!(ektar.adjacency_beta > portra.adjacency_beta);

        let e_dir: f32 = ektar
            .dir_inhibition_matrix
            .iter()
            .flatten()
            .copied()
            .sum();
        let p_dir: f32 = portra
            .dir_inhibition_matrix
            .iter()
            .flatten()
            .copied()
            .sum();
        assert!(e_dir > p_dir, "Ektar DIR sum {e_dir} should exceed Portra {p_dir}");
    }
}
