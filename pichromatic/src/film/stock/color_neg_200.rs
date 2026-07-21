//! Multilayer color-negative stock, ISO 200 class (MVP).
//!
//! Stack (top → bottom): overcoat, blue emulsion, yellow filter, green emulsion,
//! red emulsion, antihalation.
//!
//! Sensitivities: Gaussian peaks near ~450 / ~550 / ~650 nm.
//! Image dyes: yellow / magenta / cyan Gaussian ε peaks near ~450 / ~550 / ~650 nm
//! (complementary absorption).
//! Mask: residual colored coupler with broad orange absorption; spatially from
//! undeveloped coupler (no grain).
//!
//! Crystal sizes: mean diameters ~0.5–0.7 µm, σ_ln ∈ [0.25, 0.4] — order-of-magnitude
//! from published AgX emulsion surveys for ISO 200 class color negative.
//!
//! Calibration target: mid-gray ACEScg 0.185 under sunny-16 metering at box ISO
//! (absolute L ≈ K·0.185·N²/(t·S) via BaselineExposureCompensation) lands near
//! developable fraction ≈ 0.4 after per-layer `capture_k` balancing
//! ([`crate::film::constants::ABSORPTION_SIGMA_SCALE_PER_UM`] + layer `capture_k`).
//!
//! # Known MVP gaps
//! - **Single-speed layers only** — no fast/slow pairs per color; highlight latitude
//!   is narrower than real C-41 stocks.
//! - **No external densitometric stock reference** — ColorChecker ΔE tests are
//!   pipeline round-trips against the *input* patches, not measured Fuji/Kodak data.
//! - Layer `capture_k`, mask scale
//!   ([`crate::film::constants::MASK_DENSITY_FRACTION_OF_DMAX`]), and
//!   `adjacency_beta` remain empirically fit.
//! - Antihalation stack absorption vs
//!   [`crate::film::stock::AntihalationModel::reflectance`] are independently
//!   authored (same physical coating in reality; not linked yet).
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
    // Broad orange absorption: higher in blue-green, lower in deep red.
    // Provisional analytic shape for residual colored couplers (C-41 class).
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let blue = (-0.5 * ((l - 450.0) / 50.0).powi(2)).exp();
            let green = 0.6 * (-0.5 * ((l - 520.0) / 60.0).powi(2)).exp();
            0.35 * blue + 0.25 * green + 0.05
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

    // Blue-sensitive → forms yellow dye.
    let blue = EmulsionLayer {
        name: "blue_sensitive",
        depth_from_surface: Microns(1.0),
        thickness: Microns(4.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(450.0, 35.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.55_f64.ln(), // ~0.55 µm mean
            sigma_ln: 0.30,
        }),
        silver_halide_fraction: 0.18,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(450.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 2.2,
        }),
        gamma_contrast: 0.65,
        capture_k: 4.2,
        reciprocity_p: 0.88,
        is_reversal: false,
    };

    // Yellow filter between blue and green/red groups.
    let yellow_filter = EmulsionLayer {
        name: "yellow_filter",
        depth_from_surface: Microns(5.0),
        thickness: Microns(2.0),
        kind: LayerKind::Filter,
        spectral_sensitivity: Some(gaussian_curve(430.0, 40.0, 1.2)), // strong blue absorption
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: false,
    };

    // Green-sensitive → forms magenta dye.
    let green = EmulsionLayer {
        name: "green_sensitive",
        depth_from_surface: Microns(7.0),
        thickness: Microns(5.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(550.0, 40.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.60_f64.ln(),
            sigma_ln: 0.32,
        }),
        silver_halide_fraction: 0.18,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(550.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 2.2,
        }),
        gamma_contrast: 0.65,
        capture_k: 2.1,
        reciprocity_p: 0.90,
        is_reversal: false,
    };

    // Red-sensitive → forms cyan dye.
    let red = EmulsionLayer {
        name: "red_sensitive",
        depth_from_surface: Microns(12.0),
        thickness: Microns(6.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(gaussian_curve(650.0, 45.0, 1.0)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.70_f64.ln(),
            sigma_ln: 0.35,
        }),
        silver_halide_fraction: 0.18,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(650.0, 45.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 2.2,
        }),
        gamma_contrast: 0.65,
        capture_k: 1.35,
        reciprocity_p: 0.92,
        is_reversal: false,
    };

    let antihalation = EmulsionLayer {
        name: "antihalation",
        depth_from_surface: Microns(18.0),
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

    let stock = FilmStock {
        name: "ColorNeg200",
        box_iso: IsoSpeed(200.0),
        layers: vec![overcoat, blue, yellow_filter, green, red, antihalation],
        antihalation: AntihalationModel {
            reflectance: gaussian_curve(680.0, 60.0, 0.08),
            psf_local_um: 3.0,
            psf_halation_um: 80.0,
        },
        developer_diffusion_length: Microns(8.0),
        adjacency_beta: 0.35,
        dir_diffusion_length: Microns(15.0),
        dir_inhibition_matrix: vec![
            vec![0.02, 0.05, 0.03],
            vec![0.05, 0.02, 0.06],
            vec![0.03, 0.06, 0.02],
        ],
        scanner_light: SpectralCurve::constant(1.0),
        capture_luts: vec![],
        grain_kappa: vec![],
    };
    stock.finalize()
}

#[cfg(test)]
mod tests {
    use crate::film::stock::{LayerKind, StockId};

    fn argmax(samples: &[f64]) -> usize {
        samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    #[test]
    fn color_stock_validates() {
        let stock = StockId::ColorNeg200.load().unwrap();
        let emulsions: Vec<_> = stock.emulsion_layers().collect();
        assert!(emulsions.len() >= 3);
        // Yellow filter between blue and green groups.
        let names: Vec<_> = stock.layers.iter().map(|l| l.name).collect();
        let blue_i = names.iter().position(|n| *n == "blue_sensitive").unwrap();
        let filt_i = names.iter().position(|n| *n == "yellow_filter").unwrap();
        let green_i = names.iter().position(|n| *n == "green_sensitive").unwrap();
        assert!(blue_i < filt_i && filt_i < green_i);
        assert_eq!(stock.layers[filt_i].kind, LayerKind::Filter);
    }

    #[test]
    fn layer_spectral_peaks() {
        let stock = StockId::ColorNeg200.load().unwrap();
        let blue = stock
            .layers
            .iter()
            .find(|l| l.name == "blue_sensitive")
            .unwrap();
        let green = stock
            .layers
            .iter()
            .find(|l| l.name == "green_sensitive")
            .unwrap();
        let red = stock
            .layers
            .iter()
            .find(|l| l.name == "red_sensitive")
            .unwrap();
        let pb = argmax(&blue.spectral_sensitivity.as_ref().unwrap().samples);
        let pg = argmax(&green.spectral_sensitivity.as_ref().unwrap().samples);
        let pr = argmax(&red.spectral_sensitivity.as_ref().unwrap().samples);
        assert!(pb < pg && pg < pr, "peaks B={pb} G={pg} R={pr}");
    }

    #[test]
    fn dye_peaks_complementary() {
        let stock = StockId::ColorNeg200.load().unwrap();
        let yellow = stock
            .layers
            .iter()
            .find(|l| l.name == "blue_sensitive")
            .unwrap()
            .coupler
            .as_ref()
            .unwrap();
        let magenta = stock
            .layers
            .iter()
            .find(|l| l.name == "green_sensitive")
            .unwrap()
            .coupler
            .as_ref()
            .unwrap();
        let cyan = stock
            .layers
            .iter()
            .find(|l| l.name == "red_sensitive")
            .unwrap()
            .coupler
            .as_ref()
            .unwrap();
        let py = argmax(&yellow.epsilon.samples);
        let pm = argmax(&magenta.epsilon.samples);
        let pc = argmax(&cyan.epsilon.samples);
        // Yellow absorbs blue (short), magenta green (mid), cyan red (long).
        assert!(py < pm && pm < pc, "dye peaks Y={py} M={pm} C={pc}");
    }

    #[test]
    fn capture_luts_built() {
        let stock = StockId::ColorNeg200.load().unwrap();
        for (i, layer) in stock.layers.iter().enumerate() {
            if layer.kind == LayerKind::Emulsion {
                let lut = stock.capture_luts[i].as_ref().expect("LUT present");
                assert!(!lut.fraction.is_empty());
            }
        }
    }
}
