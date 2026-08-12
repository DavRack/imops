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
    // Empirical Dmin spectrum ratio @ 400-700nm (Blue 0.816, Green 0.623, Red 0.188)
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let blue = (-0.5 * ((l - 425.0) / 45.0).powi(2)).exp();
            let green = 0.65 * (-0.5 * ((l - 535.0) / 55.0).powi(2)).exp();
            0.52 * blue + 0.35 * green + 0.08
        })
        .collect();
    SpectralCurve::new(grid, samples)
}

fn blue_sensitivity_curve() -> SpectralCurve {
    // Empirical Portra 400 Blue peak @ 400nm (UV/near-violet sensitivity)
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let p1 = (-0.5 * ((l - 400.0) / 30.0).powi(2)).exp();
            let p2 = 0.4 * (-0.5 * ((l - 440.0) / 25.0).powi(2)).exp();
            (p1 + p2).min(1.0)
        })
        .collect();
    SpectralCurve::new(grid, samples)
}

fn green_sensitivity_curve() -> SpectralCurve {
    // Empirical Green peak @ 545nm with native AgHalide blue shoulder @ 430nm
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let main = (-0.5 * ((l - 545.0) / 35.0).powi(2)).exp();
            let blue_shoulder = 0.25 * (-0.5 * ((l - 430.0) / 25.0).powi(2)).exp();
            (main + blue_shoulder).min(1.0)
        })
        .collect();
    SpectralCurve::new(grid, samples)
}

fn red_sensitivity_curve() -> SpectralCurve {
    // Empirical Red peak @ 645nm with asymmetric shoulder
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| {
            let d = if l < 645.0 {
                (l - 645.0) / 45.0
            } else {
                (l - 645.0) / 30.0
            };
            (-0.5 * d * d).exp()
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
    // Empirical Blue H&D Gamma ~ 0.691, Dmax = 2.82
    let blue_fast = EmulsionLayer {
        name: "blue_fast",
        depth_from_surface: Microns(1.0),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(blue_sensitivity_curve()),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.85_f64.ln(),
            sigma_ln: 0.88,
        }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(445.0, 35.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.575, // total 1.15 split fast/slow
        }),
        gamma_contrast: 0.691,
        capture_k: 3.74,
        reciprocity_p: 0.85,
        is_reversal: false,
    };
    let blue_slow = EmulsionLayer {
        name: "blue_slow",
        depth_from_surface: Microns(3.5),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(blue_sensitivity_curve()),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.42_f64.ln(),
            sigma_ln: 0.28,
        }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: gaussian_curve(445.0, 35.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.575, // yellow slow
        }),
        gamma_contrast: 0.691,
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
    // Empirical Green H&D Gamma ~ 0.618, Dmax = 2.38
    let green_fast = EmulsionLayer {
        name: "green_fast",
        depth_from_surface: Microns(8.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(green_sensitivity_curve()),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.90_f64.ln(),
            sigma_ln: 0.88,
        }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(550.0, 35.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.666,
        }),
        gamma_contrast: 0.618,
        capture_k: 1.87,
        reciprocity_p: 0.87,
        is_reversal: false,
    };
    let green_slow = EmulsionLayer {
        name: "green_slow",
        depth_from_surface: Microns(11.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(green_sensitivity_curve()),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.45_f64.ln(),
            sigma_ln: 0.28,
        }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: gaussian_curve(550.0, 35.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.666, // magenta slow
        }),
        gamma_contrast: 0.618,
        capture_k: 1.1,
        reciprocity_p: 0.91,
        is_reversal: false,
    };

    // --- RED FAST & SLOW ---
    // Empirical Red H&D Gamma ~ 0.542, Dmax = 1.76, Cyan dye peak 680nm
    let red_fast = EmulsionLayer {
        name: "red_fast",
        depth_from_surface: Microns(14.0),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(red_sensitivity_curve()),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.95_f64.ln(),
            sigma_ln: 0.88,
        }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(680.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.704,
        }),
        gamma_contrast: 0.542,
        capture_k: 1.22,
        reciprocity_p: 0.89,
        is_reversal: false,
    };
    let red_slow = EmulsionLayer {
        name: "red_slow",
        depth_from_surface: Microns(17.5),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(red_sensitivity_curve()),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.48_f64.ln(),
            sigma_ln: 0.28,
        }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: gaussian_curve(680.0, 40.0, 1.0),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.704, // cyan slow
        }),
        gamma_contrast: 0.542,
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
            psf_local_um: 0.75,
            psf_halation_um: 70.0,
        },
        developer_diffusion_length: Microns(7.0),
        adjacency_beta: 0.30,
        dir_diffusion_length: Microns(16.0),
        dir_inhibition_matrix: dir_matrix,
        scanner_light: SpectralCurve::d50(),
        capture_luts: vec![],
        grain_kappa: vec![],
        // Kodak T-grain (tabular) morphology: plate thickness ~0.15 µm
        // (published T-grain range 0.1-0.2 µm).
        tabular_grain_thickness_um: Some(0.15),
    };
    stock.finalize()
}

#[cfg(test)]
mod runtime_calibration_tests {
    use super::*;
    use crate::film::development::reduction::reduce;
    use crate::film::FilmFormat;
    use crate::film::scan::{invert::invert_negative, normalized_dmin_acescg};
    use crate::film::types::{DyePlanes, LatentPlanes};

    #[test]
    fn fast_layer_capture_toe_has_shadow_latitude() {
        use crate::film::exposure::expose_with_pitch_and_shutter;
        use crate::film::exposure::radiance::relative_to_absolute_luminance;
        use crate::pixel::MIDDLE_GRAY;

        let stock = load().unwrap();
        const N: usize = 16;
        let shutter = 1.0f32 / stock.box_iso.0;
        let l_mid = relative_to_absolute_luminance(
            MIDDLE_GRAY as f64,
            shutter as f64,
            8.0,
            stock.box_iso.0 as f64,
        ) as f32;

        let mut fast_at = |ratio: f32| -> [f32; 3] {
            let l = l_mid * ratio;
            let rgb = vec![[l, l, l]; N * N];
            let latent = expose_with_pitch_and_shutter(&rgb, N, N, &stock, 10.0, shutter);
            // Fast layers: blue, green, red (indices 0, 2, 4).
            [
                latent.layers[0].iter().sum::<f32>() / (N * N) as f32,
                latent.layers[2].iter().sum::<f32>() / (N * N) as f32,
                latent.layers[4].iter().sum::<f32>() / (N * N) as f32,
            ]
        };

        let f_mid = fast_at(1.0);
        for (name, f) in [("blue_fast", f_mid[0]), ("green_fast", f_mid[1]), ("red_fast", f_mid[2])]
        {
            assert!(
                (0.40..=0.60).contains(&f),
                "{name} mid-gray developable {f} outside [0.40, 0.60]"
            );
        }

        let f_deep = fast_at(0.03);
        let f_shadow = fast_at(0.1);
        for (name, f) in [
            ("blue_fast", f_deep[0]),
            ("green_fast", f_deep[1]),
            ("red_fast", f_deep[2]),
        ] {
            assert!(
                f > 1.0e-4,
                "{name} developable at L/L_mid=0.03 must exceed 1e-4, got {f}"
            );
        }
        for (name, f) in [
            ("blue_fast", f_shadow[0]),
            ("green_fast", f_shadow[1]),
            ("red_fast", f_shadow[2]),
        ] {
            assert!(
                f >= 0.015,
                "{name} developable at L/L_mid=0.1 must be >= 0.015, got {f}"
            );
        }

        // Toe must stay below mid-gray calibration.
        for f in f_shadow {
            assert!(f < f_mid[0].min(f_mid[1]).min(f_mid[2]) * 0.5);
        }
    }

    #[test]
    fn spectral_sensitivity_and_dye_curves_are_finite_and_grid_resolved() {
        let stock = load().unwrap();
        let expected_sensitivity_peaks = [400.0, 400.0, 540.0, 540.0, 640.0, 640.0];
        let expected_dye_peaks = [440.0, 440.0, 540.0, 540.0, 680.0, 680.0];
        let emulsions: Vec<_> = stock.emulsion_layers().collect();
        assert_eq!(emulsions.len(), 6);

        for (i, (_, layer)) in emulsions.iter().enumerate() {
            let sensitivity = layer.spectral_sensitivity.as_ref().unwrap();
            let dye = &layer.coupler.as_ref().unwrap().epsilon;
            assert_eq!(sensitivity.grid, WavelengthGrid::mvp());
            assert_eq!(dye.grid, WavelengthGrid::mvp());
            assert_eq!(sensitivity.peak_wavelength(), expected_sensitivity_peaks[i]);
            assert_eq!(dye.peak_wavelength(), expected_dye_peaks[i]);
            assert!(sensitivity
                .samples
                .iter()
                .chain(dye.samples.iter())
                .all(|value| value.is_finite() && *value >= 0.0));
        }
    }

    #[test]
    fn density_totals_base_and_viewing_illuminant_are_finite() {
        let stock = load().unwrap();
        let emulsions: Vec<_> = stock.emulsion_layers().collect();
        let dmax = [
            emulsions[0].1.coupler.as_ref().unwrap().d_max
                + emulsions[1].1.coupler.as_ref().unwrap().d_max,
            emulsions[2].1.coupler.as_ref().unwrap().d_max
                + emulsions[3].1.coupler.as_ref().unwrap().d_max,
            emulsions[4].1.coupler.as_ref().unwrap().d_max
                + emulsions[5].1.coupler.as_ref().unwrap().d_max,
        ];
        assert_eq!(dmax, [1.15, 1.332, 1.408]);

        assert_eq!(stock.scanner_light, SpectralCurve::d50());
        assert!(stock
            .scanner_light
            .samples
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));

        let dmin = normalized_dmin_acescg(&stock);
        assert!(dmin.iter().all(|value| value.is_finite() && *value > 0.0));
        assert!((dmin[0] - 1.0).abs() < 1e-6);
        assert!(dmin[0] > dmin[1] && dmin[1] > dmin[2]);
    }

    #[test]
    fn hd_and_layered_density_curves_are_monotonic_and_finite() {
        let stock = load().unwrap();
        for (layer_index, layer) in stock.layers.iter().enumerate() {
            if layer.kind != LayerKind::Emulsion {
                continue;
            }
            let lut = stock.capture_luts[layer_index].as_ref().unwrap();
            let dmax = layer.coupler.as_ref().unwrap().d_max;
            let inv_gamma = 1.0 / layer.gamma_contrast;
            let mut previous_fraction = 0.0;
            let mut previous_density = 0.0;
            for (&fraction, _) in lut.fraction.iter().zip(&lut.log10_fluence) {
                let density = dmax * fraction.powf(inv_gamma);
                assert!(fraction.is_finite() && density.is_finite());
                assert!(fraction + 1e-6 >= previous_fraction);
                assert!(density + 1e-6 >= previous_density);
                previous_fraction = fraction;
                previous_density = density;
            }
        }

        let latent = LatentPlanes {
            width: 3,
            height: 1,
            layers: vec![vec![0.0, 0.5, 1.0]; 6],
        };
        let dyes: DyePlanes = reduce(&stock, &latent);
        for plane in &dyes.image_dye {
            assert!(plane.iter().all(|value| value.is_finite()));
            assert!(plane[0] <= plane[1] + 1e-6 && plane[1] <= plane[2] + 1e-6);
        }
    }

    #[test]
    fn midgray_negative_and_invert_reference_are_finite() {
        let stock = load().unwrap();
        let mid = crate::film::mid_negative_acescg(
            &stock,
            FilmFormat::Film35mm.pixel_pitch_um(1024),
            1.0 / stock.box_iso.0,
        );
        let dmin = normalized_dmin_acescg(&stock);
        assert!(mid.iter().all(|value| value.is_finite() && *value > 0.0));

        let mut positive = vec![mid];
        invert_negative(&mut positive, mid, dmin);
        for value in positive[0] {
            assert!(value.is_finite());
            assert!((value - crate::pixel::MIDDLE_GRAY).abs() < 1e-3);
        }
    }
}
