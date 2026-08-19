//! Kodak Portra 400 class color-negative stock definition.
//!
//! Fitted (datasheet / H&D): ISO, layer S, dyes, `γ` / `d_max` / `capture_k`.
//! Frozen kit ([`crate::film::stock::kit`]): acetate Fresnel `R`, grey AH,
//! T-grain plate thickness. DIR is off (no published matrix).

use crate::film::error::FilmError;
use crate::film::spectrum::{SpectralCurve, WavelengthGrid};
use crate::film::stock::kit;
use crate::film::stock::{
    DyeCoupler, EmulsionLayer, FilmStock, IrradiationResponse, LayerKind, LogNormalDist,
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

// Kodak Professional Portra 400 E-4050 (2016), page 4,
// "Spectral-Dye-Density Curves". Values are the spektrafilm digitization of
// that official plot (`kodak_portra_400.json`, `data.channel_density`), sampled
// at 400:20:700 nm. Small negative unmixing artifacts are clamped to zero and
// each C/M/Y channel is peak-normalized so the existing d_max scale is unchanged.
const E4050_CYAN: [f64; 16] = [
    0.2141127157,
    0.0884377415,
    0.0,
    0.0,
    0.0197136348,
    0.0306758452,
    0.0208735698,
    0.0,
    0.0,
    0.0827627683,
    0.2798804987,
    0.4792663278,
    0.6673921402,
    0.8427773324,
    0.9717129804,
    1.0,
];
const E4050_MAGENTA: [f64; 16] = [
    0.0380242729,
    0.0277833813,
    0.0,
    0.0,
    0.1277816801,
    0.4917706665,
    0.8566500335,
    1.0,
    0.8212038935,
    0.4623861007,
    0.1949634497,
    0.0725356569,
    0.0179332136,
    0.0,
    0.0,
    0.0,
];
const E4050_YELLOW: [f64; 16] = [
    0.3447954370,
    0.6782965700,
    0.9685343243,
    1.0,
    0.6972344725,
    0.2561503345,
    0.0,
    0.0,
    0.0150007151,
    0.0181651297,
    0.0153555546,
    0.0074602746,
    0.0019047223,
    0.0005695983,
    0.0,
    0.0,
];

fn e4050_image_dye(samples: [f64; 16]) -> SpectralCurve {
    SpectralCurve::new(WavelengthGrid::mvp(), samples.to_vec())
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
    // Kodak E-4050 (Portra 400, Feb 2016): no reciprocity compensation from
    // 1/10 000 s to 1 s. p = 1 also disables the invented HIRF branch below 1 ms,
    // which contradicted the sheet (no compensation down to 0.1 ms).
    const RECIPROCITY_P: f32 = 1.0;

    let overcoat = kit::overcoat(0.0);

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
            epsilon: e4050_image_dye(E4050_YELLOW),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.575, // total 1.15 split fast/slow
        }),
        gamma_contrast: 0.691,
        capture_k: 3.74,
        reciprocity_p: RECIPROCITY_P,
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
            epsilon: e4050_image_dye(E4050_YELLOW),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.575, // yellow slow
        }),
        gamma_contrast: 0.691,
        capture_k: 2.2,
        reciprocity_p: RECIPROCITY_P,
        is_reversal: false,
    };

    // Physical interlayer (Carey Lea class). Green/red sensitivity curves are
    // *layer* S with a native AgX blue shoulder, not pack S — keeping the filter
    // is not double-counting until a digitized pack-S curve is ingested.
    let yellow_filter = kit::yellow_filter(6.0, 2.0, gaussian_curve(430.0, 40.0, 1.3));

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
            epsilon: e4050_image_dye(E4050_MAGENTA),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.666,
        }),
        gamma_contrast: 0.618,
        capture_k: 1.87,
        reciprocity_p: RECIPROCITY_P,
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
            epsilon: e4050_image_dye(E4050_MAGENTA),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.666, // magenta slow
        }),
        gamma_contrast: 0.618,
        capture_k: 1.1,
        reciprocity_p: RECIPROCITY_P,
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
            epsilon: e4050_image_dye(E4050_CYAN),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.704,
        }),
        gamma_contrast: 0.542,
        capture_k: 1.22,
        reciprocity_p: RECIPROCITY_P,
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
            epsilon: e4050_image_dye(E4050_CYAN),
            mask_epsilon: Some(orange_mask_epsilon()),
            d_max: 0.704, // cyan slow
        }),
        gamma_contrast: 0.542,
        capture_k: 0.7,
        reciprocity_p: RECIPROCITY_P,
        is_reversal: false,
    };

    let antihalation = kit::antihalation_layer(21.0);

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
        // Halo fluence scales as T_AH² · R. Kit: Fresnel R + grey AH OD.
        antihalation: kit::backing(70.0),
        // Effective joint fit to Kodak E-4050 page 4 processed-film B/G/R
        // response shape
        // (Daylight exposure, C-41), PDF SHA256:
        // e83ac6775d37832a4cb466892a3e1cf4c88917a6ee93384e59d6924b1cd97e3a.
        // The split among this pre-capture component, the realized D/4 cloud
        // PSF, and post-realization adjacency is not independently identified
        // irradiation physics and is not exact Status-M calibration.
        irradiation_response: Some(IrradiationResponse {
            core_sigma_um: 1.8,
            tail_decay_um: 7.4,
            tail_weight_bgr: [0.55, 0.58, 0.79],
        }),
        developer_diffusion_length: Microns(20.2),
        adjacency_beta: 0.66,
        adjacency_beta_record: 0.0,
        adjacency_beta_cross: 0.0,
        scanner_light: SpectralCurve::d50(),
        capture_luts: vec![],
        grain_kappa: vec![],
        tabular_grain_thickness_um: Some(kit::T_GRAIN_THICKNESS_UM),
    };
    stock.finalize()
}

#[cfg(test)]
mod runtime_calibration_tests {
    use super::*;
    use crate::film::development::reduction::reduce;
    use crate::film::scan::{
        invert::invert_negative, normalized_dmin_acescg, scanner_calibration_acescg,
    };
    use crate::film::types::{DyePlanes, LatentPlanes};
    use crate::film::FilmFormat;

    #[test]
    fn portra_image_dyes_match_e4050_digitization_on_mvp_grid() {
        let stock = load().unwrap();
        let expected = [
            E4050_YELLOW,
            E4050_YELLOW,
            E4050_MAGENTA,
            E4050_MAGENTA,
            E4050_CYAN,
            E4050_CYAN,
        ];
        for ((_, layer), samples) in stock.emulsion_layers().zip(expected) {
            let epsilon = &layer.coupler.as_ref().unwrap().epsilon.samples;
            assert_eq!(epsilon.as_slice(), samples.as_slice(), "{}", layer.name);
            assert!(epsilon.iter().all(|&value| value >= 0.0));
            assert_eq!(epsilon.iter().copied().fold(0.0, f64::max), 1.0);
        }
    }

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

        let fast_at = |ratio: f32| -> [f32; 3] {
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
        for (name, f) in [
            ("blue_fast", f_mid[0]),
            ("green_fast", f_mid[1]),
            ("red_fast", f_mid[2]),
        ] {
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
        // E-4050 digitization sampled on the 20 nm MVP grid (Y, M, C).
        let expected_dye_peaks = [460.0, 460.0, 540.0, 540.0, 700.0, 700.0];
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
    fn portra_emulsions_have_no_reciprocity_failure() {
        // E-4050: no compensation 1/10 000 s–1 s; p=1 also avoids invented HIRF.
        let stock = load().unwrap();
        for (_, layer) in stock.emulsion_layers() {
            assert_eq!(layer.reciprocity_p, 1.0, "{}", layer.name);
        }
    }

    #[test]
    fn midgray_negative_and_invert_reference_are_finite() {
        let stock = load().unwrap();
        let calibration = scanner_calibration_acescg(
            &stock,
            FilmFormat::Film35mm.pixel_pitch_um(1024),
            1.0 / stock.box_iso.0,
        )
        .unwrap();
        let mid = calibration.mid;
        let dmin = calibration.dmin;
        assert!(mid.iter().all(|value| value.is_finite() && *value > 0.0));

        let mut positive = vec![mid];
        invert_negative(&mut positive, mid, dmin);
        for value in positive[0] {
            assert!(value.is_finite());
            assert!((value - crate::pixel::MIDDLE_GRAY).abs() < 1e-3);
        }
    }
}
