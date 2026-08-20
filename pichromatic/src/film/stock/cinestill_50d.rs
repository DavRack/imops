//! CineStill 50D class color-negative stock definition.
//!
//! This is CineStill's stated no-AH product: Kodak VISION3 50D (5203) ECN-2 as
//! sold by CineStill, made without the anti-halation layer, which CineStill
//! attributes to the red-halation signature. Acetate base (not PET).
//!
//! All spectral data — sensitivity curves, dye-ε curves, mid-region H&D gamma,
//! midscale-neutral dye density (0.70 / 0.73 / 0.63 above D-min), rms
//! granularity, and reciprocity range (no failure 1/1000 s to 1 s) — was
//! digitized from the official Kodak datasheet H-1-5203 (March 2026 revision,
//! pages 4-5). Measured curve peaks: sensitivity blue 465 nm / green 545 nm /
//! red 645 nm; dye peaks yellow 445 nm / magenta 535 nm / cyan 680 nm. On the
//! model's 20 nm grid these quantize to 460 / 540 / 640 nm (sensitivity) and
//! 440 / 540 / 680 nm (dyes). Relative sensitivity B:G:R = 1.0 : 0.62 : 0.57
//! (kept in the curve amplitudes; per-layer `capture_k` then balances the
//! layers to daylight-gray neutrality at mid-gray). Status M D-min:
//! B 0.82 / G 0.55 / R 0.13; diffuse D-min spectrum peaks ~455 nm at
//! 0.63 density (blue-heavy base cast).
//!
//! Kodak's March 2026 sheet also describes an AHU undercoat on current
//! VISION3 50D; that absorber is omitted here because it is not part of
//! CineStill's no-AH product. Do not mix Kodak AHU stack properties into
//! this stock.
//!
//! # Known MVP gaps
//! - No D-min parameter exists in the model: the real film's blue-heavy base
//!   cast (diffuse D-min 0.63 @ 455 nm; Status M B 0.82 / G 0.55 / R 0.13) is
//!   not represented — ECN-2 has no orange mask, so `mask_epsilon` is `None`
//!   and the simulated D-min is flat (scan-normalized).
//! - `d_max` is calibrated from the datasheet's midscale-neutral dye density
//!   at the model's mid-gray developable fraction; the true film D-max lies
//!   beyond the datasheet's plotted range (extended highlight latitude, curves
//!   still rising at log-E ≈ 2.1). Like the other stocks, the simulated H&D
//!   curve is steeper through mid-gray than the measured film (the model's
//!   capture LUT is intrinsically steep; layer `gamma_contrast` is a dye-
//!   formation exponent, not the composite curve gamma).
//! - Sensitivity is zeroed outside the wavelength range the datasheet plots
//!   (blue ≤ 480 nm, green 480-580 nm, red 580-660 nm): long-wave leakage
//!   tails are not modeled.
//! - No in-stack anti-halation absorber (CineStill construction). Backing uses
//!   the frozen kit Fresnel `R`; halation PSF (90 µm) is the existing
//!   uncalibrated interface model, not measured CineStill optical data.

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

fn measured_curve(samples: &[f64; 16]) -> SpectralCurve {
    SpectralCurve::new(WavelengthGrid::mvp(), samples.to_vec())
}

// Digitized from the H-1-5203 datasheet sensitivity curves (400-700 nm / 20 nm,
// global-max normalized).
const BLUE_SENSITIVITY: [f64; 16] = [
    0.5051, 0.6187, 0.6894, 1.0, 0.2486, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];
const GREEN_SENSITIVITY: [f64; 16] = [
    0.0, 0.0, 0.0, 0.0, 0.0284, 0.1522, 0.3057, 0.6169, 0.4894, 0.1607, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0,
];
const RED_SENSITIVITY: [f64; 16] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0482, 0.2386, 0.3486, 0.5693, 0.3786, 0.0, 0.0,
];

// Spectral dye density curves (diffuse, D-min subtracted, peak-normalized).
const YELLOW_EPSILON: [f64; 16] = [
    0.4412, 0.7256, 1.0, 0.9164, 0.5082, 0.1019, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];
const MAGENTA_EPSILON: [f64; 16] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.4392, 0.86, 1.0, 0.6795, 0.2251, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];
const CYAN_EPSILON: [f64; 16] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.051, 0.2871, 0.5328, 0.7535, 0.9224, 1.0, 0.9417,
];

pub fn load() -> Result<FilmStock, FilmError> {
    let overcoat = kit::overcoat(0.0);

    // --- BLUE FAST & SLOW ---
    // ISO-50 T-grain, finer than Ektar (100-speed); blue crystals ~2x the
    // green/red diameter per the measured rms granularity (blue 10.9 vs 5.8:
    // in this model grain κ ∝ crystal diameter for tabular grains).
    let blue_fast = EmulsionLayer {
        name: "blue_fast",
        depth_from_surface: Microns(1.0),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(measured_curve(&BLUE_SENSITIVITY)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 1.13_f64.ln(),
            sigma_ln: 0.32,
        }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: measured_curve(&YELLOW_EPSILON),
            mask_epsilon: None,
            d_max: 1.82, // total 3.65 split 50/50 fast/slow
        }),
        gamma_contrast: 0.58,
        capture_k: 2.00,
        reciprocity_p: 0.99,
        is_reversal: false,
    };
    let blue_slow = EmulsionLayer {
        name: "blue_slow",
        // capture_k is large (16.9 here, up to 29.76 for green/red slow) because
        // the physical quantity is λ = k·s²·Φ and these slow-layer crystals are
        // ~1/4 the fast-layer diameter; k is not directly comparable across
        // stocks with different crystal sizes.
        depth_from_surface: Microns(3.5),
        thickness: Microns(2.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(measured_curve(&BLUE_SENSITIVITY)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.49_f64.ln(),
            sigma_ln: 0.26,
        }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "yellow",
            epsilon: measured_curve(&YELLOW_EPSILON),
            mask_epsilon: None,
            d_max: 1.82,
        }),
        gamma_contrast: 0.42,
        capture_k: 6.80,
        reciprocity_p: 0.99,
        is_reversal: false,
    };

    let yellow_filter = kit::yellow_filter(6.0, 2.0, gaussian_curve(430.0, 40.0, 1.3));

    // --- GREEN FAST & SLOW ---
    let green_fast = EmulsionLayer {
        name: "green_fast",
        depth_from_surface: Microns(8.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(measured_curve(&GREEN_SENSITIVITY)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.60_f64.ln(),
            sigma_ln: 0.32,
        }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: measured_curve(&MAGENTA_EPSILON),
            mask_epsilon: None,
            d_max: 1.82, // total 3.64
        }),
        gamma_contrast: 0.60,
        capture_k: 7.10,
        reciprocity_p: 0.99,
        is_reversal: false,
    };
    let green_slow = EmulsionLayer {
        name: "green_slow",
        depth_from_surface: Microns(11.0),
        thickness: Microns(3.0),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(measured_curve(&GREEN_SENSITIVITY)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.26_f64.ln(),
            sigma_ln: 0.26,
        }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "magenta",
            epsilon: measured_curve(&MAGENTA_EPSILON),
            mask_epsilon: None,
            d_max: 1.82,
        }),
        gamma_contrast: 0.44,
        capture_k: 12.00,
        reciprocity_p: 0.99,
        is_reversal: false,
    };

    // --- RED FAST & SLOW ---
    let red_fast = EmulsionLayer {
        name: "red_fast",
        depth_from_surface: Microns(14.0),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(measured_curve(&RED_SENSITIVITY)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.62_f64.ln(),
            sigma_ln: 0.32,
        }),
        silver_halide_fraction: 0.16,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: measured_curve(&CYAN_EPSILON),
            mask_epsilon: None,
            d_max: 1.88, // total 3.76
        }),
        gamma_contrast: 0.52,
        capture_k: 5.55,
        reciprocity_p: 0.99,
        is_reversal: false,
    };
    let red_slow = EmulsionLayer {
        name: "red_slow",
        depth_from_surface: Microns(17.5),
        thickness: Microns(3.5),
        kind: LayerKind::Emulsion,
        spectral_sensitivity: Some(measured_curve(&RED_SENSITIVITY)),
        crystal_size: Some(LogNormalDist {
            mu_ln: 0.27_f64.ln(),
            sigma_ln: 0.26,
        }),
        silver_halide_fraction: 0.20,
        coupler: Some(DyeCoupler {
            name: "cyan",
            epsilon: measured_curve(&CYAN_EPSILON),
            mask_epsilon: None,
            d_max: 1.88,
        }),
        gamma_contrast: 0.36,
        capture_k: 10.50,
        reciprocity_p: 0.99,
        is_reversal: false,
    };

    let stock = FilmStock {
        name: "CineStill50D",
        box_iso: IsoSpeed(50.0),
        layers: vec![
            overcoat,
            blue_fast,
            blue_slow,
            yellow_filter,
            green_fast,
            green_slow,
            red_fast,
            red_slow,
        ],
        // No in-stack AH (CineStill product). Kit Fresnel R + existing halation PSF.
        antihalation: kit::backing(90.0),
        irradiation_response: None,
        developer_diffusion_length: Microns(7.0),
        inhibitor_diffusion_length: Microns(0.0),
        adjacency_beta: 0.30,
        adjacency_beta_record: 0.0,
        adjacency_beta_cross: 0.0,
        adjacency_beta_dir: 0.0,
        scanner_light: SpectralCurve::d50(),
        capture_luts: vec![],
        grain_kappa: vec![],
        tabular_grain_thickness_um: Some(kit::T_GRAIN_THICKNESS_UM),
    };
    stock.finalize()
}

#[cfg(test)]
mod tests {
    use crate::film::development::reduction::reduce;
    use crate::film::exposure::expose_with_pitch_and_shutter;
    use crate::film::exposure::radiance::{reciprocity_factor, relative_to_absolute_luminance};
    use crate::film::stock::StockId;
    use crate::pixel::MIDDLE_GRAY;

    fn mean(plane: &[f32]) -> f32 {
        plane.iter().sum::<f32>() / plane.len() as f32
    }

    #[test]
    fn cinestill50d_calibration_anchors() {
        let stock = StockId::CineStill50D.load().unwrap();
        assert_eq!(stock.name, "CineStill50D");
        assert!((stock.box_iso.0 - 50.0).abs() < 1e-6);

        // Mid-gray absolute luminance at sunny-16, box ISO 50: L = K·v·N²/(t·S)
        // with K = 12.5, v = 0.185, N = 8, t = 1/50 s, S = 50 → 148 nits.
        let l_mid =
            relative_to_absolute_luminance(MIDDLE_GRAY as f64, 1.0 / 50.0, 8.0, 50.0) as f32;
        assert!((l_mid - 148.0).abs() < 1e-3);

        const N: usize = 16;
        let rgb = vec![[l_mid, l_mid, l_mid]; N * N];
        let latent = expose_with_pitch_and_shutter(&rgb, N, N, &stock, 10.0, 1.0 / 50.0);

        // Latent planes in emulsion order: [BF, BS, GF, GS, RF, RS].
        let f_bf = mean(&latent.layers[0]);
        let f_bs = mean(&latent.layers[1]);
        let f_gf = mean(&latent.layers[2]);
        let f_gs = mean(&latent.layers[3]);
        let f_rf = mean(&latent.layers[4]);
        let f_rs = mean(&latent.layers[5]);

        for (name, f) in [
            ("blue_fast", f_bf),
            ("green_fast", f_gf),
            ("red_fast", f_rf),
        ] {
            assert!(
                (0.45..=0.65).contains(&f),
                "{name} mid-gray developable {f} outside [0.45, 0.65]"
            );
        }
        for (name, f) in [
            ("blue_slow", f_bs),
            ("green_slow", f_gs),
            ("red_slow", f_rs),
        ] {
            assert!(
                (0.10..=0.35).contains(&f),
                "{name} mid-gray developable {f} outside [0.10, 0.35]"
            );
        }
        assert!(
            (f_bf - f_gf).abs() < 0.10,
            "blue/green fast imbalance |{f_bf} - {f_gf}| = {}",
            (f_bf - f_gf).abs()
        );
        assert!(
            (f_bf - f_rf).abs() < 0.10,
            "blue/red fast imbalance |{f_bf} - {f_rf}| = {}",
            (f_bf - f_rf).abs()
        );

        // Development: per-color density = fast + slow layer (same dye).
        let dyes = reduce(&stock, &latent);
        let d_blue = mean(&dyes.image_dye[0]) + mean(&dyes.image_dye[1]);
        let d_green = mean(&dyes.image_dye[2]) + mean(&dyes.image_dye[3]);
        let d_red = mean(&dyes.image_dye[4]) + mean(&dyes.image_dye[5]);
        for (name, d) in [("blue", d_blue), ("green", d_green), ("red", d_red)] {
            assert!(
                (0.55..=0.85).contains(&d),
                "{name} mid-gray dye density {d} outside [0.55, 0.85]"
            );
        }

        // Measured: no reciprocity failure for 1/1000 s to 1 s → p ≈ 1.
        // p = 0.99 must be effectively flat in-band and show only mild loss
        // just outside it (HIRF 0.5 ms / LIRF 2 s: η = 0.5^0.01 ≈ 0.993).
        assert_eq!(reciprocity_factor(1.0 / 50.0, 0.99), 1.0);
        assert_eq!(reciprocity_factor(1.0 / 50.0, 1.0), 1.0); // p=1 control
        for (t, expect) in [(0.0005, 0.9931), (2.0, 0.9931)] {
            let eta = reciprocity_factor(t, 0.99);
            assert!(
                (eta - expect).abs() < 1e-3,
                "η({t}s, 0.99) = {eta}, expected ≈ {expect}"
            );
        }
    }
}
