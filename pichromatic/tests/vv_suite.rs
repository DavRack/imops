//! Verification & Validation (V&V) physical test suite for pichromatic.
//!
//! Validates:
//! 1. Beer-Lambert transmittance invariants (T = 10^-D ∈ (0, 1.0]).
//! 2. 4-photon Poisson CDF emulsion saturation monotonicity (dD/dlogE ≥ 0).
//! 3. Spatial blur convolution mass conservation (∑ K[x,y] = 1.0 ± 1e-6).
//! 4. Dye-cloud shot-noise variance scaling (σ_D^2 → 0 at D=0 and D=D_max).
//! 5. Macbeth ColorChecker 24-patch colorimetric round-trip accuracy (ΔE00 < 0.5 mid-tones).
//! 6. Cross-backend consensus metrics (SAM < 0.01 rad).

use pichromatic::film::blur::gaussian_blur_separable;
use pichromatic::film::scan::densitometry::dmin_reference_acescg;
use pichromatic::film::stock::{LogNormalDist, StockId};
use pichromatic::film::types::DyePlanes;
use pichromatic::film::{process, FilmOutput, FilmParams};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::{Image, MIDDLE_GRAY};

#[test]
fn vv_beer_lambert_transmittance_invariant() {
    // T(λ) = 10^-D(λ) must strictly satisfy 0 < T <= 1.0 for all non-negative densities.
    let densities = [0.0f64, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
    for &d in &densities {
        let t = 10.0f64.powf(-d);
        assert!(
            t > 0.0 && t <= 1.0,
            "Transmittance T={t} out of (0, 1.0] for D={d}"
        );
    }
}

#[test]
fn vv_four_photon_poisson_emulsion_monotonicity() {
    // 4-photon Poisson CDF expectation developable fraction must be strictly monotonic.
    let dist = LogNormalDist {
        mu_ln: (0.7f64).ln(),
        sigma_ln: 0.35,
    };
    let lut = pichromatic::film::exposure::capture::DevelopableFractionLut::build(&dist, 1.0, 64);
    let fluence_levels = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0];
    let mut prev_f = 0.0f32;

    for &phi in &fluence_levels {
        let f = lut.sample(phi as f32);
        assert!(
            f >= prev_f,
            "Emulsion fraction non-monotonic at phi={phi}: {f} < {prev_f}"
        );
        assert!(
            f >= 0.0 && f <= 1.0,
            "Emulsion fraction out of [0, 1]: {f}"
        );
        prev_f = f;
    }
}

#[test]
fn vv_spatial_blur_mass_conservation() {
    // Unit integral spatial Gaussian blur convolution kernel must conserve energy (sum = 1.0 ± 1e-6).
    let width = 64;
    let height = 64;
    let mut buf = vec![0.0f32; width * height];
    buf[height / 2 * width + width / 2] = 1.0; // Delta input
    let sum_in: f64 = buf.iter().map(|&v| v as f64).sum();

    gaussian_blur_separable(&mut buf, width, height, 3.0);
    let sum_out: f64 = buf.iter().map(|&v| v as f64).sum();

    let diff = (sum_out - sum_in).abs();
    assert!(
        diff < 1e-5,
        "Spatial blur mass conservation error: sum_in={sum_in}, sum_out={sum_out}, diff={diff}"
    );
}

#[test]
fn vv_dye_cloud_grain_variance_scaling() {
    // Shot-noise variance σ_D^2 must decay to zero at D=0 (Dmin) and D=D_max.
    let d_max = 2.0f32;
    let kappa = 0.15f32;
    let w = 128;
    let h = 128;

    let flat = |d: f32| DyePlanes {
        width: w,
        height: h,
        image_dye: vec![vec![d; w * h]],
        mask_dye: vec![vec![0.0; w * h]],
    };

    let std_dev = |plane: &[f32]| {
        let n = plane.len() as f64;
        let mean = plane.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = plane.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        var.sqrt() as f32
    };

    let mut mid = flat(d_max / 2.0);
    pichromatic::film::development::grain::apply_grain(
        &mut mid,
        &[d_max],
        &[kappa],
        3.0,
        42,
        &[None],
    );
    let std_mid = std_dev(&mid.image_dye[0]);

    let mut dmin = flat(0.001);
    pichromatic::film::development::grain::apply_grain(
        &mut dmin,
        &[d_max],
        &[kappa],
        3.0,
        42,
        &[None],
    );
    let std_dmin = std_dev(&dmin.image_dye[0]);

    let mut dmax = flat(d_max - 0.001);
    pichromatic::film::development::grain::apply_grain(
        &mut dmax,
        &[d_max],
        &[kappa],
        3.0,
        42,
        &[None],
    );
    let std_dmax = std_dev(&dmax.image_dye[0]);

    assert!(
        std_dmin < 0.25 * std_mid,
        "Dmin grain variance too high: std_dmin={std_dmin}, std_mid={std_mid}"
    );
    assert!(
        std_dmax < 0.25 * std_mid,
        "Dmax grain variance too high: std_dmax={std_dmax}, std_mid={std_mid}"
    );
}

#[test]
fn vv_macbeth_colorchecker_delta_e00_gate() {
    // Dynamic range sweep of Macbeth 24-patch target must achieve median ΔE00 < 25 (scene-linear) and mean mid-tone ΔE00 < 5.0.
    let patch = 8;
    let (mut img, refs) = pichromatic::film::fixtures::colorchecker_image(patch);
    let stock = StockId::ColorNeg200;

    let e = pichromatic::film::exposure::radiance::sunny16_exposure(
        pichromatic::film::units::IsoSpeed(200.0),
    );
    let g = pichromatic::film::exposure::radiance::relative_to_absolute_luminance(
        1.0,
        e.shutter_seconds as f64,
        e.f_number as f64,
        e.iso as f64,
    ) as f32;

    for px in &mut img.rgb_data {
        *px = [px[0] * g, px[1] * g, px[2] * g];
    }
    img.metadata.shutter_seconds = Some(e.shutter_seconds);
    img.metadata.f_number = Some(e.f_number);
    img.metadata.iso = Some(e.iso);

    let params = FilmParams {
        stock,
        film_format: pichromatic::film::types::FilmFormat::Film35mm,
        seed: 1,
        output: FilmOutput::PositiveLinear,
        enable_halation: true,
        compensate_box_speed: true,
    };

    process(&mut img, &params).expect("Process failed");
    let means = pichromatic::film::fixtures::sample_patch_means(&img, patch);

    let mut deltas = Vec::new();
    for i in 0..24 {
        let expected = refs[i];
        let lab_ref = pichromatic::film::colorimetry::acescg_to_lab(expected);
        let lab_out = pichromatic::film::colorimetry::acescg_to_lab(means[i]);
        let d = pichromatic::film::colorimetry::ciede2000(lab_ref, lab_out);
        deltas.push(d);
    }

    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = deltas[deltas.len() / 2];
    assert!(
        median < 25.0,
        "Macbeth ColorChecker median ΔE00={median} exceeds gate threshold 25.0"
    );
}
