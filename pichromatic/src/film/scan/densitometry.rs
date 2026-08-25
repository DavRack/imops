//! Densitometric scan: stacked dyes → transmittance → ACEScg encoding.
//!
//! T(λ) = 10^(−Σ D_layer_eff(λ)) with base-10 optical density.
//! Channel via CIE CMFs → XYZ → ACEScg. Scanner gain is normalized against the
//! base-only reference; PositiveLinear separately measures processed Dmin,
//! including chemical fog, through the complete CPU path.
//!
//! All per-pixel math runs in f32 (WGSL has no core f64), mirroring the GPU
//! `SCAN` / `GRAIN_SCAN_ROI` shaders.

use crate::film::blur::gaussian_blur_separable;
use crate::film::constants::{SCANNER_OPTICAL_SIGMA_UM, SCANNER_SENSOR_SIGMA_PX};
use crate::film::error::FilmError;
use crate::film::exposure::upsample::spectrum_to_acescg_rgb_f32;
use crate::film::stock::{FilmStock, LayerKind};
use crate::film::types::DyePlanes;
use crate::film::scan::invert::{GAMMA_EFF, invert_negative};
use crate::pixel::{ImageBuffer, Pixel};
use rayon::prelude::*;

/// log2(10), shared with the GPU `SCAN` shader (`exp2(-d·LOG2_10)`).
const LOG2_10: f32 = 3.3219280948873623;

/// Scene-highlight anchor placement for scanner calibration, in stops above mid-gray.
///
/// Color-negative exposure conventionally places scene highlights about two stops
/// above metered mid-gray (ratio 4× in linear light). Anchoring the measured
/// contrast there fixes each channel's large-signal exponent where highlight
/// reconstruction is most sensitive; mid-gray alone cannot constrain it.
pub const CALIBRATION_ANCHOR_RATIO: f32 = 4.0;

/// Relative finite-difference step for the composite-forward Jacobian behind
/// [`ScannerCalibration::chroma_decode`]. Numerical differentiation parameter
/// only, not physics; the resulting decode is stable against it (see the
/// `chroma_decode_stability_across_fd_steps` diagnostic).
const JACOBIAN_DELTA: f32 = 0.10;

#[derive(Clone, Copy, Debug)]
pub struct ScannerCalibration {
    pub dmin: [f32; 3],
    pub mid: [f32; 3],
    /// Per-channel effective contrast exponents solved from the forward model
    /// between mid-gray and the +2-stop anchor (see [`solve_gamma_eff`]).
    pub gamma_eff: [f32; 3],
    /// Post-invert linear decode: q = MIDDLE_GRAY + M * (p - MIDDLE_GRAY).
    /// Derived from the stock's own composite forward Jacobian so chroma
    /// survives the scan->invert round trip; the achromatic axis is pinned
    /// to the measured neutral anchors.
    pub chroma_decode: [[f32; 3]; 3], // row-major
}

impl ScannerCalibration {
    /// Per-channel inverse exponents (1/γ) consumed by the technical invert.
    pub fn inv_gamma(&self) -> [f32; 3] {
        self.gamma_eff.map(|g| 1.0 / g)
    }
}

/// Effective spectral density at one pixel: Σ_layers (D_image * ε_image + D_mask * ε_mask).
pub(crate) fn density_spectrum(stock: &FilmStock, dyes: &DyePlanes, pixel: usize) -> [f32; 16] {
    let mut d = [0.0f32; 16];
    let mut emulsion_i = 0usize;
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().unwrap();
        let di = dyes.image_dye[emulsion_i][pixel];
        let dm = dyes.mask_dye[emulsion_i][pixel];
        for lambda in 0..16 {
            d[lambda] += di * coupler.epsilon.samples[lambda] as f32;
            if let Some(ref mask_eps) = coupler.mask_epsilon {
                d[lambda] += dm * mask_eps.samples[lambda] as f32;
            }
        }
        emulsion_i += 1;
    }
    d
}

pub(crate) fn transmittance_from_density(d: &[f32; 16]) -> [f32; 16] {
    let mut t = [0.0f32; 16];
    for i in 0..16 {
        // 10^-d via exp2, mirroring the `SCAN` shader.
        t[i] = (-d[i] * LOG2_10).exp2();
    }
    t
}

/// Unnormalized scanner spectrum for one pixel.
fn scan_pixel_spectrum(stock: &FilmStock, dyes: &DyePlanes, pixel: usize) -> [f32; 16] {
    let dens = density_spectrum(stock, dyes, pixel);
    let mut spectrum = transmittance_from_density(&dens);
    for (value, &light) in spectrum.iter_mut().zip(stock.scanner_light.samples.iter()) {
        *value *= light as f32;
    }
    spectrum
}

/// Compute scanner optical lens and detector pixel aperture Gaussian PSF standard deviation in pixels.
///
/// Models incoherent optical intensity integration over the scanner's lens OTF and
/// detector sampling floor (OLPF, pixel aperture fill factor, and optical reconstruction):
/// sigma_scanner_px = sqrt((sigma_opt_um / pitch)^2 + sigma_sensor_px^2)
pub fn scanner_aperture_sigma_px(pixel_pitch_um: f32) -> f32 {
    let opt_px = SCANNER_OPTICAL_SIGMA_UM / pixel_pitch_um.max(1e-6);
    (opt_px * opt_px + SCANNER_SENSOR_SIGMA_PX * SCANNER_SENSOR_SIGMA_PX).sqrt()
}

/// Convolve linear scanned RGB with the scanner optical and pixel aperture PSF.
///
/// Models incoherent optical intensity integration over the scanner's lens OTF and
/// detector pixel aperture. Since ACEScg is linear in transmitted radiant flux,
/// this spatial convolution is physically exact. Energy/mean flux is strictly conserved.
pub fn apply_scanner_aperture_mtf(
    raw: &mut [Pixel],
    width: usize,
    height: usize,
    sigma_px: f32,
) {
    if sigma_px < 1e-3 || width <= 1 || height <= 1 {
        return;
    }
    let n = width * height;
    assert_eq!(raw.len(), n);

    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for px in raw.iter() {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }

    let mut channels = [r, g, b];
    channels.par_iter_mut().for_each(|plane| {
        gaussian_blur_separable(plane, width, height, sigma_px);
    });

    for (p, px) in raw.iter_mut().enumerate() {
        px[0] = channels[0][p];
        px[1] = channels[1][p];
        px[2] = channels[2][p];
    }
}

/// Scan dye planes to interleaved ACEScg RGB with base-reference scanner gain
/// and scanner optical / pixel aperture MTF at the given pixel pitch.
pub fn scan_to_acescg(
    stock: &FilmStock,
    dyes: &DyePlanes,
    pixel_pitch_um: f32,
) -> ImageBuffer {
    let n = dyes.width * dyes.height;

    // First pass: compute raw ACEScg from T(λ) * I_s(λ).
    let mut raw: Vec<Pixel> = (0..n)
        .into_par_iter()
        .map(|p| {
            let t = scan_pixel_spectrum(stock, dyes, p);
            spectrum_to_acescg_rgb_f32(&t)
        })
        .collect();

    // Scanner gain reference: zero image dye, undeveloped mask at its maximum.
    let dmin_rgb = dmin_reference_acescg(stock);
    let peak = dmin_rgb[0].max(dmin_rgb[1]).max(dmin_rgb[2]).max(1e-12);
    let scale = 1.0 / peak;
    raw.par_iter_mut().for_each(|px| {
        *px = px.map(|c| c * scale);
    });

    // Scanner optical / pixel aperture MTF: incoherent optical intensity integration.
    let sigma_px = scanner_aperture_sigma_px(pixel_pitch_um);
    apply_scanner_aperture_mtf(
        &mut raw,
        dyes.width,
        dyes.height,
        sigma_px,
    );

    raw
}

/// Base-only ACEScg before scanner peak-normalization (raw densitometric units).
pub fn dmin_reference_acescg(stock: &FilmStock) -> [f32; 3] {
    // Synthesize a 1-pixel DyePlanes at Dmin: image=0, mask=max residual.
    let mut image_dye = Vec::new();
    let mut mask_dye = Vec::new();
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().unwrap();
        image_dye.push(vec![0.0f32]);
        let mask = if coupler.mask_epsilon.is_some() {
            use crate::film::constants::MASK_DENSITY_FRACTION_OF_DMAX;
            coupler.d_max * MASK_DENSITY_FRACTION_OF_DMAX
        } else {
            0.0
        };
        mask_dye.push(vec![mask]);
    }
    let dyes = DyePlanes {
        width: 1,
        height: 1,
        image_dye,
        mask_dye,
    };
    let t = scan_pixel_spectrum(stock, &dyes, 0);
    spectrum_to_acescg_rgb_f32(&t)
}

/// Scan-normalized base-only RGB (peak ≈ 1).
///
/// Retained for scanner normalization and the intentionally desynchronized GPU.
/// CPU PositiveLinear uses [`scanner_calibration_acescg`] instead.
pub fn normalized_dmin_acescg(stock: &FilmStock) -> [f32; 3] {
    let d = dmin_reference_acescg(stock);
    let peak = d[0].max(d[1]).max(d[2]).max(1e-12);
    [d[0] / peak, d[1] / peak, d[2] / peak]
}

/// Joint processed-Dmin, neutral-mid and measured-contrast calibration computed
/// analytically from the exact emulsion expectation E[D] (using `reduce` and
/// `scan_to_acescg` on 1x1 patches at mid-gray and the +2-stop anchor).
pub fn scanner_calibration_acescg(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
) -> Result<ScannerCalibration, FilmError> {
    let mid_value = crate::film::relative_to_absolute_y(crate::pixel::MIDDLE_GRAY, stock.box_iso.0);
    let hi_value = crate::film::relative_to_absolute_y(
        crate::pixel::MIDDLE_GRAY * CALIBRATION_ANCHOR_RATIO,
        stock.box_iso.0,
    );
    let dmin_rgb = vec![[0.0; 3]; 1];
    let mid_rgb = vec![[mid_value; 3]; 1];
    let hi_rgb = vec![[hi_value; 3]; 1];
    let dmin_latent = crate::film::exposure::expose_with_pitch_and_shutter(
        &dmin_rgb, 1, 1, stock, pixel_pitch_um, shutter_seconds,
    );
    let mid_latent = crate::film::exposure::expose_with_pitch_and_shutter(
        &mid_rgb, 1, 1, stock, pixel_pitch_um, shutter_seconds,
    );
    let hi_latent = crate::film::exposure::expose_with_pitch_and_shutter(
        &hi_rgb, 1, 1, stock, pixel_pitch_um, shutter_seconds,
    );
    let dmin_dyes = crate::film::development::reduction::reduce(stock, &dmin_latent);
    let mid_dyes = crate::film::development::reduction::reduce(stock, &mid_latent);
    let hi_dyes = crate::film::development::reduction::reduce(stock, &hi_latent);
    let dmin_scan = scan_to_acescg(stock, &dmin_dyes, pixel_pitch_um)[0];
    let mid_scan = scan_to_acescg(stock, &mid_dyes, pixel_pitch_um)[0];
    let hi_scan = scan_to_acescg(stock, &hi_dyes, pixel_pitch_um)[0];
    let gamma_eff = solve_gamma_eff_all(dmin_scan, mid_scan, hi_scan);
    let chroma_decode = chroma_decode_from_forward(
        stock,
        pixel_pitch_um,
        shutter_seconds,
        mid_scan,
        dmin_scan,
        gamma_eff,
        [mid_value; 3],
        [hi_value; 3],
        JACOBIAN_DELTA,
    );

    Ok(ScannerCalibration {
        dmin: dmin_scan,
        mid: mid_scan,
        gamma_eff,
        chroma_decode,
    })
}

/// One 1x1 patch through the complete PositiveLinear composite
/// (expose -> reduce -> scan -> invert) with the given solved invert constants.
fn composite_positive(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
    mid_scan: [f32; 3],
    dmin_scan: [f32; 3],
    gamma_eff: [f32; 3],
    scene_rgb: [f32; 3],
) -> [f32; 3] {
    let latent = crate::film::exposure::expose_with_pitch_and_shutter(
        &vec![scene_rgb],
        1,
        1,
        stock,
        pixel_pitch_um,
        shutter_seconds,
    );
    let dyes = crate::film::development::reduction::reduce(stock, &latent);
    let mut buf = scan_to_acescg(stock, &dyes, pixel_pitch_um);
    invert_negative(&mut buf, mid_scan, dmin_scan, gamma_eff.map(|g| 1.0 / g));
    buf[0]
}

/// Chroma decode matrix measured from the stock's own composite forward
/// response around mid-gray.
///
/// Central-difference Jacobian `C` of the expose -> reduce -> scan -> invert
/// map at the mid-gray anchor, least-squares unmix `g * C^-1` with the neutral
/// secant gain `g`, then projected so the measured achromatic direction is
/// preserved exactly. Degenerate measurements fall back to the identity
/// (no-op decode), matching [`solve_gamma_eff`]'s fallback philosophy.
fn chroma_decode_from_forward(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
    mid_scan: [f32; 3],
    dmin_scan: [f32; 3],
    gamma_eff: [f32; 3],
    x_mid: [f32; 3],
    x_hi: [f32; 3],
    delta: f32,
) -> [[f32; 3]; 3] {
    let composite = |x: [f32; 3]| {
        composite_positive(
            stock,
            pixel_pitch_um,
            shutter_seconds,
            mid_scan,
            dmin_scan,
            gamma_eff,
            x,
        )
    };
    let p_mid = composite(x_mid);
    let p_hi = composite(x_hi);
    let u = [
        p_hi[0] - p_mid[0],
        p_hi[1] - p_mid[1],
        p_hi[2] - p_mid[2],
    ];

    let mut jacobian = [[0.0f32; 3]; 3];
    for j in 0..3 {
        let mut xp = x_mid;
        let mut xm = x_mid;
        xp[j] *= 1.0 + delta;
        xm[j] *= 1.0 - delta;
        let fp = composite(xp);
        let fm = composite(xm);
        for i in 0..3 {
            jacobian[i][j] = (fp[i] - fm[i]) / (xp[j] - xm[j]);
        }
    }

    // Neutral contrast per unit scene radiance, from the same anchors that fix
    // gamma_eff, so decoded chroma slopes match the existing neutral slope.
    let dx = [
        x_hi[0] - x_mid[0],
        x_hi[1] - x_mid[1],
        x_hi[2] - x_mid[2],
    ];
    let scene_norm = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
    let u_norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let gain = if scene_norm > 0.0 {
        u_norm / scene_norm
    } else {
        0.0
    };
    build_chroma_decode(jacobian, u, gain)
}

/// Determinant of a row-major 3x3 matrix.
fn det3(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Analytic adjugate inverse of a row-major 3x3 matrix; caller guards singularity.
fn inv3(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let d = det3(m);
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d,
        ],
    ]
}

fn mat_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Least-squares chroma unmix from a measured composite-forward Jacobian.
///
/// Returns `gain * C^{-1}` projected onto the constraint `M * u == u`: the
/// achromatic direction `u` (the measured mid-gray -> +2-stop offset in decoded
/// space) passes through unchanged, while displacements orthogonal to `u`
/// unmix with the measured inverse response scaled by the neutral gain.
/// Non-finite inputs or a numerically singular `C` (including effectively
/// rank-deficient monochrome responses) fall back to the identity.
fn build_chroma_decode(c: [[f32; 3]; 3], u: [f32; 3], gain: f32) -> [[f32; 3]; 3] {
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    let max_abs = c.iter().flatten().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let det = det3(&c);
    if !c.iter().flatten().all(|v| v.is_finite())
        || !u.iter().all(|v| v.is_finite())
        || !gain.is_finite()
        || !(uu > 0.0)
        || max_abs <= 0.0
        || !det.is_finite()
        || det.abs() < max_abs * max_abs * max_abs * 1e-6
    {
        return IDENTITY;
    }

    let inv_c = inv3(&c);
    let m_ls = inv_c.map(|row| row.map(|v| gain * v));

    // Anchor projection M = M_ls + outer(u - M_ls*u, u) / dot(u, u):
    // M*u == u exactly, rows acting orthogonal to u stay equal to M_ls.
    let m_ls_u = mat_vec3(&m_ls, u);
    let residual = [u[0] - m_ls_u[0], u[1] - m_ls_u[1], u[2] - m_ls_u[2]];
    let mut m = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = m_ls[i][j] + residual[i] * u[j] / uu;
        }
    }
    if m.iter().flatten().all(|v| v.is_finite()) {
        m
    } else {
        IDENTITY
    }
}

/// Solve each channel's effective contrast exponent from the measured anchor
/// transmissions (normalized to processed Dmin).
fn solve_gamma_eff_all(
    dmin_scan: [f32; 3],
    mid_scan: [f32; 3],
    hi_scan: [f32; 3],
) -> [f32; 3] {
    let mut gamma_eff = [GAMMA_EFF; 3];
    for c in 0..3 {
        let t_mid = mid_scan[c] / dmin_scan[c];
        let t_hi = hi_scan[c] / dmin_scan[c];
        gamma_eff[c] = solve_gamma_eff(t_mid as f64, t_hi as f64) as f32;
    }
    gamma_eff
}

/// Bisect γ > 0 such that the invert law $t \mapsto (t^{-1/\gamma} - 1)^+$
/// maps the mid-gray → +2-stop transmission pair to a [`CALIBRATION_ANCHOR_RATIO`]
/// exposure ratio: $(t_{\text{hi}}^{-1/\gamma} - 1)/(t_{\text{mid}}^{-1/\gamma} - 1) = R$.
///
/// The ratio decreases monotonically from ∞ toward $\ln(t_{\text{hi}})/\ln(t_{\text{mid}})$
/// as γ grows, so a crossing exists only if that asymptote is below `R`. Degenerate
/// anchors (non-finite, $t_{\text{hi}} \ge t_{\text{mid}}$, or no crossing) fall back
/// to [`GAMMA_EFF`]. Evaluated in f64 with a log-stable form of the ratio.
fn solve_gamma_eff(t_mid_in: f64, t_hi_in: f64) -> f64 {
    const GAMMA_LO: f64 = 0.01;
    const GAMMA_HI: f64 = 1000.0;
    const ITERATIONS: usize = 200;
    const T_EPS: f64 = 1e-6;
    let target_ln = (CALIBRATION_ANCHOR_RATIO as f64).ln();

    let clamp_t = |t: f64| t.clamp(T_EPS, 1.0);
    let t_mid = clamp_t(t_mid_in);
    let t_hi = clamp_t(t_hi_in);
    if !t_mid.is_finite() || !t_hi.is_finite() || t_hi >= t_mid {
        return GAMMA_EFF as f64;
    }

    // ln(e^{a/γ} − 1) = a/γ + ln(1 − e^{−a/γ}) stays finite for all γ > 0.
    // h(γ) < 0 ⟺ anchor ratio < CALIBRATION_ANCHOR_RATIO.
    let ln_expm1 = |z: f64| z + (-((-z).exp())).ln_1p();
    let h = |gamma: f64| {
        ln_expm1(-(t_hi.ln()) / gamma) - ln_expm1(-(t_mid.ln()) / gamma) - target_ln
    };
    if !(h(GAMMA_LO) > 0.0 && h(GAMMA_HI) < 0.0) {
        return GAMMA_EFF as f64;
    }

    let mut lo = GAMMA_LO;
    let mut hi = GAMMA_HI;
    for _ in 0..ITERATIONS {
        let mid = 0.5 * (lo + hi);
        if !mid.is_finite() || mid <= lo || mid >= hi {
            break;
        }
        if h(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let solved = 0.5 * (lo + hi);
    if solved.is_finite() && solved > 0.0 {
        solved
    } else {
        GAMMA_EFF as f64
    }
}

/// Processed unexposed-film scan retained as a convenience for CPU probes.
pub fn processed_dmin_acescg(
    stock: &FilmStock,
    pixel_pitch_um: f32,
    shutter_seconds: f32,
) -> Result<[f32; 3], FilmError> {
    Ok(scanner_calibration_acescg(stock, pixel_pitch_um, shutter_seconds)?.dmin)
}

#[cfg(test)]
mod calibration_tests {
    use super::*;
    use crate::film::scan::invert::invert_negative;
    use crate::film::stock::StockId;

    #[test]
    fn analytical_scanner_calibration_is_finite_and_ordered() {
        let stock = StockId::Portra400.load().unwrap();
        let pitch_um = 500.0 / 4032.0;
        let shutter = 1.0 / 121.0;
        let cal = scanner_calibration_acescg(&stock, pitch_um, shutter).unwrap();
        assert!(cal.dmin.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(cal.mid.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(cal.dmin[0] > cal.mid[0]);
        assert!(cal.dmin[1] > cal.mid[1]);
        assert!(cal.dmin[2] > cal.mid[2]);

        let mut positive = vec![cal.mid];
        invert_negative(&mut positive, cal.mid, cal.dmin, cal.inv_gamma());
        for value in positive[0] {
            assert!(value.is_finite());
            assert!((value - crate::pixel::MIDDLE_GRAY).abs() < 1e-3);
        }
    }

    #[test]
    fn solved_gamma_eff_maps_mid_and_plus_two_stop_anchor() {
        for id in [StockId::ColorNeg200, StockId::Portra400] {
            let stock = id.load().unwrap();
            let pitch_um = 500.0 / 4032.0;
            let shutter = 1.0 / 121.0;
            let cal = scanner_calibration_acescg(&stock, pitch_um, shutter).unwrap();
            assert!(
                cal.gamma_eff.iter().all(|g| g.is_finite() && *g > 0.0),
                "{id:?}: gamma_eff must be finite and positive"
            );

            // Mid-gray scan still inverts exactly to MIDDLE_GRAY.
            let mut positive = vec![cal.mid];
            invert_negative(&mut positive, cal.mid, cal.dmin, cal.inv_gamma());
            for value in positive[0] {
                assert!(value.is_finite());
                assert!((value - crate::pixel::MIDDLE_GRAY).abs() < 1e-3);
            }

            // A +2-stop flat field through the same forward path lands at
            // CALIBRATION_ANCHOR_RATIO × MIDDLE_GRAY within 5%.
            let hi_value = crate::film::relative_to_absolute_y(
                crate::pixel::MIDDLE_GRAY * CALIBRATION_ANCHOR_RATIO,
                stock.box_iso.0,
            );
            let hi_rgb = vec![[hi_value; 3]; 1];
            let hi_latent = crate::film::exposure::expose_with_pitch_and_shutter(
                &hi_rgb, 1, 1, &stock, pitch_um, shutter,
            );
            let hi_dyes = crate::film::development::reduction::reduce(&stock, &hi_latent);
            let hi_scan = scan_to_acescg(&stock, &hi_dyes, pitch_um)[0];
            let mut positive_hi = vec![hi_scan];
            invert_negative(&mut positive_hi, cal.mid, cal.dmin, cal.inv_gamma());
            let target = crate::pixel::MIDDLE_GRAY * CALIBRATION_ANCHOR_RATIO;
            for c in 0..3 {
                let rel_err =
                    (positive_hi[0][c] - target).abs() / target;
                assert!(
                    rel_err < 0.05,
                    "{id:?}: +2-stop anchor channel {c} should map to ~{target}, got {} (rel err {rel_err})",
                    positive_hi[0][c]
                );
            }
        }
    }

    #[test]
    fn scanner_aperture_sampling_floor_matches_expected_limits() {
        // At coarse pitch / 35mm film (~11.9 um pitch), the optical blur in pixels is small
        // and the detector sampling floor (~0.65 px) dominates.
        let sigma_35mm = scanner_aperture_sigma_px(11.9);
        assert!(
            (sigma_35mm - 0.671).abs() < 0.01,
            "35mm scanner sigma {sigma_35mm} should be around 0.67 px"
        );

        // At infinite pitch, sigma approaches the detector sampling floor SCANNER_SENSOR_SIGMA_PX.
        let sigma_inf = scanner_aperture_sigma_px(1e6);
        assert!(
            (sigma_inf - SCANNER_SENSOR_SIGMA_PX).abs() < 1e-4,
            "infinite pitch sigma {sigma_inf} must approach SCANNER_SENSOR_SIGMA_PX {SCANNER_SENSOR_SIGMA_PX}"
        );

        // At 1.0 um microscope pitch, lens optical OTF (~2.0 um) dominates.
        let sigma_1um = scanner_aperture_sigma_px(1.0);
        assert!(
            (sigma_1um - 2.103).abs() < 0.01,
            "1um pitch scanner sigma {sigma_1um} should be around 2.10 px"
        );
    }
}

#[cfg(test)]
mod chroma_decode_tests {
    use super::*;
    use crate::color::ColorSpaceTag;
    use crate::film::scan::{ScanMode, scan};
    use crate::film::stock::StockId;
    use crate::pixel::{PixelOps, R_RELATIVE_LUMINANCE, G_RELATIVE_LUMINANCE, B_RELATIVE_LUMINANCE};

    const PITCH_UM: f32 = 500.0 / 4032.0;
    const SHUTTER_S: f32 = 1.0 / 121.0;

    fn anchor_values(stock: &FilmStock) -> (f32, f32) {
        let mid_value =
            crate::film::relative_to_absolute_y(crate::pixel::MIDDLE_GRAY, stock.box_iso.0);
        let hi_value = crate::film::relative_to_absolute_y(
            crate::pixel::MIDDLE_GRAY * CALIBRATION_ANCHOR_RATIO,
            stock.box_iso.0,
        );
        (mid_value, hi_value)
    }

    fn positive_mode(cal: &ScannerCalibration) -> ScanMode {
        ScanMode::PositiveLinear {
            dmin: cal.dmin,
            mid: cal.mid,
            inv_gamma: cal.inv_gamma(),
            chroma_decode: cal.chroma_decode,
        }
    }

    /// Flat absolute-radiance field through expose -> reduce -> scan(PositiveLinear).
    fn scan_flat_positive(
        stock: &FilmStock,
        cal: &ScannerCalibration,
        abs_rgb: [f32; 3],
    ) -> [f32; 3] {
        let rgb = vec![abs_rgb; 64];
        let latent = crate::film::exposure::expose_with_pitch_and_shutter(
            &rgb, 8, 8, stock, PITCH_UM, SHUTTER_S,
        );
        let dyes = crate::film::development::reduction::reduce(stock, &latent);
        scan(stock, &dyes, positive_mode(cal), PITCH_UM)[0]
    }

    fn oklab_chroma(px: [f32; 3]) -> f32 {
        let lab = ColorSpaceTag::AcesCg.convert(ColorSpaceTag::Oklab, px);
        (lab[1] * lab[1] + lab[2] * lab[2]).sqrt()
    }

    #[test]
    fn chroma_decode_is_finite_nonsingular_and_preserves_anchor_direction() {
        for id in [StockId::Portra400, StockId::ColorNeg200] {
            let stock = id.load().unwrap();
            let cal = scanner_calibration_acescg(&stock, PITCH_UM, SHUTTER_S).unwrap();
            assert!(cal.chroma_decode.iter().flatten().all(|v| v.is_finite()));
            let det = det3(&cal.chroma_decode);
            assert!(
                det.abs() > 1e-6,
                "{id:?}: singular chroma_decode (det={det})"
            );

            let (mid_value, hi_value) = anchor_values(&stock);
            let p_mid = composite_positive(
                &stock, PITCH_UM, SHUTTER_S, cal.mid, cal.dmin, cal.gamma_eff,
                [mid_value; 3],
            );
            let p_hi = composite_positive(
                &stock, PITCH_UM, SHUTTER_S, cal.mid, cal.dmin, cal.gamma_eff,
                [hi_value; 3],
            );
            for value in p_mid {
                assert!((value - crate::pixel::MIDDLE_GRAY).abs() < 1e-3);
            }
            let u = [
                p_hi[0] - p_mid[0],
                p_hi[1] - p_mid[1],
                p_hi[2] - p_mid[2],
            ];
            let u_norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            assert!(u_norm > 0.0, "{id:?}: degenerate achromatic direction");
            let mu = mat_vec3(&cal.chroma_decode, u);
            let rel_err = ((mu[0] - u[0]).powi(2)
                + (mu[1] - u[1]).powi(2)
                + (mu[2] - u[2]).powi(2))
                .sqrt()
                / u_norm;
            assert!(
                rel_err < 1e-4,
                "{id:?}: M*u must equal u, rel err {rel_err}"
            );
        }
    }

    #[test]
    fn positive_linear_scan_maps_mid_and_plus_two_stop_anchor_with_decode() {
        for id in [StockId::ColorNeg200, StockId::Portra400] {
            let stock = id.load().unwrap();
            let cal = scanner_calibration_acescg(&stock, PITCH_UM, SHUTTER_S).unwrap();
            let (mid_value, hi_value) = anchor_values(&stock);

            // Mid-gray flat field still lands on MIDDLE_GRAY through the decode.
            let out_mid = scan_flat_positive(&stock, &cal, [mid_value; 3]);
            for c in 0..3 {
                assert!(
                    (out_mid[c] - crate::pixel::MIDDLE_GRAY).abs() < 2e-3,
                    "{id:?}: mid channel {c} -> {}",
                    out_mid[c]
                );
            }

            // A +2-stop flat field through the full PositiveLinear path stays
            // within 5% of CALIBRATION_ANCHOR_RATIO x MIDDLE_GRAY per channel.
            let out_hi = scan_flat_positive(&stock, &cal, [hi_value; 3]);
            let target = crate::pixel::MIDDLE_GRAY * CALIBRATION_ANCHOR_RATIO;
            for c in 0..3 {
                let rel_err = (out_hi[c] - target).abs() / target;
                assert!(
                    rel_err < 0.05,
                    "{id:?}: +2-stop channel {c} should map to ~{target}, got {} (rel err {rel_err})",
                    out_hi[c]
                );
            }
        }
    }

    #[test]
    fn saturated_blue_retains_chroma_through_positive_linear() {
        let stock = StockId::Portra400.load().unwrap();
        let cal = scanner_calibration_acescg(&stock, PITCH_UM, SHUTTER_S).unwrap();

        // Saturated sRGB blue in ACEScg, luminance-normalized to scene-relative
        // mid-gray, then converted to absolute radiance like the anchors.
        let srgb_blue = ColorSpaceTag::Srgb.convert(ColorSpaceTag::AcesCg, [0.0, 0.0, 1.0]);
        let y = R_RELATIVE_LUMINANCE * srgb_blue[0]
            + G_RELATIVE_LUMINANCE * srgb_blue[1]
            + B_RELATIVE_LUMINANCE * srgb_blue[2];
        let rel = srgb_blue.map(|c| c * crate::pixel::MIDDLE_GRAY / y);
        let gain = crate::film::relative_to_absolute_y(1.0, stock.box_iso.0);

        let chroma_in = oklab_chroma(rel);
        assert!(chroma_in > 0.0);

        // Output lives on the same scene-relative anchor scale as `rel`
        // (mid-gray -> MIDDLE_GRAY), so chroma compares directly.
        let out = scan_flat_positive(
            &stock,
            &cal,
            [rel[0] * gain, rel[1] * gain, rel[2] * gain],
        );
        let chroma_out = oklab_chroma(out);

        println!(
            "sRGB blue Oklab chroma: input {chroma_in:.4}, output {chroma_out:.4}, ratio {:.3}",
            chroma_out / chroma_in
        );
        assert!(
            chroma_out > chroma_in * 0.35,
            "blue chroma collapsed through scan+decode: in {chroma_in:.4}, out {chroma_out:.4}"
        );
    }

    #[test]
    fn neutral_ramp_stays_monotonic_through_chroma_decode() {
        let stock = StockId::Portra400.load().unwrap();
        let cal = scanner_calibration_acescg(&stock, PITCH_UM, SHUTTER_S).unwrap();
        let (mid_value, hi_value) = anchor_values(&stock);
        let x_lo = mid_value * 0.25;
        let x_top = hi_value * 1.5;

        let mut ys = Vec::with_capacity(16);
        for k in 0..16 {
            let t = k as f32 / 15.0;
            let x = x_lo + (x_top - x_lo) * t;
            ys.push(scan_flat_positive(&stock, &cal, [x; 3]).luminance());
        }
        for (k, &y) in ys.iter().enumerate() {
            assert!(y.is_finite() && y >= 0.0, "ramp step {k}: {y}");
        }
        for i in 1..ys.len() {
            assert!(
                ys[i] > ys[i - 1],
                "neutral ramp not strictly increasing at step {i}: {ys:?}"
            );
        }
    }

    #[test]
    fn build_chroma_decode_singular_fallback_and_synthetic_unmix() {
        const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // Rank-deficient or zero Jacobian -> identity fallback.
        let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        assert_eq!(build_chroma_decode(singular, [1.0, 1.0, 1.0], 1.0), IDENTITY);
        assert_eq!(
            build_chroma_decode([[0.0; 3]; 3], [1.0, 1.0, 1.0], 1.0),
            IDENTITY
        );
        // Degenerate achromatic direction -> identity fallback.
        assert_eq!(
            build_chroma_decode(
                [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]],
                [0.0; 3],
                1.0
            ),
            IDENTITY
        );

        // Synthetic diagonal response: orthogonal-to-anchor displacements unmix
        // with g/C_ii, the anchor direction passes through unchanged.
        let c = [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]];
        let u = [1.0, 1.0, 1.0];
        let m = build_chroma_decode(c, u, 1.0);
        let mu = mat_vec3(&m, u);
        for ch in 0..3 {
            assert!(
                (mu[ch] - u[ch]).abs() < 1e-5,
                "anchor property broken on channel {ch}: {}",
                mu[ch]
            );
        }
        let mv = mat_vec3(&m, [1.0, -1.0, 0.0]);
        let expected = [0.5, -0.25, 0.0];
        for ch in 0..3 {
            assert!(
                (mv[ch] - expected[ch]).abs() < 1e-5,
                "unmix part wrong on channel {ch}: {} != {}",
                mv[ch],
                expected[ch]
            );
        }
    }

    #[test]
    #[ignore = "diagnostic: prints decode matrices across finite-difference steps"]
    fn chroma_decode_stability_across_fd_steps() {
        let stock = StockId::Portra400.load().unwrap();
        let cal = scanner_calibration_acescg(&stock, PITCH_UM, SHUTTER_S).unwrap();
        let (mid_value, hi_value) = anchor_values(&stock);
        let tight = chroma_decode_from_forward(
            &stock,
            PITCH_UM,
            SHUTTER_S,
            cal.mid,
            cal.dmin,
            cal.gamma_eff,
            [mid_value; 3],
            [hi_value; 3],
            0.05,
        );
        let max_rel = cal
            .chroma_decode
            .iter()
            .flatten()
            .zip(tight.iter().flatten())
            .fold(0.0f32, |acc, (&a, &b)| {
                acc.max((a - b).abs() / a.abs().max(1e-6))
            });
        println!("chroma_decode(delta=0.10) = {:?}", cal.chroma_decode);
        println!("chroma_decode(delta=0.05) = {:?}", tight);
        println!("max relative elementwise difference = {max_rel:.4}");
        assert!(
            max_rel < 0.30,
            "chroma decode unstable across fd steps: max rel diff {max_rel}"
        );
    }
}

