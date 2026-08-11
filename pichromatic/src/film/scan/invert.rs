//! Analytical scanner invert: negative → positive (linear ACEScg).
//!
//! Linear scanner pass without stylistic looks or artificial S-curves:
//! 1. Substrate Color Correction: D_img = −log10(clamp(T_neg / Dmin, ε, ∞))
//! 2. Film Dynamic Range (Gamma) Reconstruction: E_scene = 10^(D_img / γ_eff) − 1.0
//! 3. Mid-Gray Anchor: Scale factor g_c = MIDDLE_GRAY / E_scene(mid_c) per channel.

use crate::film::constants::FOG_OFFSET;
use crate::pixel::{ImageBuffer, MIDDLE_GRAY};
use rayon::prelude::*;

/// log2(10), shared with the GPU `SCAN` shader.
const LOG2_10: f32 = 3.3219280948873623;
/// log10(2), shared with the GPU `SCAN` shader.
const LOG10_2: f32 = 0.3010299956639812;

/// Target effective contrast gamma of developed color negative film (~0.6).
pub const GAMMA_EFF: f32 = 0.6;

/// Shared CPU/GPU calibration for the diagnostic negative invert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvertConstants {
    pub dmin: [f32; 3],
    /// Precomputed `1/dmin`, baked so CPU and GPU divide by the same f32 value.
    pub inv_dmin: [f32; 3],
    pub gain: [f32; 3],
    pub slope: f32,
    pub gamma_eff: f32,
    /// Precomputed `1/gamma_eff`.
    pub inv_gamma: f32,
    /// `1/gamma_eff · log2(10)` precombined so both sides use one rounded constant.
    pub inv_gamma_log2_10: f32,
    pub eps: f32,
    pub fog_offset: f32,
}

/// Effective density after fog toe, fed into the gamma `exp2` transfer.
///
/// C¹ quadratic toe (same shape as a filmistic shoulder/toe join):
/// - `d_img ≤ 0` (Dmin) → `0`
/// - `0 < d_img < fog` → `d_img² / (2·fog)` (preserves sub-fog dye-cloud structure)
/// - `d_img ≥ fog` → `d_img − fog/2`
///
/// At `d_img = fog`, both pieces equal `fog/2` with matching slope 1 — no cliff.
#[inline]
fn fog_effective_density(d_img: f32, fog_offset: f32) -> f32 {
    if d_img <= 0.0 {
        0.0
    } else if d_img < fog_offset {
        (d_img * d_img) / (2.0 * fog_offset)
    } else {
        d_img - 0.5 * fog_offset
    }
}

/// Map image density `d_img = −log10(T/Dmin)` to linear scene exposure.
#[inline]
pub fn density_img_to_exposure(
    d_img: f32,
    inv_gamma_log2_10: f32,
    _slope: f32,
    fog_offset: f32,
) -> f32 {
    let d_eff = fog_effective_density(d_img, fog_offset);
    if d_eff <= 0.0 {
        0.0
    } else {
        (d_eff * inv_gamma_log2_10).exp2() - 1.0
    }
}

pub fn invert_constants(mid_negative: [f32; 3], dmin_negative: [f32; 3]) -> InvertConstants {
    let eps = 1e-6f32;
    let dmin = dmin_negative.map(|v| v.max(eps));
    let inv_dmin = dmin.map(|v| 1.0 / v);
    let mid_t = [
        (mid_negative[0] * inv_dmin[0]).clamp(eps, 1.0),
        (mid_negative[1] * inv_dmin[1]).clamp(eps, 1.0),
        (mid_negative[2] * inv_dmin[2]).clamp(eps, 1.0),
    ];
    let inv_gamma = 1.0 / GAMMA_EFF;
    let slope = 10.0f32.ln() / GAMMA_EFF;
    let inv_gamma_log2_10 = inv_gamma * LOG2_10;
    let d_mid = mid_t.map(|v| (-v.log10()).max(eps));
    let e_mid = d_mid.map(|d_img| {
        density_img_to_exposure(d_img, inv_gamma_log2_10, slope, FOG_OFFSET).max(0.005)
    });
    InvertConstants {
        dmin,
        inv_dmin,
        gain: e_mid.map(|v| (MIDDLE_GRAY / v).min(25.0)),
        slope: 10.0f32.ln() / GAMMA_EFF,
        gamma_eff: GAMMA_EFF,
        inv_gamma,
        inv_gamma_log2_10: inv_gamma * LOG2_10,
        eps,
        fog_offset: FOG_OFFSET,
    }
}

/// Linear scanner invert for PositiveLinear.
///
/// Mirrors the GPU `SCAN` shader invert block (log10 via `log2·LOG10_2`,
/// `10^(d/γ)` via `exp2(d·inv_gamma_log2_10)`).
pub fn invert_negative(buffer: &mut ImageBuffer, mid_negative: [f32; 3], dmin_negative: [f32; 3]) {
    let constants = invert_constants(mid_negative, dmin_negative);

    buffer.par_iter_mut().for_each(|px| {
        let t = [
            (px[0] * constants.inv_dmin[0]).max(constants.eps),
            (px[1] * constants.inv_dmin[1]).max(constants.eps),
            (px[2] * constants.inv_dmin[2]).max(constants.eps),
        ];

        let d_img = [
            -(t[0].log2() * LOG10_2),
            -(t[1].log2() * LOG10_2),
            -(t[2].log2() * LOG10_2),
        ];

        let e_scene = [
            density_img_to_exposure(
                d_img[0],
                constants.inv_gamma_log2_10,
                constants.slope,
                constants.fog_offset,
            ),
            density_img_to_exposure(
                d_img[1],
                constants.inv_gamma_log2_10,
                constants.slope,
                constants.fog_offset,
            ),
            density_img_to_exposure(
                d_img[2],
                constants.inv_gamma_log2_10,
                constants.slope,
                constants.fog_offset,
            ),
        ];

        *px = [
            constants.gain[0] * e_scene[0],
            constants.gain[1] * e_scene[1],
            constants.gain[2] * e_scene[2],
        ];
    });
}

/// Mean RGB of a buffer (flat-field mid / Dmin probes).
pub fn mean_rgb(buffer: &ImageBuffer) -> [f32; 3] {
    let n = buffer.len().max(1) as f64;
    let mut acc = [0.0f64; 3];
    for px in buffer {
        acc[0] += px[0] as f64;
        acc[1] += px[1] as f64;
        acc[2] += px[2] as f64;
    }
    [
        (acc[0] / n) as f32,
        (acc[1] / n) as f32,
        (acc[2] / n) as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelOps;

    #[test]
    fn dmin_maps_to_black_mid_to_middle_gray() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let mut buf = vec![dmin, mid];
        invert_negative(&mut buf, mid, dmin);
        for c in 0..3 {
            assert!(
                buf[0][c] < 1e-5,
                "Dmin channel {c} should be ~0, got {}",
                buf[0][c]
            );
            assert!(
                (buf[1][c] - MIDDLE_GRAY).abs() < 1e-3,
                "mid channel {c} should be MIDDLE_GRAY, got {}",
                buf[1][c]
            );
        }
    }

    #[test]
    fn pathological_blue_mid_gain_stays_bounded() {
        // Mimic ColorNeg200: mid blue sits almost on Dmin → linear g_b ≫ g_r.
        let dmin = [1.0f32, 0.45, 0.127];
        let mid = [0.65f32, 0.315, 0.122];
        let bright = [0.16f32, 0.087, 0.058]; // dense highlight negative
        let mut buf = vec![bright];
        invert_negative(&mut buf, mid, dmin);
        let p = buf[0];
        let y = p.luminance();
        assert!(
            y.is_finite() && y > 0.0,
            "highlight should be finite+, got {p:?}"
        );
        assert!(
            p[0] < 50.0 && p[1] < 50.0 && p[2] < 50.0,
            "channels must be bounded by max gain threshold 25.0, got {p:?}"
        );
    }

    #[test]
    fn orange_mask_shadow_stays_near_neutral() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let shadow_neg = [0.95f32, 0.42, 0.125];
        let mut buf = vec![shadow_neg];
        invert_negative(&mut buf, mid, dmin);
        let p = buf[0];
        let mean = (p[0] + p[1] + p[2]) / 3.0;
        assert!(
            mean > 0.0 && mean < 0.1,
            "shadow positive should be dark & positive, got {mean}"
        );
        let max_chan_diff = (p[0] - p[1])
            .abs()
            .max((p[1] - p[2]).abs())
            .max((p[0] - p[2]).abs());
        assert!(
            max_chan_diff < 0.05,
            "shadow should stay near neutral, max diff {max_chan_diff}"
        );
    }

    #[test]
    fn saturated_dye_keeps_chroma_without_neon() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.63f32, 0.30, 0.12];
        let red_scene_neg = [0.15f32, 0.35, 0.11];
        let mut buf = vec![red_scene_neg];
        invert_negative(&mut buf, mid, dmin);
        let p = buf[0];
        let max = p[0].max(p[1]).max(p[2]);
        let min = p[0].min(p[1]).min(p[2]).max(1e-6);
        let ratio = max / min;
        assert!(
            ratio > 2.0,
            "channel ratio too low (B&W): {p:?} ratio={ratio}"
        );
        assert!(p[0] > p[1] && p[0] > p[2], "expected reddish {p:?}");
    }

    #[test]
    fn near_dmin_fluctuation_preserves_structure() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let n = 256;
        let mut buf = Vec::with_capacity(n);
        for i in 0..n {
            let wave = ((i % 17) as f32 + 0.5) / 17.0 * 0.008;
            buf.push([
                dmin[0] - wave,
                dmin[1] - wave * 0.6,
                dmin[2] - wave * 0.3,
            ]);
        }
        invert_negative(&mut buf, mid, dmin);
        let zero_frac = buf
            .iter()
            .filter(|px| px[0] == 0.0 && px[1] == 0.0 && px[2] == 0.0)
            .count() as f32
            / n as f32;
        assert!(
            zero_frac < 0.05,
            "near-Dmin toe should not collapse all pixels to zero, zero_frac={zero_frac}"
        );
        let std_r = {
            let mean = buf.iter().map(|px| px[0] as f64).sum::<f64>() / n as f64;
            (buf
                .iter()
                .map(|px| {
                    let e = px[0] as f64 - mean;
                    e * e
                })
                .sum::<f64>()
                / n as f64)
                .sqrt() as f32
        };
        assert!(
            std_r > 1e-7,
            "near-Dmin fluctuations should survive invert, std_r={std_r}"
        );
        let mean_y = mean_rgb(&buf).luminance();
        assert!(
            mean_y < 0.01,
            "near-Dmin patch should stay dark, mean Y={mean_y}"
        );
    }

    #[test]
    fn fog_toe_maps_dmin_to_zero_and_preserves_subfog_slope() {
        let inv_gamma_log2_10 = (1.0 / GAMMA_EFF) * LOG2_10;
        let slope = 10.0f32.ln() / GAMMA_EFF;
        assert!(
            density_img_to_exposure(0.0, inv_gamma_log2_10, slope, FOG_OFFSET).abs() < 1e-7,
            "Dmin density maps to zero exposure"
        );
        let low = density_img_to_exposure(0.001, inv_gamma_log2_10, slope, FOG_OFFSET);
        let high = density_img_to_exposure(0.002, inv_gamma_log2_10, slope, FOG_OFFSET);
        assert!(low > 0.0 && high > low, "sub-fog densities rise monotonically");
        let below = fog_effective_density(FOG_OFFSET - 1e-6, FOG_OFFSET);
        let at = fog_effective_density(FOG_OFFSET, FOG_OFFSET);
        let above_d = fog_effective_density(FOG_OFFSET + 1e-6, FOG_OFFSET);
        assert!(
            (at - 0.5 * FOG_OFFSET).abs() < 1e-6,
            "knee density must be fog/2, got {at}"
        );
        assert!(
            (below - at).abs() < 1e-5 && (above_d - at).abs() < 1e-5,
            "C1 toe must be continuous at fog knee"
        );
        let knee = density_img_to_exposure(FOG_OFFSET, inv_gamma_log2_10, slope, FOG_OFFSET);
        let above = density_img_to_exposure(FOG_OFFSET + 0.001, inv_gamma_log2_10, slope, FOG_OFFSET);
        assert!(above >= knee, "exposure continues above fog knee");
    }

    #[test]
    fn noisy_shadow_has_no_spurious_color_pedestal() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let n = 10000;
        let mut buf = Vec::with_capacity(n);
        for i in 0..n {
            let r_noise = ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0 - 0.5;
            let g_noise = ((i * 214013 + 2531011) % 1000) as f32 / 1000.0 - 0.5;
            let b_noise = ((i * 1664525 + 1013904223) % 1000) as f32 / 1000.0 - 0.5;
            buf.push([
                dmin[0] + r_noise * 0.02,
                dmin[1] + g_noise * 0.01,
                dmin[2] + b_noise * 0.005,
            ]);
        }
        invert_negative(&mut buf, mid, dmin);
        let mean = mean_rgb(&buf);
        let y = mean.luminance();
        assert!(y < 0.01, "noisy Dmin mean Y should stay dark, got {y}");
    }
}
