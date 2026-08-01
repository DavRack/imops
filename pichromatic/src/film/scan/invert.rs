//! Analytical scanner invert: negative → positive (linear ACEScg).
//!
//! Linear scanner pass without stylistic looks or artificial S-curves:
//! 1. Substrate Color Correction: D_img = −log10(clamp(T_neg / Dmin, ε, ∞))
//! 2. Film Dynamic Range (Gamma) Reconstruction: E_scene = 10^(D_img / γ_eff) − 1.0
//! 3. Mid-Gray Anchor: Scale factor g_c = MIDDLE_GRAY / E_scene(mid_c) per channel.

use crate::pixel::{ImageBuffer, MIDDLE_GRAY};
use rayon::prelude::*;

/// log2(10), shared with the GPU `SCAN` shader.
const LOG2_10: f32 = 3.3219280948873623;
/// log10(2), shared with the GPU `SCAN` shader.
const LOG10_2: f32 = 0.3010299956639812;

/// Target effective contrast gamma of developed color negative film (~0.6).
pub const GAMMA_EFF: f32 = 0.6;

/// Substrate fog density offset threshold above reference Dmin (~0.005).
pub const FOG_OFFSET: f32 = 0.005;

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

pub fn invert_constants(mid_negative: [f32; 3], dmin_negative: [f32; 3]) -> InvertConstants {
    let eps = 1e-6f32;
    let dmin = dmin_negative.map(|v| v.max(eps));
    let inv_dmin = dmin.map(|v| 1.0 / v);
    let mid_t = [
        (mid_negative[0] * inv_dmin[0]).clamp(eps, 1.0),
        (mid_negative[1] * inv_dmin[1]).clamp(eps, 1.0),
        (mid_negative[2] * inv_dmin[2]).clamp(eps, 1.0),
    ];
    let d_mid = mid_t.map(|v| (-v.log10() - FOG_OFFSET).max(eps));
    let e_mid = d_mid.map(|v| (10.0f32.powf(v / GAMMA_EFF) - 1.0).max(0.005));
    let inv_gamma = 1.0 / GAMMA_EFF;
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

        let d_clamped = [
            (d_img[0] - constants.fog_offset).max(0.0),
            (d_img[1] - constants.fog_offset).max(0.0),
            (d_img[2] - constants.fog_offset).max(0.0),
        ];

        // C1 continuous exposure transfer function matching slope (ln 10)/gamma at D = 0
        let e_scene = [
            if d_clamped[0] > 0.0 {
                (d_clamped[0] * constants.inv_gamma_log2_10).exp2() - 1.0
            } else {
                constants.slope * d_clamped[0]
            },
            if d_clamped[1] > 0.0 {
                (d_clamped[1] * constants.inv_gamma_log2_10).exp2() - 1.0
            } else {
                constants.slope * d_clamped[1]
            },
            if d_clamped[2] > 0.0 {
                (d_clamped[2] * constants.inv_gamma_log2_10).exp2() - 1.0
            } else {
                constants.slope * d_clamped[2]
            },
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
