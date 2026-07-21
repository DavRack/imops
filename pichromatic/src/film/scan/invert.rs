//! Optional densitometric invert: negative → positive (linear ACEScg).
//!
//! Uses stock **Dmin** (film base) and **mid-gray** negative:
//!
//! 1. Clear the orange mask: `T = neg / Dmin`.
//! 2. Soft densitometric invert per channel:
//!    `soft(inv) = inv / (1 + inv / (HEADROOM · inv_mid))` with `inv = 1/T − 1`.
//!    Mid maps to [`MIDDLE_GRAY`] in every channel (neutrality), while channels
//!    whose mid density sits near Dmin (typical orange-mask blue) compress
//!    early instead of exploding in highlights.
//! 3. Soften chroma toward luminance (`CHROMA_KEEP`), fading to neutral in
//!    deep shadows so toe/mask noise does not leave a color pedestal.

use crate::pixel::{ImageBuffer, PixelOps, MIDDLE_GRAY};
use rayon::prelude::*;

/// How much of the per-channel invert's chroma to keep (0 = gray, 1 = full).
/// Tuned between the neon (`1.0`) and B&W (`0`) failure modes.
const CHROMA_KEEP: f32 = 0.55;

/// Highlight headroom in units of each channel's mid invert.
/// Channel asymptote is `MIDDLE_GRAY * (HEADROOM + 1)`.
const HEADROOM: f32 = 6.0;

/// Scene-referred Y below which chroma is faded toward neutral.
const SHADOW_CHROMA_FADE_Y: f32 = 0.02;

#[inline]
fn soft_inv(inv: f32, shoulder: f32) -> f32 {
    let inv = inv.max(0.0);
    let s = shoulder.max(1e-6);
    inv / (1.0 + inv / s)
}

/// Mid/Dmin densitometric invert for PositiveLinear.
pub fn invert_negative(
    buffer: &mut ImageBuffer,
    mid_negative: [f32; 3],
    dmin_negative: [f32; 3],
) {
    let eps = 1e-6f32;
    let dmin = [
        dmin_negative[0].max(eps),
        dmin_negative[1].max(eps),
        dmin_negative[2].max(eps),
    ];
    let mid_t = [
        (mid_negative[0] / dmin[0]).clamp(eps, 1.0 - eps),
        (mid_negative[1] / dmin[1]).clamp(eps, 1.0 - eps),
        (mid_negative[2] / dmin[2]).clamp(eps, 1.0 - eps),
    ];
    let inv_mid = [
        (1.0 / mid_t[0] - 1.0).max(0.0),
        (1.0 / mid_t[1] - 1.0).max(0.0),
        (1.0 / mid_t[2] - 1.0).max(0.0),
    ];
    // Per-channel shoulder tracks mid invert so a near-Dmin mid channel
    // compresses as soon as density appears, instead of applying a huge linear gain.
    let shoulder = [
        (inv_mid[0] * HEADROOM).max(eps),
        (inv_mid[1] * HEADROOM).max(eps),
        (inv_mid[2] * HEADROOM).max(eps),
    ];
    let soft_mid = [
        soft_inv(inv_mid[0], shoulder[0]).max(eps),
        soft_inv(inv_mid[1], shoulder[1]).max(eps),
        soft_inv(inv_mid[2], shoulder[2]).max(eps),
    ];
    let g = [
        MIDDLE_GRAY / soft_mid[0],
        MIDDLE_GRAY / soft_mid[1],
        MIDDLE_GRAY / soft_mid[2],
    ];

    buffer.par_iter_mut().for_each(|px| {
        // T>1 (mask-clearing toe) → that channel contributes 0, not a whole-pixel crush.
        let t = [
            (px[0] / dmin[0]).clamp(eps, 1.0),
            (px[1] / dmin[1]).clamp(eps, 1.0),
            (px[2] / dmin[2]).clamp(eps, 1.0),
        ];
        let inv = [
            (1.0 / t[0] - 1.0).max(0.0),
            (1.0 / t[1] - 1.0).max(0.0),
            (1.0 / t[2] - 1.0).max(0.0),
        ];
        let raw = [
            (g[0] * soft_inv(inv[0], shoulder[0])).max(0.0),
            (g[1] * soft_inv(inv[1], shoulder[1])).max(0.0),
            (g[2] * soft_inv(inv[2], shoulder[2])).max(0.0),
        ];
        let y = raw.luminance().max(0.0);
        // Deep shadows: fade chroma toward luminance so a single-channel toe
        // spike cannot leave a colored pedestal in blacks.
        let k = if y <= 0.0 {
            0.0
        } else if y >= SHADOW_CHROMA_FADE_Y {
            CHROMA_KEEP
        } else {
            CHROMA_KEEP * (y / SHADOW_CHROMA_FADE_Y)
        };
        *px = [
            (y + k * (raw[0] - y)).max(0.0),
            (y + k * (raw[1] - y)).max(0.0),
            (y + k * (raw[2] - y)).max(0.0),
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
        assert!(y.is_finite() && y > 0.0, "highlight should be finite+, got {p:?}");
        assert!(
            p[0] < 2.5 && p[1] < 2.5 && p[2] < 2.5,
            "channel blow from unbounded blue gain: {p:?}"
        );
        let max = p[0].max(p[1]).max(p[2]);
        let min = p[0].min(p[1]).min(p[2]).max(1e-6);
        assert!(
            max / min < 4.0,
            "highlight cast too strong after soft invert: {p:?}"
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
        assert!(mean > 0.0, "shadow positive should be > 0");
        let max = p[0].max(p[1]).max(p[2]);
        let min = p[0].min(p[1]).min(p[2]);
        assert!(
            max - min < 0.05 + 0.5 * mean,
            "near-shadow cast too strong: {p:?}"
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
            ratio < 14.0,
            "channel ratio too high (neon): {p:?} ratio={ratio}"
        );
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
            // Gaussian-like pseudo-random noise near Dmin
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
        let max = mean[0].max(mean[1]).max(mean[2]);
        let min = mean[0].min(mean[1]).min(mean[2]);
        assert!(
            max - min < 0.002 + 0.5 * y,
            "noisy Dmin mean should stay near-neutral, got {mean:?}"
        );
        assert!(y < 0.01, "noisy Dmin mean Y should stay dark, got {y}");
    }
}
