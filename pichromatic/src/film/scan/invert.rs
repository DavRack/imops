//! Optional densitometric invert: negative → positive (linear ACEScg).
//!
//! Uses stock **Dmin** (film base) and **mid-gray** negative only:
//!
//! 1. Clear the orange mask: `T = neg / Dmin`.
//! 2. Per-channel densitometric invert
//!    `pos'_c = g_c · (1/T_c − 1)` with `g` from mid → [`MIDDLE_GRAY`].
//! 3. Soften chroma toward luminance so dye imbalance does not neon-blow,
//!    without collapsing to B&W (`CHROMA_KEEP`).

use crate::pixel::{ImageBuffer, PixelOps, MIDDLE_GRAY};
use rayon::prelude::*;

/// How much of the per-channel invert's chroma to keep (0 = gray, 1 = full).
/// Tuned between the neon (`1.0`) and B&W (`0`) failure modes.
const CHROMA_KEEP: f32 = 0.55;

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
    let g = [
        MIDDLE_GRAY * mid_t[0] / (1.0 - mid_t[0]),
        MIDDLE_GRAY * mid_t[1] / (1.0 - mid_t[1]),
        MIDDLE_GRAY * mid_t[2] / (1.0 - mid_t[2]),
    ];

    buffer.par_iter_mut().for_each(|px| {
        // T>1 (mask-clearing toe) → that channel contributes 0, not a whole-pixel crush.
        let t = [
            (px[0] / dmin[0]).clamp(eps, 1.0),
            (px[1] / dmin[1]).clamp(eps, 1.0),
            (px[2] / dmin[2]).clamp(eps, 1.0),
        ];
        let raw = [
            (g[0] * (1.0 / t[0] - 1.0)).max(0.0),
            (g[1] * (1.0 / t[1] - 1.0)).max(0.0),
            (g[2] * (1.0 / t[2] - 1.0)).max(0.0),
        ];
        let y = raw.luminance().max(0.0);
        // Keep luminance; pull RGB toward gray by (1 - CHROMA_KEEP).
        *px = [
            (y + CHROMA_KEEP * (raw[0] - y)).max(0.0),
            (y + CHROMA_KEEP * (raw[1] - y)).max(0.0),
            (y + CHROMA_KEEP * (raw[2] - y)).max(0.0),
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
}
