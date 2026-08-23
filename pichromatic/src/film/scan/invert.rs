//! Technical scanner invert: negative → unbounded linear HDR positive in ACEScg.
//!
//! Reconstructs scene exposure using the physical inverse-transmission formula:
//! $$e = \left(\left(\frac{T}{D_{\min}}\right)^{-1/\gamma_{\text{eff}}} - 1.0\right)^+$$
//! scaled per channel such that the calibration mid-gray scan maps exactly to [`MIDDLE_GRAY`].
//!
//! The per-channel exponents are measured from the forward model at calibration
//! time (mid-gray and a +2-stop anchor); [`GAMMA_EFF`] is only the degenerate-case
//! fallback when the anchors admit no solvable contrast.

use crate::pixel::{ImageBuffer, MIDDLE_GRAY};
use rayon::prelude::*;

/// Fallback effective contrast exponent used by the invert when the measured
/// anchor pair is degenerate (no crossing in the solver bracket).
pub const GAMMA_EFF: f32 = 0.6;

/// Reconstructs unbounded scene-linear exposure in ACEScg from scanned negative transmission:
/// - $t_{\text{rel}} = \frac{T}{D_{\min}} \in (0, 1]$
/// - $e_{\text{raw}} = (t_{\text{rel}}^{-1/\gamma_c} - 1.0)^+$ with per-channel $\gamma_c$
///   measured from the forward model (see [`crate::film::scan::densitometry`])
/// - $g_c = \frac{\text{MIDDLE\_GRAY}}{e_{\text{mid}, c}}$ where $e_{\text{mid}, c} = \left(\left(\frac{\text{mid\_negative}[c]}{D_{\min}[c]}\right)^{-1/\gamma_c} - 1.0\right)^+$
/// - $\text{px}[c] = e_{\text{raw}, c} \cdot g_c$
/// - Result: Unexposed $T = D_{\min} \to 0.0$, Neutral Midtone $T = \text{mid} \to \text{MIDDLE\_GRAY} = 0.18$, Highlights $T \ll D_{\min} \to [0, \infty)$ in ACEScg.
pub fn invert_negative(
    buffer: &mut ImageBuffer,
    mid_negative: [f32; 3],
    dmin_negative: [f32; 3],
    inv_gamma: [f32; 3],
) {
    let eps = 1e-6f32;
    let inv_dmin = [
        1.0 / dmin_negative[0].max(eps),
        1.0 / dmin_negative[1].max(eps),
        1.0 / dmin_negative[2].max(eps),
    ];

    let mut gain = [0.0f32; 3];
    for c in 0..3 {
        let t_mid = (mid_negative[c] * inv_dmin[c]).max(eps);
        let e_mid = (t_mid.powf(-inv_gamma[c]) - 1.0).max(0.0);
        gain[c] = if e_mid > eps {
            MIDDLE_GRAY / e_mid
        } else {
            0.0
        };
    }

    buffer.par_iter_mut().for_each(|px| {
        *px = [
            ((px[0] * inv_dmin[0]).max(eps).powf(-inv_gamma[0]) - 1.0).max(0.0) * gain[0],
            ((px[1] * inv_dmin[1]).max(eps).powf(-inv_gamma[1]) - 1.0).max(0.0) * gain[1],
            ((px[2] * inv_dmin[2]).max(eps).powf(-inv_gamma[2]) - 1.0).max(0.0) * gain[2],
        ];
    });
}

/// Deprecated alias for [`invert_negative`].
#[deprecated(note = "Use invert_negative instead")]
pub fn invert_negative_inverse_hd(
    buffer: &mut ImageBuffer,
    mid_negative: [f32; 3],
    dmin_negative: [f32; 3],
    inv_gamma: [f32; 3],
) {
    invert_negative(buffer, mid_negative, dmin_negative, inv_gamma);
}

/// Shared CPU/GPU calibration for negative invert.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvertConstants {
    pub dmin: [f32; 3],
    pub inv_dmin: [f32; 3],
    pub gain: [f32; 3],
    pub gamma_eff: [f32; 3],
    pub inv_gamma: [f32; 3],
    pub eps: f32,
    pub exponent: [f32; 3],
}

pub fn invert_constants(
    mid_negative: [f32; 3],
    dmin_negative: [f32; 3],
    inv_gamma: [f32; 3],
) -> InvertConstants {
    let eps = 1e-6f32;
    let dmin = dmin_negative.map(|v| v.max(eps));
    let inv_dmin = dmin.map(|v| 1.0 / v);
    let mid_t = [
        (mid_negative[0] * inv_dmin[0]).max(eps),
        (mid_negative[1] * inv_dmin[1]).max(eps),
        (mid_negative[2] * inv_dmin[2]).max(eps),
    ];
    let mut gain = [0.0f32; 3];
    for c in 0..3 {
        let e_mid = (mid_t[c].powf(-inv_gamma[c]) - 1.0).max(0.0);
        gain[c] = if e_mid > eps {
            MIDDLE_GRAY / e_mid
        } else {
            0.0
        };
    }
    let exponent = mid_t.map(|t| {
        let t_clamped = t.clamp(eps, 1.0 - eps);
        (1.0 - MIDDLE_GRAY).ln() / t_clamped.ln()
    });

    InvertConstants {
        dmin,
        inv_dmin,
        gain,
        gamma_eff: inv_gamma.map(|ig| 1.0 / ig),
        inv_gamma,
        eps,
        exponent,
    }
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

    const TEST_INV_GAMMA: [f32; 3] = [1.0 / GAMMA_EFF; 3];

    #[test]
    fn dmin_maps_to_black_mid_to_middle_gray() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let mut buf = vec![dmin, mid];
        invert_negative(&mut buf, mid, dmin, TEST_INV_GAMMA);
        for c in 0..3 {
            assert!(
                buf[0][c] < 1e-5,
                "Dmin channel {c} should be ~0, got {}",
                buf[0][c]
            );
            assert!(
                (buf[1][c] - MIDDLE_GRAY).abs() < 1e-4,
                "mid channel {c} should be MIDDLE_GRAY, got {}",
                buf[1][c]
            );
        }
    }

    #[test]
    #[allow(deprecated)]
    fn inverse_hd_alias_matches_invert_negative() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let mut buf1 = vec![dmin, mid];
        let mut buf2 = vec![dmin, mid];
        invert_negative(&mut buf1, mid, dmin, TEST_INV_GAMMA);
        invert_negative_inverse_hd(&mut buf2, mid, dmin, TEST_INV_GAMMA);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn output_is_monotonic_and_unbounded_hdr() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let fractions = [1.0, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001];
        let mut buf: ImageBuffer = fractions
            .into_iter()
            .map(|t| [dmin[0] * t, dmin[1] * t, dmin[2] * t])
            .collect();
        invert_negative(&mut buf, mid, dmin, TEST_INV_GAMMA);

        for px in &buf {
            assert!(
                px.iter().all(|&v| v.is_finite() && v >= 0.0),
                "invert produced negative or non-finite {px:?}"
            );
        }
        for pair in buf.windows(2) {
            for c in 0..3 {
                assert!(
                    pair[1][c] > pair[0][c],
                    "denser negative must produce strictly brighter positive exposure: {pair:?}"
                );
            }
        }
        // Highlights (small transmission fractions) should exceed 1.0 (HDR unbounded)
        let highlight_px = buf.last().unwrap();
        for c in 0..3 {
            assert!(
                highlight_px[c] > 1.0,
                "highlights should be HDR (> 1.0), got {}",
                highlight_px[c]
            );
        }
    }

    #[test]
    fn near_dmin_fluctuation_preserves_structure() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let n = 256;
        let mut buf = Vec::with_capacity(n);
        for i in 0..n {
            let wave = ((i % 17) as f32 + 0.5) / 17.0 * 0.008;
            buf.push([dmin[0] - wave, dmin[1] - wave * 0.6, dmin[2] - wave * 0.3]);
        }
        invert_negative(&mut buf, mid, dmin, TEST_INV_GAMMA);
        let zero_frac = buf
            .iter()
            .filter(|px| px[0] == 0.0 && px[1] == 0.0 && px[2] == 0.0)
            .count() as f32
            / n as f32;
        assert!(
            zero_frac < 0.05,
            "near-Dmin variations should not collapse to zero, zero_frac={zero_frac}"
        );
        let std_r = {
            let mean = buf.iter().map(|px| px[0] as f64).sum::<f64>() / n as f64;
            (buf.iter()
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
    }
}
