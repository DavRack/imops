//! Analytical scanner invert: negative → positive (linear ACEScg).
//!
//! Linear scanner pass without stylistic looks or artificial S-curves:
//! 1. Substrate Color Correction: D_2 = −log2(clamp(T_neg / Dmin, ε, ∞))
//! 2. Film Dynamic Range (Gamma) Reconstruction: E_scene = 2^(D_eff2 / γ_eff) − 1.0
//! 3. Mid-Gray Anchor: Scale factor g_c = MIDDLE_GRAY / E_scene(mid_c) per channel.
//!
//! Formulated entirely in native base-2 arithmetic (log2 / exp2) to eliminate
//! f32 transcendental base-conversion roundtrip errors and optimize ALU throughput.

use crate::film::constants::FOG_OFFSET;
use crate::pixel::{ImageBuffer, MIDDLE_GRAY};
use rayon::prelude::*;

/// log2(10), used for converting base-10 constants to native base-2.
const LOG2_10: f32 = 3.3219280948873623;
/// log10(2), algebraic inverse of LOG2_10.
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
    /// `1/gamma_eff · log2(10)` precombined so GPU scan shader can use one rounded constant.
    pub inv_gamma_log2_10: f32,
    pub eps: f32,
    pub fog_offset: f32,
    /// Precomputed base-2 fog offset: `fog_offset * LOG2_10`.
    pub fog2: f32,
}

/// Effective density in base-2 after fog toe, fed directly into the gamma `exp2` transfer.
///
/// C¹ quadratic toe in base-2 density space ($D_2 = D_{10} \cdot \log_2(10)$, $\mathrm{fog}_2 = \mathrm{fog}_{10} \cdot \log_2(10)$):
/// - `d2 ≤ 0` (Dmin) → `0`
/// - `0 < d2 < fog2` → `d2² / (2·fog2)` (preserves sub-fog dye-cloud structure)
/// - `d2 ≥ fog2` → `d2 − fog2/2`
///
/// At `d2 = fog2`, both pieces equal `fog2/2` with matching slope 1 — continuous first derivative (C¹).
/// Because $\log_2(10)$ factors out quadratically in the numerator and linearly in the denominator,
/// $\frac{(d_{10}\log_2 10)^2}{2(\mathrm{fog}_{10}\log_2 10)} = \frac{d_{10}^2}{2\mathrm{fog}_{10}} \log_2 10$,
/// making the base-2 formulation algebraically exact.
#[inline]
pub fn fog_effective_density_base2(d2: f32, fog2: f32) -> f32 {
    if d2 <= 0.0 {
        0.0
    } else if d2 < fog2 {
        (d2 * d2) / (2.0 * fog2)
    } else {
        d2 - 0.5 * fog2
    }
}

/// Map base-2 image density `d2 = −log2(T/Dmin)` directly to linear scene exposure.
///
/// Reconstructs scene exposure using native base-2 arithmetic:
/// `E_scene = 2^(d_eff2 / γ) − 1.0 = exp2(d_eff2 * inv_gamma) − 1.0`.
///
/// ### Mathematical Identity & Precision Rationale
/// Conventional optical density is decadic ($D_{10} = -\log_{10} T$). Inverting $D_{10}$ requires
/// computing $10^{D_{10}/\gamma} = 2^{(D_{10}/\gamma)\log_2 10}$.
/// On hardware ALUs lacking native base-10 transcendentals, evaluating decadic density requires:
/// 1. $D_{10} = -\log_2(T) \cdot \log_{10}(2)$
/// 2. $E = \exp_2(D_{10} \cdot \frac{1}{\gamma} \cdot \log_2(10)) - 1.0$
///
/// Since $\log_{10}(2) \cdot \log_2(10) \equiv 1.0$, the conversion factors cancel algebraically.
/// Performing the intermediate multiplication by `LOG10_2` followed by `LOG2_10` in IEEE 754 `f32`
/// introduces intermediate rounding and truncation errors. Furthermore, because $dE/dD \propto 2^{D/\gamma}$,
/// any float precision loss at high negative densities (bright scene highlights) is exponentially amplified.
/// Working directly with base-2 density $D_2 = -\log_2 T$ executes exclusively in native hardware
/// transcendentals (`log2` and `exp2`) with zero intermediate conversion error and optimal ALU throughput.
#[inline]
pub fn density2_to_exposure(d2: f32, inv_gamma: f32, fog2: f32) -> f32 {
    let d_eff2 = fog_effective_density_base2(d2, fog2);
    if d_eff2 <= 0.0 {
        0.0
    } else {
        (d_eff2 * inv_gamma).exp2() - 1.0
    }
}

/// Legacy base-10 image density mapper (delegates to native base-2 formulation).
#[inline]
pub fn density_img_to_exposure(
    d_img: f32,
    inv_gamma_log2_10: f32,
    _slope: f32,
    fog_offset: f32,
) -> f32 {
    let d2 = d_img * LOG2_10;
    let fog2 = fog_offset * LOG2_10;
    let inv_gamma = inv_gamma_log2_10 * LOG10_2;
    density2_to_exposure(d2, inv_gamma, fog2)
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
    let fog2 = FOG_OFFSET * LOG2_10;
    let d_mid2 = mid_t.map(|v| (-v.log2()).max(eps));
    let e_mid = d_mid2.map(|d2| {
        density2_to_exposure(d2, inv_gamma, fog2).max(0.005)
    });
    InvertConstants {
        dmin,
        inv_dmin,
        gain: e_mid.map(|v| (MIDDLE_GRAY / v).min(25.0)),
        slope,
        gamma_eff: GAMMA_EFF,
        inv_gamma,
        inv_gamma_log2_10,
        eps,
        fog_offset: FOG_OFFSET,
        fog2,
    }
}

/// Linear scanner invert for PositiveLinear.
///
/// Reconstructs linear ACEScg positive scene exposure directly from negative transmittance
/// using native base-2 optical density:
/// 1. Substrate Color Correction: $T_c = \mathrm{clamp}(T_{\mathrm{neg}, c} \cdot \mathrm{inv\_dmin}_c, \varepsilon, 1.0)$
/// 2. Base-2 Optical Density: $D_{2, c} = -\log_2(T_c)$
/// 3. C¹ Fog Toe & Gamma Reversal: $E_c = \exp_2(D_{\mathrm{eff}2, c} \cdot \mathrm{inv\_gamma}) - 1.0$
/// 4. Mid-Gray Anchor: $\mathrm{RGB}_c = g_c \cdot E_c$
pub fn invert_negative(buffer: &mut ImageBuffer, mid_negative: [f32; 3], dmin_negative: [f32; 3]) {
    let constants = invert_constants(mid_negative, dmin_negative);

    buffer.par_iter_mut().for_each(|px| {
        let t = [
            (px[0] * constants.inv_dmin[0]).max(constants.eps),
            (px[1] * constants.inv_dmin[1]).max(constants.eps),
            (px[2] * constants.inv_dmin[2]).max(constants.eps),
        ];

        let d2 = [
            -t[0].log2(),
            -t[1].log2(),
            -t[2].log2(),
        ];

        let e_scene = [
            density2_to_exposure(d2[0], constants.inv_gamma, constants.fog2),
            density2_to_exposure(d2[1], constants.inv_gamma, constants.fog2),
            density2_to_exposure(d2[2], constants.inv_gamma, constants.fog2),
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
        let inv_gamma = 1.0 / GAMMA_EFF;
        let fog2 = FOG_OFFSET * LOG2_10;
        assert!(
            density2_to_exposure(0.0, inv_gamma, fog2).abs() < 1e-7,
            "Dmin density maps to zero exposure"
        );
        let low = density2_to_exposure(0.001 * LOG2_10, inv_gamma, fog2);
        let high = density2_to_exposure(0.002 * LOG2_10, inv_gamma, fog2);
        assert!(low > 0.0 && high > low, "sub-fog densities rise monotonically");
        let below = fog_effective_density_base2(fog2 - 1e-6, fog2);
        let at = fog_effective_density_base2(fog2, fog2);
        let above_d = fog_effective_density_base2(fog2 + 1e-6, fog2);
        assert!(
            (at - 0.5 * fog2).abs() < 1e-6,
            "knee density must be fog2/2, got {at}"
        );
        assert!(
            (below - at).abs() < 1e-5 && (above_d - at).abs() < 1e-5,
            "C1 toe must be continuous at fog knee"
        );
        let knee = density2_to_exposure(fog2, inv_gamma, fog2);
        let above = density2_to_exposure(fog2 + 0.001, inv_gamma, fog2);
        assert!(above >= knee, "exposure continues above fog knee");
    }

    #[test]
    fn base2_and_base10_algebraic_equivalence() {
        let inv_gamma = 1.0 / GAMMA_EFF;
        let inv_gamma_log2_10 = inv_gamma * LOG2_10;
        let slope = 10.0f32.ln() / GAMMA_EFF;
        let fog2 = FOG_OFFSET * LOG2_10;

        for d10_int in 0..500 {
            let d10 = d10_int as f32 * 0.01;
            let d2 = d10 * LOG2_10;

            let exp_base2 = density2_to_exposure(d2, inv_gamma, fog2);
            let exp_base10 = density_img_to_exposure(d10, inv_gamma_log2_10, slope, FOG_OFFSET);

            let rel_diff = (exp_base2 - exp_base10).abs() / (exp_base2.max(exp_base10) + 1e-5);
            assert!(
                rel_diff < 1e-4,
                "base2 vs base10 divergence at d10={d10}: base2={exp_base2}, base10={exp_base10}, rel_diff={rel_diff}"
            );
        }
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
