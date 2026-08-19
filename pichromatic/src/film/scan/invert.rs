//! Technical scanner invert: negative → bounded positive ACEScg.
//!
//! This is the technical half of a Negadoctor-like workflow, not an optical
//! print simulation and not an inverse-H&D reconstruction. Each scanner channel
//! is divided by processed-film Dmin, then mapped by `1 - t^a`; the three global
//! exponents make the simulated neutral mid scan map to [`MIDDLE_GRAY`].

use crate::film::constants::FOG_OFFSET;
use crate::pixel::{ImageBuffer, MIDDLE_GRAY};
use rayon::prelude::*;

/// log2(10), used for converting base-10 constants to native base-2.
const LOG2_10: f32 = 3.3219280948873623;
/// Legacy GPU target; CPU PositiveLinear no longer reconstructs inverse H&D.
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
    /// Per-channel exponent for the bounded CPU scanner inversion.
    pub exponent: [f32; 3],
}

/// Legacy GPU effective density after its fog toe.
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
fn fog_effective_density_base2(d2: f32, fog2: f32) -> f32 {
    if d2 <= 0.0 {
        0.0
    } else if d2 < fog2 {
        (d2 * d2) / (2.0 * fog2)
    } else {
        d2 - 0.5 * fog2
    }
}

/// Legacy GPU inverse-H&D mapper retained while CPU and GPU are intentionally desynchronized.
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
fn density2_to_exposure(d2: f32, inv_gamma: f32, fog2: f32) -> f32 {
    let d_eff2 = fog_effective_density_base2(d2, fog2);
    if d_eff2 <= 0.0 {
        0.0
    } else {
        (d_eff2 * inv_gamma).exp2() - 1.0
    }
}

pub fn invert_constants(mid_negative: [f32; 3], dmin_negative: [f32; 3]) -> InvertConstants {
    let eps = 1e-6f32;
    let dmin = dmin_negative.map(|v| v.max(eps));
    let inv_dmin = dmin.map(|v| 1.0 / v);
    let mid_t = [
        (mid_negative[0] * inv_dmin[0]).clamp(eps, 1.0 - eps),
        (mid_negative[1] * inv_dmin[1]).clamp(eps, 1.0 - eps),
        (mid_negative[2] * inv_dmin[2]).clamp(eps, 1.0 - eps),
    ];
    let exponent = mid_t.map(|t| (1.0 - MIDDLE_GRAY).ln() / t.ln());

    // GPU compatibility only: the GPU still consumes the legacy inverse-H&D
    // fields until the approved CPU path is committed and ported.
    let inv_gamma = 1.0 / GAMMA_EFF;
    let slope = 10.0f32.ln() / GAMMA_EFF;
    let inv_gamma_log2_10 = inv_gamma * LOG2_10;
    let fog2 = FOG_OFFSET * LOG2_10;
    let d_mid2 = mid_t.map(|v| (-v.log2()).max(eps));
    let e_mid = d_mid2.map(|d2| density2_to_exposure(d2, inv_gamma, fog2).max(0.005));
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
        exponent,
    }
}

/// Bounded technical scanner invert for PositiveLinear.
///
/// `dmin_negative` is the mean scan of developed unexposed film, so it already
/// includes chemical fog. Applying the old `FOG_OFFSET` density toe here would
/// subtract that same fog a second time. Film base and fog are therefore removed
/// once by `T / Dmin`; all remaining variation comes from the realized scan.
pub fn invert_negative(buffer: &mut ImageBuffer, mid_negative: [f32; 3], dmin_negative: [f32; 3]) {
    let constants = invert_constants(mid_negative, dmin_negative);

    buffer.par_iter_mut().for_each(|px| {
        *px = [
            1.0 - (px[0] * constants.inv_dmin[0])
                .clamp(constants.eps, 1.0)
                .powf(constants.exponent[0]),
            1.0 - (px[1] * constants.inv_dmin[1])
                .clamp(constants.eps, 1.0)
                .powf(constants.exponent[1]),
            1.0 - (px[2] * constants.inv_dmin[2])
                .clamp(constants.eps, 1.0)
                .powf(constants.exponent[2]),
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
    fn output_is_monotonic_and_bounded() {
        let dmin = [1.0f32, 0.45, 0.13];
        let mid = [0.23f32, 0.15, 0.10];
        let mut buf: ImageBuffer = [1.0, 0.75, 0.5, 0.25, 0.0]
            .into_iter()
            .map(|t| [dmin[0] * t, dmin[1] * t, dmin[2] * t])
            .collect();
        invert_negative(&mut buf, mid, dmin);

        for px in &buf {
            assert!(
                px.iter()
                    .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)),
                "bounded invert produced {px:?}"
            );
        }
        for pair in buf.windows(2) {
            for c in 0..3 {
                assert!(
                    pair[1][c] >= pair[0][c],
                    "denser negative must produce brighter positive: {pair:?}"
                );
            }
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
        invert_negative(&mut buf, mid, dmin);
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
