//! Developable-fraction LUT from a log-normal crystal size distribution.
//!
//! For crystal diameter s ~ LogNormal(μ, σ) (s in µm), the expected developable
//! fraction at absorbed fluence Φ is:
//!
//!   f(Φ) = 1 − E_s[ P(X < 4; λ = k · s² · Φ) ]
//!
//! Photon arrivals are Poisson; developability at the single-crystal level uses
//! the T = 4 silver-atom sensitivity-speck threshold.
//!
//! Equivalence: a crystal is treated as developable once the Poisson mean of
//! absorbed photons (∝ s² Φ) yields a non-zero probability of ≥ T latent atoms;
//! the continuum population average collapses to the expectation of the Poisson CDF
//! with `k` absorbing QE and geometric factors.
//!
//! This LUT is a memoized physical integral, not an artist curve. The expectation
//! is evaluated with a ≥64-node trapezoidal quadrature on the standardized normal
//! underlying ln(s) — the change-of-variable form of Gauss–Hermite on that Gaussian
//! (film-implementation.md §5.5 requires ≥32 nodes).
//!
//! The grid and fractions are stored as f32 (rounded from the f64 quadrature) and
//! sampled in f32 with the same log10 + linear-interpolation sequence as the GPU
//! `LUT` shader (`shaders::lut_sample`).

use crate::film::stock::LogNormalDist;

/// Precomputed 1D LUT: log-spaced fluence → developable fraction.
#[derive(Clone, Debug, PartialEq)]
pub struct DevelopableFractionLut {
    /// Log10 of fluence samples (photons/µm²), f32 as uploaded to the GPU.
    pub log10_fluence: Vec<f32>,
    /// Developable fraction f ∈ [0, 1] at each sample, f32 as uploaded to the GPU.
    pub fraction: Vec<f32>,
    /// Absorption/quantum calibration factor k in f = 1 − E[P(X<4; k s² Φ)].
    pub k: f64,
}

impl DevelopableFractionLut {
    /// Build LUT with ≥64 log-spaced entries using ≥32-node quadrature.
    /// Quadrature stays in f64 (per-stock precompute); stored values are rounded
    /// to f32 exactly as `gpu::bake_consts` uploads them.
    pub fn build(dist: &LogNormalDist, k: f64, n_entries: usize) -> Self {
        assert!(n_entries >= 64);
        assert!(k > 0.0);

        let log_min = -4.0_f64;
        let log_max = 8.0_f64;
        let mut log10_fluence = Vec::with_capacity(n_entries);
        let mut fraction = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let t = i as f64 / (n_entries - 1) as f64;
            let log_phi = log_min + t * (log_max - log_min);
            let phi = 10f64.powf(log_phi);
            let f = expected_developable_fraction(dist, k, phi);
            log10_fluence.push(log_phi as f32);
            fraction.push(f as f32);
        }
        Self {
            log10_fluence,
            fraction,
            k,
        }
    }

    /// Linear interpolate fraction for fluence `phi` (photons/µm²). Clamped.
    ///
    /// Mirrors the GPU `LUT` shader `lut_sample` (log10 via `ln(Φ)·log10(e)`,
    /// step binning over the uniform log10 grid, linear lerp).
    pub fn sample(&self, phi: f32) -> f32 {
        if !(phi > 0.0) {
            return self.fraction[0];
        }
        let lp = phi.ln() * INV_LN10;
        let logs = &self.log10_fluence;
        let fracs = &self.fraction;
        let lo0 = logs[0];
        let lo63 = logs[63];
        if lp <= lo0 {
            return fracs[0];
        }
        if lp >= lo63 {
            return fracs[63];
        }
        let step = logs[1] - logs[0];
        let lo = ((lp - lo0) / step).floor() as usize;
        let hi = lo + 1;
        let t = (lp - logs[lo]) / (logs[hi] - logs[lo]);
        fracs[lo] * (1.0 - t) + fracs[hi] * t
    }
}

/// log10(e), shared with the GPU `LUT` shader constant `INV_LN10`.
const INV_LN10: f32 = 0.4342944819032518;

/// E_s[ P(X < 4; λ = k s² Φ) ] for ln(s) ~ N(μ, σ²).
fn expected_survival(dist: &LogNormalDist, k: f64, phi: f64) -> f64 {
    const N: usize = 64;
    const Z_MAX: f64 = 8.0;
    let dz = (2.0 * Z_MAX) / N as f64;
    let inv_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().recip();
    let mut acc = 0.0;
    for i in 0..=N {
        let z = -Z_MAX + i as f64 * dz;
        let trap_w = if i == 0 || i == N { 0.5 } else { 1.0 };
        let pdf = inv_sqrt_2pi * (-0.5 * z * z).exp();
        let s = (dist.mu_ln + dist.sigma_ln * z).exp();
        let lambda = k * s * s * phi;
        let p_not_dev = (-lambda).exp() * (1.0 + lambda + lambda * lambda / 2.0 + lambda * lambda * lambda / 6.0);
        acc += trap_w * pdf * p_not_dev * dz;
    }
    acc
}

pub(crate) fn expected_developable_fraction(dist: &LogNormalDist, k: f64, phi: f64) -> f64 {
    (1.0 - expected_survival(dist, k, phi)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dist() -> LogNormalDist {
        LogNormalDist {
            mu_ln: 0.7_f64.ln(),
            sigma_ln: 0.35,
        }
    }

    #[test]
    fn lut_monotonic() {
        let lut = DevelopableFractionLut::build(&test_dist(), 1.0, 64);
        let mut prev = -1.0f32;
        for i in 0..64 {
            let log_phi = -4.0 + i as f32 * (12.0 / 63.0);
            let phi = 10f32.powf(log_phi);
            let f = lut.sample(phi);
            assert!(
                f + 1e-6 >= prev,
                "non-monotonic at i={i}: prev={prev}, f={f}"
            );
            prev = f;
        }
    }

    #[test]
    fn lut_limits() {
        let lut = DevelopableFractionLut::build(&test_dist(), 1.0, 64);
        let f0 = lut.sample(1e-6);
        let f_inf = lut.sample(1e10);
        assert!(f0 < 0.05, "toe: f={f0}");
        assert!(f_inf > 0.99, "shoulder: f={f_inf}");
    }

    #[test]
    fn lut_shoulder_toe_curvature() {
        let lut = DevelopableFractionLut::build(&test_dist(), 1.0, 64);
        let n = 40;
        let mut fs = Vec::with_capacity(n);
        for i in 0..n {
            let log_phi = -3.0 + i as f32 * (10.0 / (n - 1) as f32);
            fs.push(lut.sample(10f32.powf(log_phi)));
        }
        let mut max_slope = 0.0;
        let mut max_i = 1usize;
        for i in 1..n {
            let slope = fs[i] - fs[i - 1];
            if slope > max_slope {
                max_slope = slope;
                max_i = i;
            }
        }
        let toe_i = (max_i / 3).max(2);
        let d2_toe = (fs[toe_i] - fs[toe_i - 1]) - (fs[toe_i - 1] - fs[toe_i - 2]);
        let sh_i = ((max_i + n) / 2).min(n - 1).max(max_i + 2);
        let d2_shoulder = (fs[sh_i] - fs[sh_i - 1]) - (fs[sh_i - 1] - fs[sh_i - 2]);
        // f32 rounding can pin the plateau to exactly 1.0 → second-diff 0; accept.
        assert!(
            d2_toe >= 0.0,
            "toe second-diff should be ≥0, got {d2_toe} (toe_i={toe_i}, max_i={max_i})"
        );
        assert!(
            d2_shoulder <= 0.0,
            "shoulder second-diff should be ≤0, got {d2_shoulder} (sh_i={sh_i})"
        );
    }

    #[test]
    fn lut_deterministic() {
        let a = DevelopableFractionLut::build(&test_dist(), 1.0, 64);
        let b = DevelopableFractionLut::build(&test_dist(), 1.0, 64);
        assert_eq!(a, b);
    }
}
