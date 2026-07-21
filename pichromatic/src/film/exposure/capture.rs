//! Developable-fraction LUT from a log-normal crystal size distribution.
//!
//! For crystal diameter s ~ LogNormal(μ, σ) (s in µm), the expected developable
//! fraction at absorbed fluence Φ is:
//!
//!   f(Φ) = 1 − E_s[ exp(−k · s² · Φ) ]
//!
//! Photon arrivals are Poisson; developability at the single-crystal level uses
//! the T = 4 silver-atom sensitivity-speck threshold folded into calibration
//! constant `k` (see film-implementation.md §5.5 and DEVELOPABILITY_THRESHOLD_ATOMS).
//!
//! Equivalence: a crystal is treated as developable once the Poisson mean of
//! absorbed photons (∝ s² Φ) yields a non-zero probability of ≥ T latent atoms;
//! the continuum population average collapses to the Laplace-transform form above
//! with `k` absorbing QE and geometric factors.
//!
//! This LUT is a memoized physical integral, not an artist curve. The expectation
//! is evaluated with a ≥64-node trapezoidal quadrature on the standardized normal
//! underlying ln(s) — the change-of-variable form of Gauss–Hermite on that Gaussian
//! (film-implementation.md §5.5 requires ≥32 nodes).

use crate::film::stock::LogNormalDist;

/// Precomputed 1D LUT: log-spaced fluence → developable fraction.
#[derive(Clone, Debug, PartialEq)]
pub struct DevelopableFractionLut {
    /// Log10 of fluence samples (photons/µm²).
    pub log10_fluence: Vec<f64>,
    /// Developable fraction f ∈ [0, 1] at each sample.
    pub fraction: Vec<f64>,
    /// Absorption/quantum calibration factor k in f = 1 − E[exp(−k s² Φ)].
    pub k: f64,
}

impl DevelopableFractionLut {
    /// Build LUT with ≥64 log-spaced entries using ≥32-node quadrature.
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
            log10_fluence.push(log_phi);
            fraction.push(f);
        }
        Self {
            log10_fluence,
            fraction,
            k,
        }
    }

    /// Linear interpolate fraction for fluence `phi` (photons/µm²). Clamped.
    pub fn sample(&self, phi: f64) -> f64 {
        if !(phi > 0.0) {
            return self.fraction[0];
        }
        let log_phi = phi.log10();
        let logs = &self.log10_fluence;
        let fracs = &self.fraction;
        if log_phi <= logs[0] {
            return fracs[0];
        }
        let last = logs.len() - 1;
        if log_phi >= logs[last] {
            return fracs[last];
        }
        let mut lo = 0usize;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if logs[mid] <= log_phi {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (log_phi - logs[lo]) / (logs[hi] - logs[lo]);
        fracs[lo] * (1.0 - t) + fracs[hi] * t
    }
}

/// E_s[ exp(−k s² Φ) ] for ln(s) ~ N(μ, σ²).
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
        let g = (-k * s * s * phi).exp();
        acc += trap_w * pdf * g * dz;
    }
    acc
}

fn expected_developable_fraction(dist: &LogNormalDist, k: f64, phi: f64) -> f64 {
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
        let mut prev = -1.0;
        for i in 0..64 {
            let log_phi = -4.0 + i as f64 * (12.0 / 63.0);
            let phi = 10f64.powf(log_phi);
            let f = lut.sample(phi);
            assert!(
                f + 1e-12 >= prev,
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
            let log_phi = -3.0 + i as f64 * (10.0 / (n - 1) as f64);
            fs.push(lut.sample(10f64.powf(log_phi)));
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
        assert!(
            d2_toe > 0.0,
            "toe second-diff should be >0, got {d2_toe} (toe_i={toe_i}, max_i={max_i})"
        );
        assert!(
            d2_shoulder < 0.0,
            "shoulder second-diff should be <0, got {d2_shoulder} (sh_i={sh_i})"
        );
    }

    #[test]
    fn lut_deterministic() {
        let a = DevelopableFractionLut::build(&test_dist(), 1.0, 64);
        let b = DevelopableFractionLut::build(&test_dist(), 1.0, 64);
        assert_eq!(a, b);
    }
}
