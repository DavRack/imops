use std::f64::consts::PI;

#[derive(Clone, Debug, PartialEq)]
pub struct LogNormalDist {
    pub mu_ln: f64,
    pub sigma_ln: f64,
}

fn expected_survival(dist: &LogNormalDist, k: f64, phi: f64) -> f64 {
    const N: usize = 64;
    const Z_MAX: f64 = 8.0;
    let dz = (2.0 * Z_MAX) / N as f64;
    let inv_sqrt_2pi = (2.0 * PI).sqrt().recip();
    let mut acc = 0.0;
    for i in 0..=N {
        let z = -Z_MAX + i as f64 * dz;
        let trap_w = if i == 0 || i == N { 0.5 } else { 1.0 };
        let pdf = inv_sqrt_2pi * (-0.5 * z * z).exp();
        let s = (dist.mu_ln + dist.sigma_ln * z).exp();
        let lambda = k * s * s * phi;
        let p_not_dev = (-lambda).exp() * (1.0 + lambda);
        acc += trap_w * pdf * p_not_dev * dz;
    }
    acc
}

pub fn expected_developable_fraction(dist: &LogNormalDist, k: f64, phi: f64) -> f64 {
    (1.0 - expected_survival(dist, k, phi)).clamp(0.0, 1.0)
}

#[test]
fn sweep_test() {
    let phi_mid = 1.5; // Arbitrary mid-gray reference point

    // Current parameters for the 6 emulsions (Red fast, Red slow, Green fast, Green slow, Blue fast, Blue slow)
    // format: (name, mu_exp, sigma, k, thickness, packing)
    let current_params: Vec<(&str, f64, f64, f64, f64, f64)> = vec![
        ("Blue Fast", 0.85, 0.65, 3.74, 4.0, 0.16),
        ("Blue Slow", 0.42, 0.28, 2.20, 4.0, 0.20),
        ("Green Fast", 0.90, 0.65, 1.87, 3.0, 0.16),
        ("Green Slow", 0.45, 0.28, 1.10, 3.0, 0.20),
        ("Red Fast", 0.95, 0.65, 1.22, 3.5, 0.16),
        ("Red Slow", 0.48, 0.28, 0.70, 3.5, 0.20),
    ];

    // We want to scale crystals by c = 0.35 to boost rho_areal by ~23x.
    // And we want to increase sigma_ln by 0.15 to open the toe.
    let c = 0.22f64;
    let sigma_boost = 0.15f64;

    for (name, old_mu_exp, old_sigma, old_k, thickness, packing) in current_params {
        let old_dist = LogNormalDist {
            mu_ln: old_mu_exp.ln(),
            sigma_ln: old_sigma,
        };

        // Find the f_mid that the OLD parameters produced at phi=1.5
        let target_f_mid = expected_developable_fraction(&old_dist, old_k, phi_mid);

        // Compute new mu and sigma
        let new_mu_exp = old_mu_exp * c;
        let new_sigma = old_sigma + sigma_boost;
        let new_dist = LogNormalDist {
            mu_ln: new_mu_exp.ln(),
            sigma_ln: new_sigma,
        };

        // Binary search for new k to match target_f_mid exactly
        let mut k_lo = 0.0;
        let mut k_hi = 1000.0;
        for _ in 0..60 {
            let k_mid = (k_lo + k_hi) / 2.0;
            let f = expected_developable_fraction(&new_dist, k_mid, phi_mid);
            if f < target_f_mid {
                k_lo = k_mid;
            } else {
                k_hi = k_mid;
            }
        }
        let new_k = (k_lo + k_hi) / 2.0;

        let old_mean_s = (old_dist.mu_ln + 0.5 * old_dist.sigma_ln * old_dist.sigma_ln).exp();
        let old_vol = PI * (old_mean_s * 0.5).max(1e-6).powi(2) * thickness;
        let old_rho = packing * thickness / old_vol;

        let new_mean_s = (new_dist.mu_ln + 0.5 * new_dist.sigma_ln * new_dist.sigma_ln).exp();
        let new_vol = PI * (new_mean_s * 0.5).max(1e-6).powi(2) * thickness;
        let new_rho = packing * thickness / new_vol;

        println!("{}:", name);
        println!(
            "  OLD: mu={:.2}, sigma={:.2}, k={:.2} -> rho={:.2}, target_f={:.4}",
            old_mu_exp, old_sigma, old_k, old_rho, target_f_mid
        );
        println!(
            "  NEW: mu={:.2}_f64.ln(), sigma={:.2}, capture_k: {:.2}",
            new_mu_exp, new_sigma, new_k
        );
        println!("  NEW rho={:.2}", new_rho);
    }
}
