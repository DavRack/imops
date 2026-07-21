//! ACEScg RGB → 16-sample spectral radiance proxy (Meng-style smooth basis).
//!
//! Method: non-negative combination of three smooth Gaussian basis spectra whose
//! integrated XYZ (via CIE 1931 2° CMFs) form an invertible 3×3 map from basis
//! weights → ACEScg. Closed-form per pixel: solve the 3×3 system, clamp small
//! negatives to zero (Meng-style non-negativity), no iterative solver in the hot path.
//!
//! Basis: Gaussians peaked near ~450 / ~550 / ~650 nm (smooth reflectance class).
//! CMF source: CIE 1931 2° standard observer, subsampled at 400–700 nm step 20 nm
//! (CIE 15 / ASTM E308 tabulated rows).
//! XYZ↔ACEScg matrices: Academy Color Encoding System AP1 / ACEScg (AMPAS).

use crate::film::spectrum::{SpectralCurve, WavelengthGrid};

/// CIE 1931 2° x̄ at λ = 400 + 20k nm (k = 0..15).
/// Source: CIE 1931 standard observer (CIE 15 tabulated values at exact 20 nm rows).
pub const CIE1931_XBAR: [f64; 16] = [
    0.01431, 0.13438, 0.34828, 0.29080, 0.09564, 0.00490, 0.06327, 0.29040, 0.59450, 0.91630,
    1.06220, 0.85445, 0.44790, 0.16490, 0.04677, 0.01136,
];
/// CIE 1931 2° ȳ (same sampling). Source: CIE 15.
pub const CIE1931_YBAR: [f64; 16] = [
    0.000396, 0.004000, 0.023000, 0.060000, 0.139020, 0.323000, 0.710000, 0.954000, 0.995000,
    0.870000, 0.631000, 0.381000, 0.175000, 0.061000, 0.017000, 0.004100,
];
/// CIE 1931 2° z̄ (same sampling). Source: CIE 15.
pub const CIE1931_ZBAR: [f64; 16] = [
    0.06785, 0.64560, 1.74706, 1.66920, 0.81295, 0.27200, 0.07825, 0.02030, 0.00390, 0.00165,
    0.00080, 0.00019, 0.00002, 0.00000, 0.00000, 0.00000,
];

/// XYZ (D65) → ACEScg (AP1). Source: Academy ACES AP1 specification (AMPAS).
/// Row-major: out_i = Σ_j M[i][j] * xyz_j.
pub const XYZ_D65_TO_ACESCG: [[f64; 3]; 3] = [
    [1.6410233797, -0.3248032942, -0.2364246952],
    [-0.6636628587, 1.6153315917, 0.0167563477],
    [0.0117218943, -0.0082844165, 0.9883948585],
];

/// ACEScg → XYZ (D65). Source: Academy ACES AP1 specification (AMPAS).
pub const ACESCG_TO_XYZ_D65: [[f64; 3]; 3] = [
    [0.6624541811, 0.1340042065, 0.1561876870],
    [0.2722287168, 0.6740817658, 0.0536895174],
    [-0.0055746495, 0.0040607335, 1.0103391003],
];

fn matvec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn inv3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let id = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
        ],
    ]
}

fn gaussian_basis(peak_nm: f64, sigma_nm: f64) -> [f64; 16] {
    let grid = WavelengthGrid::mvp();
    let mut s = [0.0f64; 16];
    for (i, &lambda) in grid.wavelengths_nm.iter().enumerate() {
        let d = (lambda - peak_nm) / sigma_nm;
        s[i] = (-0.5 * d * d).exp();
    }
    s
}

fn integrate_cmf(spectrum: &[f64; 16], cmf: &[f64; 16]) -> f64 {
    // Trapezoidal over Δλ = 20 nm.
    let dlambda = 20.0;
    let mut acc = 0.0;
    for i in 0..15 {
        let a = spectrum[i] * cmf[i];
        let b = spectrum[i + 1] * cmf[i + 1];
        acc += 0.5 * (a + b) * dlambda;
    }
    acc
}

fn spectrum_to_xyz(spectrum: &[f64; 16]) -> [f64; 3] {
    [
        integrate_cmf(spectrum, &CIE1931_XBAR),
        integrate_cmf(spectrum, &CIE1931_YBAR),
        integrate_cmf(spectrum, &CIE1931_ZBAR),
    ]
}

fn spectrum_to_acescg(spectrum: &[f64; 16]) -> [f64; 3] {
    matvec(XYZ_D65_TO_ACESCG, spectrum_to_xyz(spectrum))
}

/// Precomputed basis spectra and ACEScg←weights matrix inverse.
struct UpsampleBasis {
    b0: [f64; 16],
    b1: [f64; 16],
    b2: [f64; 16],
    /// Maps ACEScg → basis weights: w = M_inv * rgb.
    acescg_to_weights: [[f64; 3]; 3],
}

impl UpsampleBasis {
    fn new() -> Self {
        // Peaks near short / middle / long; σ chosen for smooth overlap (Meng-style).
        let b0 = gaussian_basis(450.0, 40.0);
        let b1 = gaussian_basis(550.0, 40.0);
        let b2 = gaussian_basis(650.0, 40.0);
        // Columns of M are ACEScg of each unit-weight basis.
        let c0 = spectrum_to_acescg(&b0);
        let c1 = spectrum_to_acescg(&b1);
        let c2 = spectrum_to_acescg(&b2);
        let m = [
            [c0[0], c1[0], c2[0]],
            [c0[1], c1[1], c2[1]],
            [c0[2], c1[2], c2[2]],
        ];
        Self {
            b0,
            b1,
            b2,
            acescg_to_weights: inv3(m),
        }
    }

    fn upsample(&self, rgb: [f64; 3]) -> [f64; 16] {
        let mut w = matvec(self.acescg_to_weights, rgb);
        // Non-negative clamp (Meng-style). Small negatives from gamut/rounding → 0.
        for wi in &mut w {
            if *wi < 0.0 {
                *wi = 0.0;
            }
        }
        let mut s = [0.0f64; 16];
        for i in 0..16 {
            s[i] = w[0] * self.b0[i] + w[1] * self.b1[i] + w[2] * self.b2[i];
        }
        s
    }
}

fn basis() -> &'static UpsampleBasis {
    use std::sync::OnceLock;
    static BASIS: OnceLock<UpsampleBasis> = OnceLock::new();
    BASIS.get_or_init(UpsampleBasis::new)
}

/// Upsample one ACEScg pixel to a 16-sample spectral radiance proxy (≥0).
pub fn upsample_acescg(rgb: [f32; 3]) -> [f64; 16] {
    basis().upsample([rgb[0] as f64, rgb[1] as f64, rgb[2] as f64])
}

/// Upsample to a [`SpectralCurve`] on the MVP grid.
pub fn upsample_to_curve(rgb: [f32; 3]) -> SpectralCurve {
    let samples = upsample_acescg(rgb).to_vec();
    SpectralCurve::new(WavelengthGrid::mvp(), samples)
}

/// Integrate a spectrum against CIE CMFs → XYZ → ACEScg (for round-trip tests / scan).
pub fn spectrum_to_acescg_rgb(spectrum: &[f64]) -> [f64; 3] {
    assert_eq!(spectrum.len(), 16);
    let mut s = [0.0f64; 16];
    s.copy_from_slice(spectrum);
    spectrum_to_acescg(&s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::MIDDLE_GRAY;

    #[test]
    fn upsample_mid_gray_nonnegative() {
        let g = MIDDLE_GRAY;
        let s = upsample_acescg([g, g, g]);
        for (i, &v) in s.iter().enumerate() {
            assert!(v >= 0.0, "sample {i} negative: {v}");
        }
    }

    #[test]
    fn upsample_roundtrip_gray() {
        let g = MIDDLE_GRAY as f64;
        let s = upsample_acescg([g as f32, g as f32, g as f32]);
        let rgb = spectrum_to_acescg_rgb(&s);
        for (i, &c) in rgb.iter().enumerate() {
            assert!(
                (c - g).abs() < 0.02,
                "channel {i}: got {c}, expected ~{g}"
            );
        }
    }

    #[test]
    fn upsample_primary_peak_order() {
        // Saturated red-ish ACEScg: more energy at long λ than short λ.
        let s = upsample_acescg([0.8, 0.05, 0.02]);
        let short: f64 = s[0..4].iter().sum(); // 400–460
        let long: f64 = s[12..16].iter().sum(); // 640–700
        assert!(
            long > short,
            "red-ish should favour long λ: short={short}, long={long}"
        );
    }
}
