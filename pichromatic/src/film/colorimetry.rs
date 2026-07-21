//! Colorimetric helpers for film tests and invert neutrality (ACEScg → XYZ → Lab).
//!
//! ACEScg↔XYZ matrices: Academy ACES AP1 (AMPAS).
//! CIEDE2000: Sharma / CIE 15 formulation.

fn matvec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// ACEScg → XYZ (D65). Source: Academy ACES AP1 specification (AMPAS).
const ACESCG_TO_XYZ: [[f64; 3]; 3] = [
    [0.6624541811, 0.1340042065, 0.1561876870],
    [0.2722287168, 0.6740817658, 0.0536895174],
    [-0.0055746495, 0.0040607335, 1.0103391003],
];

/// ACEScg → CIE XYZ (D65).
pub fn acescg_to_xyz(rgb: [f64; 3]) -> [f64; 3] {
    matvec(ACESCG_TO_XYZ, rgb)
}

fn lab_f(t: f64) -> f64 {
    let delta: f64 = 6.0 / 29.0;
    if t > delta.powi(3) {
        t.cbrt()
    } else {
        t / (3.0 * delta * delta) + 4.0 / 29.0
    }
}

/// XYZ → CIELAB (D65 white).
pub fn xyz_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    let xn = 0.95047;
    let yn = 1.00000;
    let zn = 1.08883;
    let fx = lab_f(xyz[0] / xn);
    let fy = lab_f(xyz[1] / yn);
    let fz = lab_f(xyz[2] / zn);
    [
        116.0 * fy - 16.0,
        500.0 * (fx - fy),
        200.0 * (fy - fz),
    ]
}

pub fn acescg_to_lab(rgb: [f32; 3]) -> [f64; 3] {
    xyz_to_lab(acescg_to_xyz([rgb[0] as f64, rgb[1] as f64, rgb[2] as f64]))
}

pub fn chroma_ab(lab: [f64; 3]) -> f64 {
    (lab[1] * lab[1] + lab[2] * lab[2]).sqrt()
}

/// CIEDE2000 colour difference (Sharma et al. / CIE 15).
pub fn ciede2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let (l1, a1, b1) = (lab1[0], lab1[1], lab1[2]);
    let (l2, a2, b2) = (lab2[0], lab2[1], lab2[2]);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) / 2.0;
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let h1p = b1.atan2(a1p).to_degrees().rem_euclid(360.0);
    let h2p = b2.atan2(a2p).to_degrees().rem_euclid(360.0);
    let dl = l2 - l1;
    let dc = c2p - c1p;
    let dh = {
        if c1p * c2p < 1e-15 {
            0.0
        } else if (h2p - h1p).abs() <= 180.0 {
            h2p - h1p
        } else if h2p <= h1p {
            h2p - h1p + 360.0
        } else {
            h2p - h1p - 360.0
        }
    };
    let dh_p = 2.0 * (c1p * c2p).sqrt() * (dh.to_radians() / 2.0).sin();
    let l_bar = (l1 + l2) / 2.0;
    let c_bar_p = (c1p + c2p) / 2.0;
    let h_bar = {
        if c1p * c2p < 1e-15 {
            h1p + h2p
        } else if (h1p - h2p).abs() <= 180.0 {
            (h1p + h2p) / 2.0
        } else if h1p + h2p < 360.0 {
            (h1p + h2p + 360.0) / 2.0
        } else {
            (h1p + h2p - 360.0) / 2.0
        }
    };
    let t = 1.0 - 0.17 * (h_bar - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar).to_radians().cos()
        + 0.32 * (3.0 * h_bar + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar - 63.0).to_radians().cos();
    let sl = 1.0 + (0.015 * (l_bar - 50.0).powi(2)) / (20.0 + (l_bar - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let dtheta = 30.0 * (-((h_bar - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25f64.powi(7))).sqrt();
    let rt = -rc * (2.0 * dtheta).to_radians().sin();
    ((dl / sl).powi(2) + (dc / sc).powi(2) + (dh_p / sh).powi(2) + rt * (dc / sc) * (dh_p / sh))
        .sqrt()
}
