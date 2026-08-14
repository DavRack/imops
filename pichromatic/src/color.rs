//! Internal zero-dependency, physics-exact color space transformations and tags.
//!
//! Provides standard color space tags ([`ColorSpaceTag`]), exact CIE chromaticity
//! matrices, Oklab/Oklch perceptual conversions, and piecewise transfer functions.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Supported color space tags across the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorSpaceTag {
    #[serde(alias = "linearsrgb", alias = "linear_srgb", alias = "linear-srgb")]
    LinearSrgb,
    #[serde(alias = "srgb")]
    Srgb,
    #[serde(alias = "acescg", alias = "aces_cg", alias = "aces-cg")]
    AcesCg,
    #[serde(alias = "oklab")]
    Oklab,
    #[serde(alias = "oklch")]
    Oklch,
    #[serde(alias = "xyzd65", alias = "xyz_d65", alias = "xyz-d65", alias = "xyz", alias = "XYZ")]
    XyzD65,
    #[serde(alias = "displayp3", alias = "display_p3", alias = "display-p3", alias = "p3", alias = "P3")]
    DisplayP3,
    #[serde(alias = "linearp3", alias = "linear_p3", alias = "linear-p3")]
    LinearP3,
    #[serde(alias = "rec2020", alias = "rec_2020", alias = "rec-2020", alias = "bt2020")]
    Rec2020,
    #[serde(alias = "linearrec2020", alias = "linear_rec2020", alias = "linear-rec2020", alias = "linearbt2020")]
    LinearRec2020,
}

impl fmt::Display for ColorSpaceTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinearSrgb => write!(f, "LinearSrgb"),
            Self::Srgb => write!(f, "Srgb"),
            Self::AcesCg => write!(f, "AcesCg"),
            Self::Oklab => write!(f, "Oklab"),
            Self::Oklch => write!(f, "Oklch"),
            Self::XyzD65 => write!(f, "XyzD65"),
            Self::DisplayP3 => write!(f, "DisplayP3"),
            Self::LinearP3 => write!(f, "LinearP3"),
            Self::Rec2020 => write!(f, "Rec2020"),
            Self::LinearRec2020 => write!(f, "LinearRec2020"),
        }
    }
}

impl FromStr for ColorSpaceTag {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let clean = s.trim().to_lowercase().replace(['-', '_', ' '], "");
        match clean.as_str() {
            "linearsrgb" => Ok(Self::LinearSrgb),
            "srgb" => Ok(Self::Srgb),
            "acescg" => Ok(Self::AcesCg),
            "oklab" => Ok(Self::Oklab),
            "oklch" => Ok(Self::Oklch),
            "xyzd65" | "xyz" => Ok(Self::XyzD65),
            "displayp3" | "p3" => Ok(Self::DisplayP3),
            "linearp3" => Ok(Self::LinearP3),
            "rec2020" | "bt2020" => Ok(Self::Rec2020),
            "linearrec2020" | "linearbt2020" => Ok(Self::LinearRec2020),
            _ => Err(format!("Unknown color space: {}", s)),
        }
    }
}

// ─── Transformation Matrices (f32) ──────────────────────────────────────────

/// Linear sRGB (Rec. 709 primaries, D65 white) -> CIE 1931 XYZ (D65)
pub const LINEAR_SRGB_TO_XYZ_D65: [[f32; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

/// CIE 1931 XYZ (D65) -> Linear sRGB
pub const XYZ_D65_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

/// ACEScg (AP1 primaries, AMPAS) -> CIE 1931 XYZ (D65)
pub const ACESCG_TO_XYZ_D65: [[f32; 3]; 3] = [
    [0.66245418, 0.13400421, 0.15618769],
    [0.27222872, 0.67408177, 0.05368952],
    [-0.00557465, 0.00406073, 1.01033910],
];

/// CIE 1931 XYZ (D65) -> ACEScg (AP1, AMPAS)
pub const XYZ_D65_TO_ACESCG: [[f32; 3]; 3] = [
    [1.6410234, -0.3248033, -0.2364247],
    [-0.6636629, 1.6153316, 0.01675635],
    [0.01172189, -0.00828442, 0.98839486],
];

/// Direct ACEScg -> Linear sRGB
pub const ACESCG_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
    [1.7309784, -0.6039468, -0.0800931],
    [-0.1316219, 1.1348677, -0.0086794],
    [-0.0245741, -0.1257807, 1.0658931],
];

/// Direct Linear sRGB -> ACEScg
pub const LINEAR_SRGB_TO_ACESCG: [[f32; 3]; 3] = [
    [0.6032027, 0.3263271, 0.0479840],
    [0.0701291, 0.9198951, 0.0127605],
    [0.0221818, 0.1160757, 0.9407929],
];

/// Linear Display P3 (DCI-P3 primaries, D65 white) -> CIE 1931 XYZ (D65)
pub const LINEAR_P3_TO_XYZ_D65: [[f32; 3]; 3] = [
    [0.48657094, 0.26566769, 0.19821729],
    [0.22897456, 0.69173934, 0.07928610],
    [0.00000000, 0.04511330, 1.04394437],
];

/// CIE 1931 XYZ (D65) -> Linear Display P3
pub const XYZ_D65_TO_LINEAR_P3: [[f32; 3]; 3] = [
    [2.49349691, -0.93138362, -0.40271078],
    [-0.82948897, 1.76266406, 0.02362524],
    [0.03584583, -0.07617239, 0.95688452],
];

/// Linear Rec.2020 (ITU-R BT.2020 primaries, D65 white) -> CIE 1931 XYZ (D65)
pub const LINEAR_REC2020_TO_XYZ_D65: [[f32; 3]; 3] = [
    [0.63695805, 0.14461690, 0.16888097],
    [0.26270021, 0.67799807, 0.05930172],
    [0.00000000, 0.02807269, 1.06098506],
];

/// CIE 1931 XYZ (D65) -> Linear Rec.2020
pub const XYZ_D65_TO_LINEAR_REC2020: [[f32; 3]; 3] = [
    [1.71665119, -0.35567078, -0.25336628],
    [-0.66668435, 1.61648124, 0.01576855],
    [0.01763986, -0.04277061, 0.94210312],
];

// ─── Oklab Matrices (Björn Ottosson, 2020) ───────────────────────────────────

/// Linear sRGB -> LMS cone responses
pub const OKLAB_SRGB_TO_LMS: [[f32; 3]; 3] = [
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
];

/// Non-linear LMS -> Oklab (L, a, b)
pub const OKLAB_LMS_TO_LAB: [[f32; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

/// Oklab (L, a, b) -> Non-linear LMS
pub const OKLAB_LAB_TO_LMS: [[f32; 3]; 3] = [
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
];

/// Linear LMS -> Linear sRGB
pub const OKLAB_LMS_TO_SRGB: [[f32; 3]; 3] = [
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010],
];

// ─── Matrix Multiplication ──────────────────────────────────────────────────

/// Multiplies a 3x3 matrix by a 3-element vector: `M * v`.
#[inline(always)]
pub fn mat3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0].mul_add(v[0], m[0][1].mul_add(v[1], m[0][2] * v[2])),
        m[1][0].mul_add(v[0], m[1][1].mul_add(v[1], m[1][2] * v[2])),
        m[2][0].mul_add(v[0], m[2][1].mul_add(v[1], m[2][2] * v[2])),
    ]
}

// ─── Transfer Functions ─────────────────────────────────────────────────────

/// sRGB EOTF (electro-optical transfer function: encoded non-linear sRGB -> linear).
#[inline]
pub fn srgb_eotf_f32(c: f32) -> f32 {
    let abs_c = c.abs();
    let decoded = if abs_c <= 0.04045 {
        abs_c / 12.92
    } else {
        ((abs_c + 0.055) / 1.055).powf(2.4)
    };
    if c >= 0.0 { decoded } else { -decoded }
}

/// sRGB OETF (opto-electronic transfer function: linear -> encoded non-linear sRGB).
#[inline]
pub fn srgb_oetf_f32(c: f32) -> f32 {
    let abs_c = c.abs();
    let encoded = if abs_c <= 0.0031308 {
        12.92 * abs_c
    } else {
        1.055 * abs_c.powf(1.0 / 2.4) - 0.055
    };
    if c >= 0.0 { encoded } else { -encoded }
}

/// ITU-R BT.2020 OETF (linear -> non-linear BT.2020).
#[inline]
pub fn rec2020_oetf_f32(c: f32) -> f32 {
    const ALPHA: f32 = 1.0992968;
    const BETA: f32 = 0.01805397;
    let abs_c = c.abs();
    let encoded = if abs_c < BETA {
        4.5 * abs_c
    } else {
        ALPHA * abs_c.powf(0.45) - (ALPHA - 1.0)
    };
    if c >= 0.0 { encoded } else { -encoded }
}

/// ITU-R BT.2020 EOTF (non-linear BT.2020 -> linear).
#[inline]
pub fn rec2020_eotf_f32(c: f32) -> f32 {
    const ALPHA: f32 = 1.0992968;
    const BETA: f32 = 0.01805397;
    let abs_c = c.abs();
    let decoded = if abs_c < 4.5 * BETA {
        abs_c / 4.5
    } else {
        ((abs_c + ALPHA - 1.0) / ALPHA).powf(1.0 / 0.45)
    };
    if c >= 0.0 { decoded } else { -decoded }
}

// ─── Oklab and Oklch Transformations ────────────────────────────────────────

/// Converts Linear sRGB to Oklab `[L, a, b]`.
#[inline]
pub fn linear_srgb_to_oklab(rgb: [f32; 3]) -> [f32; 3] {
    let lms = mat3_mul(&OKLAB_SRGB_TO_LMS, rgb);
    let lms_c = [lms[0].cbrt(), lms[1].cbrt(), lms[2].cbrt()];
    mat3_mul(&OKLAB_LMS_TO_LAB, lms_c)
}

/// Converts Oklab `[L, a, b]` to Linear sRGB.
#[inline]
pub fn oklab_to_linear_srgb(lab: [f32; 3]) -> [f32; 3] {
    let lms_c = mat3_mul(&OKLAB_LAB_TO_LMS, lab);
    let lms = [
        lms_c[0] * lms_c[0] * lms_c[0],
        lms_c[1] * lms_c[1] * lms_c[1],
        lms_c[2] * lms_c[2] * lms_c[2],
    ];
    mat3_mul(&OKLAB_LMS_TO_SRGB, lms)
}

/// Converts Oklab `[L, a, b]` to Oklch `[L, C, h]` (hue `h` in degrees `[0, 360)`).
#[inline]
pub fn oklab_to_oklch(lab: [f32; 3]) -> [f32; 3] {
    let [l, a, b] = lab;
    let c = (a * a + b * b).sqrt();
    let mut h = b.atan2(a).to_degrees();
    if h < 0.0 {
        h += 360.0;
    }
    [l, c, h]
}

/// Converts Oklch `[L, C, h]` (hue `h` in degrees `[0, 360)`) to Oklab `[L, a, b]`.
#[inline]
pub fn oklch_to_oklab(lch: [f32; 3]) -> [f32; 3] {
    let [l, c, h] = lch;
    let h_rad = h.to_radians();
    let a = c * h_rad.cos();
    let b = c * h_rad.sin();
    [l, a, b]
}

// ─── Linear Conversions ─────────────────────────────────────────────────────

#[inline]
fn linear_to_xyz_d65(space: ColorSpaceTag, pixel: [f32; 3]) -> [f32; 3] {
    match space {
        ColorSpaceTag::LinearSrgb => mat3_mul(&LINEAR_SRGB_TO_XYZ_D65, pixel),
        ColorSpaceTag::AcesCg => mat3_mul(&ACESCG_TO_XYZ_D65, pixel),
        ColorSpaceTag::LinearP3 => mat3_mul(&LINEAR_P3_TO_XYZ_D65, pixel),
        ColorSpaceTag::LinearRec2020 => mat3_mul(&LINEAR_REC2020_TO_XYZ_D65, pixel),
        ColorSpaceTag::XyzD65 => pixel,
        _ => unreachable!("expected a linear space"),
    }
}

#[inline]
fn xyz_d65_to_linear(target: ColorSpaceTag, xyz: [f32; 3]) -> [f32; 3] {
    match target {
        ColorSpaceTag::LinearSrgb => mat3_mul(&XYZ_D65_TO_LINEAR_SRGB, xyz),
        ColorSpaceTag::AcesCg => mat3_mul(&XYZ_D65_TO_ACESCG, xyz),
        ColorSpaceTag::LinearP3 => mat3_mul(&XYZ_D65_TO_LINEAR_P3, xyz),
        ColorSpaceTag::LinearRec2020 => mat3_mul(&XYZ_D65_TO_LINEAR_REC2020, xyz),
        ColorSpaceTag::XyzD65 => xyz,
        _ => unreachable!("expected a linear space"),
    }
}

#[inline]
fn convert_linear(from: ColorSpaceTag, to: ColorSpaceTag, pixel: [f32; 3]) -> [f32; 3] {
    if from == to {
        return pixel;
    }
    match (from, to) {
        (ColorSpaceTag::AcesCg, ColorSpaceTag::LinearSrgb) => mat3_mul(&ACESCG_TO_LINEAR_SRGB, pixel),
        (ColorSpaceTag::LinearSrgb, ColorSpaceTag::AcesCg) => mat3_mul(&LINEAR_SRGB_TO_ACESCG, pixel),
        (ColorSpaceTag::LinearSrgb, ColorSpaceTag::XyzD65) => mat3_mul(&LINEAR_SRGB_TO_XYZ_D65, pixel),
        (ColorSpaceTag::XyzD65, ColorSpaceTag::LinearSrgb) => mat3_mul(&XYZ_D65_TO_LINEAR_SRGB, pixel),
        (ColorSpaceTag::AcesCg, ColorSpaceTag::XyzD65) => mat3_mul(&ACESCG_TO_XYZ_D65, pixel),
        (ColorSpaceTag::XyzD65, ColorSpaceTag::AcesCg) => mat3_mul(&XYZ_D65_TO_ACESCG, pixel),
        (ColorSpaceTag::LinearP3, ColorSpaceTag::XyzD65) => mat3_mul(&LINEAR_P3_TO_XYZ_D65, pixel),
        (ColorSpaceTag::XyzD65, ColorSpaceTag::LinearP3) => mat3_mul(&XYZ_D65_TO_LINEAR_P3, pixel),
        (ColorSpaceTag::LinearRec2020, ColorSpaceTag::XyzD65) => mat3_mul(&LINEAR_REC2020_TO_XYZ_D65, pixel),
        (ColorSpaceTag::XyzD65, ColorSpaceTag::LinearRec2020) => mat3_mul(&XYZ_D65_TO_LINEAR_REC2020, pixel),
        _ => {
            let xyz = linear_to_xyz_d65(from, pixel);
            xyz_d65_to_linear(to, xyz)
        }
    }
}

// ─── Main Conversion Method ─────────────────────────────────────────────────

impl ColorSpaceTag {
    /// Converts a pixel from `self` color space to `target` color space.
    pub fn convert(&self, target: ColorSpaceTag, pixel: [f32; 3]) -> [f32; 3] {
        if *self == target {
            return pixel;
        }

        // Direct Oklab <-> Oklch shortcut
        match (*self, target) {
            (Self::Oklab, Self::Oklch) => return oklab_to_oklch(pixel),
            (Self::Oklch, Self::Oklab) => return oklch_to_oklab(pixel),
            _ => {}
        }

        // 1. Linearize source pixel to its corresponding linear space
        let (lin_space, lin_pixel) = match self {
            Self::LinearSrgb => (Self::LinearSrgb, pixel),
            Self::Srgb => (
                Self::LinearSrgb,
                [srgb_eotf_f32(pixel[0]), srgb_eotf_f32(pixel[1]), srgb_eotf_f32(pixel[2])],
            ),
            Self::AcesCg => (Self::AcesCg, pixel),
            Self::LinearP3 => (Self::LinearP3, pixel),
            Self::DisplayP3 => (
                Self::LinearP3,
                [srgb_eotf_f32(pixel[0]), srgb_eotf_f32(pixel[1]), srgb_eotf_f32(pixel[2])],
            ),
            Self::LinearRec2020 => (Self::LinearRec2020, pixel),
            Self::Rec2020 => (
                Self::LinearRec2020,
                [rec2020_eotf_f32(pixel[0]), rec2020_eotf_f32(pixel[1]), rec2020_eotf_f32(pixel[2])],
            ),
            Self::XyzD65 => (Self::XyzD65, pixel),
            Self::Oklab => (Self::LinearSrgb, oklab_to_linear_srgb(pixel)),
            Self::Oklch => (Self::LinearSrgb, oklab_to_linear_srgb(oklch_to_oklab(pixel))),
        };

        // 2. Direct path if target is Oklab or Oklch
        if matches!(target, Self::Oklab | Self::Oklch) {
            let lin_srgb = convert_linear(lin_space, Self::LinearSrgb, lin_pixel);
            let lab = linear_srgb_to_oklab(lin_srgb);
            return match target {
                Self::Oklab => lab,
                Self::Oklch => oklab_to_oklch(lab),
                _ => unreachable!(),
            };
        }

        // 3. Identify target linear space
        let target_lin_space = match target {
            Self::LinearSrgb | Self::Srgb => Self::LinearSrgb,
            Self::AcesCg => Self::AcesCg,
            Self::LinearP3 | Self::DisplayP3 => Self::LinearP3,
            Self::LinearRec2020 | Self::Rec2020 => Self::LinearRec2020,
            Self::XyzD65 => Self::XyzD65,
            Self::Oklab | Self::Oklch => unreachable!(),
        };

        let target_lin_pixel = convert_linear(lin_space, target_lin_space, lin_pixel);

        // 4. Apply target transfer function / encoding
        match target {
            Self::LinearSrgb | Self::AcesCg | Self::LinearP3 | Self::LinearRec2020 | Self::XyzD65 => {
                target_lin_pixel
            }
            Self::Srgb | Self::DisplayP3 => [
                srgb_oetf_f32(target_lin_pixel[0]),
                srgb_oetf_f32(target_lin_pixel[1]),
                srgb_oetf_f32(target_lin_pixel[2]),
            ],
            Self::Rec2020 => [
                rec2020_oetf_f32(target_lin_pixel[0]),
                rec2020_oetf_f32(target_lin_pixel[1]),
                rec2020_oetf_f32(target_lin_pixel[2]),
            ],
            Self::Oklab | Self::Oklch => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_conversions() {
        let p = [0.35, 0.62, 0.18];
        for cs in [
            ColorSpaceTag::LinearSrgb,
            ColorSpaceTag::Srgb,
            ColorSpaceTag::AcesCg,
            ColorSpaceTag::Oklab,
            ColorSpaceTag::Oklch,
            ColorSpaceTag::XyzD65,
            ColorSpaceTag::DisplayP3,
            ColorSpaceTag::LinearP3,
            ColorSpaceTag::Rec2020,
            ColorSpaceTag::LinearRec2020,
        ] {
            let out = cs.convert(cs, p);
            assert_eq!(out, p, "Identity failed for {:?}", cs);
        }
    }

    #[test]
    fn test_srgb_transfer_roundtrip() {
        for &v in &[0.0, 0.001, 0.0031308, 0.01, 0.04045, 0.18, 0.5, 1.0, -0.5] {
            let encoded = srgb_oetf_f32(v);
            let decoded = srgb_eotf_f32(encoded);
            assert!((decoded - v).abs() < 1e-5, "srgb roundtrip failed for {}: got {}", v, decoded);
        }
    }

    #[test]
    fn test_rec2020_transfer_roundtrip() {
        for &v in &[0.0, 0.001, 0.01805397, 0.05, 0.18, 0.5, 1.0, -0.5] {
            let encoded = rec2020_oetf_f32(v);
            let decoded = rec2020_eotf_f32(encoded);
            assert!((decoded - v).abs() < 1e-5, "rec2020 roundtrip failed for {}: got {}", v, decoded);
        }
    }

    #[test]
    fn test_oklab_oklch_roundtrip() {
        let lab = [0.65, 0.12, -0.08];
        let lch = oklab_to_oklch(lab);
        let lab_back = oklch_to_oklab(lch);
        assert!((lab[0] - lab_back[0]).abs() < 1e-5);
        assert!((lab[1] - lab_back[1]).abs() < 1e-5);
        assert!((lab[2] - lab_back[2]).abs() < 1e-5);
    }

    #[test]
    fn test_srgb_oklab_roundtrip() {
        let rgb = [0.4, 0.7, 0.2];
        let lab = linear_srgb_to_oklab(rgb);
        let rgb_back = oklab_to_linear_srgb(lab);
        assert!((rgb[0] - rgb_back[0]).abs() < 1e-4);
        assert!((rgb[1] - rgb_back[1]).abs() < 1e-4);
        assert!((rgb[2] - rgb_back[2]).abs() < 1e-4);
    }

    #[test]
    fn test_acescg_linear_srgb_roundtrip() {
        let p = [0.5, 0.3, 0.8];
        let srgb = ColorSpaceTag::AcesCg.convert(ColorSpaceTag::LinearSrgb, p);
        let acescg = ColorSpaceTag::LinearSrgb.convert(ColorSpaceTag::AcesCg, srgb);
        assert!((p[0] - acescg[0]).abs() < 1e-4);
        assert!((p[1] - acescg[1]).abs() < 1e-4);
        assert!((p[2] - acescg[2]).abs() < 1e-4);
    }

    #[test]
    fn test_acescg_srgb_roundtrip() {
        let p = [0.5, 0.3, 0.8];
        let srgb = ColorSpaceTag::AcesCg.convert(ColorSpaceTag::Srgb, p);
        let acescg = ColorSpaceTag::Srgb.convert(ColorSpaceTag::AcesCg, srgb);
        assert!((p[0] - acescg[0]).abs() < 1e-4);
        assert!((p[1] - acescg[1]).abs() < 1e-4);
        assert!((p[2] - acescg[2]).abs() < 1e-4);
    }

    #[test]
    fn test_from_str_and_display() {
        let tag = ColorSpaceTag::AcesCg;
        assert_eq!(tag.to_string(), "AcesCg");
        assert_eq!(ColorSpaceTag::from_str("AcesCg").unwrap(), ColorSpaceTag::AcesCg);
        assert_eq!(ColorSpaceTag::from_str("acescg").unwrap(), ColorSpaceTag::AcesCg);
        assert_eq!(ColorSpaceTag::from_str("aces_cg").unwrap(), ColorSpaceTag::AcesCg);
        assert_eq!(ColorSpaceTag::from_str("linear-srgb").unwrap(), ColorSpaceTag::LinearSrgb);
        assert_eq!(ColorSpaceTag::from_str("rec2020").unwrap(), ColorSpaceTag::Rec2020);
        assert_eq!(ColorSpaceTag::from_str("p3").unwrap(), ColorSpaceTag::DisplayP3);
    }
}
