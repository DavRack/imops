//! Macbeth ColorChecker–like 24-patch ACEScg fixture for Checkpoint 12.
//!
//! sRGB patch centres from the published ColorChecker Classic chart (Pascale /
//! X-Rite representative values), converted sRGB→ACEScg via the `color` crate
//! path XYZ↔linear. Spatial layout: 6 columns × 4 rows.
//!
//! **These patches feed both the synthetic scene and the ΔE reference**.
//! PositiveLinear is a mid/Dmin densitometric invert (no print-paper curve).
//! There is no independent densitometric characterization of a commercial stock.
use color::ColorSpaceTag;
use crate::image::ImageMetadata;
use crate::pixel::Image;

/// Classic ColorChecker sRGB 8-bit centres (approx), row-major 6×4.
/// Source: published ColorChecker Classic patch sRGB values (X-Rite / BabelColor).
const COLORCHECKER_SRGB_U8: [[u8; 3]; 24] = [
    // Row 1 — natural colours
    [115, 82, 68],
    [194, 150, 130],
    [98, 122, 157],
    [87, 108, 67],
    [133, 128, 177],
    [103, 189, 170],
    // Row 2 — miscellaneous
    [214, 126, 44],
    [80, 91, 166],
    [193, 90, 99],
    [94, 60, 108],
    [157, 188, 64],
    [224, 163, 46],
    // Row 3 — primary/secondary
    [56, 61, 150],
    [70, 148, 73],
    [175, 54, 60],
    [231, 199, 31],
    [187, 86, 149],
    [8, 133, 161],
    // Row 4 — neutrals (19–24)
    [243, 243, 242],
    [200, 200, 200],
    [160, 160, 160],
    [122, 122, 121],
    [85, 85, 85],
    [52, 52, 52],
];

fn srgb_u8_to_acescg(rgb: [u8; 3]) -> [f32; 3] {
    let lin = [
        srgb_to_linear(rgb[0] as f32 / 255.0),
        srgb_to_linear(rgb[1] as f32 / 255.0),
        srgb_to_linear(rgb[2] as f32 / 255.0),
    ];
    // sRGB → XYZ D65 → ACEScg via color crate.
    ColorSpaceTag::Srgb.convert(ColorSpaceTag::AcesCg, lin)
}

fn srgb_to_linear(u: f32) -> f32 {
    if u <= 0.04045 {
        u / 12.92
    } else {
        ((u + 0.055) / 1.055).powf(2.4)
    }
}

/// Reference ACEScg colours for the 24 patches.
pub fn colorchecker_acescg() -> [[f32; 3]; 24] {
    let mut out = [[0.0f32; 3]; 24];
    for i in 0..24 {
        out[i] = srgb_u8_to_acescg(COLORCHECKER_SRGB_U8[i]);
    }
    out
}

/// Build a `cols*patch` × `rows*patch` image with solid ColorChecker patches.
pub fn colorchecker_image(patch: usize) -> (Image, [[f32; 3]; 24]) {
    let cols = 6usize;
    let rows = 4usize;
    let width = cols * patch;
    let height = rows * patch;
    let refs = colorchecker_acescg();
    let mut rgb_data = vec![[0.0f32; 3]; width * height];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let color = refs[idx];
            for y in 0..patch {
                for x in 0..patch {
                    let yy = r * patch + y;
                    let xx = c * patch + x;
                    rgb_data[yy * width + xx] = color;
                }
            }
        }
    }
    let image = Image {
        rgb_data,
        raw_data: vec![],
        metadata: ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::AcesCg),
            ..Default::default()
        },
    };
    (image, refs)
}

/// Mean ACEScg of each patch from a processed ColorChecker image.
pub fn sample_patch_means(image: &Image, patch: usize) -> [[f32; 3]; 24] {
    let cols = 6usize;
    let rows = 4usize;
    let width = image.metadata.width;
    let mut means = [[0.0f32; 3]; 24];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let mut acc = [0.0f64; 3];
            let mut n = 0usize;
            // Sample centre 50% of patch to avoid edge bleed.
            let margin = patch / 4;
            for y in margin..(patch - margin).max(margin + 1) {
                for x in margin..(patch - margin).max(margin + 1) {
                    let yy = r * patch + y;
                    let xx = c * patch + x;
                    let px = image.rgb_data[yy * width + xx];
                    acc[0] += px[0] as f64;
                    acc[1] += px[1] as f64;
                    acc[2] += px[2] as f64;
                    n += 1;
                }
            }
            let n = n.max(1) as f64;
            means[idx] = [(acc[0] / n) as f32, (acc[1] / n) as f32, (acc[2] / n) as f32];
        }
    }
    means
}
