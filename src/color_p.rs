use core::{any::TypeId, f32};

#[inline]
fn matmul(m: &[[f32; 3]; 3], x: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2],
        m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2],
        m[2][0] * x[0] + m[2][1] * x[1] + m[2][2] * x[2],
    ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum ColorSpaceTag {
    LinearSrgb = 1,
    Oklab = 5,
    Oklch = 6,
    XyzD65 = 14,
    // NOTICE: If a new value is added, be sure to modify `MAX_VALUE` in the bytemuck impl. Also
    // note the variants' integer values are not necessarily in order, allowing newly added color
    // space tags to be grouped with related color spaces.
}

/// The layout of a color space, particularly the hue component.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum ColorSpaceLayout {
    /// Rectangular, no hue component.
    Rectangular,
    /// Cylindrical, hue is first component.
    HueFirst,
    /// Cylindrical, hue is third component.
    HueThird,
}

pub trait ColorSpace: Clone + Copy + 'static {
    /// Whether the color space is linear.
    ///
    /// Calculations in linear color spaces can sometimes be simplified,
    /// for example it is not necessary to undo premultiplication when
    /// converting.
    const IS_LINEAR: bool = false;

    /// The layout of the color space.
    ///
    /// The layout primarily identifies the hue channel for cylindrical
    /// color spaces, which is important because hue is not premultiplied.
    const LAYOUT: ColorSpaceLayout = ColorSpaceLayout::Rectangular;

    /// The tag corresponding to this color space, if a matching tag exists.
    const TAG: Option<ColorSpaceTag> = None;

    /// The component values for the color white within this color space.
    const WHITE_COMPONENTS: [f32; 3];

    /// Convert an opaque color to linear sRGB.
    ///
    /// Values are likely to exceed [0, 1] for wide-gamut and HDR colors.
    fn to_xyz(src: [f32; 3]) -> [f32; 3];

    /// Convert an opaque color from linear sRGB.
    ///
    /// In general, this method should not do any gamut clipping.
    fn from_xyz(src: [f32; 3]) -> [f32; 3];

    /// Scale the chroma by the given amount.
    ///
    /// In color spaces with a natural representation of chroma, scale
    /// directly. In other color spaces, equivalent results as scaling
    /// chroma in Oklab.
    // fn scale_chroma(src: [f32; 3], scale: f32) -> [f32; 3] {
    //     let rgb = Self::to_linear_srgb(src);
    //     let scaled = LinearSrgb::scale_chroma(rgb, scale);
    //     Self::from_linear_srgb(scaled)
    // }

    /// Convert to a different color space.
    ///
    /// The default implementation is a no-op if the color spaces
    /// are the same, otherwise converts from the source to linear
    /// sRGB, then from that to the target. Implementations are
    /// encouraged to specialize further (using the [`TypeId`] of
    /// the color spaces), effectively finding a shortest path in
    /// the conversion graph.
    fn convert<TargetCS: ColorSpace>(src: [f32; 3]) -> [f32; 3] {
        if TypeId::of::<Self>() == TypeId::of::<TargetCS>() {
            src
        } else {
            let lin_rgb = Self::to_xyz(src);
            TargetCS::from_xyz(lin_rgb)
        }
    }

    /// Clip the color's components to fit within the natural gamut of the color space.
    ///
    /// There are many possible ways to map colors outside of a color space's gamut to colors
    /// inside the gamut. Some methods are perceptually better than others (for example, preserving
    /// the mapped color's hue is usually preferred over preserving saturation). This method will
    /// generally do the mathematically simplest thing, namely clamping the individual color
    /// components' values to the color space's natural limits of those components, bringing
    /// out-of-gamut colors just onto the gamut boundary. The resultant color may be perceptually
    /// quite distinct from the original color.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use color::{ColorSpace, Srgb, XyzD65};
    ///
    /// assert_eq!(Srgb::clip([0.4, -0.2, 1.2]), [0.4, 0., 1.]);
    /// assert_eq!(XyzD65::clip([0.4, -0.2, 1.2]), [0.4, -0.2, 1.2]);
    /// ```
    fn clip(src: [f32; 3]) -> [f32; 3];
}

#[derive(Clone, Copy, Debug)]
pub struct XyzD65;

impl ColorSpace for XyzD65 {
    const IS_LINEAR: bool = true;

    const TAG: Option<ColorSpaceTag> = Some(ColorSpaceTag::XyzD65);

    const WHITE_COMPONENTS: [f32; 3] = [3127. / 3290., 1., 3583. / 3290.];

    fn to_xyz(src: [f32; 3]) -> [f32; 3] {
        // const XYZ_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
        //     [3.240_97, -1.537_383_2, -0.498_610_76],
        //     [-0.969_243_65, 1.875_967_5, 0.041_555_06],
        //     [0.055_630_08, -0.203_976_96, 1.056_971_5],
        // ];
        // matmul(&XYZ_TO_LINEAR_SRGB, src)
        src
    }

    fn from_xyz(src: [f32; 3]) -> [f32; 3] {
        // const LINEAR_SRGB_TO_XYZ: [[f32; 3]; 3] = [
        //     [0.412_390_8, 0.357_584_33, 0.180_480_8],
        //     [0.212_639, 0.715_168_65, 0.072_192_32],
        //     [0.019_330_818, 0.119_194_78, 0.950_532_14],
        // ];
        // matmul(&LINEAR_SRGB_TO_XYZ, src)
        src
    }

    fn clip([x, y, z]: [f32; 3]) -> [f32; 3] {
        [x, y, z]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Oklab;

// Matrices taken from [Oklab] blog post, precision reduced to f32
//
// [Oklab]: https://bottosson.github.io/posts/oklab/
const OKLAB_LAB_TO_LMS: [[f32; 3]; 3] = [
    [1.0, 0.396_337_78, 0.215_803_76],
    [1.0, -0.105_561_346, -0.063_854_17],
    [1.0, -0.089_484_18, -1.291_485_5],
];

// const OKLAB_LMS_TO_SRGB: [[f32; 3]; 3] = [
//     [4.076_741_7, -3.307_711_6, 0.230_969_94],
//     [-1.268_438, 2.609_757_4, -0.341_319_38],
//     [-0.004_196_086_3, -0.703_418_6, 1.707_614_7],
// ];

const OKLAB_LMS_TO_XYZ: [[f32; 3]; 3] = [
    [ 0.98232712, -0.88026228,  0.71652658,],
    [ 0.02524606,  1.37215964,  -0.3649593,],
    [-0.11829507, -0.30102463,  1.57160392,],
];

// const OKLAB_SRGB_TO_LMS: [[f32; 3]; 3] = [
//     [0.412_221_46, 0.536_332_55, 0.051_445_995],
//     [0.211_903_5, 0.680_699_5, 0.107_396_96],
//     [0.088_302_46, 0.281_718_85, 0.629_978_7],
// ];
const OKLAB_XYZ_TO_LMS: [[f32; 3]; 3] = [
    [ 0.96619195,  0.55127368, -0.31248951,],
    [ 0.00165046,  0.76884006,  0.17778831,],
    [ 0.07304166,  0.188758,    0.64682497,],
];

const OKLAB_LMS_TO_LAB: [[f32; 3]; 3] = [
    [0.210_454_26, 0.793_617_8, -0.004_072_047],
    [1.977_998_5, -2.428_592_2, 0.450_593_7],
    [0.025_904_037, 0.782_771_77, -0.808_675_77],
];

impl ColorSpace for Oklab {
    const TAG: Option<ColorSpaceTag> = Some(ColorSpaceTag::Oklab);

    const WHITE_COMPONENTS: [f32; 3] = [1., 0., 0.];

    #[inline]
    fn to_xyz(src: [f32; 3]) -> [f32; 3] {
        let lms = matmul(&OKLAB_LAB_TO_LMS, src).map(|x| x.powi(3));
        matmul(&OKLAB_LMS_TO_XYZ, lms)
    }

    #[inline]
    fn from_xyz(src: [f32; 3]) -> [f32; 3] {
        let lms = matmul(&OKLAB_XYZ_TO_LMS, src).map(f32::cbrt);
        matmul(&OKLAB_LMS_TO_LAB, lms)
    }

    // fn scale_chroma([l, a, b]: [f32; 3], scale: f32) -> [f32; 3] {
    //     [l, a * scale, b * scale]
    // }

    fn clip([l, a, b]: [f32; 3]) -> [f32; 3] {
        [l.clamp(0., 1.), a, b]
    }
}

/// Rectangular to cylindrical conversion.
#[inline]
fn lab_to_lch([l, a, b]: [f32; 3]) -> [f32; 3] {
    let mut h = b.atan2(a) * (180. / f32::consts::PI);
    if h < 0.0 {
        h += 360.0;
    }
    let c = b.hypot(a);
    [l, c, h]
}

/// Cylindrical to rectangular conversion.
#[inline]
fn lch_to_lab([l, c, h]: [f32; 3]) -> [f32; 3] {
    let (sin, cos) = (h * (f32::consts::PI / 180.)).sin_cos();
    let a = c * cos;
    let b = c * sin;
    [l, a, b]
}

/// 🌌 The cylindrical version of the [Oklab] color space.
///
/// Its components are `[L, C, h]` with
/// - `L` - the lightness as in [`Oklab`];
/// - `C` - the chromatic intensity, the natural lower bound of 0 being achromatic, usually not
///    exceeding 0.5; and
/// - `h` - the hue angle in degrees.
#[derive(Clone, Copy, Debug)]
pub struct Oklch;

impl ColorSpace for Oklch {
    const TAG: Option<ColorSpaceTag> = Some(ColorSpaceTag::Oklch);

    const LAYOUT: ColorSpaceLayout = ColorSpaceLayout::HueThird;

    const WHITE_COMPONENTS: [f32; 3] = [1., 0., 90.];

    fn from_xyz(src: [f32; 3]) -> [f32; 3] {
        lab_to_lch(Oklab::from_xyz(src))
    }

    fn to_xyz(src: [f32; 3]) -> [f32; 3] {
        Oklab::to_xyz(lch_to_lab(src))
    }

    // fn scale_chroma([l, c, h]: [f32; 3], scale: f32) -> [f32; 3] {
    //     [l, c * scale, h]
    // }

    fn convert<TargetCS: ColorSpace>(src: [f32; 3]) -> [f32; 3] {
        if TypeId::of::<Self>() == TypeId::of::<TargetCS>() {
            src
        } else if TypeId::of::<TargetCS>() == TypeId::of::<Oklab>() {
            lch_to_lab(src)
        } else {
            let lin_rgb = Self::to_xyz(src);
            TargetCS::from_xyz(lin_rgb)
        }
    }

    fn clip([l, c, h]: [f32; 3]) -> [f32; 3] {
        [l.clamp(0., 1.), c.max(0.), h]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LinearSrgb;

// Matrices for XYZ D65 <-> Linear sRGB conversion.
// Using standard values with f32 precision.
const XYZ_D65_TO_LINEAR_SRGB: [[f32; 3]; 3] = [
    [3.240969941904521, -1.537383177570093, -0.498610760293003],
    [-0.969243636280879, 1.87596750150772, 0.041555057407175],
    [0.055630079696993, -0.20397695888897, 1.056971514242879],
];

const LINEAR_SRGB_TO_XYZ_D65: [[f32; 3]; 3] = [
    [0.412390799265959, 0.357584339383878, 0.180480788401834],
    [0.21263900587151, 0.715168678767756, 0.072192315360733],
    [0.019330818715592, 0.119194779794626, 0.950532152249661],
];

impl ColorSpace for LinearSrgb {
    const IS_LINEAR: bool = true;
    const TAG: Option<ColorSpaceTag> = Some(ColorSpaceTag::LinearSrgb);
    const WHITE_COMPONENTS: [f32; 3] = [1.0, 1.0, 1.0];

    fn to_xyz(src: [f32; 3]) -> [f32; 3] {
        matmul(&LINEAR_SRGB_TO_XYZ_D65, src)
    }

    fn from_xyz(src: [f32; 3]) -> [f32; 3] {
        matmul(&XYZ_D65_TO_LINEAR_SRGB, src)
    }

    fn clip(src: [f32; 3]) -> [f32; 3] {
        [src[0].clamp(0.0, 1.0), src[1].clamp(0.0, 1.0), src[2].clamp(0.0, 1.0)]
    }
}

/// Adjusts the saturation of a color in the XYZ D65 color space without converting to another color space.
///
/// This is achieved by moving the color's chromaticity coordinates (x, y) towards or away from
/// the D65 white point in the xy chromaticity diagram, while preserving luminance (Y).
///
/// A `saturation_factor` of 1.0 means no change.
/// A factor greater than 1.0 increases saturation.
/// A factor less than 1.0 decreases saturation.
/// A factor of 0.0 results in a grayscale color with the same luminance.
pub fn adjust_xyz_saturation(xyz: [f32; 3], saturation_factor: f32) -> [f32; 3] {
    let [x, y, z] = xyz;

    // D65 white point chromaticity coordinates
    const XW: f32 = 0.3127;
    const YW: f32 = 0.3290;

    let sum = x + y + z;

    if sum < 1e-5 {
        // It's black, no change
        return xyz;
    }

    // Chromaticity of the color
    let cx = x / sum;
    let cy = y / sum;

    // New chromaticity, interpolated towards/from white point
    let new_cx = XW + saturation_factor * (cx - XW);
    let new_cy = YW + saturation_factor * (cy - YW);

    if new_cy < 1e-5 {
        // Cannot compute, return original color.
        // This can happen if the new chromaticity is on the x-axis of the diagram,
        // which is outside the gamut of real colors.
        return xyz;
    }

    // Calculate new X and Z, preserving Y
    let new_x = (new_cx / new_cy) * y;
    let new_z = ((1.0 - new_cx - new_cy) / new_cy) * y;

    [new_x, y, new_z]
}

/// Checks if a color in the Linear SRGB space is within the gamut (all components in [0, 1]).
pub fn is_in_srgb_gamut(linear_srgb: [f32; 3]) -> bool {
    linear_srgb[0] >= -1e-5 && linear_srgb[0] <= 1.0 + 1e-5 &&
    linear_srgb[1] >= -1e-5 && linear_srgb[1] <= 1.0 + 1e-5 &&
    linear_srgb[2] >= -1e-5 && linear_srgb[2] <= 1.0 + 1e-5
}

pub fn map_xyz_to_srgb_gamut(xyz: [f32; 3]) -> [f32; 3] {
    // First, check if the color is already in gamut.
    if is_in_srgb_gamut(LinearSrgb::from_xyz(xyz)) {
        return xyz;
    }

    // Convert to Oklch to work with chroma and lightness.
    let oklch = Oklch::from_xyz(xyz);
    let l = oklch[0];
    let c = oklch[1];
    let h = oklch[2];

    // Binary search for the intersection `t` on the path to white.
    // The path is a line from the original color (t=0) to white (t=1).
    let mut low_t = 0.0;
    let mut high_t = 1.0;

    for _ in 0..15 { // 15 iterations are enough for f32 precision.
        let mid_t = (low_t + high_t) / 2.0;
        
        let l_t = l * (1.0 - mid_t) + mid_t; // L towards 1.0
        let c_t = c * (1.0 - mid_t);       // C towards 0.0

        let test_oklch = [l_t, c_t, h];
        let test_xyz = Oklch::to_xyz(test_oklch);
        
        if is_in_srgb_gamut(LinearSrgb::from_xyz(test_xyz)) {
            // This point is in gamut, so the boundary is at a smaller `t` (closer to original color).
            // The current `mid_t` is our new best guess for the boundary on the in-gamut side.
            high_t = mid_t;
        } else {
            // This point is out of gamut, so we need to move closer to white (larger `t`).
            low_t = mid_t;
        }
    }

    // After the search, `high_t` is our best approximation for the `t`
    // that lies on the gamut boundary. Calculate the final color using this `t`.
    let final_l = l * (1.0 - high_t) + high_t;
    let final_c = c * (1.0 - high_t);
    let final_oklch = [final_l, final_c, h];
    
    Oklch::to_xyz(final_oklch)
}

// sRGB non-linear transfer functions (gamma correction)
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Performs gamut clipping on a non-linear sRGB color while preserving hue.
///
/// It converts the color to XYZ, maps it to the sRGB gamut using the hue-preserving
/// `map_xyz_to_srgb_gamut` function, and then converts it back to non-linear sRGB.
/// sRGB / Rec.709 Luminance Coefficients.
const SRGB_LUMA_COEFFS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// A simplified implementation of the ACES Reference Gamut Compression (RGC),
/// adapted for the sRGB/Rec.709 primaries. This is a private helper function.
fn gamut_compress_aces_simplified(linear_rgb: [f32; 3], limit: f32, threshold: f32) -> [f32; 3] {
    let [r, g, b] = linear_rgb;

    // 1. Calculate Achromatic Luminance (Y) using sRGB coefficients.
    let ach = r * SRGB_LUMA_COEFFS[0] + 
              g * SRGB_LUMA_COEFFS[1] + 
              b * SRGB_LUMA_COEFFS[2];

    // 2. Calculate "Reach" (Distance from Achromatic Axis)
    let max_val = r.max(g).max(b);
    let distance = max_val - ach;

    if distance <= threshold {
        return linear_rgb;
    }

    // 3. Apply Soft Compression
    let scale_factor = limit - threshold;
    if scale_factor <= 0.0 {
        return linear_rgb;
    }
    let compressed_dist = scale_factor * ((distance - threshold) / scale_factor).tanh();
    
    // 4. Reconstruct the Pixel
    if distance < 1e-6 {
        return linear_rgb;
    }
    let scale = (compressed_dist + threshold) / distance;

    [
        ach + (r - ach) * scale,
        ach + (g - ach) * scale,
        ach + (b - ach) * scale,
    ]
}

/// Performs gamut clipping on a non-linear sRGB color using a simplified
/// ACES "Nugget" reference gamut compression algorithm.
struct HueParams {
    hue_angle: f32,
    threshold: f32,
    limit: f32,
}

// A lookup table for hue-dependent compression parameters.
// These values are illustrative, based on the general shape of the sRGB gamut.
// Using 60-degree increments for simplicity. Red is duplicated for wraparound interpolation.
const HUE_PARAMS: [HueParams; 7] = [
    HueParams { hue_angle: 0.0,   threshold: 0.82, limit: 1.15 }, // Red
    HueParams { hue_angle: 60.0,  threshold: 0.90, limit: 1.10 }, // Yellow
    HueParams { hue_angle: 120.0, threshold: 0.85, limit: 1.10 }, // Green
    HueParams { hue_angle: 180.0, threshold: 0.85, limit: 1.10 }, // Cyan
    HueParams { hue_angle: 240.0, threshold: 0.80, limit: 1.20 }, // Blue
    HueParams { hue_angle: 300.0, threshold: 0.80, limit: 1.20 }, // Magenta
    HueParams { hue_angle: 360.0, threshold: 0.82, limit: 1.15 }, // Red (wraparound)
];

// /// Calculates adaptive compression parameters by interpolating based on hue.
// fn get_adaptive_params(h: f32) -> (f32, f32) {
//     // Find the two hue parameter points the current hue `h` falls between.
//     let (p1, p2) = HUE_PARAMS
//         .windows(2)
//         .find(|win| h >= win[0].hue_angle && h <= win[1].hue_angle)
//         .unwrap_or((&HUE_PARAMS[0], &HUE_PARAMS[1])); // Fallback for safety

//     // Calculate the interpolation factor `t`.
//     let t = (h - p1.hue_angle) / (p2.hue_angle - p1.hue_angle);

//     // Interpolate threshold and limit.
//     let threshold = p1.threshold * (1.0 - t) + p2.threshold * t;
//     let limit = p1.limit * (1.0 - t) + p2.limit * t;

//     (threshold, limit)
// }

// /// Performs gamut clipping on a non-linear sRGB color using a hue-adaptive,
// /// simplified ACES "Nugget" reference gamut compression algorithm.
// pub fn gamut_clip_aces(rgb: [f32; 3]) -> [f32; 3] {
//     // 1. Convert to linear
//     let linear_rgb = [
//         srgb_to_linear(rgb[0]),
//         srgb_to_linear(rgb[1]),
//         srgb_to_linear(rgb[2]),
//     ];

//     // Handle achromatic colors separately for efficiency and to avoid hue calculation on greys.
//     let r = linear_rgb[0];
//     let g = linear_rgb[1];
//     let b = linear_rgb[2];
//     if (r - g).abs() < 1e-5 && (g - b).abs() < 1e-5 {
//         let clipped = r.clamp(0.0, 1.0);
//         let non_linear = linear_to_srgb(clipped);
//         return [non_linear, non_linear, non_linear];
//     }

//     // 2. Get Hue for adaptive parameters
//     let xyz = LinearSrgb::to_xyz(linear_rgb);
//     let oklch = Oklch::from_xyz(xyz);
//     let h = oklch[2];

//     // 3. Get hue-dependent threshold and limit
//     let (threshold, limit) = get_adaptive_params(h);

//     // 4. Apply gamut compression
//     let compressed_linear_rgb = gamut_compress_aces_simplified(linear_rgb, limit, threshold);

//     // 5. Clip to the [0, 1] range, as compression doesn't guarantee the result is perfectly in gamut.
//     let clipped_linear_rgb = [
//         compressed_linear_rgb[0].clamp(0.0, 1.0),
//         compressed_linear_rgb[1].clamp(0.0, 1.0),
//         compressed_linear_rgb[2].clamp(0.0, 1.0),
//     ];

//     // 6. Convert back to non-linear sRGB
//     [
//         linear_to_srgb(clipped_linear_rgb[0]),
//         linear_to_srgb(clipped_linear_rgb[1]),
//         linear_to_srgb(clipped_linear_rgb[2]),
//     ]
// }

