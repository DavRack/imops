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
