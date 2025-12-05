use std::{any, usize};
use std::time::Instant;


use color::{ColorSpace, Oklch, Srgb, XyzD65, AcesCg};
use rawler::{imgop::xyz::Illuminant, pixarray::RgbF32, RawImage};
// use sealed::Cache;
use serde::{Deserialize, Serialize};
use toml::map::Map;
use crate::chromaCompresion::RgcParams;
// use crate::color_p::gamut_clip_aces;
use crate::mask::{Mask};
use crate::tone_mapping::sigmoid;
use crate::{chromaCompresion, helpers::*, pixels, tone_mapping};
use crate::pixels::*;
use crate::demosaic;

use crate::conditional_paralell::prelude::*;

const R: usize = 0;
const G: usize = 1;
const B: usize = 2;

// working color space
type WCS = AcesCg;

pub trait GenericModule {
    fn set_cache(&mut self, cache: PipelineImage);
    fn get_cache(&self) -> Option<PipelineImage>;
    fn get_name(&self) -> String;
    fn get_mask(&self) -> Option<Box<dyn Mask>>;
}

pub trait PipelineModule: GenericModule{
    fn process(&self, image: PipelineImage, raw_image: &RawImage) -> PipelineImage;
}

impl<T> GenericModule for Module<T> {
    fn set_cache(&mut self, cache: PipelineImage) {
        self.cache = Some(cache)
    }
    fn get_cache(&self) -> Option<PipelineImage>{
        match &self.cache {
            Some(cache) => Some(cache.clone()),
            None => None,
        }
    }
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn get_mask(&self) -> Option<Box<dyn Mask>> {
        if let Some(v) = &self.mask{
            let a = dyn_clone::clone_box(&**v);
            return Some(a)
        }else {
            return None
        }
    }
}

#[derive(Clone)]
pub struct FormedImage {
    pub raw_image: RawImage,
    pub data: RgbF32,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Default)]
pub struct PipelineImage {
    pub data: pixels::ImageBuffer,
    pub height: usize,
    pub width: usize,
    pub max_raw_value: SubPixel,
}

pub struct Module<T>{
    pub name: String,
    pub cache: Option<PipelineImage>,
    pub config: T,
    pub mask: Option<Box<dyn Mask>>
}

impl<T> Module<T>{
    pub fn from_toml<'a>(module: Map<String, toml::Value>) -> Box<Self>
    where
        T: Deserialize<'a> + Default,
        Self: Sized
    {
        let cfg: T = module.clone().try_into::<T>().expect(any::type_name::<Self>());
        let module = Module{
            name: module["name"].to_string(),
            cache: None,
            config: cfg,
            mask: None
        };
        Box::new(module)
    }
}

pub fn subtractive_saturation_acescg(rgb: [f32; 3], saturation: f32) -> [f32; 3] {
    // 1. ACEScg Luminance Weights (AP1 Primaries)
    // These are necessary to determine the "Achromatic Axis" (Visual Neutral).
    const LUMA_WEIGHTS: [f32; 3] = [0.2722287168, 0.6740817658, 0.0536895174];

    // 2. Safety Clamp
    // We cannot take the log of 0 or negative numbers.
    // 1e-5 prevents NaNs while preserving deep blacks.
    let r = rgb[0].max(1e-5);
    let g = rgb[1].max(1e-5);
    let b = rgb[2].max(1e-5);

    // 3. Convert Linear Transmission to Optical Density (The "CMY" conversion)
    // D = -ln(T). Higher density = More dye = Less light.
    let d_r = -r.ln(); // Density of Cyan (absorbs Red)
    let d_g = -g.ln(); // Density of Magenta (absorbs Green)
    let d_b = -b.ln(); // Density of Yellow (absorbs Blue)

    // 4. Calculate Achromatic Density (The "Grey" point)
    // We use a weighted average based on ACEScg luminance to preserve perceptual brightness.
    // Note: In pure chemical physics, this might be an unweighted average, 
    // but for image processing, weighting prevents hue shifts in perceived lightness.
    let d_achromatic = (d_r * LUMA_WEIGHTS[0]) + 
                       (d_g * LUMA_WEIGHTS[1]) + 
                       (d_b * LUMA_WEIGHTS[2]);

    // 5. Expand Density (Apply Saturation)
    // We expand the distance of each channel's density from the neutral axis.
    let d_r_sat = d_achromatic + (d_r - d_achromatic) * saturation;
    let d_g_sat = d_achromatic + (d_g - d_achromatic) * saturation;
    let d_b_sat = d_achromatic + (d_b - d_achromatic) * saturation;

    // 6. Convert Density back to Linear Transmission
    // T = e^(-D)
    [
        (-d_r_sat).exp(),
        (-d_g_sat).exp(),
        (-d_b_sat).exp()
    ]
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct  LCH{
    pub lc: SubPixel,
    pub cc: SubPixel,
    pub hc: SubPixel,
}
pub fn gamut_compress_smooth(pix: [SubPixel; 3]) -> [SubPixel; 3] {
    // 1. ACEScg Luminance Coefficients
    // These define the "axis" of brightness we desaturate towards.
    let [r, g, b] = pix;
    let luma = r * 0.2722287168 + g * 0.6740817658 + b * 0.0536895174;

    // 2. Find the channel that is most "out of bounds"
    let min_channel = r.min(g).min(b);

    // 3. Define the Threshold (the "Knee")
    // Values above this are linear. Values below are compressed.
    // 0.04 (4%) is standard for linear rendering pipelines.
    let threshold = 0.04;

    // If the lowest channel is already safe, do nothing.
    if min_channel >= threshold {
        return [r, g, b];
    }

    // 4. Calculate the "Target" value for the minimum channel.
    // Instead of hard-clipping to 0.0, we use an exponential decay.
    // This ensures a smooth slope (C1 continuity) at the threshold.
    let target_min = if min_channel < threshold {
         threshold * ((min_channel - threshold) / threshold).exp()
    } else {
        min_channel
    };

    // 5. Calculate the Scale Factor (Saturation compression)
    // We want to interpolate between the Original Color and Gray (Luma)
    // such that 'min_channel' is pulled up to 'target_min'.
    //
    // Formula derivation: 
    // NewColor = Luma + k * (OldColor - Luma)
    // TargetMin = Luma + k * (MinChannel - Luma)
    // k = (TargetMin - Luma) / (MinChannel - Luma)
    
    // Safety: If the pixel is pure black or negative luma, avoid div by zero
    if luma <= 1e-9 {
        return [0.0, 0.0, 0.0];
    }
    
    // Calculate the distance of the lowest channel from the achromatic center
    let chromatic_dist = min_channel - luma;
    
    // If the distance is basically zero, we can't compress, just return gray.
    if chromatic_dist.abs() < 1e-9 {
        return [luma, luma, luma];
    }

    // Calculate the compression factor needed
    let scale = (target_min - luma) / chromatic_dist;

    // 6. Apply the scaling to the vector (relative to Luma)
    [
        luma + scale * (r - luma),
        luma + scale * (g - luma),
        luma + scale * (b - luma)
    ]
}

impl PipelineModule for Module<LCH> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        image.data.par_iter_mut().for_each(
            |p| {
                // let pix = subtractive_saturation_acescg(*p, self.config.cc);

                let [l, c, h] = WCS::convert::<Oklch>(*p);
                let pix = Oklch::convert::<WCS>([l*self.config.lc, c*self.config.cc, h*self.config.hc]);
                *p = pix;
            }
        );
        return image
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct  Crop{
    pub factor: usize,
}

impl PipelineModule for Module<Crop> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let width = image.width;
        let height = image.height;
        let new_width = width/self.config.factor;
        let new_height = height/self.config.factor;
        let mut result = vec![[0.0;3] ; (new_width)*(new_height)];
        let mut i = 0;
        for row in (0..image.height).step_by(self.config.factor) {
            for col in (0..image.width).step_by(self.config.factor) {
                result[i] = image.data[row*width+col];
                i+=1;
            }
        }
        image.data = result;
        image.width = new_width;
        image.height = new_height;
        image
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct HighlightReconstruction {
}

impl PipelineModule for Module<HighlightReconstruction> {

    fn process(&self, mut image: PipelineImage, raw_image: &RawImage) -> PipelineImage {
        let [_, clip_g, _, _] = raw_image.wb_coeffs;
        image.data.par_iter_mut().for_each(|pixel|{
            let [r, g, b] = *pixel;
            let factor = g/clip_g;
            let reconstructed_g = ((1.0-factor)*g) + (factor*(r+b)*(1.0/2.0));
            pixel[G] = reconstructed_g;
        });
        return image
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ChromaDenoise {
	  pub a: SubPixel,
    pub b: SubPixel,
    pub strength: SubPixel,
    #[serde(default)]
    pub use_ai: bool,
}

impl PipelineModule for Module<ChromaDenoise> {
    fn process(&self, image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        // image.data = crate::chroma_nr::denoise_chroma(
        //     image.data, 
        //     image.width, 
        //     image.height, 
        //     3, 
        //     self.config.strength, 
        //     self.config.use_ai
        // );
        return image;
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Exp {
    pub ev: SubPixel
}

impl PipelineModule for Module<Exp> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let value = (2.0 as SubPixel).powf(self.config.ev);
        image.data.par_iter_mut().for_each(
            |p| *p = p.map(|x| x*value)
        );
        return image;
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Sigmoid {
    pub c: SubPixel
}
/// A 3-component vector representing color (XYZ or RGB).
pub type Vec3 = [SubPixel; 3];

/// A 3x3 Matrix stored in row-major order.
/// [ m0, m1, m2,
///   m3, m4, m5,
///   m6, m7, m8 ]
pub type Mat3 = [SubPixel; 9];
const XYZ_TO_REC2020: Mat3 = [
         1.7166512, -0.3556708, -0.2533663,
        -0.6666844,  1.6164812,  0.0157685,
         0.0176399, -0.0427706,  0.9421031,
    ];

/// Matrix: Linear Rec.2020 -> CIE XYZ D65
const REC2020_TO_XYZ: Mat3 = [
    0.6369580, 0.1446169, 0.1688810,
    0.2627002, 0.6779981, 0.0593017,
    0.0000000, 0.0280727, 1.0609851,
];
#[inline(always)]
fn mat3_mul_vec3(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}
pub fn subtractive_desaturation(pixel_xyz: Vec3, amount: SubPixel) -> Vec3 {
    // 1. Convert to Linear Rec.2020
    // (We use Rec.2020 as the working space for the channel split)
    let rgb = mat3_mul_vec3(&XYZ_TO_REC2020, &pixel_xyz);

    // 2. Epsilon to prevent -inf in log2 (Singularity at 0.0 light)
    const EPSILON: SubPixel = 1e-8;

    // 3. Convert Linear Light -> Optical Density
    // Density = -log2(Light). 
    // We invert the signal to work in "Absorbance" space.
    let density = [
        -(rgb[0].max(EPSILON)).log2(),
        -(rgb[1].max(EPSILON)).log2(),
        -(rgb[2].max(EPSILON)).log2(),
    ];

    // 4. Calculate Average Density
    // Note: Arithmetic mean in Log space = Geometric mean in Linear space.
    let avg_density = (density[0] + density[1] + density[2]) / 3.0;

    // 5. Interpolate towards the Average Density
    let factor = amount.clamp(0.0, 1.0);

    let mix_density = [
        density[0] + (avg_density - density[0]) * factor,
        density[1] + (avg_density - density[1]) * factor,
        density[2] + (avg_density - density[2]) * factor,
    ];

    // 6. Convert Optical Density -> Linear Light
    // Light = 2^(-Density)
    let out_rgb = [
        (-mix_density[0]).exp2(),
        (-mix_density[1]).exp2(),
        (-mix_density[2]).exp2(),
    ];

    // 7. Convert Linear Rec.2020 -> XYZ
    mat3_mul_vec3(&REC2020_TO_XYZ, &out_rgb)
}

pub fn apply_filmic_saturation(pixel: [SubPixel; 3], saturation: SubPixel) -> [SubPixel; 3] {
            let [r, g, b] = pixel;

            // ACEScg (AP1) Luminance Weights
            const W_R: SubPixel = 0.2722287168;
            const W_G: SubPixel = 0.6740817658;
            const W_B: SubPixel = 0.0536895174;

            // 1. Calculate Perceived Luminance
            let luma = r * W_R + g * W_G + b * W_B;

            // 2. Stable Saturation Mixing (Lerp)
            let s = saturation.max(0.0);
            let inv_s = 1.0 - s;

            // This is where negatives are born if s > 1.0 and channel < luma
            let mut r_sat = r * s + luma * inv_s;
            let mut g_sat = g * s + luma * inv_s;
            let mut b_sat = b * s + luma * inv_s;

            // 3. Subtractive Density (Only applies when adding saturation)
            // (This step reduces brightness but doesn't fix negative signs)
            if s > 1.0 {
                let sat_gain = s - 1.0;
                let density = 1.0 / (1.0 + sat_gain * 0.05);

                r_sat *= density;
                g_sat *= density;
                b_sat *= density;
            }

            [r_sat, g_sat, b_sat]
        }
impl PipelineModule for Module<Sigmoid> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        image.data.iter_mut().for_each(|p|{
            let pix = *p;
            let params = &RgcParams::default();
            let pix = chromaCompresion::gamut_compress_pixel(pix, params);
            let pix = pix.map(tone_mapping::sigmoid);
            *p = pix;
        });
        return image
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Contrast {
    pub c: SubPixel
}

impl PipelineModule for Module<Contrast> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        const MIDDLE_GRAY: SubPixel = 0.1845;
        // let f = |x| (MIDDLE_GRAY*(x/MIDDLE_GRAY)).powf(self.c);
        image.data.par_iter_mut().for_each( |p|{
            *p = p.map(|x| MIDDLE_GRAY*(x/MIDDLE_GRAY).powf(self.config.c))
        });
        return image
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CFACoeffs {
}

impl PipelineModule for Module<CFACoeffs> {
    fn process(&self, mut image: PipelineImage, raw_image: &RawImage) -> PipelineImage {
        let [rv, gv, bv, _] = raw_image.wb_coeffs;
        image.data.par_iter_mut().for_each(
            |p|{
                let [r, g, b] = p;
                *p = [*r*rv, *g*gv, *b*bv]
            }
        );
        return image
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct LocalExpousure {
    pub c: SubPixel,
    pub m: SubPixel,
    pub p: SubPixel,
}

impl PipelineModule for Module<LocalExpousure> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let f = |x: SubPixel| x*((self.config.c*((2.0 as SubPixel).powf(-((x-self.config.p).powf(2.0)/self.config.m))))+1.0);
        let result = image.data.par_iter().map(|p|p.map(f));
        image.data = result.collect();
        return image

    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct LS<'a> {
    #[serde(skip_deserializing)]
    pub cache: Option<&'a PipelineImage>,
    pub transition_width: SubPixel,
    pub shadows_exp: SubPixel,
    pub highlits_exp: SubPixel,
    pub pivot: SubPixel, //ev
}

impl<'a> PipelineModule for Module<LS<'a>> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let config = self;
        let p: SubPixel = config.config.pivot;
        let d: SubPixel = config.config.transition_width;
        let f2 = |x: SubPixel, m: SubPixel| p*(x/p).powf(m);
        let f = |x: SubPixel, pf: SubPixel| 1.0/(1.0+(1.0/((2.0 as SubPixel).powf(d*(x-(p*pf))))));

        // shadows
        let ms: SubPixel = 1.0/config.config.shadows_exp;
        let pfs: SubPixel = 0.8;
        // let result = image.data.data.par_iter().map(|p|p.map(|x| (f2(x, ms)*(1.0-f(x, pf)))+f(x, pf)*x));
        //
        //// heights
        let mh: SubPixel = config.config.highlits_exp;
        let pfh: SubPixel = 1.2;

        let complete_f = |x| (((f2(x, ms)*(1.0-f(x, pfs)))+f(x, pfs)*x)+((f2(x, mh)*f(x, pfh))+((1.0-f(x, pfh))*x)))/2.0;

        let result = image.data.par_iter().map(|p|p.map(|x| complete_f(x)));

        image.data = result.collect();
        return image

    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CST {
    pub color_space: ColorSpaceMatrix,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub enum ColorSpaceMatrix {
    #[default]
    CameraToWCS,
    WCSTOsRGB,
    WCSTORGB,
    RGBTOsRGB,
}
pub fn acescg_to_srgb_display_safe(pixel: [SubPixel; 3]) -> [SubPixel; 3] {
    // 1. Matrix Convert: ACEScg (AP1) -> Linear sRGB (Rec.709)
    // --------------------------------------------------------
    let r_lin =  1.70485868 * pixel[0] - 0.62171602 * pixel[1] - 0.08329937 * pixel[2];
    let g_lin = -0.13007725 * pixel[0] + 1.14073577 * pixel[1] - 0.01055984 * pixel[2];
    let b_lin = -0.02396418 * pixel[0] - 0.12897547 * pixel[1] + 1.15301402 * pixel[2];

    let mut rgb = [r_lin, g_lin, b_lin];

    // 2. Gamut Mapping (Handle Negatives)
    // --------------------------------------------------------
    // Before compressing highlights, we must fix impossible colors (negatives).
    // We use the same Luminance-Preserving Desaturation from the previous step.
    let min_val = r_lin.min(g_lin).min(b_lin);

    if min_val < 0.0 {
        // Rec.709 Luminance weights
        let luma = r_lin * 0.2126 + g_lin * 0.7152 + b_lin * 0.0722;
        // Project towards Luma until min_val hits 0.0
        let factor = luma / (luma - min_val + 1e-6);
        
        rgb[0] = luma + (rgb[0] - luma) * factor;
        rgb[1] = luma + (rgb[1] - luma) * factor;
        rgb[2] = luma + (rgb[2] - luma) * factor;
    }

    // 3. Tone Mapping (Handle Highs > 1.0) using Reinhard-Jodie
    // --------------------------------------------------------
    // Standard Reinhard (x / 1+x) desaturates too much.
    // Luminance-only Reinhard (preserve ratios) clips colors ugly.
    // "Jodie" mixes the two to approximate film saturation roll-off.

    let [r, g, b] = rgb;
    
    // Recalculate luma (gamut mapping might have changed it slightly)
    let luma = r * 0.2126 + g * 0.7152 + b * 0.0722;

    // A. Apply Curve to Luminance only
    let tone_mapped_luma = luma / (1.0 + luma);

    // B. Apply Curve to RGB channels individually
    let r_tm = r / (1.0 + r);
    let g_tm = g / (1.0 + g);
    let b_tm = b / (1.0 + b);

    // C. The Jodie Mix
    // We blend between A and B based on how saturated the pixel is.
    // This prevents bright red becoming pink (desaturation) or clipping (hue skew).
    
    // Avoid div by zero
    if luma > 1e-6 {
        return [
            tone_mapped_luma.lerp(r_tm, r_tm / r),
            tone_mapped_luma.lerp(g_tm, g_tm / g),
            tone_mapped_luma.lerp(b_tm, b_tm / b),
        ];
    }

    // Fallback for black
    [0.0, 0.0, 0.0]
}

// Helper trait for Linear Interpolation if you don't have one
trait Lerp {
    fn lerp(self, other: Self, t: Self) -> Self;
}

impl Lerp for SubPixel {
    fn lerp(self, other: SubPixel, t: SubPixel) -> SubPixel {
        self * (1.0 - t) + other * t
    }
}


impl PipelineModule for Module<CST> {
    fn process(&self, mut image: PipelineImage, raw_image: &RawImage) -> PipelineImage {
        match self.config.color_space {
            ColorSpaceMatrix::WCSTOsRGB => {
                // let matrix = [
                //     [3.240479,  -1.537150, -0.498535,],
                //     [-0.969256,  1.875991,  0.041556,],
                //     [0.055648,  -0.204043,  1.057311]
                // ];
                // let foward_matrix = rawler::imgop::matrix::normalize(matrix);
                // image.data.par_iter_mut().for_each(|pixel|{
                //     let pix = acescg_to_srgb_display_safe(*pixel);
                //     let srgb = Srgb::from_linear_srgb(pix);
                //     *pixel = srgb;
                // });
                image.data.par_iter_mut().for_each(|pixel|{
                    *pixel = WCS::convert::<Srgb>(*pixel);
                });
            },
            ColorSpaceMatrix::WCSTORGB => {
                image.data.par_iter_mut().for_each(|pixel|{
                    let srgb = WCS::to_linear_srgb(*pixel);
                    // let srgb = gamut_clip_aces(srgb);
                    *pixel = srgb.map(|subp| subp.abs() );
                });
            },
            ColorSpaceMatrix::RGBTOsRGB => {
                image.data.par_iter_mut().for_each(|pixel|{
                    let srgb = Srgb::from_linear_srgb(*pixel);
                    // let srgb = gamut_clip_aces(srgb);
                    *pixel = srgb.map(|subp| subp.abs() );
                });
            },
            ColorSpaceMatrix::CameraToWCS => {
                let d65 = raw_image.camera.color_matrix[&Illuminant::D65].clone();
                let components = d65.len() / 3;
                let mut xyz2cam: [Pixel; 3] = [[0.0; 3]; 3];
                for i in 0..components {
                    for j in 0..3 {
                        xyz2cam[i][j] = d65[i * 3 + j];
                    }
                }
                let xyz2cam_normalized = rawler::imgop::matrix::normalize(xyz2cam);
                let foward_matrix = rawler::imgop::matrix::pseudo_inverse(xyz2cam_normalized);
                image.data.par_iter_mut().for_each(|p|{
                    let [r, g, b] = *p;
                    let pix = [
                        foward_matrix[0][0] * r + foward_matrix[0][1] * g + foward_matrix[0][2] * b,
                        foward_matrix[1][0] * r + foward_matrix[1][1] * g + foward_matrix[1][2] * b,
                        foward_matrix[2][0] * r + foward_matrix[2][1] * g + foward_matrix[2][2] * b,
                    ];
                    *p = XyzD65::convert::<WCS>(pix)

                });
            },
        };

        return image
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Demosaic {
    pub algorithm: String,
}

impl PipelineModule for Module<Demosaic> {
    fn process(
        &self,
        image: PipelineImage,
        raw_image: &RawImage,
    ) -> PipelineImage {
        let mut new_image = image;
        let debayer_image: RgbF32;
        let max_raw_value: f32;
        if let rawler::RawImageData::Integer(ref im) = raw_image.data {
            let cfa = raw_image.camera.cfa.clone();
            let cfa = demosaic::get_cfa(cfa, raw_image.crop_area.unwrap());
            let (nim, width, height) = demosaic::crop(
                raw_image.dim(),
                raw_image.crop_area.unwrap(),
                im
            );
            let black_level = raw_image.blacklevel.as_bayer_array()[0];
            let white_level = raw_image.whitelevel.as_bayer_array()[0];
            let range = white_level - black_level;
            let factor = 1.0/range;
            let nim: Vec<f32> = nim.par_iter().map(|pix| {
                (*pix as f32 - black_level) * factor
            }).collect();

            max_raw_value = nim
            .par_iter()
            .fold(|| f32::NEG_INFINITY, |acc, &x| acc.max(x))
            .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));

            debayer_image =
                match self.config.algorithm.as_str() {
                    "markesteijn" => demosaic::DemosaicAlgorithms::markesteijn(width, height, cfa, nim),
                    "linear" => demosaic::DemosaicAlgorithms::linear_interpolation(width, height, cfa, nim),
                    "fast" => demosaic::DemosaicAlgorithms::fast(width, height, cfa, nim),
                    _ => panic!("Unknown demosaic algorithm"),
                };
        } else {
            panic!("Don't know how to process non-integer raw files");
        }

        new_image.data = debayer_image.data;
        new_image.width = debayer_image.width;
        new_image.height = debayer_image.height;
        new_image.max_raw_value = max_raw_value;
        new_image
    }
}

// pub fn lineal_mask(height: usize, width: usize) -> Vec<SubPixel> {
//     let mut result = vec![0.0; width*height];
//     result.par_iter_mut().enumerate().for_each(|(i, val)|{
//         let x = i % width;
//         let y = (i - x) / width;
//         // *val = 1.0/(((height-y) as SubPixel * 0.01) + 1.0)
//         *val = 1.0
//     });
//     result
// }
// pub fn get_channel(c: usize, data: &mut Array3<SubPixel>) -> Array3<SubPixel>{
//     let shape = data.shape();

//     let mut final_image = Array3::<SubPixel>::zeros((shape[0], shape[1], 3));
//     final_image.slice_mut(s![.., ..,c]).assign(&data.slice(s![.., .., c]));
//     return final_image
// }

// pub fn film_curve(p: f64, d: f64, a: f64, b: f64, p2: f64, data: &mut Array3<SubPixel>) -> Array3<SubPixel> {
//     let f1 = |x: f64| d*(x-p)+p;

//     let c2 = (d*a)/f1(a);
//     let c1 = (d*a)/(c2*(a.powf(c2)));
//     let c4 = (-d*(p2-b))/(f1(b)-1.0);
//     let c3 = (-d)/(c4*(p2-b).powf(c4-1.0));
    
//     let f2 = |x: f64| c1*(x.powf(c2));
//     let f3 = |x: f64| c3*(-x+p2).powf(c4)+1.0;

//     let f = |x: f64| 
//         if x > p2 { 1.0 }
//         else if b < x && x <= p2 { f3(x) }
//         else if a < x && x <= b { f1(x) }
//         else if 0.0 <= x && x <= a { f2(x) }
//         else {0.0}
//     ;

//     data.par_mapv_inplace(|x| f(x as f64) as SubPixel);
//     return data.clone();
// }

// fn small(v: Array3<SubPixel>) -> Array3<SubPixel> {
//     let f = 1;
//     let s = v.shape();
//     let mut nv = Array3::zeros(((s[0] / f) + 1, (s[1] / f) + 1, s[2]));
//     for i in (0..s[0]).step_by(f) {
//         for j in (0..s[1]).step_by(f) {
//             for x in 0..3 {
//                 nv[[i / f, j / f, x]] = v[[i, j, x]];
//             }
//         }
//     }
//     return nv;
// }
