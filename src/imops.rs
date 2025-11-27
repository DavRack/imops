use std::{any, usize};

use color::{ColorSpace, Oklch, Srgb, XyzD65};
use rawler::{imgop::xyz::Illuminant, pixarray::RgbF32, RawImage};
// use sealed::Cache;
use serde::{Deserialize, Serialize};
use toml::map::Map;
// use crate::color_p::gamut_clip_aces;
use crate::mask::{Mask};
use crate::{helpers::*, pixels, tone_mapping};
use crate::pixels::*;
use crate::demosaic;

use crate::conditional_paralell::prelude::*;

const R: usize = 0;
const G: usize = 1;
const B: usize = 2;

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


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct  LCH{
    pub lc: SubPixel,
    pub cc: SubPixel,
    pub hc: SubPixel,
}

impl PipelineModule for Module<LCH> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        image.data.par_iter_mut().for_each(
            |p| {
                let [l, c, h] = XyzD65::convert::<Oklch>(*p);
                *p = Oklch::convert::<XyzD65>([l*self.config.lc, c*self.config.cc, h*self.config.hc])
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
pub type Vec3 = [f32; 3];

/// A 3x3 Matrix stored in row-major order.
/// [ m0, m1, m2,
///   m3, m4, m5,
///   m6, m7, m8 ]
pub type Mat3 = [f32; 9];
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
pub fn subtractive_desaturation(pixel_xyz: Vec3, amount: f32) -> Vec3 {
    // 1. Convert to Linear Rec.2020
    // (We use Rec.2020 as the working space for the channel split)
    let rgb = mat3_mul_vec3(&XYZ_TO_REC2020, &pixel_xyz);

    // 2. Epsilon to prevent -inf in log2 (Singularity at 0.0 light)
    const EPSILON: f32 = 1e-8;

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

#[inline(always)]
fn aces_film(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    (x * (a * x + b)) / (x * (c * x + d) + e).max(f32::EPSILON)
}

impl PipelineModule for Module<Sigmoid> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        #[derive(Debug)]
        enum SigmoidMethod {
            Custom,
            Gemini,
            PlainSigmoid,
        }
        let method = SigmoidMethod::PlainSigmoid;

        match method {
            SigmoidMethod::PlainSigmoid => {
                let max_current_value = image.data.iter().fold(0.0, |current_max, pixel| pixel.norm().max(current_max));
                let scaled_one = (1.0/image.max_raw_value)*max_current_value;
                let sigmoid_normalization_constant = 1.0 + (1.0/(scaled_one*self.config.c)).powi(2);

                fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
                    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
                }
                const REC709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

                fn sanitize_rgb(rgb: [f32; 3]) -> [f32; 3] {
                    let min_val = rgb[0].min(rgb[1]).min(rgb[2]);

                    // If all channels are positive, the color is valid. Return as is.
                    if min_val >= 0.0 {
                        return rgb;
                    }

                    // The color is out of gamut (negative). 
                    // We need to desaturate it towards its own Luminance until it fits.
                    let luma = dot(rgb, REC709_LUMA);

                    // If the color is so weird it has negative luminance, just black it out.
                    if luma <= 0.0 {
                        return [0.0, 0.0, 0.0]; 
                    }

                    // Solve for x:  min_val + x * (luma - min_val) = 0
                    // This finds the exact intersection with the gamut boundary.
                    let sub = luma - min_val;
                    if sub.abs() < 1e-6 { return [luma, luma, luma]; } // Avoid div by zero

                    let ratio = -min_val / sub;

                    [
                        rgb[0] + ratio * (luma - rgb[0]),
                        rgb[1] + ratio * (luma - rgb[1]),
                        rgb[2] + ratio * (luma - rgb[2]),
                    ]
                }

                image.data.iter_mut().for_each(|p|{
                    let pix = *p;
                    let lum = pix.luminance();
                    let transformed_lum = tone_mapping::sigmoid(lum);
                    let factor = transformed_lum/lum;
                    let pix = pix.map(|x| (x * factor).clamp(0.0, 1.0));
                    pix.iter().for_each(|x| if *x < 0.0 || *x > 1.0 { println!("{:?}, {:?}",x, *p)});
                    let [l, c, h] = XyzD65::convert::<Oklch>(pix);
                    let pix = Oklch::convert::<XyzD65>([l, c*(1.0-transformed_lum), h]);
                    *p = pix
                });
            },
            SigmoidMethod::Custom => {
                let max_current_value = image.data.iter().fold(0.0, |current_max, pixel| pixel.luminance().max(current_max));
                let scaled_one = (1.0/image.max_raw_value)*max_current_value;
                let sigmoid_normalization_constant = 1.0 + (1.0/(scaled_one*self.config.c)).powi(2);

                fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
                    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
                }
                                const REC709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

                fn sanitize_rgb(rgb: [f32; 3]) -> [f32; 3] {
                    let min_val = rgb[0].min(rgb[1]).min(rgb[2]);

                    // If all channels are positive, the color is valid. Return as is.
                    if min_val >= 0.0 {
                        return rgb;
                    }

                    // The color is out of gamut (negative). 
                    // We need to desaturate it towards its own Luminance until it fits.
                    let luma = dot(rgb, REC709_LUMA);

                    // If the color is so weird it has negative luminance, just black it out.
                    if luma <= 0.0 {
                        return [0.0, 0.0, 0.0]; 
                    }

                    // Solve for x:  min_val + x * (luma - min_val) = 0
                    // This finds the exact intersection with the gamut boundary.
                    let sub = luma - min_val;
                    if sub.abs() < 1e-6 { return [luma, luma, luma]; } // Avoid div by zero

                    let ratio = -min_val / sub;

                    [
                        rgb[0] + ratio * (luma - rgb[0]),
                        rgb[1] + ratio * (luma - rgb[1]),
                        rgb[2] + ratio * (luma - rgb[2]),
                    ]
                }
                image.data.iter_mut().for_each(|p|{
                    let lum = p.luminance();
                    let new_lum = (sigmoid_normalization_constant / (1.0 + (1.0/(self.config.c*lum)))).powi(2);
                    let [r, g, b] = *p;
                    let factor = new_lum/lum;
                    let pix = [r*factor, g*factor, b*factor];
                    let n = new_lum;
                    let [l, c, h] = XyzD65::convert::<Oklch>(pix);
                    let pix = Oklch::to_linear_srgb([l, c*(1.0-n.powf(1.5)), h]);
                    let pix = sanitize_rgb(pix);
                    *p = Srgb::from_linear_srgb(pix)
                });
            },
            SigmoidMethod::Gemini => {
                use std::f32::consts::PI;

                // ==========================================
                // 1. STRUCTS & CONSTANTS
                // ==========================================

                #[derive(Clone, Copy, Debug)]
                struct Chromaticities {
                    red: [f32; 2],
                    green: [f32; 2],
                    blue: [f32; 2],
                    white: [f32; 2],
                }

                // Standard Rec.709 Primaries (sRGB)
                const REC709_PRIMARIES: Chromaticities = Chromaticities {
                    red: [0.64, 0.33],
                    green: [0.30, 0.60],
                    blue: [0.15, 0.06],
                    white: [0.3127, 0.3290], // D65
                };

                // ==========================================
                // 2. MATH HELPERS (From Camera-AgX-Lib.h)
                // ==========================================

                fn mat3_mul_vec3(m: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
                    [
                        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
                        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
                        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
                    ]
                }

                // Computes the XYZ conversion matrix from Primaries
                // (Standard CIE 1931 algorithm)
                fn primaries_to_xyz_mat(p: Chromaticities) -> [f32; 9] {
                    let [xr, yr] = p.red;
                    let [xg, yg] = p.green;
                    let [xb, yb] = p.blue;
                    let [xw, yw] = p.white;

                    let zr = 1.0 - xr - yr;
                    let zg = 1.0 - xg - yg;
                    let zb = 1.0 - xb - yb;
                    let zw = 1.0 - xw - yw;

                    let rw = xw / yw;
                    let gw = 1.0;
                    let bw = zw / yw;

                    let det = xr * (yg * zb - zg * yb) - xg * (yr * zb - zr * yb) + xb * (yr * zg - zr * yg);

                    let sr = (det.recip()) * (rw * (yg * zb - zg * yb) - gw * (yr * zb - zr * yb) + bw * (yr * zg - zr * yg));
                    let sg = (det.recip()) * (-rw * (xg * zb - zg * xb) + gw * (xr * zb - zr * xb) - bw * (xr * zg - zr * xg));
                    let sb = (det.recip()) * (rw * (xg * yb - yg * xb) - gw * (xr * yb - yr * xb) + bw * (xr * yg - yr * xg));

                    [
                        sr * xr, sg * xg, sb * xb,
                        sr * yr, sg * yg, sb * yb,
                        sr * zr, sg * zg, sb * zb,
                    ]
                }

                // Inverts a 3x3 Matrix (for Outset)
                fn mat3_invert(m: &[f32; 9]) -> [f32; 9] {
                    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) -
                    m[1] * (m[3] * m[8] - m[5] * m[6]) +
                    m[2] * (m[3] * m[7] - m[4] * m[6]);
                    let inv_det = 1.0 / det;

                    [
                        (m[4] * m[8] - m[5] * m[7]) * inv_det,
                        (m[2] * m[7] - m[1] * m[8]) * inv_det,
                        (m[1] * m[5] - m[2] * m[4]) * inv_det,
                        (m[5] * m[6] - m[3] * m[8]) * inv_det,
                        (m[0] * m[8] - m[2] * m[6]) * inv_det,
                        (m[2] * m[3] - m[0] * m[5]) * inv_det,
                        (m[3] * m[7] - m[4] * m[6]) * inv_det,
                        (m[1] * m[6] - m[0] * m[7]) * inv_det,
                        (m[0] * m[4] - m[1] * m[3]) * inv_det,
                    ]
                }

                // ==========================================
                // 3. CORE AGX FUNCTIONS (The "Correct" Parts)
                // ==========================================

                // Sobotka's "Insetcalcmatrix" from Camera-AgX-Lib.h
                // This dynamically calculates the compression matrix based on attenuation.
                fn calc_inset_matrix(p: Chromaticities, att_r: f32, att_g: f32, att_b: f32) -> [f32; 9] {
                    // 1. Calculate Scale factors (1 / (1-attenuation)^2)
                    let scale_r = 1.0 / (1.0 - att_r).powi(2);
                    let scale_g = 1.0 / (1.0 - att_g).powi(2);
                    let scale_b = 1.0 / (1.0 - att_b).powi(2);

                    // 2. Adjust Primaries towards White Point
                    // Formula: (Primary - White) * Scale + White
                    let adj_red = [
                        (p.red[0] - p.white[0]) * scale_r + p.white[0],
                        (p.red[1] - p.white[1]) * scale_r + p.white[1],
                    ];
                    let adj_green = [
                        (p.green[0] - p.white[0]) * scale_g + p.white[0],
                        (p.green[1] - p.white[1]) * scale_g + p.white[1],
                    ];
                    let adj_blue = [
                        (p.blue[0] - p.white[0]) * scale_b + p.white[0],
                        (p.blue[1] - p.white[1]) * scale_b + p.white[1],
                    ];

                    let adj_chroma = Chromaticities {
                        red: adj_red,
                        green: adj_green,
                        blue: adj_blue,
                        white: p.white,
                    };

                    // 3. Compute Transform: RGB -> XYZ -> Adjusted RGB
                    let rgb_to_xyz = primaries_to_xyz_mat(p);
                    let xyz_to_adj = mat3_invert(&primaries_to_xyz_mat(adj_chroma));

                    // Multiply matrices: XYZ_to_Adj * RGB_to_XYZ
                    // (Note: In Sobotka's code it's `mult_f33_f33(In2XYZ, XYZ2Adj)`)
                    // We implement row-by-column multiplication here:
                    let m1 = xyz_to_adj;
                    let m2 = rgb_to_xyz;

                    [
                        m1[0]*m2[0]+m1[1]*m2[3]+m1[2]*m2[6], m1[0]*m2[1]+m1[1]*m2[4]+m1[2]*m2[7], m1[0]*m2[2]+m1[1]*m2[5]+m1[2]*m2[8],
                        m1[3]*m2[0]+m1[4]*m2[3]+m1[5]*m2[6], m1[3]*m2[1]+m1[4]*m2[4]+m1[5]*m2[7], m1[3]*m2[2]+m1[4]*m2[5]+m1[5]*m2[8],
                        m1[6]*m2[0]+m1[7]*m2[3]+m1[8]*m2[6], m1[6]*m2[1]+m1[7]*m2[4]+m1[8]*m2[7], m1[6]*m2[2]+m1[7]*m2[5]+m1[8]*m2[8],
                    ]
                }

                // Sobotka's Normalized Log2 Encoding (tf == 10)
                fn agx_log2_resolve(val: f32) -> f32 {
                    // Standard AgX Resolve Middle Grey
                    let v = val / 0.18; 

                    // Log2
                    let log_v = v.max(1e-10).log2();

                    // Clamp to Range [-10.0, 6.5]
                    let clamped = log_v.clamp(-10.0, 6.5);

                    // Normalize to [0.0, 1.0] (Total range 16.5 stops)
                    (clamped + 10.0) / 16.5
                }

                // The AgX Base Sigmoid (Golden Curve)
                fn agx_sigmoid(x: f32) -> f32 {
                    let t = x.clamp(0.0, 1.0);
                    let t2 = t * t;
                    let t4 = t2 * t2;

                    // Coefficients from AgX implementation
                    0.01826068 
                    + 0.90696317 * t 
                    + 0.18342209 * t2 
                    - 0.5284358  * (t2 * t) 
                    + 0.6402758  * t4 
                    - 0.3204968  * (t4 * t)
                }

                // ==========================================
                // 4. MAIN PIPELINE
                // ==========================================

                // Pre-calculate matrices (Do this once outside the loop!)
                // Default "AgX Kraken" settings from README:
                // Attenuation: 0.2
                let inset_mat = calc_inset_matrix(REC709_PRIMARIES, 0.2, 0.2, 0.2);
                let outset_mat = mat3_invert(&inset_mat); // Restore gamut
// const REC709_PRIMARIES: Chromaticities = Chromaticities {
//     red: [0.64, 0.33],
//     green: [0.30, 0.60],
//     blue: [0.15, 0.06],
//     white: [0.3127, 0.3290], // D65
// };

// ==========================================
// 2. THE PIPELINE
// ==========================================

// Pre-calc Matrices (as before)
// We calculate the matrices specifically for Rec.709 primaries.
// This means we DO NOT need to convert to ARRI LogC. 
// We can stay in Linear Rec.709, which is a lossless conversion from your XYZ D65.
let inset_mat = calc_inset_matrix(REC709_PRIMARIES, 0.2, 0.2, 0.2);
let outset_mat = mat3_invert(&inset_mat);

image.data.iter_mut().for_each(|p| {
    // 1. INPUT: XYZ D65 Linear -> Linear Rec.709
    // Since your input is XYZ D65 Linear, and we built the AgX matrices 
    // using Rec.709 primaries above, we must convert to Linear Rec.709 here.
    let mut rgb = XyzD65::to_linear_srgb(*p);

    // 2. Sanitize / Clip Negatives
    // Linear conversions from XYZ can produce negatives for out-of-gamut colors.
    // AgX math requires positive input.
    rgb[0] = rgb[0].max(1e-6);
    rgb[1] = rgb[1].max(1e-6);
    rgb[2] = rgb[2].max(1e-6);

    // 3. AgX Inset (Gamut Compression)
    let inset = mat3_mul_vec3(&inset_mat, rgb);

    // 4. Log2 Encoding + Sigmoid
    // The library uses a specific normalization for Rec.709/sRGB inputs.
    // Middle Grey (0.18) is the anchor.
    let transformed = [
        agx_sigmoid(agx_log2_resolve(inset[0])),
        agx_sigmoid(agx_log2_resolve(inset[1])),
        agx_sigmoid(agx_log2_resolve(inset[2])),
    ];

    // 5. AgX Outset (Gamut Restoration)
    let outset = mat3_mul_vec3(&outset_mat, transformed);

    // 6. OUTPUT: Linear Rec.709 -> XYZ D65 Linear
    // We strictly clamp to 0.0 to avoid producing invalid XYZ.
    let final_rgb = [
        outset[0].max(0.0),
        outset[1].max(0.0),
        outset[2].max(0.0),
    ];
    
    *p = Srgb::from_linear_srgb(final_rgb);
});},
        }
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
    CameraToXYZ,
    XYZTOsRGB,
    XYZTORGB,
    RGBTOsRGB,
}


impl PipelineModule for Module<CST> {
    fn process(&self, mut image: PipelineImage, raw_image: &RawImage) -> PipelineImage {
        match self.config.color_space {
            ColorSpaceMatrix::XYZTOsRGB => {
                // let matrix = [
                //     [3.240479,  -1.537150, -0.498535,],
                //     [-0.969256,  1.875991,  0.041556,],
                //     [0.055648,  -0.204043,  1.057311]
                // ];
                // let foward_matrix = rawler::imgop::matrix::normalize(matrix);
                image.data.par_iter_mut().for_each(|pixel|{
                    let srgb = XyzD65::convert::<Srgb>(*pixel);
                    // let srgb = gamut_clip_aces(srgb);
                    // if srgb[0] < 0.0 || srgb[1] < 0.0 || srgb[2] < 0.0 {
                    //     srgb[0] = if srgb[0] < 0.0 {1.0} else {0.0};
                    //     srgb[1] = if srgb[1] < 0.0 {1.0} else {0.0};
                    //     srgb[2] = if srgb[2] < 0.0 {1.0} else {0.0};
                    // }
                    *pixel = srgb.map(|subp| subp );
                });
            },
            ColorSpaceMatrix::XYZTORGB => {
                image.data.par_iter_mut().for_each(|pixel|{
                    let srgb = XyzD65::to_linear_srgb(*pixel);
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
            ColorSpaceMatrix::CameraToXYZ => {
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
                    *p = [
                        foward_matrix[0][0] * r + foward_matrix[0][1] * g + foward_matrix[0][2] * b,
                        foward_matrix[1][0] * r + foward_matrix[1][1] * g + foward_matrix[1][2] * b,
                        foward_matrix[2][0] * r + foward_matrix[2][1] * g + foward_matrix[2][2] * b,
                    ]
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
        let debayer_image: FormedImage;
        let mut raw_image = raw_image.clone();
        let _ = raw_image.apply_scaling();
        if let rawler::RawImageData::Float(ref im) = raw_image.data {
            let cfa = raw_image.camera.cfa.clone();
            let cfa = demosaic::get_cfa(cfa, raw_image.crop_area.unwrap());
            let (nim, width, height) =
                demosaic::crop(raw_image.dim(), raw_image.crop_area.unwrap(), im.to_vec());

            debayer_image = FormedImage {
                raw_image: raw_image.clone(),
                data: match self.config.algorithm.as_str() {
                    "markesteijn" => demosaic::DemosaicAlgorithms::markesteijn(width, height, cfa, nim),
                    _ => panic!("Unknown demosaic algorithm"),
                },
            };
        } else {
            panic!("Don't know how to process non-integer raw files");
        }

        let max_raw_value = debayer_image.data.data.iter().fold(0.0, |acc, pix|{
            let [r, g, b] = pix;
            r.max(*g).max(*b).max(acc)
        });

        new_image.data = debayer_image.data.data;
        new_image.width = debayer_image.data.width;
        new_image.height = debayer_image.data.height;
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
