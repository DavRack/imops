use std::{any, fmt, usize};

use color::{ColorSpace, Oklch, Srgb, XyzD65};
use rawler::{imgop::xyz::Illuminant, pixarray::RgbF32, RawImage};
// use sealed::Cache;
use serde::{Deserialize, Serialize};
use toml::map::Map;
// use crate::color_p::gamut_clip_aces;
use crate::mask::{Mask};
use crate::{helpers::*, pixels};
use crate::pixels::*;
use crate::wavelet_nl_means;
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
            let factor = (g/clip_g);
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
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
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

impl PipelineModule for Module<Sigmoid> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let max_current_value = image.data.iter().fold(0.0, |current_max, pixel| pixel.luminance().max(current_max));
        let scaled_one = (1.0/image.max_raw_value)*max_current_value;
        let c = 1.0 + (1.0/(scaled_one*self.config.c)).powi(2);

        image.data.iter_mut().for_each(|p|{
            let lum = p.luminance();
            let new_lum = (c / (1.0 + (1.0/(self.config.c*lum)))).powi(2);
            let [r, g, b] = *p;
            let factor = new_lum/lum;
            let pix = [r*factor, g*factor, b*factor];
            let lum = pix.luminance().clamp(0.0, 1.0);
            let [l, c, h] = XyzD65::convert::<Oklch>(pix);
            *p = Oklch::convert::<XyzD65>([l, c*(1.0-(lum.powf(0.8))), h])
            // *p=agx::transform(*p);
            // *p = subtractive_desaturation(pix, lum.powf(2.0));
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
