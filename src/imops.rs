use std::{any,  usize};

use color::{ColorSpace, Oklab, XyzD65};
use rawler::{imgop::xyz::Illuminant, pixarray::RgbF32, RawImage};
use serde::{Deserialize, Serialize};
use toml::map::Map;
use crate::cst::{oklab_to_xyz, xyz_to_oklab, xyz_to_oklab_l};
use crate::{chroma_nr, helpers::*};

use crate::conditional_paralell::prelude::*;

pub trait PipelineModule{
    fn process(&self, image: PipelineImage, raw_image: &RawImage) -> PipelineImage;
    fn get_name(&self) -> String;
    fn set_cache(&mut self, cache: PipelineImage);
    fn get_cache(&self) -> PipelineImage;
}

const CHANNELS_PER_PIXEL: usize = 3;

const R: usize = 0;
const G: usize = 0;
const B: usize = 0;

#[derive(Clone)]
pub struct FormedImage {
    pub raw_image: RawImage,
    pub data: RgbF32,
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, Default)]
pub struct PipelineImage {
    pub data: Vec<[f32; CHANNELS_PER_PIXEL]>,
    pub height: usize,
    pub width: usize,
    pub max_raw_value: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Module<T>{
    pub cache: PipelineImage,
    pub config: T,
}

impl<T> Module<T>{
    pub fn from_toml<'a>(module: Map<String, toml::Value>) -> Box<Self>
    where
        T: Deserialize<'a> + Default,
        Self: Sized
    {
        let cfg: T = module.try_into::<T>().expect(any::type_name::<Self>());
        let module = Module{
            cache: PipelineImage::default(),
            config: cfg
        };
        Box::new(module)
    }
}


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct  LCH{
    pub lc: f32,
    pub cc: f32,
    pub hc: f32,
}

impl PipelineModule for Module<LCH> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        image.data.par_iter_mut().for_each(
            |p| {
                let [l, c, h] = XyzD65::convert::<Oklab>(*p);
                *p = Oklab::convert::<XyzD65>([l*self.config.lc, c*self.config.cc, h*self.config.hc])
            }
        );
        return image
    }

    fn get_name(&self) -> String{
        return "LCH".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
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

    fn get_name(&self) -> String{
        return "Crop".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct HighlightReconstruction {
}

#[inline]
fn avg_pixels(sorrounding_pixels: &Vec<[f32; 3]>) -> [f32; 3]{
    let len = sorrounding_pixels.len() as f32;
    let px = sorrounding_pixels.into_iter().fold([0., 0., 0.], |acc, pixel|{
        let [r, g, b] = pixel;
        let [ar, ag, ab] = acc;
        [
            r+ar,
            g+ag,
            b+ab
        ]
    });
    px.map(|x|x/len)
}

impl PipelineModule for Module<HighlightReconstruction> {

    fn process(&self, mut image: PipelineImage, raw_image: &RawImage) -> PipelineImage {
        let d = image.clone();
        let [clip_r, clip_g, clip_b, _] = raw_image.wb_coeffs;


        image.data.par_iter_mut().enumerate().for_each(|(idx, pixel)|{
            let [r, g, b] = *pixel;
            let [cliped_r, cliped_g, cliped_b] = [
                r >= clip_r,
                g >= clip_g,
                b >= clip_b,
            ];

            if cliped_g || cliped_r || cliped_b{
                let sorrounding_pixels = d.get_px_tail(1, idx);
                let [r, g, b] = avg_pixels(&sorrounding_pixels);


                if cliped_r {
                    pixel[R] = g+b
                }

                if cliped_g {
                    pixel[G] = r+b;
                }

                if cliped_b {
                    pixel[B] = r+g;
                }
            }
        });
        return image
    }

    fn get_name(&self) -> String{
        return "HighlightReconstruction".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ChromaDenoise {
	  a: f32,
    b: f32,
    strength: f32
}

impl PipelineModule for Module<ChromaDenoise> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {

        let data = image.data.clone();
        let data_l = data.par_iter().map(|p|{
            xyz_to_oklab_l(p)
        });

        image.data = chroma_nr::denoise_rgb(image.data, image.width, image.height, 3, 1);

        image.data.par_iter_mut().zip(data_l).for_each(|(pixel, l)|{
            let [_, c, h] = xyz_to_oklab(pixel);
            *pixel = oklab_to_xyz(&[l, c, h])
        });

        return image;
    }

    fn get_name(&self) -> String{
        return "ChromaDenoise".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Exp {
    pub ev: f32
}

impl PipelineModule for Module<Exp> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let value = 2.0_f32.powf(self.config.ev);
        image.data.par_iter_mut().for_each(
            |p| *p = p.map(|x| x*value)
        );
        return image;
    }

    fn get_name(&self) -> String{
        return "Exp".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Sigmoid {
    pub c: f32
}

impl PipelineModule for Module<Sigmoid> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let max_current_value = image.data.iter().fold(0.0, |current_max, [r, g, b]| r.max(*g).max(*b).max(current_max));
        let scaled_one = (1.0/image.max_raw_value)*max_current_value;
        let c = 1.0 + (1.0/scaled_one).powi(2);
        image.data.par_iter_mut().for_each(|p|{
            *p = p.map(|x| (c / (1.0 + (1.0/(self.config.c*x)))).powi(2))
        });
        return image
    }

    fn get_name(&self) -> String{
        return "Sigmoid".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Contrast {
    pub c: f32
}

impl PipelineModule for Module<Contrast> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        const MIDDLE_GRAY: f32 = 0.1845;
        // let f = |x| (MIDDLE_GRAY*(x/MIDDLE_GRAY)).powf(self.c);
        image.data.par_iter_mut().for_each( |p|{
            *p = p.map(|x| MIDDLE_GRAY*(x/MIDDLE_GRAY).powf(self.config.c))
        });
        return image
    }

    fn get_name(&self) -> String{
        return "Contrast".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
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

    fn get_name(&self) -> String{
        return "CFACoeffs".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct LocalExpousure {
    pub c: f32,
    pub m: f32,
    pub p: f32,
}

impl PipelineModule for Module<LocalExpousure> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let f = |x: f32| x*((self.config.c*(2.0_f32.powf(-((x-self.config.p).powf(2.0)/self.config.m))))+1.0);
        let result = image.data.par_iter().map(|p|p.map(f));
        image.data = result.collect();
        return image

    }

    fn get_name(&self) -> String{
        return "LocalExpousure".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct LS {
    #[serde(skip_deserializing)]
    pub cache: PipelineImage,
    pub transition_width: f32,
    pub shadows_exp: f32,
    pub highlits_exp: f32,
    pub pivot: f32, //ev
}

impl PipelineModule for Module<LS> {
    fn process(&self, mut image: PipelineImage, _raw_image: &RawImage) -> PipelineImage {
        let config = self.clone();
        let p: f32 = config.config.pivot;
        let d: f32 = config.config.transition_width;
        let f2 = |x: f32, m: f32| p*(x/p).powf(m);
        let f = |x: f32, pf: f32| 1.0/(1.0+(1.0/(2.0_f32.powf(d*(x-(p*pf))))));

        // shadows
        let ms: f32 = 1.0/config.config.shadows_exp;
        let pfs: f32 = 0.8;
        // let result = image.data.data.par_iter().map(|p|p.map(|x| (f2(x, ms)*(1.0-f(x, pf)))+f(x, pf)*x));
        //
        //// heights
        let mh: f32 = config.config.highlits_exp;
        let pfh: f32 = 1.2;

        let complete_f = |x| (((f2(x, ms)*(1.0-f(x, pfs)))+f(x, pfs)*x)+((f2(x, mh)*f(x, pfh))+((1.0-f(x, pfh))*x)))/2.0;

        let result = image.data.par_iter().map(|p|p.map(|x| complete_f(x)));

        image.data = result.collect();
        return image

    }

    fn get_name(&self) -> String{
        return "LS".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
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
    XYZTOsRGB
}


impl PipelineModule for Module<CST> {
    fn process(&self, mut image: PipelineImage, raw_image: &RawImage) -> PipelineImage {
        let foward_matrix = match self.config.color_space {
            ColorSpaceMatrix::XYZTOsRGB => {
                let matrix = [
                    [3.240479,  -1.537150, -0.498535,],
                    [-0.969256,  1.875991,  0.041556,],
                    [0.055648,  -0.204043,  1.057311]
                ];
                let xyz2srgb_normalized = rawler::imgop::matrix::normalize(matrix);
                xyz2srgb_normalized
            },
            ColorSpaceMatrix::CameraToXYZ => {
                let d65 = raw_image.camera.color_matrix[&Illuminant::D65].clone();
                let components = d65.len() / 3;
                let mut xyz2cam: [[f32; 3]; 3] = [[0.0; 3]; 3];
                for i in 0..components {
                    for j in 0..3 {
                        xyz2cam[i][j] = d65[i * 3 + j];
                    }
                }
                let xyz2cam_normalized = rawler::imgop::matrix::normalize(xyz2cam);
                let cam2xyz = rawler::imgop::matrix::pseudo_inverse(xyz2cam_normalized);
                cam2xyz
            },
        };

        image.data.par_iter_mut().for_each(|p|{
            let [r, g, b] = *p;
            *p = [
                foward_matrix[0][0] * r + foward_matrix[0][1] * g + foward_matrix[0][2] * b,
                foward_matrix[1][0] * r + foward_matrix[1][1] * g + foward_matrix[1][2] * b,
                foward_matrix[2][0] * r + foward_matrix[2][1] * g + foward_matrix[2][2] * b,
            ]
        });
        return image
    }

    fn get_name(&self) -> String{
        return "CST".to_string()
    }

    fn set_cache(&mut self, cache: PipelineImage){
        self.cache = cache
    }

    fn get_cache(&self) -> PipelineImage{
        self.cache.clone()
    }
}

// pub fn lineal_mask(height: usize, width: usize) -> Vec<f32> {
//     let mut result = vec![0.0; width*height];
//     result.par_iter_mut().enumerate().for_each(|(i, val)|{
//         let x = i % width;
//         let y = (i - x) / width;
//         // *val = 1.0/(((height-y) as f32 * 0.01) + 1.0)
//         *val = 1.0
//     });
//     result
// }
// pub fn get_channel(c: usize, data: &mut Array3<f32>) -> Array3<f32>{
//     let shape = data.shape();

//     let mut final_image = Array3::<f32>::zeros((shape[0], shape[1], 3));
//     final_image.slice_mut(s![.., ..,c]).assign(&data.slice(s![.., .., c]));
//     return final_image
// }

// pub fn film_curve(p: f64, d: f64, a: f64, b: f64, p2: f64, data: &mut Array3<f32>) -> Array3<f32> {
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

//     data.par_mapv_inplace(|x| f(x as f64) as f32);
//     return data.clone();
// }

// fn small(v: Array3<f32>) -> Array3<f32> {
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
