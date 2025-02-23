use std::usize;

use color::{ColorSpace, Oklch, XyzD65};
use rawler::{imgop::xyz::Illuminant, pixarray::RgbF32, RawImage};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::denoise;

#[derive(Clone)]
pub struct FormedImage {
    pub raw_image: RawImage,
    pub max_raw_value: f32,
    pub data: RgbF32,
}

pub trait PipelineModule {
    fn process(&self, image: FormedImage) -> FormedImage;
    fn get_name(&self) -> String;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct  LCH{
    pub lc: f32,
    pub cc: f32,
    pub hc: f32,
}

impl PipelineModule for LCH {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        image.data.data = image.data.data.par_iter().map(
            |p| {
                let [l, c, h] = XyzD65::convert::<Oklch>(*p);
                let xyz = Oklch::convert::<XyzD65>([l*self.lc, c*self.cc, h*self.hc]);
                xyz
            }
        ).collect();
        return image
    }

    fn get_name(&self) -> String{
        return "LCH".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChromaDenoise {
}

impl PipelineModule for ChromaDenoise {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let lch_data = image.data.data.par_iter().map(|p|{
            XyzD65::convert::<Oklch>(*p)
        });
        // let lambda = 0.00189442719;
        // let convergence_threshold = 10_f64.powi(-10);
        // let max_iter = 100;
        // let tau: f64 = 1.0 / 2_f64.sqrt();
        // let sigma: f64 = 1_f64 / (8.0 * tau);
        // let gamma: f64 = 0.35 * lambda;

        // let c_data: Vec<f32> = lch_data.clone().map(|[_, c, _]| c).collect();
        let mut c_data = image.data.clone();
        c_data.data = lch_data.clone().map(|[_,c,_]| [c as f32, c as f32, c as f32]).collect();

        let denoised_c = denoise::denoise(c_data.data, image.data.height, image.data.width);
        image.data.data = denoised_c;
        // let denoise_image = lch_data.zip(denoised_c.data).map(|([l,_,h], [_, c, _])| [l, c, h]);
        // image.data.data = denoise_image.map(|p|Oklch::convert::<XyzD65>(p)).collect();
        // let denoise_image = lch_data.zip(denoised_c.data).map(|([l,_,h], [_, c, _])| [c, c, c]).collect();
        // image.data.data = denoise_image;
        return image;
    }

    fn get_name(&self) -> String{
        return "ChromaDenoise".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Exp {
    pub ev: f32
}

impl PipelineModule for Exp {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let value = (2.0 as f32).powf(self.ev);
        let result = image.data.data.par_iter().map(|p| p.map(|x|x*value));
        image.data = RgbF32::new_with(result.collect(), image.data.width, image.data.height);
        return image;
    }

    fn get_name(&self) -> String{
        return "Exp".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Sigmoid {
    pub c: f32
}

impl PipelineModule for Sigmoid {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let max_current_value = image.data.data.as_flattened().iter().cloned().reduce(f32::max).unwrap();
        let scaled_one = (1.0/image.max_raw_value)*max_current_value;
        let c = 1.0 + (1.0/scaled_one).powf(2.0);
        println!("scaled one {:}", scaled_one);
        let result = image.data.data.par_iter().map(|p| p.map(|x| (c / (1.0 + (1.0/(self.c*x)))).powf(2.0)));
        image.data = RgbF32::new_with(result.collect(), image.data.width, image.data.height);
        return image
    }

    fn get_name(&self) -> String{
        return "Sigmoid".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Contrast {
    pub c: f32
}

impl PipelineModule for Contrast {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        const MIDDLE_GRAY: f32 = 0.1845;
        let f = |x: f32| (MIDDLE_GRAY*(x/MIDDLE_GRAY)).powf(self.c);
        image.data.data = image.data.data.par_iter().map(|p|p.map(f)).collect();
        return image
    }

    fn get_name(&self) -> String{
        return "Contrast".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CFACoeffs {
}

impl PipelineModule for CFACoeffs {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let rv = image.raw_image.wb_coeffs[0];
        let gv = image.raw_image.wb_coeffs[1]; 
        let bv = image.raw_image.wb_coeffs[2];
        let result = image.data.data.par_iter().map(|[r,g,b]| [r*rv, g*gv, b*bv]);
        image.data = RgbF32::new_with(result.collect(), image.data.width, image.data.height);
        return image
    }

    fn get_name(&self) -> String{
        return "CFACoeffs".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalExpousure {
    pub c: f32,
    pub m: f32,
    pub p: f32,
}

impl PipelineModule for LocalExpousure {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let f = |x: f32| x*((self.c*(2.0_f32.powf(-((x-self.p).powf(2.0)/self.m))))+1.0);
        let result = image.data.data.par_iter().map(|p|p.map(f));
        image.data = RgbF32::new_with(result.collect(), image.data.width, image.data.height);
        return image

    }

    fn get_name(&self) -> String{
        return "LocalExpousure".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LS {
    pub transition_width: f32,
    pub shadows_exp: f32,
    pub highlits_exp: f32,
    pub pivot: f32, //ev
}

impl PipelineModule for LS {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let p: f32 = self.pivot;
        let d: f32 = self.transition_width;
        let f2 = |x: f32, m: f32| p*(x/p).powf(m);
        let f = |x: f32, pf: f32| 1.0/(1.0+(1.0/(2.0_f32.powf(d*(x-(p*pf))))));

        // shadows
        let ms: f32 = 1.0/self.shadows_exp;
        let pfs: f32 = 0.8;
        // let result = image.data.data.par_iter().map(|p|p.map(|x| (f2(x, ms)*(1.0-f(x, pf)))+f(x, pf)*x));
        //
        //// heights
        let mh: f32 = self.highlits_exp;
        let pfh: f32 = 1.2;

        let complete_f = |x| (((f2(x, ms)*(1.0-f(x, pfs)))+f(x, pfs)*x)+((f2(x, mh)*f(x, pfh))+((1.0-f(x, pfh))*x)))/2.0;

        let result = image.data.data.par_iter().map(|p|p.map(|x| complete_f(x)));

        image.data = RgbF32::new_with(result.collect(), image.data.width, image.data.height);
        return image

    }

    fn get_name(&self) -> String{
        return "LS".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CST {
    pub color_space: ColorSpaceMatrix,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ColorSpaceMatrix {
    CameraToXYZ,
    XYZTOsRGB
}


impl PipelineModule for CST {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let foward_matrix = match self.color_space {
            ColorSpaceMatrix::XYZTOsRGB => {
                let matrix = [
                    [3.240479, -1.537150, -0.498535,],
                    [-0.969256,  1.875991,  0.041556,],
                    [0.055648, -0.204043,  1.057311]
                ];
                let xyz2srgb_normalized = rawler::imgop::matrix::normalize(matrix);
                xyz2srgb_normalized
            },
            ColorSpaceMatrix::CameraToXYZ => {
                let d65 = image.raw_image.camera.color_matrix[&Illuminant::D65].clone();
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

        image.data.data = image.data.data.par_iter().map(|[rp, gp, bp]|{
            [
                foward_matrix[0][0] * rp + foward_matrix[0][1] * gp + foward_matrix[0][2] * bp,
                foward_matrix[1][0] * rp + foward_matrix[1][1] * gp + foward_matrix[1][2] * bp,
                foward_matrix[2][0] * rp + foward_matrix[2][1] * gp + foward_matrix[2][2] * bp,
            ]
        }).collect();
        return image
    }

    fn get_name(&self) -> String{
        return "CST".to_string()
    }
}

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
