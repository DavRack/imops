use std::usize;

use ndarray::{s, Array3};
use rawler::{imgop::xyz::Illuminant, pixarray::RgbF32, RawImage};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[derive(Clone)]
pub struct FormedImage {
    pub raw_image: RawImage,
    pub data: RgbF32,
}

pub trait PipelineModule {
    fn process(&self, image: FormedImage) -> FormedImage;
    fn get_name(&self) -> String;
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
        let result = image.data.data.par_iter().map(|p| p.map(|x| (1.0 / (1.0 + (1.0/(self.c*x)))).powf(2.0)));
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
    pub c: f32,
    pub m: f32,
    pub p: f32,
}

impl PipelineModule for LS {
    fn process(&self, mut image: FormedImage) -> FormedImage {
        let f = |x: f32| x*((self.c/(1.0+(1.0/(2.0_f32.powf((self.m/self.p)*(x-self.p))))))+1.0);
        let result = image.data.data.par_iter().map(|p|p.map(f));
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

pub fn get_channel(c: usize, data: &mut Array3<f32>) -> Array3<f32>{
    let shape = data.shape();

    let mut final_image = Array3::<f32>::zeros((shape[0], shape[1], 3));
    final_image.slice_mut(s![.., ..,c]).assign(&data.slice(s![.., .., c]));
    return final_image
}

pub fn film_curve(p: f64, d: f64, a: f64, b: f64, p2: f64, data: &mut Array3<f32>) -> Array3<f32> {
    let f1 = |x: f64| d*(x-p)+p;

    let c2 = (d*a)/f1(a);
    let c1 = (d*a)/(c2*(a.powf(c2)));
    let c4 = (-d*(p2-b))/(f1(b)-1.0);
    let c3 = (-d)/(c4*(p2-b).powf(c4-1.0));
    
    let f2 = |x: f64| c1*(x.powf(c2));
    let f3 = |x: f64| c3*(-x+p2).powf(c4)+1.0;

    let f = |x: f64| 
        if x > p2 { 1.0 }
        else if b < x && x <= p2 { f3(x) }
        else if a < x && x <= b { f1(x) }
        else if 0.0 <= x && x <= a { f2(x) }
        else {0.0}
    ;

    data.par_mapv_inplace(|x| f(x as f64) as f32);
    return data.clone();
}
