use std::usize;

use color::{ColorSpace, Oklch, XyzD65};
use rawler::{imgop::xyz::Illuminant, pixarray::RgbF32, RawImage};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use crate::{chroma_nr, helpers::*};

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
pub struct HighlightReconstruction {
}

fn get_cliped_channels(image: &FormedImage, pixel: [f32; 3]) -> [bool; 3]{
    let [clip_r, clip_g, clip_b, _] = image.raw_image.wb_coeffs;
    let [r, g, b] = pixel;
    [
        r >= clip_r,
        g >= clip_g,
        b >= clip_b,
    ]
}

fn reconstruct_pixel<I>(sorrounding_pixels: &I, channel: usize) -> f32
where 
    I: Iterator<Item = [f32; 3]> + Clone + ExactSizeIterator + Sync,
{

    let other_channels = (
        (channel + 1) % 3,
        (channel + 2) % 3
        );

    let len = sorrounding_pixels.len() as f32;

    let px = sorrounding_pixels.clone().reduce(|[ar, ag, ab], [r, g, b]| [ar + r, ag + g, ab + b]).unwrap();
    (px[other_channels.0]+px[other_channels.1])/(2.0*len)
}

impl PipelineModule for HighlightReconstruction {

    fn process(&self, mut image: FormedImage) -> FormedImage {
        let corrected_pixels= image.data.data.par_iter().enumerate().map(|(idx, pixel)|{
            let sorrounding_pixels = image.data.get_px_tail(1, idx).into_iter();
            let mut reconstructed_pixel: [f32; 3] = *pixel;
            let [cliped_r, cliped_g, cliped_b] = get_cliped_channels(&image, *pixel);

            if cliped_r {
                    reconstructed_pixel[0] = reconstruct_pixel(&sorrounding_pixels, 0);
            }

            if cliped_g {
                    reconstructed_pixel[1] = reconstruct_pixel(&sorrounding_pixels, 1);
            }

            if cliped_b {
                    reconstructed_pixel[2] = reconstruct_pixel(&sorrounding_pixels, 2);
            }
            reconstructed_pixel
        });
        image.data.data = corrected_pixels.collect();
        return image
    }

    fn get_name(&self) -> String{
        return "HighlightReconstruction".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChromaDenoise {
	  a: f32,
    b: f32,
    strength: f32
}

impl PipelineModule for ChromaDenoise {
    fn process(&self, mut image: FormedImage) -> FormedImage {

        let data_l = image.data.data.par_iter().map(|p|{
            let [l, _h, _c] = XyzD65::convert::<Oklch>(*p);
            l
        });

        let vres: Vec<Vec<f32>> = (0..3).into_par_iter().map(|channel|{
            let channel_data = image.data.data.par_iter().map(|pixel|pixel[channel]).collect();
            chroma_nr::denoise(channel_data, image.data.width, image.data.height, 6, 3)
        }).collect();

        let res = (0..(image.data.width*image.data.height)).into_par_iter().map(|idx|{
            [
                vres[0][idx],
                vres[1][idx],
                vres[2][idx],
            ]
        });

        let denoised: Vec<[f32; 3]> = res.into_par_iter().zip(data_l).map(|([r, g, b], l)|{
            let [_l, c, h] = XyzD65::convert::<Oklch>([r,g, b]);
            Oklch::convert::<XyzD65>([l, c, h])
        }).collect();

        image.data.data = denoised;
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
        let result = image.data.data.par_iter().map(|p| p.map(|x| x*value));
        // image.data = RgbF32::new_with(result.collect(), image.data.width, image.data.height);
        image.data.data = result.collect();
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
        let c = 1.0 + (1.0/scaled_one).powi(2);
        let result = image.data.data.par_iter().map(|p|{
            p.map(|x| (c / (1.0 + (1.0/(self.c*x)))).powi(2))
        });
        image.data.data = result.collect();
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
