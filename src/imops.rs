use std::usize;

use ndarray::{s, Array2, Array3};
use rawler::{imgop::xyz::Illuminant, RawImage};

#[derive(Clone)]
pub struct FormedImage {
    pub raw_image: RawImage,
    pub data: Array3<f32>,
}

pub trait PipelineModule {
    fn process(&self, image: &mut FormedImage) -> FormedImage;
}

pub struct Exp {
    pub ev: f32
}

pub struct Sigmoid {
    pub c: f32
}

pub struct Contrast {
    pub c: f32
}

pub struct Wb {
}

pub enum ColorSpaceMatrix {
    CameraToXYZ,
    XYZTOsRGB
}

pub struct CST {
    pub color_space: ColorSpaceMatrix,
}

impl PipelineModule for Exp {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        let value = (2.0 as f32).powf(self.ev);
        image.data.par_mapv_inplace(|v| v*value);
        return image.clone();
    }
}

impl PipelineModule for Sigmoid {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        image.data.par_mapv_inplace(|v| (1.0 / (1.0 + (1.0/(self.c*v)))).powf(2.0));
        return image.clone();
    }
}

impl PipelineModule for Contrast {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        const MIDDLE_GRAY: f32 = 0.18;
        image.data.par_mapv_inplace(|v| (MIDDLE_GRAY*(v/MIDDLE_GRAY)).powf(self.c));
        return image.clone();
    }
}

impl PipelineModule for Wb {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        let rv = image.raw_image.wb_coeffs[0];
        let gv = image.raw_image.wb_coeffs[1]; 
        let bv = image.raw_image.wb_coeffs[2];
        // println!("{:?}", image.raw_image.wb_coeffs);

        let r = image.data.clone();
        let r1 = r.slice(s![.., ..,0]);

        let g = image.data.clone();
        let g1 = g.slice(s![.., ..,1]);

        let b = image.data.clone();
        let b1 = b.slice(s![.., ..,2]);

        image.data.slice_mut(s![.., ..,0]).assign(&r1.map(|r|r*rv).clone());
        image.data.slice_mut(s![.., ..,1]).assign(&g1.map(|g|g*gv).clone());
        image.data.slice_mut(s![.., ..,2]).assign(&b1.map(|b|b*bv).clone());
        return image.clone()
    }
}

impl PipelineModule for CST {
    fn process(&self, image: &mut FormedImage) -> FormedImage {
        let foward_matrix = match self.color_space {
            ColorSpaceMatrix::XYZTOsRGB => {
                let matrix = [
                    [3.240479, -1.537150, -0.498535,],
                    [-0.969256,  1.875991,  0.041556,],
                    [0.055648, -0.204043,  1.057311]
                ];
                let xyz2srgb_normalized = rawler::imgop::matrix::normalize(matrix);
                let xyz_to_srgb_matrix = Array2::<f32>::from_shape_vec((3,3), xyz2srgb_normalized.to_vec().as_flattened().to_vec()).unwrap();
                xyz_to_srgb_matrix
            },
            ColorSpaceMatrix::CameraToXYZ => {
                let d65 = image.raw_image.camera.color_matrix[&Illuminant::D65].clone();
                println!("{:?}", d65);
                let components = d65.len() / 3;
                let mut xyz2cam: [[f32; 3]; 3] = [[0.0; 3]; 3];
                for i in 0..components {
                    for j in 0..3 {
                        xyz2cam[i][j] = d65[i * 3 + j];
                    }
                }
                let xyz2cam_normalized = rawler::imgop::matrix::normalize(xyz2cam);
                let cam2xyz = rawler::imgop::matrix::pseudo_inverse(xyz2cam_normalized);
                let cam2xyz_matrix = Array2::<f32>::from_shape_vec((3,3), cam2xyz.to_vec().as_flattened().to_vec()).unwrap();
                cam2xyz_matrix
            },
        };

        let (rows, cols, _) = image.data.dim();
        let mut corrected_image = Array3::<f32>::zeros((rows, cols, 3));
        for r in 0..rows {
            for c in 0..cols {
                let pixel = image.data.slice(s![r, c, ..]);
                let rp = pixel[0];
                let gp = pixel[1];
                let bp = pixel[2];
                corrected_image[[r, c, 0]] = foward_matrix[[0, 0]] * rp + foward_matrix[[0, 1]] * gp + foward_matrix[[0, 2]] * bp;
                corrected_image[[r, c, 1]] = foward_matrix[[1, 0]] * rp + foward_matrix[[1, 1]] * gp + foward_matrix[[1, 2]] * bp;
                corrected_image[[r, c, 2]] = foward_matrix[[2, 0]] * rp + foward_matrix[[2, 1]] * gp + foward_matrix[[2, 2]] * bp;
            }
        }
        image.data = corrected_image;
        return image.clone()
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
