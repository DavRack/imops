use core::panic;

use image::buffer;
use rand::seq::IndexedRandom;
use rawler::pixarray::{PixF32, RgbF32};
use rayon::prelude::*;

use crate::helpers::{PixelTail, Stats};

#[derive(Clone, Copy)]
pub struct ChromaDenoise<'a> {
    pub tail_radious: usize,
    pub image: &'a RgbF32
}
impl <'a> ChromaDenoise<'a> {
    pub fn ai_denoise(self) -> Vec<f32>{
        vec![]
    }
    // pub fn denoise_image(self) -> Vec<[f32; 3]>{
    //     // // let new_data: Vec<[f32; 3]> = self.image.data.par_iter().enumerate().map(|(idx, _)|{
    //     // //     self.filtered_pixel(idx)
    //     // // }).collect();

    //     // // return new_data;
    //     // self.ai_denoise();
    //     // let input_img: Vec<f32> = self.image.data;
    //     // let mut filter_output = vec![0.0f32; input_img.len()];

    //     // let device = oidn::Device::new();
    //     // filter_output
    // }

    fn filtered_pixel(self, idx: usize) -> [f32; 3]{
        let tail = self.image.get_tail(self.tail_radious, idx);
        let center_pixel = tail[tail.len()/2].1;
        let r_pixels: Vec<f32> = tail.iter().map(|(_, [r, _, _])|*r).collect();
        let g_pixels: Vec<f32> = tail.iter().map(|(_, [_, g, _])|*g).collect();
        let b_pixels: Vec<f32> = tail.iter().map(|(_, [_, _, b])|*b).collect();

        let g_variance = g_pixels.iter().variance();

        let [center_r, center_g, center_b] = center_pixel;
        let r_mean = r_pixels.iter().mean();
        let g_mean = g_pixels.iter().mean();

        let r_sign = (r_mean-center_r)/(r_mean-center_r).abs();
        let new_r = center_r+(r_sign*g_variance);

        let b_mean = b_pixels.iter().mean();

        let b_sign = (b_mean-center_b)/(b_mean-center_b).abs();
        let new_b = center_b+(b_sign*g_variance);

        return [r_mean, g_mean, b_mean]
    }
}

use ndarray::{Array2, s, Axis};
use rustfft::{FftPlanner, num_complex::Complex, FftDirection};
use image_dwt::{self, transform::ATrousTransformInput, ATrousTransform};
use image_dwt::decompose::WaveletDecompose;
use image_dwt::RecomposableWaveletLayers;
// use ndarray::{Array2, s};

pub fn denoise(
    image: &[f32],
    width: usize,
    height: usize,
    num_scales: usize,
    strength: f32,
) -> Vec<f32> {
    return image.to_vec()
    // let img_array = Array2::from_shape_vec((height, width), image.to_vec())
    //     .expect("Invalid image dimensions");

    // // Step 1: Perform à trous wavelet decomposition
    //     let gray_image = ATrousTransformInput::Grayscale { data: Array2::from_shape_vec((width, height), image.to_vec()).unwrap() };
    //     let levels = 10;
    //     let wavelet = ATrousTransform{
    //         input: gray_image,
    //         levels,
    //         kernel: image_dwt::Kernel::B3SplineKernel,
    //         current_level: 0,
    //     };
    // // let data = match wavelet {
    // //     image_dwt::layer::WaveletLayerBuffer::Grayscale { data } => data,
    // //     _ => panic!()
        
    // // };
    // // wavelet.recompose_into_vec(width, height, image_dwt::recompose::OutputLayer::Grayscale);

    // let mut im = Array2::from_shape_vec((height, width), image.to_vec()).unwrap();
    // let buffer = im.wavelet_decompose(image_dwt::Kernel::B3SplineKernel, num_scales);
    
    // let buffer_data = match buffer {
    //     image_dwt::layer::WaveletLayerBuffer::Grayscale { data } => data,
    //     _ => panic!()
    // };
    // let raw_data = buffer_data.into_raw_vec();

    // return image.iter().zip(raw_data).map(|(og, d)| *og-d).collect()

    // // let data = wavelet.recompose_into_vec(width, height, image_dwt::recompose::OutputLayer::Grayscale);

    // // let wavelet = ATrousTransform::new(num_scales, Padding::Symmetric);
    // // let (approx, mut details) = wavelet.decompose(&img_array);

    // // Step 2: Noise estimation from finest detail coefficients
    // // println!("{:}", wavelet);
    // // let noise_sigma = estimate_noise(wavelet.) * strength;

    // // // Step 3: Bayesian shrinkage for each scale
    // // for (scale, detail) in details.iter_mut().enumerate() {
    // //     let scale_factor = 1.0 / (1 << scale) as f32;
    // //     let threshold = noise_sigma * scale_factor * ((2.0 * (detail.len() as f32).ln()) as f32 ).powf(0.5);
        
    // //     // Apply adaptive thresholding
    // //     detail.mapv_inplace(|x| {
    // //         let sign = x.signum();
    // //         let abs_val = x.abs();
    // //         sign * (abs_val - threshold).max(0.0)
    // //     });
    // // }

    // // // Step 4: Reconstruct from processed coefficients
    // // wavelet.reconstruct(approx, &details).into_raw_vec()
}

fn estimate_noise(detail: &Array2<f32>) -> f32 {
    // Median Absolute Deviation (MAD) estimator
    let mut abs_coeffs: Vec<f32> = detail.iter().map(|&x| x.abs()).collect();
    abs_coeffs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    
    let median = abs_coeffs[abs_coeffs.len() / 2];
    let mut deviations: Vec<f32> = abs_coeffs.iter()
        .map(|&x| (x - median).abs())
        .collect();
    deviations.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    
    deviations[deviations.len() / 2] / 0.6745
}
