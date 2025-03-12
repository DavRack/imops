use core::panic;

use image::buffer;
use rand::seq::IndexedRandom;
use rawler::pixarray::{PixF32, RgbF32};
use rayon::prelude::*;
use rustfft::{num_complex::ComplexFloat, num_traits::real::Real};

use crate::helpers::{self, PixelTail, Stats};

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

use ndarray::Array2;
use image_dwt::{
    self, layer::{WaveletLayer, WaveletLayerBuffer}, recompose::OutputLayer, transform::ATrousTransformInput, ATrousTransform, RecomposableWaveletLayers, Rescale
};

pub fn denoise(
    image: Vec<f32>,
    width: usize,
    height: usize,
    num_scales: usize,
    v: usize,
) -> Vec<f32> {
    let grayscale_image = ATrousTransformInput::Grayscale {
        data: Array2::from_shape_vec((height, width), image).unwrap()
    };

    // let transform = ATrousTransform{
    //     input: grayscale_image.clone(),
    //     kernel: image_dwt::Kernel::B3SplineKernel,
    //     levels: num_scales,
    //     current_level: 0,
    // };

    // for item in transform.into_iter(){
    //     let data = match item.buffer {
    //         WaveletLayerBuffer::Grayscale { data } => data,
    //         _ => panic!(),
    //     };
    //     println!("{:?}", data);
    //     println!("{:?}", item.pixel_scale);
    // }

    let transform = ATrousTransform{
        input: grayscale_image,
        kernel: image_dwt::Kernel::B3SplineKernel,
        levels: num_scales,
        current_level: 0,
    };

    let threshold = [
        0.1,
        0.2,
        0.3,
        0.0,
        0.0,
        0.0,
    ];

    let recomposed = transform
        .into_iter()
        // Skip pixel scale 0 layer for noise removal
        .skip(0)
        // Only take layers where pixel scale is less than 2
        // .filter(|item| item.pixel_scale.is_some_and(|scale| scale == 0))
        .map(|item|{
            if let Some(scale) = item.pixel_scale {
                if scale < v {
                    let data = match item.buffer {
                        WaveletLayerBuffer::Grayscale { data } => data,
                        _ => panic!(),
                    };

                    let new_data: Vec<f32> = data.clone().into_raw_vec().into_par_iter().map(|v|{
                        v*threshold[scale]
                    }).collect();

                    // let len = new_data.len();
                    // let new_data = PixF32::new_with(new_data, width, height);
                    // let new_data = (0..len).into_par_iter().map(|idx| {
                    //     new_data.get_px_tail(1, idx).iter().median()
                    // }).collect();

                    // let new_data = PixF32::new_with(new_data, width, height);
                    // let new_data = (0..len).into_par_iter().map(|idx| new_data.get_px_tail(2, idx).iter().mean()).collect();


                    let filtered_data = Array2::from_shape_vec(data.dim(), new_data).unwrap();

                    let new = WaveletLayer{
                        buffer: WaveletLayerBuffer::Grayscale { data: filtered_data },
                        pixel_scale: item.pixel_scale
                    };
                    new
                }else{
                    item
                }
            }else{
                item
            }
        })
        // Recompose processed layers into final image
        .recompose_into_vec(height, width, OutputLayer::Grayscale);
    recomposed
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
