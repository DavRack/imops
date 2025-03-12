use core::panic;
use rayon::prelude::*;
use ndarray::Array2;
use image_dwt::{
    self, layer::WaveletLayerBuffer, recompose::OutputLayer, transform::ATrousTransformInput, ATrousTransform, RecomposableWaveletLayers
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
        .map(|mut item|{
            if item.pixel_scale.is_some_and(|scale| scale < v ){
                let scale = item.pixel_scale.unwrap();
                let data = match item.buffer {
                    WaveletLayerBuffer::Grayscale { data } => data,
                    _ => panic!(),
                };

                let new_data: Vec<f32> = data.into_par_iter().map(|v|{
                    v*threshold[scale]
                }).collect();


                let filtered_data = Array2::from_shape_vec((height, width), new_data).unwrap();
                item.buffer = WaveletLayerBuffer::Grayscale { data: filtered_data };

                item
            }else{
                item
            }
        })
        .recompose_into_vec(height, width, OutputLayer::Grayscale);
    recomposed
}
