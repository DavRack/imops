use rayon::prelude::*;
use ndarray::Array2;
use image_dwt::{
    self, layer::WaveletLayerBuffer, transform::ATrousTransformInput, ATrousTransform
};

#[inline(always)]
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
        0.5,
        0.0,
        0.0,
    ];

    let layers: Vec<Vec<f32>> = transform
        .into_iter()
        .map(|item|{
            let data = match item.buffer {
                WaveletLayerBuffer::Grayscale { data } => data.into_raw_vec(),
                _ => panic!(),
            };
            if item.pixel_scale.is_some_and(|scale| scale < v ){
                let scale = item.pixel_scale.unwrap();
                let th = threshold[scale]/1.0;

                let new_data: Vec<f32> = data.into_iter().map(|v|{
                    v*th
                }).collect();

                new_data
            }else{
                data
            }
        }).collect();

    let layers: Vec<f32> = (0..width*height).into_par_iter().map(|idx|{
            layers.iter().map(|layer|layer[idx]).sum::<f32>()
        }).collect() ;
    return layers
}
