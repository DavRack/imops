use crate::wavelets::{Kernel, WaveletDecompose};
use crate::conditional_paralell::prelude::*;

#[derive(Clone)]
pub struct ATrousTransform {
    pub input: Vec<[f32; 3]>,
    pub width: usize,
    pub height: usize,
    pub levels: usize,
    pub kernel: Kernel,
    pub current_level: usize,
}
#[derive(Clone)]
pub struct WaveletLayer {
    pub buffer: Vec<[f32; 3]>,
    pub pixel_scale: Option<usize>,
}

impl Iterator for ATrousTransform {
    type Item = WaveletLayer;

    fn next(&mut self) -> Option<Self::Item> {
        let pixel_scale = self.current_level;
        self.current_level += 1;

        if pixel_scale > self.levels {
            return None;
        }

        if pixel_scale == self.levels {
            return Some(WaveletLayer {
                buffer: self.input.clone(),
                pixel_scale: None,
            });
        }

        let layer_buffer = self.input.wavelet_decompose(self.height, self.width, self.kernel, pixel_scale);
        Some(WaveletLayer {
            pixel_scale: Some(pixel_scale),
            buffer: layer_buffer,
        })
    }
}

// pub fn denoise(
//     image: Vec<f32>,
//     width: usize,
//     height: usize,
//     num_scales: usize,
//     v: usize,
// ) -> Vec<f32> {
//     // let grayscale_image = ATrousTransformInput::Grayscale {
//     //     data: Array2::from_shape_vec((height, width), image).unwrap()
//     // };

//     let transform = ATrousTransform{
//         input: image,
//         height,
//         width,
//         kernel: Kernel::B3SplineKernel,
//         levels: num_scales,
//         current_level: 0,
//     };

//     let threshold = [
//         0.1,
//         0.2,
//         0.3,
//         0.5,
//         0.0,
//         0.0,
//     ];

//     let layers: Vec<Vec<f32>> = transform
//         .into_iter()
//         .map(|item|{
//             let data = item.buffer;
//             if item.pixel_scale.is_some_and(|scale| scale < v ){
//                 let scale = item.pixel_scale.unwrap();
//                 let th = threshold[scale]/1.0;

//                 let new_data: Vec<f32> = data.into_iter().map(|v|{
//                     v*th
//                 }).collect();

//                 new_data
//             }else{
//                 data
//             }
//         }).collect();

//     let layers: Vec<f32> = (0..width*height).into_par_iter().map(|idx|{
//             layers.iter().map(|layer|layer[idx]).sum::<f32>()
//         }).collect() ;
//     return layers
// }
pub fn denoise_rgb(
    image: Vec<[f32; 3]>,
    width: usize,
    height: usize,
    num_scales: usize,
    v: usize,
) -> Vec<[f32; 3]> {
    // let grayscale_image = ATrousTransformInput::Grayscale {
    //     data: Array2::from_shape_vec((height, width), image).unwrap()
    // };

    let transform = ATrousTransform{
        input: image,
        height,
        width,
        kernel: Kernel::B3SplineKernel,
        levels: num_scales,
        current_level: 0,
    };

    let threshold = [
        0.2,
        0.2,
        0.3,
        0.5,
        0.0,
        0.0,
    ];

    let layers: Vec<[f32; 3]> = transform
        .into_iter().skip(1)
        .map(|item|{
            let mut data = item.buffer;
            if item.pixel_scale.is_some_and(|scale| scale < v ){
                let scale = item.pixel_scale.unwrap();
                let th = threshold[scale];

                data.iter_mut().for_each(|v|{
                    *v = v.map(|x| x*th);
                });
            }
            data
        }).reduce(|acc, val| val.par_iter().zip(acc).map(|(a, b)|{
            let [a, x, c] = a;
            let [r, g, b] = b;
            [
                a + r,
                x + g,
                c + b,
            ]
        }).collect()).unwrap();

    return layers
}
