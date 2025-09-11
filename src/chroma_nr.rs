use crate::wavelets::{Kernel, WaveletDecompose};
use crate::conditional_paralell::prelude::*;
use crate::cst::{xyz_to_oklab, oklab_to_xyz};

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

pub fn denoise_chroma(
    image: Vec<[f32; 3]>,
    width: usize,
    height: usize,
    num_scales: usize,
    strength: f32,
) -> Vec<[f32; 3]> {
    // Convert image to Oklab
    let oklab_image: Vec<[f32; 3]> = image.par_iter().map(|p| xyz_to_oklab(p)).collect();

    let transform = ATrousTransform {
        input: oklab_image,
        height,
        width,
        kernel: Kernel::B3SplineKernel,
        levels: num_scales,
        current_level: 0,
    };

    let threshold = [
        0.1,
        0.1,
        0.3,
        0.5,
        0.0,
        0.0,
    ];

    let layers: Vec<[f32; 3]> = transform
        .into_iter()
        .skip(1)
        .map(|item| {
            let mut data = item.buffer;
            if let Some(scale) = item.pixel_scale {
                if scale < threshold.len() {
                    let th = 1.0 - (threshold[scale] * strength).min(1.0);
                    data.par_iter_mut().for_each(|val| {
                        // val is [L, a, b]
                        // Only denoise chroma channels 'a' and 'b'
                        val[1] *= th;
                        val[2] *= th;
                    });
                }
            }
            data
        })
        .reduce(|acc, val| {
            val.par_iter()
                .zip(acc)
                .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                .collect()
        })
        .unwrap();

    // Convert back to XYZ
    layers.par_iter().map(|p| oklab_to_xyz(p)).collect()
}