use crate::wavelets::{Kernel, WaveletDecompose};
use crate::conditional_paralell::prelude::*;
use crate::cst::{xyz_to_oklab, oklab_to_xyz};
use crate::nl_means;
use crate::chroma_nr::{ATrousTransform}; // Re-use the transform struct

pub fn denoise(
    image: Vec<[f32; 3]>,
    width: usize,
    height: usize,
    num_scales: usize,
    patch_radius: usize,
    search_radius: usize,
    h: f32,
) -> Vec<[f32; 3]> {
    // 1. Convert image to Oklab
    let oklab_image: Vec<[f32; 3]> = image.par_iter().map(|p| xyz_to_oklab(p)).collect();

    // 2. Decompose into wavelet layers
    let transform = ATrousTransform {
        input: oklab_image,
        height,
        width,
        kernel: Kernel::B3SplineKernel,
        levels: num_scales,
        current_level: 0,
    };

    // 3. Denoise chroma channels of each detail layer with NL-Means
    let layers: Vec<[f32; 3]> = transform
        .into_iter()
        .skip(1) // Skip the base layer
        .map(|item| {
            let mut data = item.buffer;
            if let Some(scale) = item.pixel_scale {
                // Only denoise the first few scales
                if scale < 3 {
                    // Extract 'a' and 'b' channels
                    let a_channel: Vec<f32> = data.iter().map(|p| p[1]).collect();
                    let b_channel: Vec<f32> = data.iter().map(|p| p[2]).collect();

                    // Denoise each channel
                    // Adjust 'h' based on wavelet scale. Finer scales have more noise.
                    let scale_h = h / (scale as f32 + 1.0);
                    let denoised_a = nl_means::denoise(&a_channel, width, height, patch_radius, search_radius, scale_h);
                    let denoised_b = nl_means::denoise(&b_channel, width, height, patch_radius, search_radius, scale_h);

                    // Recombine channels
                    data.iter_mut().zip(denoised_a).zip(denoised_b).for_each(|((pixel, new_a), new_b)| {
                        pixel[1] = new_a;
                        pixel[2] = new_b;
                    });
                }
            }
            data
        })
        .reduce(|acc, val| { // 4. Reconstruct image
            val.par_iter()
                .zip(acc)
                .map(|(a, b)| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
                .collect()
        })
        .unwrap();

    // 5. Convert back to XYZ
    layers.par_iter().map(|p| oklab_to_xyz(p)).collect()
}
