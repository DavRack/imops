use core::panic;
use std::usize;

use image_dwt::RecomposableWaveletLayers;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
pub fn denoise(img: Vec<[f32; 3]>, height: usize, width: usize) -> Vec<[f32; 3]> {
    let radius: i32 = 2; // Radius of the kernel
    let spatial_sigma:f32 = 10.0; // Spatial sigma
    let sigma_spatial_squared = spatial_sigma.powi(2);

    let denoised_img = img.par_iter().enumerate().map(|(index, _)|{
        let x = index % width;
        let y = (index-x)/width;

        let mut weights_and_values = Vec::with_capacity((radius*2).pow(2) as usize);
        for wy in -radius..=radius {
            for wx in -radius..=radius {
                let neighbor_y = (y as i32 + wy).clamp(0, (height - 1) as i32) as usize;
                let neighbor_x = (x as i32 + wx).clamp(0, (width - 1) as i32) as usize;
                let neighbor_pixel = img[neighbor_y * width + neighbor_x];

                // Calculate spatial distance
                let spatial_distance_squared = (wx as f32).powi(2) + (wy as f32).powi(2);
                let spatial_weight = (spatial_distance_squared*-0.5 / sigma_spatial_squared).exp();

                weights_and_values.push((spatial_weight, neighbor_pixel));
            }
        }
        let denoised_pixel = weighted_average_vec(weights_and_values.into_iter());
        denoised_pixel
    }).collect();
    denoised_img
}

fn weighted_average_vec(weights_and_values: impl Iterator<Item = (f32, [f32; 3])>) -> [f32; 3] {
    let mut weights_sum = 0.0;
    let mut weighted_channel_sums = [0.0, 0.0, 0.0];

    for (weight, pixel) in weights_and_values {
        weights_sum += weight;
        for i in 0..3 {
            weighted_channel_sums[i] += weight * pixel[i];
        }
    }

    if weights_sum == 0.0 {
        return [0.0, 0.0, 0.0]; // Avoid division by zero if weights_sum is zero
    }

    let channel_averages = weighted_channel_sums.map(|x| x/weights_sum);
    channel_averages
}

pub fn denoise_w(
    image: Vec<f32>,
    width: usize,
    height: usize,
    a: f32,
    b: f32,
    strength: f32,
) -> Vec<f32> {
    // Apply Variance Stabilizing Transform (VST)
    let precond: Vec<f32> = image
        .iter()
        .map(|&x| 2.0 * ((x + b) / a).sqrt())
        .collect();

    let max_scale = 4; // Number of wavelet scales

    // Decompose the image into wavelet details
    let mut current = precond.clone();
    let mut details = Vec::with_capacity(max_scale);

    for s in 0..max_scale {
        let stride = 1 << s; // 2^s
        let next = atrous_convolve(&current, width, height, stride);
        let detail: Vec<f32> = current
            .iter()
            .zip(&next)
            .map(|(c, n)| c - n)
            .collect();
        details.push(detail);
        current = next;
    }

    // Apply thresholding to each detail layer
    for detail in details.iter_mut() {
        let var = compute_variance(detail);
        let sigma = var.sqrt();
        let threshold = strength * sigma;

        for coeff in detail.iter_mut() {
            let abs_coeff = coeff.abs();
            if abs_coeff > threshold {
                *coeff = coeff.signum() * (abs_coeff - threshold);
            } else {
                *coeff = 0.0;
            }
        }
    }

    // Reconstruct the image from the wavelet details
    let mut denoised = current;
    for s in (0..max_scale).rev() {
        let stride = 1 << s;
        let upsampled = atrous_convolve(&denoised, width, height, stride);
        denoised = upsampled
            .iter()
            .zip(&details[s])
            .map(|(u, d)| u + d)
            .collect();
    }

    // Apply inverse VST
    denoised
        .into_iter()
        .map(|x| {
            let val = (x / 2.0).powi(2) * a - b;
            val.max(0.0)
        })
        .collect()
}

// Helper function to compute the à-trous convolution
fn atrous_convolve(image: &[f32], width: usize, height: usize, stride: usize) -> Vec<f32> {
    let temp = convolve_horizontal(image, width, height, stride);
    convolve_vertical(&temp, width, height, stride)
}

// Horizontal convolution with symmetric padding
fn convolve_horizontal(image: &[f32], width: usize, height: usize, stride: usize) -> Vec<f32> {
    let kernel = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let mut result = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for (i, &k) in kernel.iter().enumerate() {
                let offset = i as isize - 2;
                let pos = x as isize + offset * stride as isize;
                let pos = if pos < 0 {
                    (-pos - 1) as usize
                } else if pos >= width as isize {
                    (2 * (width as isize) - pos - 1) as usize
                } else {
                    pos as usize
                };

                sum += image[y * width + pos] * k;
            }

            result[y * width + x] = sum;
        }
    }

    result
}

// Vertical convolution with symmetric padding
fn convolve_vertical(image: &[f32], width: usize, height: usize, stride: usize) -> Vec<f32> {
    let kernel = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let mut result = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for (i, &k) in kernel.iter().enumerate() {
                let offset = i as isize - 2;
                let pos = y as isize + offset * stride as isize;
                let pos = if pos < 0 {
                    (-pos - 1) as usize
                } else if pos >= height as isize {
                    (2 * (height as isize) - pos - 1) as usize
                } else {
                    pos as usize
                };

                sum += image[pos * width + x] * k;
            }

            result[y * width + x] = sum;
        }
    }

    result
}

// Compute the variance of a slice of f32 values
fn compute_variance(data: &[f32]) -> f32 {
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>()
        / data.len() as f32
}
use ndarray::Array2;
use image_dwt::{
    layer::WaveletLayer, recompose::{self, OutputLayer}, transform::ATrousTransformInput, ATrousTransform, Kernel
    // Kernel, // Assuming Kernel is re-exported or available
};

pub fn apply_wavelet_denoising(
    mut image: Array2<f32>,
    levels: usize
) -> Vec<f32>{
    let gray_image_input = ATrousTransformInput::Grayscale { data: image.clone() }; // Clone here
    let transform = ATrousTransform{
        input: gray_image_input,
        levels,
        kernel: Kernel::B3SplineKernel,
        current_level: 0,
    };

    // Collect wavelet coefficients into a Vec to modify them
    let mut wavelet_items: Vec<WaveletLayer> = transform.clone()
        .into_iter()
        .skip(1) // Skip level 0 (original image) to process detail levels
        .collect();

    // Noise Thresholding (Soft Thresholding) - Apply to each detail level
    let sigma_est = estimate_sigma_atrous(&wavelet_items); // Estimate noise from wavelet coefficients
    // Corrected line: use image.len() to get total number of elements
    let universal_threshold = sigma_est * (2.0 * (image.len() as f32).ln()).sqrt(); // Universal threshold

    for item in wavelet_items.iter_mut() {
        // Access the coefficient Array2<f32> directly as item.data
        let detail_array = match &mut item.buffer {
            image_dwt::layer::WaveletLayerBuffer::Grayscale{data} => data,
            _ => panic!(""),
        }; // Get a mutable reference to the Array2

        detail_array.par_map_inplace(|coeff| {
            *coeff = soft_threshold(*coeff, universal_threshold);
        });
    }

    let [height, width] = image.shape() else {panic!("")}; // Corrected variable name to width
    let recomposed: Vec<WaveletLayer> = transform
        .into_iter()
        .skip(1) // Skip level 0 (original image) to process detail levels
        .map(|mut item: WaveletLayer| { // Use map to modify WaveletLayer in place
            let detail_array = match &mut item.buffer {
                image_dwt::layer::WaveletLayerBuffer::Grayscale{data} => data,
                _ => panic!(""),
            }; // Get a mutable reference to the Array2

            detail_array.par_map_inplace(|coeff| {
                *coeff = soft_threshold(*coeff, universal_threshold);
            });
            item // Return the modified WaveletLayer
        }).collect();
    let recomposed = recomposed.into_iter()
        .recompose_into_vec(*height as usize, *width as usize, OutputLayer::Grayscale); // Corrected width/height order


    // Corrected line: return recomposed directly as it's already Vec<f32>
    recomposed // Return directly as Vec<f32>
}


// Soft thresholding function (same as before)
fn soft_threshold(coefficient: f32, threshold: f32) -> f32 {
    if coefficient > threshold {
        coefficient - threshold
    } else if coefficient < -threshold {
        coefficient + threshold
    } else {
        0.0
    }
}

// Estimate noise standard deviation (sigma) for ATrous transform
// Here, we estimate from the coefficients of all detail levels combined for simplicity
fn estimate_sigma_atrous(wavelet_items: &Vec<WaveletLayer>) -> f32 {
    let mut all_detail_coeffs: Vec<f32> = Vec::new();
    for item in wavelet_items.iter() {
        // Access coefficients as item.data.as_slice().unwrap()
        all_detail_coeffs.extend(
            match &item.buffer {
                image_dwt::layer::WaveletLayerBuffer::Grayscale{data} => data,
                _ => panic!(""),
            } // Get a mutable reference to the Array2
        ); // Flatten detail array
    }

    if all_detail_coeffs.is_empty() {
        return 1.0; // Fallback if no detail coefficients found
    }

    let mut abs_coeffs: Vec<f32> = all_detail_coeffs.iter().map(|coeff| coeff.abs()).collect();
    abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_abs_coeffs = if abs_coeffs.len() % 2 == 0 {
        (abs_coeffs[abs_coeffs.len() / 2 - 1] + abs_coeffs[abs_coeffs.len() / 2]) / 2.0
    } else {
        abs_coeffs[abs_coeffs.len() / 2]
    };

    median_abs_coeffs / 0.6745 // Robust MAD estimator
}
