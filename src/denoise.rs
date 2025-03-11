// use std::vec::Vec;
// use std::f64;
// use fastcwt::{Transform, Wavelet, cwt2d, icwt2d};

// // --- Daubechies Wavelet ---
// // fastcwt supports Daubechies wavelets. We'll use db2.
// // No need to define filters manually, fastcwt provides them.


// // --- 1D Daubechies Wavelet Transform (Decomposition) using fastcwt ---
// fn daubechies_wavelet_1d_fastcwt(data: &Vec<f32>, level: usize) -> Vec<Vec<f32>> {
//     let wavelet = Wavelet::Daubechies { family: Wavelet::D2 };
//     let mut current_level_signal: Vec<f64> = data.iter().map(|&x| x as f64).collect();
//     let mut all_levels_coeffs: Vec<Vec<f32>> = Vec::new();

//     for _ in 0..level {
//         if current_level_signal.len() < 2 { // Stop if signal is too short
//             break;
//         }
//         let coeffs = Transform::new(&current_level_signal, wavelet, 1).unwrap(); // Decompose one level
//         let detail_coeffs: Vec<f32> = coeffs.high().iter().map(|&x| x as f32).collect();
//         all_levels_coeffs.push(detail_coeffs);
//         current_level_signal = coeffs.low().to_vec(); // Use approximation as next level input
//     }
//     all_levels_coeffs.push(current_level_signal.iter().map(|&x| x as f32).collect()); // Push final approx coefficients (LL)
//     all_levels_coeffs.reverse(); // Reverse to have LL at the end
//     all_levels_coeffs
// }


// // --- Inverse 1D Daubechies Wavelet Transform (Reconstruction) using fastcwt ---
// fn inverse_daubechies_wavelet_1d_fastcwt(wavelet_coeffs: &mut Vec<Vec<f32>>) -> Vec<f32> {
//     let wavelet = Wavelet::Daubechies { family: Wavelet পরিবার::D2 };
//     let mut current_level_a: Vec<f64> = wavelet_coeffs.pop().unwrap().iter().map(|&x| x as f64).collect(); // Start with coarsest approximation (LL)

//     while !wavelet_coeffs.is_empty() {
//         let level_d: Vec<f64> = wavelet_coeffs.pop().unwrap().iter().map(|&x| x as f64).collect(); // Detail coeffs
//         let reconstructed = Transform::inverse(&level_d, &current_level_a, wavelet, 1).unwrap();
//         current_level_a = reconstructed; // Reconstruct one level
//     }
//     current_level_a.iter().map(|&x| x as f32).collect()
// }



// // --- 2D Daubechies Wavelet Transform (Decomposition) - Multi-level using fastcwt ---
// fn daubechies_wavelet_2d_fastcwt(image: &Vec<f32>, width: usize, height: usize, levels: usize) -> Vec<Vec<Vec<f32>>> {
//     let wavelet = Wavelet::Daubechies { family: Wavelet পরিবার::D2 };
//     let mut current_image: Vec<f64> = image.iter().map(|&x| x as f64).collect();
//     let mut wavelet_coeffs_2d: Vec<Vec<Vec<f32>>> = Vec::with_capacity(levels);

//     let mut current_width = width;
//     let mut current_height = height;

//     for _ in 0..levels {
//         if current_width < 2 || current_height < 2 {
//             break;
//         }
//         let coeffs2d_struct = cwt2d(&current_image, current_width, wavelet, 1).unwrap();

//         // Extract subbands and convert to Vec<f32>
//         let level_lh: Vec<f32> = coeffs2d_struct.horizontal_details().iter().map(|&x| x as f32).collect();
//         let level_hl: Vec<f32> = coeffs2d_struct.vertical_details().iter().map(|&x| x as f32).collect();
//         let level_hh: Vec<f32> = coeffs2d_struct.diagonal_details().iter().map(|&x| x as f32).collect();

//         wavelet_coeffs_2d.push(vec![level_lh, level_hl, level_hh]); // Store LH, HL, HH
//         current_image = coeffs2d_struct.approx_coarse().to_vec(); // Next level input is LL
//         current_width /= 2;
//         current_height /= 2;
//     }
//     let final_ll: Vec<f32> = current_image.iter().map(|&x| x as f32).collect();
//     wavelet_coeffs_2d.push(vec![final_ll]); // Store final LL coefficients
//     wavelet_coeffs_2d.reverse(); // Finest to coarsest
//     wavelet_coeffs_2d
// }



// // --- Inverse 2D Daubechies Wavelet Transform (Reconstruction) - Multi-level using fastcwt ---
// fn inverse_daubechies_wavelet_2d_fastcwt(wavelet_coeffs_2d: &mut Vec<Vec<Vec<f32>>>, original_width: usize, original_height: usize) -> Vec<f32> {
//     if wavelet_coeffs_2d.is_empty() {
//         return vec![0.0; original_width * original_height]; // Handle empty coefficients gracefully
//     }

//     let wavelet = Wavelet::Daubechies { family: Wavelet পরিবার::D2 };
//     let mut current_ll: Vec<f64> = wavelet_coeffs_2d.pop().unwrap()[0].iter().map(|&x| x as f64).collect(); // Start with coarsest LL
//     let mut current_width = original_width / (1 << wavelet_coeffs_2d.len());
//     let mut current_height= original_height/ (1 << wavelet_coeffs_2d.len());


//     while !wavelet_coeffs_2d.is_empty() {
//         let level_coeffs = wavelet_coeffs_2d.pop().unwrap();
//         let level_lh: Vec<f64> = level_coeffs[0].iter().map(|&x| x as f64).collect();
//         let level_hl: Vec<f64> = level_coeffs[1].iter().map(|&x| x as f64).collect();
//         let level_hh: Vec<f64> = level_coeffs[2].iter().map(|&x| x as f64).collect();

//         let reconstructed_level = icwt2d(
//             &current_ll,
//             &level_hl,
//             &level_lh,
//             &level_hh,
//             current_width,
//             wavelet,
//             1
//         ).unwrap();
//         current_ll = reconstructed_level;
//         current_width *= 2;
//         current_height *= 2;
//     }
//     current_ll.iter().map(|&x| x as f32).collect()
// }


// // --- Robust Noise Variance Estimation (using Median Absolute Deviation - MAD) ---
// fn estimate_noise_variance_mad(wavelet_coeffs_hh: &Vec<f32>) -> f32 {
//     if wavelet_coeffs_hh.is_empty() {
//         return 0.0;
//     }
//     let mut abs_coeffs: Vec<f32> = wavelet_coeffs_hh.iter().map(|&coeff| coeff.abs()).collect();
//     abs_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap()); // Sort absolute coefficients
//     let median_abs_coeff = abs_coeffs[abs_coeffs.len() / 2]; // Median of absolute coefficients
//     let mad = median_abs_coeff / 0.6745; // Scale factor for Gaussian distribution assumption
//     mad * mad // Variance estimate (MAD^2)
// }



// // --- Bayesian Wavelet Denoising function (with strength parameter) ---
// pub fn bayesian_wavelet_denoising(noisy_image: Vec<f32>, width: usize, height: usize, levels: usize, denoising_strength: f32) -> Vec<f32> {
//     if width < (1 << levels) || height < (1 << levels) {
//         println!("Warning: Image dimensions are too small for {} wavelet levels. Reducing levels.", levels);
//         return noisy_image; // Or adjust levels... for simplicity, return noisy image
//     }


//     // 1. Multi-level 2D Wavelet Decomposition (using fastcwt)
//     let mut wavelet_coeffs_2d = daubechies_wavelet_2d_fastcwt(&noisy_image, width, height, levels);


//     // 2. Noise Variance Estimation (from finest level HH subband)
//     let finest_level_hh = wavelet_coeffs_2d[0][2].clone(); // HH coefficients of finest level
//     let noise_variance = estimate_noise_variance_mad(&finest_level_hh);


//     // 3. Bayesian Shrinkage (Wiener-like, applied to detail subbands at each level)
//     for level_index in 0..wavelet_coeffs_2d.len() {
//         let current_level_noise_variance = noise_variance * denoising_strength; // Apply strength parameter here

//         // Apply shrinkage to LH, HL, HH subbands
//         wavelet_coeffs_2d[level_index][0] = bayesian_shrinkage(&wavelet_coeffs_2d[level_index][0], current_level_noise_variance); // LH
//         wavelet_coeffs_2d[level_index][1] = bayesian_shrinkage(&wavelet_coeffs_2d[level_index][1], current_level_noise_variance); // HL
//         wavelet_coeffs_2d[level_index][2] = bayesian_shrinkage(&wavelet_coeffs_2d[level_index][2], current_level_noise_variance); // HH
//     }


//     // 4. Inverse 2D Wavelet Transform (using fastcwt)
//     inverse_daubechies_wavelet_2d_fastcwt(&mut wavelet_coeffs_2d, width, height)
// }



// // --- Simplified Bayesian Shrinkage function (Wiener-like) ---
// fn bayesian_shrinkage(coefficients: &Vec<f32>, noise_variance: f32) -> Vec<f32> {
//     let mut denoised_coeffs = Vec::with_capacity(coefficients.len());
//     let signal_variance_estimate = estimate_variance(coefficients); // Signal variance in this subband


//     for &coeff in coefficients {
//         let wiener_filter = if signal_variance_estimate > 0.0 {
//              signal_variance_estimate / (signal_variance_estimate + noise_variance)
//         } else {
//              0.0 // If signal variance is very low, shrink to zero
//         };
//         denoised_coeffs.push(coeff * wiener_filter.max(0.0).min(1.0)); // Ensure filter is in [0, 1]
//     }
//     denoised_coeffs
// }


// // --- Helper function to estimate variance (sample variance) ---
// fn estimate_variance(data: &Vec<f32>) -> f32 {
//     if data.is_empty() {
//         return 0.0;
//     }
//     let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
//     let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
//     variance
// }



// #[cfg(test)]
// mod tests {
//     use super::*;
//     use image::{GrayImage, ImageBuffer};
//     use std::path::Path;
//     use rand::thread_rng;
//     use rand_distr::{Normal, Distribution};

//     #[test]
//     fn test_daubechies_wavelet_1d_forward_inverse_fastcwt() {
//         let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//         let wavelet_coeffs = daubechies_wavelet_1d_fastcwt(&original_data, 2);
//         let reconstructed_data = inverse_daubechies_wavelet_1d_fastcwt(&mut wavelet_coeffs.clone());

//         for i in 0..original_data.len() {
//             assert!((original_data[i] - reconstructed_data[i]).abs() < 1e-4, "Mismatch at index {}", i);
//         }
//     }


//     #[test]
//     fn test_daubechies_wavelet_2d_forward_inverse_fastcwt() {
//         let original_image = vec![
//             1.0, 2.0, 5.0, 6.0,
//             3.0, 4.0, 7.0, 8.0,
//             9.0, 10.0, 13.0, 14.0,
//             11.0, 12.0, 15.0, 16.0,
//         ];
//         let width = 4;
//         let height = 4;
//         let wavelet_coeffs_2d = daubechies_wavelet_2d_fastcwt(&original_image, width, height, 1); // 1 level
//         let reconstructed_image = inverse_daubechies_wavelet_2d_fastcwt(&mut wavelet_coeffs_2d.clone(), width, height);

//          for i in 0..original_image.len() {
//             assert!((original_image[i] - reconstructed_image[i]).abs() < 1e-4, "Mismatch at index {}", i);
//         }
//     }


//     #[test]
//     fn test_bayesian_denoising_on_small_image_fastcwt() {
//         let noisy_image = vec![
//             50.0, 52.0, 130.0, 132.0,
//             53.0, 55.0, 133.0, 135.0,
//             180.0, 182.0, 250.0, 252.0,
//             183.0, 185.0, 253.0, 255.0,
//         ];
//         let width = 4;
//         let height = 4;
//         let levels = 1;
//         let strength = 1.0;

//         let denoised_image = bayesian_wavelet_denoising(noisy_image.clone(), width, height, levels, strength);
//         assert_eq!(denoised_image.len(), noisy_image.len());

//         println!("Noisy Image (fastcwt): {:?}", noisy_image);
//         println!("Denoised Image (fastcwt): {:?}", denoised_image);
//     }


//     #[test]
//     fn test_bayesian_denoising_on_image_file_fastcwt() {
//         // --- Requires the 'image' crate ---
//         // To run this test, add 'image = "0.24.7"' to your Cargo.toml dependencies

//         // 1. Load a grayscale image (replace "test_image.png" with a real grayscale image file in your project)
//         let image_path = Path::new("test_image.png"); // You'll need to create or provide a test_image.png
//         let gray_image = image::open(image_path).unwrap().to_luma8();
//         let (width, height) = gray_image.dimensions();

//         let mut noisy_pixels: Vec<f32> = gray_image.pixels().map(|p| p[0] as f32).collect();

//         // Simulate noise (e.g., Gaussian noise)
//         let noise_std_dev = 20.0;
//         let mut rng = thread_rng();
//         let normal = Normal::new(0.0, noise_std_dev).unwrap();

//         for pixel in noisy_pixels.iter_mut() {
//             *pixel += normal.sample(&mut rng);
//             *pixel = pixel.max(0.0).min(255.0); // Clip to valid range [0, 255]
//         }


//         // 2. Denoise the image
//         let levels = 3; // Example levels - adjust as needed
//         let strength = 1.0; // Example strength - adjust to control denoising
//         let denoised_pixels: Vec<f32> = bayesian_wavelet_denoising(noisy_pixels.clone(), width as usize, height as usize, levels, strength);


//         // 3. Convert back to image and save (optional, for visual inspection)
//         let mut denoised_image_buffer: GrayImage = ImageBuffer::new(width, height);
//         for (x, y, pixel) in denoised_image_buffer.enumerate_pixels_mut() {
//             pixel[0] = denoised_pixels[(y * width + x) as usize] as u8; // Assuming pixel values are in 0-255 range
//         }

//         let output_path = Path::new("denoised_image_db2_fastcwt.png");
//         denoised_image_buffer.save(output_path).unwrap();

//         println!("Denoised image saved to denoised_image_db2_fastcwt.png");
//         // --- To actually *test* image quality, you'd need to compare against a ground truth and use metrics like PSNR, SSIM ---
//     }
// }
