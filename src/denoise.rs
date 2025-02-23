use std::usize;

use rawler::pixarray::RgbF32;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};


// Helper function to calculate positive gradient along x-axis (axis 1)
fn positive_gradient_x(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut gradient = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let current_index = y * width + x;
            let next_x = x + 1;
            if next_x < width {
                let next_index = y * width + next_x;
                gradient[current_index] = data[next_index] - data[current_index];
            } // else gradient is 0 at the boundary (implicit zero padding)
        }
    }
    gradient
}

// Helper function to calculate positive gradient along y-axis (axis 0)
fn positive_gradient_y(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut gradient = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let current_index = y * width + x;
            let next_y = y + 1;
            if next_y < height {
                let next_index = next_y * width + x;
                gradient[current_index] = data[next_index] - data[current_index];
            } // else gradient is 0 at the boundary (implicit zero padding)
        }
    }
    gradient
}

// Helper function to calculate negative gradient along x-axis (axis 1)
fn negative_gradient_x(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut gradient = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let current_index = y * width + x;
            let prev_x = x as isize - 1;
            if prev_x >= 0 {
                let prev_index = y * width + prev_x as usize;
                gradient[current_index] = data[current_index] - data[prev_index];
            } // else gradient is 0 at the boundary (implicit zero padding)
        }
    }
    gradient
}

// Helper function to calculate negative gradient along y-axis (axis 0)
fn negative_gradient_y(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut gradient = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let current_index = y * width + x;
            let prev_y = y as isize - 1;
            if prev_y >= 0 {
                let prev_index = prev_y as usize * width + x;
                gradient[current_index] = data[current_index] - data[prev_index];
            } // else gradient is 0 at the boundary (implicit zero padding)
        }
    }
    gradient
}

// Helper function for vector length at each pixel (assuming dual_a and dual_b are the x and y components)
fn vector_len(dual_a: &[f32], dual_b: &[f32]) -> Vec<f32> {
    dual_a
        .iter()
        .zip(dual_b.iter())
        .map(|(&da, &db)| (da * da + db * db).sqrt().max(1.0)) // max(1, sqrt(da^2 + db^2)) as in original code's logic
        .collect()
}

// Helper function for weighted average (simplified to linear interpolation with lambda)
fn weighted_average(current: &[f32], original: &[f32], lambda: f64) -> Vec<f32> {
    current
        .iter()
        .zip(original.iter())
        .map(|(&c, &o)| c + lambda as f32 * (o - c)) // Simplified to current + lambda * (original - current) which can be interpreted as moving towards original
        .collect()
}

// Helper function to calculate norm of a vector
fn norm(data: &[f32]) -> f64 {
    data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

// pub fn denoise_vec(
//     mut image: FormedImage
// ) -> FormedImage {
//     for _ in 0..10{
//         for _ in 0..40{
//             image.data = filter3x3(image.clone(), &[0.0967, 0.1176, 0.0967, 0.1176, 0.143, 0.1176, 0.0967, 0.1176, 0.0967].clone());
//         }
//         image.data = filter3x3(image.clone(), &[0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0].clone());
//     }
//     for _ in 0..5{
//         image.data = filter3x3(image.clone(), &[0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0].clone());
//     }
//     // image.data = filter3x3(image.clone(), &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0].clone());
//     return image
// }
pub fn filter3x3(image: RgbF32, kernel: &[f32]) -> RgbF32{ // Specify FormedImage<RgbF32>

    let height = image.height;
    let width = image.width;

    let mut out_data = image.clone();

    if kernel.len() != 9 {
        panic!("Kernel must be of size 9 for a 3x3 filter.");
    }


    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let kernel_val = kernel[ky * 3 + kx]; // Kernel index: row-major order
                    let sample_x = (x as i32 - 1 + kx as i32) as usize;
                    let sample_y = (y as i32 - 1 + ky as i32) as usize;

                    let sample_pixel = image.at(sample_y, sample_x);

                    sum_r += sample_pixel[0] * kernel_val;
                    sum_g += sample_pixel[1] * kernel_val;
                    sum_b += sample_pixel[2] * kernel_val;
                }
            }

            // Calculate index into the flat buffer for pixel (x, y)
            let base_index = y as usize * width as usize + x as usize; // * 3 for RGB

            out_data.data[base_index][0] = sum_r; // R channel
            out_data.data[base_index][1] = sum_g; // G channel
            out_data.data[base_index][2] = sum_b; // B channel
        }
    }

    out_data
}

pub fn denoise_poly(image: RgbF32) -> RgbF32 {
    let height = image.height as i32;
    let width = image.width as i32;

    let mut out_data = image.clone();
    let mix_value = 0.8;

    for y in 2..(height - 2) {
        for x in 2..(width - 2) {
            let v = |i1: i32, i2: i32| {
                let ny = y + i1;
                let nx = x + i2;

                if ny >= 0 && ny < height && nx >= 0 && nx < width {
                    image.at(ny as usize, nx as usize)[0]
                } else {
                    0.0 // Handle out-of-bounds access
                }
            };

            let dx = |a: f32, b: f32| {
                let x2mx1 = b-a;

                if x2mx1 != 0.0 {
                    -1.0/x2mx1
                }else{
                    0.0
                }
            };

            let ds = [
                [dx(v(-2, -2), v(-1, -1)), dx(v(1, 1), v(2, 2)), v(-1, -1), v(1, 1)],
                [dx(v(0, -2),  v(0, -1)),  dx(v(0, 1), v(0, 2)), v(0, -1),  v(0, 1)],
                [dx(v(-2, -2), v(-1, -1)), dx(v(1, 1), v(2, 2)), v(-1, -1), v(1, 1)],
                [dx(v(-2, -2), v(-1, -1)), dx(v(1, 1), v(2, 2)), v(-1, -1), v(1, 1)],
            ];

            let xs = [
                [3.0, -2.0,  1.0, 0.0],
                [3.0,  2.0,  1.0, 0.0],
                [-1.0, 1.0, -1.0, 1.0],
                [1.0,  1.0,  1.0, 1.0],
            ];
            

            let interpolated_value: f32 = ds.iter().map(|val| {
                let sol = match solve_linear_system(&xs, val) {
                    Some(solution) => solution[3],
                    _ => v(0, 0), // Use center pixel if solve fails
                };
                if sol < 0.0{0.0}else{sol / 4.0}
            }).sum();

            let new_value = (v(0,0)*mix_value) + ((1.0-mix_value)*interpolated_value);
            // let new_value = interpolated_value;

            let base_index = (y * width + x) as usize;
            out_data.data[base_index][0] = new_value;
            out_data.data[base_index][1] = new_value;
            out_data.data[base_index][2] = new_value;
        }
    }

    out_data
}
fn solve_linear_system(matrix: &[[f32; 4]; 4], vector: &[f32; 4]) -> Option<[f32; 4]> {
    let n = 4;
    let mut augmented_matrix: [[f32; 5]; 4] = [[0.0; 5]; 4];

    // Create augmented matrix [matrix | vector]
    for i in 0..n {
        for j in 0..n {
            augmented_matrix[i][j] = matrix[i][j];
        }
        augmented_matrix[i][n] = vector[i];
    }

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot row
        let mut max_row = i;
        for k in i + 1..n {
            if augmented_matrix[k][i].abs() > augmented_matrix[max_row][i].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            augmented_matrix.swap(i, max_row);
        }

        // If pivot is zero, continue to next column (or return None if needed for singular matrix detection)
        if augmented_matrix[i][i].abs() < 1e-6 { // Using a small epsilon for float comparison
            if augmented_matrix[i][n].abs() > 1e-6 { // Check for inconsistency if pivot is near zero and RHS is not
                return None; // No solution (or singular and inconsistent)
            }
            continue; // Singular or dependent system, but let's continue for possible solutions if they exist
        }

        // Eliminate below current row
        for j in i + 1..n {
            let factor = augmented_matrix[j][i] / augmented_matrix[i][i];
            for k in i..n + 1 {
                augmented_matrix[j][k] -= factor * augmented_matrix[i][k];
            }
        }
    }

    // Back substitution
    let mut solution = [0.0; 4];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in i + 1..n {
            sum += augmented_matrix[i][j] * solution[j];
        }
        if augmented_matrix[i][i].abs() < 1e-6 {
            if augmented_matrix[i][n].abs() > 1e-6 {
                return None; // No solution (inconsistent)
            } else {
                // System may be singular or have infinite solutions. For simplicity, returning None for non-unique solution cases.
                return None;
            }
        }
        solution[i] = (augmented_matrix[i][n] - sum) / augmented_matrix[i][i];
    }

    Some(solution)
}

pub fn denoise(img: Vec<[f32; 3]>, height: usize, width: usize) -> Vec<[f32; 3]> {
    let radius: i32 = 3; // Radius of the kernel
    let spatial_sigma:f32 = 10.0; // Spatial sigma
    let color_sigma:f32 = 10.0; // Color sigma
    let sigma_spatial_squared = spatial_sigma.powi(2);
    let sigma_color_squared = color_sigma.powi(2);

    let denoised_img = img.par_iter().enumerate().map(|(index, center_pixel)|{
        let x = index % width;
        let y = (index-x)/width;

        let mut weights_and_values = Vec::with_capacity(radius.pow(2) as usize);
        for wy in -radius..=radius {
            for wx in -radius..=radius {
                let neighbor_y = (y as i32 + wy).clamp(0, (height - 1) as i32) as usize;
                let neighbor_x = (x as i32 + wx).clamp(0, (width - 1) as i32) as usize;
                let neighbor_pixel = img[neighbor_y * width + neighbor_x];

                // Calculate spatial distance
                let spatial_distance_squared = (wx as f32).powi(2) + (wy as f32).powi(2);
                let spatial_weight = gaussian_weight(spatial_distance_squared, sigma_spatial_squared);

                // Calculate color distance
                let euclidean_color_distance_squared = center_pixel.iter()
                    .zip(neighbor_pixel.iter())
                    .map(|(c1, c2)| (c1 - c2).powi(2))
                    .sum::<f32>();
                let color_weight = gaussian_weight(euclidean_color_distance_squared, sigma_color_squared);

                let weight = spatial_weight * color_weight;
                weights_and_values.push((weight, neighbor_pixel));
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

    let mut channel_averages = [0.0, 0.0, 0.0];
    for i in 0..3 {
        channel_averages[i] = weighted_channel_sums[i] / weights_sum;
    }
    channel_averages
}

/// Un-normalized Gaussian Weight
fn gaussian_weight(x_squared: f32, sigma_squared: f32) -> f32 {
    if sigma_squared == 0.0 {
        if x_squared == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (-0.5 * x_squared / sigma_squared).exp()
    }
}
