use rayon::prelude::*;

/// A helper function to get the value of a pixel at (x, y) with boundary clamping.
#[inline]
fn get_pixel(data: &[f32], width: usize, height: usize, x: i32, y: i32) -> f32 {
    let x = x.clamp(0, width as i32 - 1) as usize;
    let y = y.clamp(0, height as i32 - 1) as usize;
    data[y * width + x]
}

/// Calculates the squared Euclidean distance between two patches.
/// Patches are defined by their center coordinates and a radius.
fn patch_distance(
    data: &[f32],
    width: usize,
    height: usize,
    x1: usize, y1: usize,
    x2: usize, y2: usize,
    patch_radius: usize,
) -> f32 {
    let mut ssd = 0.0;
    let patch_rad = patch_radius as i32;

    for dy in -patch_rad..=patch_rad {
        for dx in -patch_rad..=patch_rad {
            let p1_val = get_pixel(data, width, height, x1 as i32 + dx, y1 as i32 + dy);
            let p2_val = get_pixel(data, width, height, x2 as i32 + dx, y2 as i32 + dy);
            let diff = p1_val - p2_val;
            ssd += diff * diff;
        }
    }
    
    // Normalize by patch size
    let patch_size = (2 * patch_radius + 1).pow(2) as f32;
    ssd / patch_size
}

/// Denoises a single-channel f32 image using the Non-Local Means algorithm.
///
/// # Arguments
/// * `input` - A flat buffer of the single-channel image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `patch_radius` - The radius of the patches to compare (e.g., 3 for a 7x7 patch).
/// * `search_radius` - The radius of the search window (e.g., 10 for a 21x21 window).
/// * `h` - The filtering parameter. A larger h results in more smoothing.
///
/// # Returns
/// A `Vec<f32>` containing the denoised image data.
pub fn denoise(
    input: &[f32],
    width: usize,
    height: usize,
    patch_radius: usize,
    search_radius: usize,
    h: f32,
) -> Vec<f32> {
    let output: Vec<f32> = (0..height * width)
        .into_par_iter()
        .map(|idx| {
            let y = idx / width;
            let x = idx % width;

            let mut total_weight = 0.0;
            let mut weighted_sum = 0.0;
            
            let search_rad = search_radius as i32;

            // Define the search window boundaries
            let y_min = (y as i32 - search_rad).max(0) as usize;
            let y_max = (y as i32 + search_rad).min(height as i32 - 1) as usize;
            let x_min = (x as i32 - search_rad).max(0) as usize;
            let x_max = (x as i32 + search_rad).min(width as i32 - 1) as usize;

            for sy in y_min..=y_max {
                for sx in x_min..=x_max {
                    // Calculate distance between the patch at (x, y) and the patch at (sx, sy)
                    let dist_sq = patch_distance(input, width, height, x, y, sx, sy, patch_radius);

                    // Calculate weight from distance
                    let weight = (-dist_sq / (h * h)).exp();

                    total_weight += weight;
                    weighted_sum += weight * input[sy * width + sx];
                }
            }

            if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                input[idx] // Use index directly
            }
        })
        .collect();

    output
}
