use crate::pixel::{Image, ImageBuffer, SubPixel};
use rawler::{RawImage, imgop::{Dim2, Rect}};
use rayon::prelude::*;

pub trait DemosaicAlgorithm {
    fn demosaic(self, width: usize, height: usize, cfa: rawler::CFA, input: Vec<SubPixel>) -> Image;
}

pub fn demosaic(raw_image: RawImage, demosaic_algorithm: impl DemosaicAlgorithm) -> Image {
    // 1. Extract Metadata early
    let crop_area = raw_image.crop_area.unwrap();
    let width = crop_area.d.w;
    let height = crop_area.d.h;
    
    // Calculate normalization factors once
    let black_level = raw_image.blacklevel.as_bayer_array()[0] as u16;
    let white_level = raw_image.whitelevel.as_bayer_array()[0] as u16;
    let range = (white_level - black_level) as f32;
    let factor = 1.0 / range;

    let nim: Vec<f32>;

    if let rawler::RawImageData::Integer(ref im) = raw_image.data {
        // 2. Adjust CFA for the crop offset
        // We clone only if necessary. 
        // Note: Make sure get_cfa handles the shift correctly based on crop_area.p
        let cfa = raw_image.camera.cfa.clone(); 
        let cfa = get_cfa(cfa, crop_area);

        // 3. Fused Crop + Normalize
        // We go directly from Raw Integer -> Cropped Float
        nim = crop_and_normalize(
            raw_image.dim(),
            crop_area,
            im,
            black_level,
            factor
        );
        // println!("crop/norm time: {}ms", t0.elapsed().as_millis());

        let debayer_image = demosaic_algorithm.demosaic(width, height, cfa, nim);

        return debayer_image
    } else {
        panic!("Don't know how to process non-integer raw files");
    }
}

/// Fuses cropping, type conversion (u16->f32), and normalization into a single pass.
fn crop_and_normalize(
    dim: Dim2, 
    crop_rect: Rect, 
    data: &[u16], 
    black_level: u16, 
    factor: f32
) -> Vec<f32> {
    let crop_w = crop_rect.d.w;
    let crop_h = crop_rect.d.h;
    let full_w = dim.w;
    let start_x = crop_rect.p.x;
    let start_y = crop_rect.p.y;

    let len = crop_w * crop_h;
    let mut output = vec![0.0; len];

    // Parallel Iteration over Output Rows
    output.par_chunks_exact_mut(crop_w)
        .enumerate()
        .for_each(|(i, out_row)| {
            let src_row_idx = start_y + i;
            let begin = src_row_idx * full_w + start_x;
            let end = begin + crop_w;
            
            // Slice the source row
            let src_slice = &data[begin..end];

            out_row.iter_mut().zip(src_slice).for_each(|(out_pix, &in_pix)|{
                *out_pix = (in_pix - black_level) as f32 * factor;
            });
        }
        );

    output
}

pub fn get_cfa(cfa: rawler::CFA, crop_rect: Rect) -> rawler::CFA {
    let x = crop_rect.p.x;
    let y = crop_rect.p.y;
    let new_cfa = cfa.shift(x, y);
    return new_cfa;
}

pub mod demosaic_algorithms {
    use super::*;

    pub struct Markesteijn{}
    pub struct Fast{}

    impl DemosaicAlgorithm for Markesteijn{
        fn demosaic(
            self,
            width: usize,
            height: usize,
            cfa: rawler::CFA, // Changed to reference to avoid ownership issues, adjust as needed
            input: Vec<SubPixel> // Changed to slice to avoid moving/cloning
        ) -> Image {
            // demosaic_neon( width, height, &cfa, &input) // // 1. Allocate Result Buffer Once
            let mut rgb: ImageBuffer = vec![[0.0; 3]; width * height];

            // 2. Parallel Iteration by Row
            // using par_chunks_exact_mut avoids manual index math (idx / width)
            rgb.par_chunks_exact_mut(width)
                .enumerate()
                .for_each(|(row, row_pixels)| {

                    // Pre-calculate vertical bounds for this row
                    let y_min = row.saturating_sub(1);
                    let y_max = (row + 1).min(height - 1);

                    for (col, pixel_out) in row_pixels.iter_mut().enumerate() {
                        // Get the raw Bayer color for this specific pixel (0=R, 1=G, 2=B)
                        let center_cfa_color = cfa.color_at(row, col);

                        // Pre-calculate horizontal bounds
                        let x_min = col.saturating_sub(1);
                        let x_max = (col + 1).min(width - 1);
                        let center_idx = row * width + col;

                        // 3. Fill all 3 channels (R, G, B) for this pixel
                        for channel in 0..3 {
                            if channel == center_cfa_color {
                                // CASE A: The channel matches the Bayer filter color.
                                // Just use the raw value.
                                pixel_out[channel] = input[center_idx];
                            } else {
                                // CASE B: The channel is missing.
                                // We must interpolate from neighbors.
                                let mut sum = 0.0;
                                let mut count = 0.0;

                                // Iterate 3x3 Grid
                                for ny in y_min..=y_max {
                                    for nx in x_min..=x_max {
                                        // Skip the center pixel (we know it's not the color we want)
                                        if ny == row && nx == col {
                                            continue;
                                        }

                                        // If this neighbor is the color we are looking for, accumulate it
                                        if cfa.color_at(ny, nx) == channel {
                                            let n_idx = ny * width + nx;
                                            sum += input[n_idx];
                                            count += 1.0;
                                        }
                                    }
                                }

                                // Avoid division by zero at image edges
                                if count > 0.0 {
                                    pixel_out[channel] = sum / count;
                                }
                            }
                        }
                    }
                });

            return Image{
                data: rgb,
                height: height,
                width: width,
                color_space: None
            }
        }
    }


    impl DemosaicAlgorithm for Fast {
        fn demosaic(
            self,
            width: usize,
            height: usize,
            cfa: rawler::CFA,
            input: Vec<SubPixel>,
        ) -> Image {
            let new_width = width / 2;
            let new_height = height / 2;
            let mut rgb: ImageBuffer = vec![[0.0; 3]; new_width * new_height];

            let c00 = cfa.color_at(0, 0);
            let c01 = cfa.color_at(0, 1);
            let c10 = cfa.color_at(1, 0);
            let c11 = cfa.color_at(1, 1);

            rgb.par_iter_mut().enumerate().for_each(|(idx, pix)| {
                let new_col = idx % new_width;
                let new_row = idx / new_width;

                let r = new_row * 2;
                let c = new_col * 2;

                let tl_idx = r * width + c;

                let mut values = [0.0, 0.0, 0.0];

                values[c00] = input[tl_idx];
                values[c01] = input[tl_idx + 1];
                values[c10] = input[tl_idx + width];
                values[c11] = input[tl_idx + width + 1];

                *pix = values;
            });
            return Image{
                data: rgb,
                height: new_height,
                width: new_width,
                color_space: None
            }
        }
    }
}
