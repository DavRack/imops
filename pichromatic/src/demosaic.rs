use std::time::Instant;

use crate::image::ImageMetadata;
use crate::pixel::{Image, ImageBuffer, SubPixel};
use crate::cfa::CFA;
// use rawler::{RawImage, imgop::{Dim2, Rect}};
use rayon::prelude::*;

/// Descriptor of a two-dimensional area
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dim2 {
  pub w: usize,
  pub h: usize,
}


/// A simple x/y point
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point {
  pub x: usize,
  pub y: usize,
}

/// Rectangle by a point and dimension
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
  pub p: Point,
  pub d: Dim2,
}

pub trait DemosaicAlgorithm {
    fn demosaic(self, width: usize, height: usize, cfa: CFA, input: Vec<SubPixel>) -> Image;
}

pub fn demosaic(
    raw_image_data: &[u16],
    image_metadata: ImageMetadata,
    demosaic_algorithm: impl DemosaicAlgorithm
) -> Image {
    // Calculate normalization factors once
    let range = image_metadata.white_level.unwrap() - image_metadata.black_level.unwrap();
    let factor = 1.0 / range;

    let cfa = get_cfa(
        image_metadata.cfa.expect("A CFA must be set to demosaic raw images"),
        image_metadata.crop_area.expect("Crop area must be set to demosaic raw images")
    );

    let normalized_raw_data = crop_and_normalize(
        image_metadata.width,
        image_metadata.crop_area.expect("crop_area must be set to demosaic raw images"),
        raw_image_data,
        image_metadata.black_level.expect("black level must be set to demosaic raw images"),
        factor
    );

    let debayer_image = demosaic_algorithm.demosaic(
        image_metadata.width,
        image_metadata.height,
        cfa,
        normalized_raw_data
    );
    return debayer_image
}


fn crop_and_normalize(
    width: usize, 
    crop_rect: Rect, 
    data: &[u16], 
    black_level: f32, 
    factor: f32
) -> Vec<f32> {
    let crop_w = crop_rect.d.w;
    let crop_h = crop_rect.d.h;
    let full_w = width;
    let start_x = crop_rect.p.x;
    let start_y = crop_rect.p.y;

    let len = crop_w * crop_h;
    let mut output = vec![0.0; len];
    let bias = -black_level * factor;

    // turns out in this specific loop is faster to NOT do a parallel iterator
    output.chunks_exact_mut(crop_w)
        .enumerate()
        .for_each(|(i, out_row)| {
            let src_row_idx = start_y + i;
            let begin = src_row_idx * full_w + start_x;
            let end = begin + crop_w;
            
            // Slice the source row
            let src_slice = &data[begin..end];

            out_row.iter_mut().zip(src_slice).for_each(|(out_pix, &in_pix)|{
                *out_pix = (in_pix as f32).mul_add(factor, bias);
            });
        });
    output
}

pub fn get_cfa(cfa: CFA, crop_rect: Rect) -> CFA {
    let x = crop_rect.p.x;
    let y = crop_rect.p.y;
    let new_cfa = cfa.shift(x, y);
    return new_cfa;
}

pub mod demosaic_algorithms {
    use super::*;

    pub struct Markesteijn{}
    pub struct Fast{}
    pub struct SuperFast{}

    impl DemosaicAlgorithm for Markesteijn{
        fn demosaic(
            self,
            width: usize,
            height: usize,
            cfa: CFA, // Changed to reference to avoid ownership issues, adjust as needed
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

                    row_pixels.iter_mut().enumerate().for_each(|(col, pixel_out)| {
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
                    });
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
            cfa: CFA,
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

    impl DemosaicAlgorithm for SuperFast {
        fn demosaic(
            self,
            width: usize,
            height: usize,
            cfa: CFA,
            input: Vec<SubPixel>,
        ) -> Image {
            // Target: 1/4th of the original width and height
            // This results in an image 1/16th the size of the RAW file (Thumbnail/Preview size)
            let new_width = width / 4;
            let new_height = height / 4;

            // Initialize output buffer
            let mut rgb: ImageBuffer = vec![[0.0; 3]; new_width * new_height];

            // Pre-calculate CFA color indices for the top-left 2x2 block
            let c00 = cfa.color_at(0, 0);
            let c01 = cfa.color_at(0, 1);
            let c10 = cfa.color_at(1, 0);
            let c11 = cfa.color_at(1, 1);

            rgb.par_iter_mut().enumerate().for_each(|(idx, pix)| {
                let new_col = idx % new_width;
                let new_row = idx / new_width;

                // STRIDE CALCULATION:
                // We multiply by 4 to skip lines and columns.
                // This selects the top-left pixel of every 4x4 block in the original image.
                let r = new_row * 4;
                let c = new_col * 4;

                // Calculate the 1D index for the top-left corner of the block
                let tl_idx = r * width + c;

                let mut values = [0.0, 0.0, 0.0];

                // We only read the immediate 2x2 neighbors to form a color.
                // We ignore the surrounding pixels (skipping), which is what makes this "SuperFast".

                // Check bounds to ensure we don't read past the buffer (optional safety for edge cases)
                if tl_idx + width + 1 < input.len() {
                    values[c00] = input[tl_idx];
                    values[c01] = input[tl_idx + 1];
                    values[c10] = input[tl_idx + width];
                    values[c11] = input[tl_idx + width + 1];
                }

                *pix = values;
            });

            return Image {
                data: rgb,
                height: new_height,
                width: new_width,
                color_space: None
            }
        }
    }
}
