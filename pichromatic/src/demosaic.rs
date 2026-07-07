use crate::image::ImageMetadata;
use crate::pixel::{Image, ImageBuffer, SubPixel};
use crate::cfa::CFA;
// use rawler::{RawImage, imgop::{Dim2, Rect}};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Descriptor of a two-dimensional area
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Dim2 {
  pub w: usize,
  pub h: usize,
}


/// A simple x/y point
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Point {
  pub x: usize,
  pub y: usize,
}

/// Rectangle by a point and dimension
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Serialize, Deserialize)]
pub struct Rect {
  pub p: Point,
  pub d: Dim2,
}

pub trait DemosaicAlgorithm {
    fn demosaic(self, width: usize, height: usize, cfa: CFA, input: Vec<SubPixel>) -> Image;
}

pub fn demosaic(
    image: Image,
    demosaic_algorithm: impl DemosaicAlgorithm
) -> Image {

    let cfa = get_cfa(
        image.metadata.cfa.as_ref().expect("A CFA must be set to demosaic raw images"),
        image.metadata.crop_area.expect("Crop area must be set to demosaic raw images")
    );

    let mut original_metadata = image.metadata.clone();

    let mut debayer_image = demosaic_algorithm.demosaic(
        image.metadata.width,
        image.metadata.height,
        cfa,
        image.raw_data
    );

    original_metadata.width = debayer_image.metadata.width;
    original_metadata.height = debayer_image.metadata.height;
    debayer_image.metadata = original_metadata;

    return debayer_image
}


pub fn crop_and_normalize(
    image: &Image, 
) -> Vec<f32> {
    let crop_rect = image.metadata.crop_area.expect("crop_area must be set to demosaic raw images");
    let crop_w = crop_rect.d.w;
    let crop_h = crop_rect.d.h;
    let full_w = image.metadata.width;
    let start_x = crop_rect.p.x;
    let start_y = crop_rect.p.y;

    let len = crop_w * crop_h;
    let mut output = vec![0.0; len];

    // turns out in this specific loop is faster to NOT do a parallel iterator
    output.chunks_exact_mut(crop_w)
        .enumerate()
        .for_each(|(i, out_row)| {
            let src_row_idx = start_y + i;
            let begin = src_row_idx * full_w + start_x;
            let end = begin + crop_w;
            
            // Slice the source row
            let src_slice = &image.raw_data[begin..end];

            out_row.iter_mut().zip(src_slice).for_each(|(out_pix, &in_pix)|{
                // *out_pix = (in_pix as f32).mul_add(factor, bias);
                *out_pix = in_pix;
            });
        });
    output
}

pub fn get_cfa(cfa: &CFA, crop_rect: Rect) -> CFA {
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
    pub struct SuperSuperFast{}

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

            let mut image_metadata = ImageMetadata::default();
            image_metadata.height = height;
            image_metadata.width = width;

            return Image {
                rgb_data: rgb,
                raw_data: vec![],
                metadata: image_metadata,
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
            let mut image_metadata = ImageMetadata::default();
            image_metadata.height = new_height;
            image_metadata.width = new_width;

            return Image {
                rgb_data: rgb,
                raw_data: vec![],
                metadata: image_metadata,
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

            rgb.par_chunks_exact_mut(new_width).enumerate().for_each(|(idw, pixs)| {
                pixs.iter_mut().enumerate().for_each(|(idi, pix)|{
                    let idx = (new_width*idw)+idi;
                    let new_col = idx % new_width;
                    let new_row = idx / new_width;

                    // STRIDE CALCULATION:
                    // We multiply by 4 to skip lines and columns.
                    // This selects the top-left pixel of every 4x4 block in the original image.
                    let r = new_row * 4;
                    let c = new_col * 4;

                    // Calculate the 1D index for the top-left corner of the block
                    let tl_idx = r * width + c;

                    // We only read the immediate 2x2 neighbors to form a color.
                    // We ignore the surrounding pixels (skipping), which is what makes this "SuperFast".

                    // Check bounds to ensure we don't read past the buffer (optional safety for edge cases)
                    pix[c00] = input[tl_idx];
                    pix[c01] = input[tl_idx + 1];
                    pix[c10] = input[tl_idx + width];
                    pix[c11] = input[tl_idx + width + 1];

                });
            });

            let mut image_metadata = ImageMetadata::default();
            image_metadata.height = new_height;
            image_metadata.width = new_width;

            return Image {
                rgb_data: rgb,
                raw_data: vec![],
                metadata: image_metadata,
            }
        }
    }
    impl DemosaicAlgorithm for SuperSuperFast {
        fn demosaic(
            self,
            width: usize,
            height: usize,
            cfa: CFA,
            input: Vec<SubPixel>,
        ) -> Image {
            // Target: 1/4th of the original width and height
            // This results in an image 1/16th the size of the RAW file (Thumbnail/Preview size)
            let factor = 8;
            let new_width = width / factor;
            let new_height = height / factor;

            // Initialize output buffer
            let mut rgb: ImageBuffer = vec![[0.0; 3]; new_width * new_height];

            // Pre-calculate CFA color indices for the top-left 2x2 block
            let c00 = cfa.color_at(0, 0);
            let c01 = cfa.color_at(0, 1);
            let c10 = cfa.color_at(1, 0);
            let c11 = cfa.color_at(1, 1);

            rgb.par_chunks_exact_mut(new_width).enumerate().for_each(|(idw, pixs)| {
                pixs.iter_mut().enumerate().for_each(|(idi, pix)|{
                    let idx = (new_width*idw)+idi;
                    let new_col = idx % new_width;
                    let new_row = idx / new_width;

                    // STRIDE CALCULATION:
                    // We multiply by 4 to skip lines and columns.
                    // This selects the top-left pixel of every 4x4 block in the original image.
                    let r = new_row * factor;
                    let c = new_col * factor;

                    // Calculate the 1D index for the top-left corner of the block
                    let tl_idx = r * width + c;

                    // We only read the immediate 2x2 neighbors to form a color.
                    // We ignore the surrounding pixels (skipping), which is what makes this "SuperFast".

                    // Check bounds to ensure we don't read past the buffer (optional safety for edge cases)
                    pix[c00] = input[tl_idx];
                    pix[c01] = input[tl_idx + 1];
                    pix[c10] = input[tl_idx + width];
                    pix[c11] = input[tl_idx + width + 1];

                });
            });

            let mut image_metadata = ImageMetadata::default();
            image_metadata.height = new_height;
            image_metadata.width = new_width;

            return Image {
                rgb_data: rgb,
                raw_data: vec![],
                metadata: image_metadata,
            }
        }
    }

    pub struct Amaze {
        pub clip_pt: f32,
    }

    impl Default for Amaze {
        fn default() -> Self {
            Self { clip_pt: 1.0 }
        }
    }

    impl DemosaicAlgorithm for Amaze {
        fn demosaic(
            self,
            width: usize,
            height: usize,
            cfa: CFA,
            input: Vec<SubPixel>,
        ) -> Image {
            let mut rgb: ImageBuffer = vec![[0.0; 3]; width * height];

            struct UnsafeSlice<'a> {
                slice: *mut [f32; 3],
                _marker: std::marker::PhantomData<&'a mut [[f32; 3]]>,
            }
            unsafe impl<'a> Send for UnsafeSlice<'a> {}
            unsafe impl<'a> Sync for UnsafeSlice<'a> {}
            impl<'a> UnsafeSlice<'_> {
                fn write_pixel(&self, idx: usize, r: f32, g: f32, b: f32) {
                    unsafe {
                        let pixel_ptr = self.slice.add(idx);
                        *pixel_ptr = [r, g, b];
                    }
                }
            }

            let out_slice = UnsafeSlice {
                slice: rgb.as_mut_ptr(),
                _marker: std::marker::PhantomData,
            };

            let ts_step = 128 - 32; // 96
            let mut tiles = Vec::new();
            let mut top = -16;
            while top < height as i32 {
                let mut left = -16;
                while left < width as i32 {
                    tiles.push((top, left));
                    left += ts_step;
                }
                top += ts_step;
            }

            tiles.into_par_iter().for_each(|(top, left)| {
                const TS: usize = 128;
                const TSH: usize = 64;

                const V1: usize = 128;
                const V2: usize = 256;
                const V3: usize = 384;
                const P1: usize = 127;
                const P2: usize = 254;
                const P3: usize = 381;
                const M1: usize = 129;
                const M2: usize = 258;
                const M3: usize = 387;

                let eps = 1e-5f32;
                let epssq = 1e-10f32;
                let arthresh = 0.75f32;
                let gaussodd = [0.14659727707323927f32, 0.103592713382435f32, 0.0732036125103057f32, 0.0365543548389495f32];
                let nyqthresh = 0.5f32;
                let gaussgrad = [
                    nyqthresh * 0.07384411893421103f32,
                    nyqthresh * 0.06207511968171489f32,
                    nyqthresh * 0.0521818194747806f32,
                    nyqthresh * 0.03687419286733595f32,
                    nyqthresh * 0.03099732204057846f32,
                    nyqthresh * 0.018413194161458882f32,
                ];
                let gausseven = [0.13719494435797422f32, 0.05640252782101291f32];
                let gquinc = [0.169917f32, 0.108947f32, 0.069855f32, 0.0287182f32];

                let sqr = |x: f32| x * x;
                let get_in = |r: i32, c: i32| -> f32 {
                    let r_clamped = r.clamp(0, height as i32 - 1) as usize;
                    let c_clamped = c.clamp(0, width as i32 - 1) as usize;
                    input[r_clamped * width + c_clamped]
                };
                let color_at = |r: i32, c: i32| -> usize {
                    cfa.color_at(
                        ((r % 48 + 48) % 48) as usize,
                        ((c % 48 + 48) % 48) as usize,
                    )
                };

                let lim = |x: f32, min_val: f32, max_val: f32| -> f32 {
                    min_val.max(x.min(max_val))
                };
                let ulim = |x: f32, y: f32, z: f32| -> f32 {
                    if y < z { lim(x, y, z) } else { lim(x, z, y) }
                };
                let clampnan = |x: f32, m: f32, m_max: f32| -> f32 {
                    if x.is_infinite() {
                        if x < m { m } else if x > m_max { m_max } else { x }
                    } else if x.is_nan() {
                        (m + m_max) * 0.5
                    } else {
                        x
                    }
                };
                let interpolatef = |w: f32, a: f32, b: f32| -> f32 {
                    w * a + (1.0 - w) * b
                };

                let mut rgbgreen = vec![0.0f32; 16384];
                let mut delhvsqsum = vec![0.0f32; 16384];
                let mut dirwts0 = vec![0.0f32; 16384];
                let mut dirwts1 = vec![0.0f32; 16384];
                let mut vcd = vec![0.0f32; 16384];
                let mut hcd = vec![0.0f32; 16384];
                let mut vcdalt = vec![0.0f32; 16384];
                let mut hcdalt = vec![0.0f32; 16384];
                let mut cddiffsq = vec![0.0f32; 16384];
                let mut hvwt = vec![0.0f32; 8192];
                let mut dgrb0 = vec![0.0f32; 8192];
                let mut dgrb1 = vec![0.0f32; 8192];
                let mut delp = vec![0.0f32; 8192];
                let mut delm = vec![0.0f32; 8192];
                let mut rbint = vec![0.0f32; 8192];
                let mut dgrb2_h = vec![0.0f32; 8192];
                let mut dgrb2_v = vec![0.0f32; 8192];
                let mut dgintv = vec![0.0f32; 16384];
                let mut dginth = vec![0.0f32; 16384];
                let mut dgrbsq1m = vec![0.0f32; 8192];
                let mut dgrbsq1p = vec![0.0f32; 8192];
                let mut cfa_buf = vec![0.0f32; 16384];
                let mut pmwt = vec![0.0f32; 8192];
                let mut rbm = vec![0.0f32; 8192];
                let mut rbp = vec![0.0f32; 8192];
                let mut nyquist = vec![0u8; 8192];
                let mut nyquist2 = vec![0u8; 8192];
                let mut nyqutest = vec![0.0f32; 8192];

                let bottom = (top + 128).min(height as i32 + 16);
                let right_edge = (left + 128).min(width as i32 + 16);
                let rr1 = (bottom - top) as usize;
                let cc1 = (right_edge - left) as usize;
                let rrmin = if top < 0 { 16 } else { 0 };
                let ccmin = if left < 0 { 16 } else { 0 };
                let rrmax = if bottom > height as i32 { (height as i32 - top) as usize } else { rr1 };
                let ccmax = if right_edge > width as i32 { (width as i32 - left) as usize } else { cc1 };

                // Fill upper border
                if rrmin > 0 {
                    for rr in 0..16 {
                        for cc in ccmin..ccmax {
                            let row = 32 - rr as i32 + top;
                            let val = get_in(row, cc as i32 + left);
                            cfa_buf[rr * TS + cc] = val;
                            rgbgreen[rr * TS + cc] = val;
                        }
                    }
                }

                // Fill inner part
                for rr in rrmin..rrmax {
                    let row = rr as i32 + top;
                    for cc in ccmin..ccmax {
                        let val = get_in(row, cc as i32 + left);
                        let idx = rr * TS + cc;
                        cfa_buf[idx] = val;
                        rgbgreen[idx] = val;
                    }
                }

                // Fill lower border
                if rrmax < rr1 {
                    for rr in 0..16 {
                        for cc in ccmin..ccmax {
                            let val = get_in(height as i32 - rr as i32 - 2, left + cc as i32);
                            let idx = (rrmax + rr) * TS + cc;
                            cfa_buf[idx] = val;
                            rgbgreen[idx] = val;
                        }
                    }
                }

                // Fill left border
                if ccmin > 0 {
                    for rr in rrmin..rrmax {
                        for cc in 0..16 {
                            let row = rr as i32 + top;
                            let val = get_in(row, 32 - cc as i32 + left);
                            let idx = rr * TS + cc;
                            cfa_buf[idx] = val;
                            rgbgreen[idx] = val;
                        }
                    }
                }

                // Fill right border
                if ccmax < cc1 {
                    for rr in rrmin..rrmax {
                        for cc in 0..16 {
                            let val = get_in(top + rr as i32, width as i32 - cc as i32 - 2);
                            let idx = rr * TS + ccmax + cc;
                            cfa_buf[idx] = val;
                            rgbgreen[idx] = val;
                        }
                    }
                }

                // Corners
                if rrmin > 0 && ccmin > 0 {
                    for rr in 0..16 {
                        for cc in 0..16 {
                            let val = get_in(32 - rr as i32, 32 - cc as i32);
                            cfa_buf[rr * TS + cc] = val;
                            rgbgreen[rr * TS + cc] = val;
                        }
                    }
                }

                if rrmax < rr1 && ccmax < cc1 {
                    for rr in 0..16 {
                        for cc in 0..16 {
                            let val = get_in(height as i32 - rr as i32 - 2, width as i32 - cc as i32 - 2);
                            let idx = (rrmax + rr) * TS + ccmax + cc;
                            cfa_buf[idx] = val;
                            rgbgreen[idx] = val;
                        }
                    }
                }

                if rrmin > 0 && ccmax < cc1 {
                    for rr in 0..16 {
                        for cc in 0..16 {
                            let val = get_in(32 - rr as i32, width as i32 - cc as i32 - 2);
                            let idx = rr * TS + ccmax + cc;
                            cfa_buf[idx] = val;
                            rgbgreen[idx] = val;
                        }
                    }
                }

                if rrmax < rr1 && ccmin > 0 {
                    for rr in 0..16 {
                        for cc in 0..16 {
                            let val = get_in(height as i32 - rr as i32 - 2, 32 - cc as i32);
                            let idx = (rrmax + rr) * TS + cc;
                            cfa_buf[idx] = val;
                            rgbgreen[idx] = val;
                        }
                    }
                }

                // Horizontal and vertical gradients
                for rr in 2..rr1 - 2 {
                    for cc in 2..cc1 - 2 {
                        let indx = rr * TS + cc;
                        let delh = (cfa_buf[indx + 1] - cfa_buf[indx - 1]).abs();
                        let delv = (cfa_buf[indx + V1] - cfa_buf[indx - V1]).abs();
                        dirwts0[indx] = eps + (cfa_buf[indx + V2] - cfa_buf[indx]).abs() + (cfa_buf[indx] - cfa_buf[indx - V2]).abs() + delv;
                        dirwts1[indx] = eps + (cfa_buf[indx + 2] - cfa_buf[indx]).abs() + (cfa_buf[indx] - cfa_buf[indx - 2]).abs() + delh;
                        delhvsqsum[indx] = sqr(delh) + sqr(delv);
                    }
                }

                let clip_pt8 = self.clip_pt * 0.8;

                // Interpolate vertical and horizontal colour differences
                for rr in 4..rr1 - 4 {
                    let mut fcswitch = (color_at(top + rr as i32, left + 4) & 1) != 0;

                    for cc in 4..cc1 - 4 {
                        let indx = rr * TS + cc;

                        let cru = cfa_buf[indx - V1] * (dirwts0[indx - V2] + dirwts0[indx])
                            / (dirwts0[indx - V2] * (eps + cfa_buf[indx]) + dirwts0[indx] * (eps + cfa_buf[indx - V2]));
                        let crd = cfa_buf[indx + V1] * (dirwts0[indx + V2] + dirwts0[indx])
                            / (dirwts0[indx + V2] * (eps + cfa_buf[indx]) + dirwts0[indx] * (eps + cfa_buf[indx + V2]));
                        let crl = cfa_buf[indx - 1] * (dirwts1[indx - 2] + dirwts1[indx])
                            / (dirwts1[indx - 2] * (eps + cfa_buf[indx]) + dirwts1[indx] * (eps + cfa_buf[indx - 2]));
                        let crr = cfa_buf[indx + 1] * (dirwts1[indx + 2] + dirwts1[indx])
                            / (dirwts1[indx + 2] * (eps + cfa_buf[indx]) + dirwts1[indx] * (eps + cfa_buf[indx + 2]));

                        // G interpolated in vert/hor directions using Hamilton-Adams method
                        let guha = cfa_buf[indx - V1] + (cfa_buf[indx] - cfa_buf[indx - V2]) * 0.5;
                        let gdha = cfa_buf[indx + V1] + (cfa_buf[indx] - cfa_buf[indx + V2]) * 0.5;
                        let glha = cfa_buf[indx - 1] + (cfa_buf[indx] - cfa_buf[indx - 2]) * 0.5;
                        let grha = cfa_buf[indx + 1] + (cfa_buf[indx] - cfa_buf[indx + 2]) * 0.5;

                        // G interpolated in vert/hor directions using adaptive ratios
                        let guar = if (1.0 - cru).abs() < arthresh { cfa_buf[indx] * cru } else { guha };
                        let gdar = if (1.0 - crd).abs() < arthresh { cfa_buf[indx] * crd } else { gdha };
                        let glar = if (1.0 - crl).abs() < arthresh { cfa_buf[indx] * crl } else { glha };
                        let grar = if (1.0 - crr).abs() < arthresh { cfa_buf[indx] * crr } else { grha };

                        // adaptive weights for vertical/horizontal directions
                        let hwt = dirwts1[indx - 1] / (dirwts1[indx - 1] + dirwts1[indx + 1]);
                        let vwt = dirwts0[indx - V1] / (dirwts0[indx + V1] + dirwts0[indx - V1]);

                        // interpolated G via adaptive weights of cardinal evaluations
                        let gintvha = vwt * gdha + (1.0 - vwt) * guha;
                        let ginthha = hwt * grha + (1.0 - hwt) * glha;

                        // interpolated colour differences
                        let (mut vcd_val, mut hcd_val, vcdalt_val, hcdalt_val) = if fcswitch {
                            (
                                cfa_buf[indx] - (vwt * gdar + (1.0 - vwt) * guar),
                                cfa_buf[indx] - (hwt * grar + (1.0 - hwt) * glar),
                                cfa_buf[indx] - gintvha,
                                cfa_buf[indx] - ginthha,
                            )
                        } else {
                            (
                                (vwt * gdar + (1.0 - vwt) * guar) - cfa_buf[indx],
                                (hwt * grar + (1.0 - hwt) * glar) - cfa_buf[indx],
                                gintvha - cfa_buf[indx],
                                ginthha - cfa_buf[indx],
                            )
                        };

                        fcswitch = !fcswitch;

                        let (guar_final, gdar_final, glar_final, grar_final) = if cfa_buf[indx] > clip_pt8 || gintvha > clip_pt8 || ginthha > clip_pt8 {
                            vcd_val = vcdalt_val;
                            hcd_val = hcdalt_val;
                            (guha, gdha, glha, grha)
                        } else {
                            (guar, gdar, glar, grar)
                        };

                        vcd[indx] = vcd_val;
                        hcd[indx] = hcd_val;
                        vcdalt[indx] = vcdalt_val;
                        hcdalt[indx] = hcdalt_val;

                        // differences of interpolations in opposite directions
                        dgintv[indx] = sqr(guha - gdha).min(sqr(guar_final - gdar_final));
                        dginth[indx] = sqr(glha - grha).min(sqr(glar_final - grar_final));
                    }
                }

                for rr in 4..rr1 - 4 {
                    for cc in 4..cc1 - 4 {
                        let indx = rr * TS + cc;
                        let c = (color_at(top + rr as i32, left + cc as i32) & 1) != 0;

                        let hcdvar = 3.0 * (sqr(hcd[indx - 2]) + sqr(hcd[indx]) + sqr(hcd[indx + 2]))
                            - sqr(hcd[indx - 2] + hcd[indx] + hcd[indx + 2]);
                        let hcdaltvar = 3.0 * (sqr(hcdalt[indx - 2]) + sqr(hcdalt[indx]) + sqr(hcdalt[indx + 2]))
                            - sqr(hcdalt[indx - 2] + hcdalt[indx] + hcdalt[indx + 2]);
                        let vcdvar = 3.0 * (sqr(vcd[indx - V2]) + sqr(vcd[indx]) + sqr(vcd[indx + V2]))
                            - sqr(vcd[indx - V2] + vcd[indx] + vcd[indx + V2]);
                        let vcdaltvar = 3.0 * (sqr(vcdalt[indx - V2]) + sqr(vcdalt[indx]) + sqr(vcdalt[indx + V2]))
                            - sqr(vcdalt[indx - V2] + vcdalt[indx] + vcdalt[indx + V2]);

                        if hcdaltvar < hcdvar {
                            hcd[indx] = hcdalt[indx];
                        }
                        if vcdaltvar < vcdvar {
                            vcd[indx] = vcdalt[indx];
                        }

                        let ginth: f32;
                        let gintv: f32;

                        if c {
                            // G site
                            ginth = -hcd[indx] + cfa_buf[indx];
                            gintv = -vcd[indx] + cfa_buf[indx];

                            if hcd[indx] > 0.0 {
                                if 3.0 * hcd[indx] > (ginth + cfa_buf[indx]) {
                                    hcd[indx] = -ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]) + cfa_buf[indx];
                                } else {
                                    let hwt = 1.0 - 3.0 * hcd[indx] / (eps + ginth + cfa_buf[indx]);
                                    hcd[indx] = hwt * hcd[indx] + (1.0 - hwt) * (-ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]) + cfa_buf[indx]);
                                }
                            }

                            if vcd[indx] > 0.0 {
                                if 3.0 * vcd[indx] > (gintv + cfa_buf[indx]) {
                                    vcd[indx] = -ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]) + cfa_buf[indx];
                                } else {
                                    let vwt = 1.0 - 3.0 * vcd[indx] / (eps + gintv + cfa_buf[indx]);
                                    vcd[indx] = vwt * vcd[indx] + (1.0 - vwt) * (-ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]) + cfa_buf[indx]);
                                }
                            }

                            if ginth > self.clip_pt {
                                hcd[indx] = -ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]) + cfa_buf[indx];
                            }
                            if gintv > self.clip_pt {
                                vcd[indx] = -ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]) + cfa_buf[indx];
                            }
                        } else {
                            // R or B site
                            ginth = hcd[indx] + cfa_buf[indx];
                            gintv = vcd[indx] + cfa_buf[indx];

                            if hcd[indx] < 0.0 {
                                if 3.0 * hcd[indx] < -(ginth + cfa_buf[indx]) {
                                    hcd[indx] = ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]) - cfa_buf[indx];
                                } else {
                                    let hwt = 1.0 + 3.0 * hcd[indx] / (eps + ginth + cfa_buf[indx]);
                                    hcd[indx] = hwt * hcd[indx] + (1.0 - hwt) * (ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]) - cfa_buf[indx]);
                                }
                            }

                            if vcd[indx] < 0.0 {
                                if 3.0 * vcd[indx] < -(gintv + cfa_buf[indx]) {
                                    vcd[indx] = ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]) - cfa_buf[indx];
                                } else {
                                    let vwt = 1.0 + 3.0 * vcd[indx] / (eps + gintv + cfa_buf[indx]);
                                    vcd[indx] = vwt * vcd[indx] + (1.0 - vwt) * (ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]) - cfa_buf[indx]);
                                }
                            }

                            if ginth > self.clip_pt {
                                hcd[indx] = ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]) - cfa_buf[indx];
                            }
                            if gintv > self.clip_pt {
                                vcd[indx] = ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]) - cfa_buf[indx];
                            }

                            cddiffsq[indx] = sqr(vcd[indx] - hcd[indx]);
                        }
                    }
                }

                for rr in 6..rr1 - 6 {
                    let mut cc = 6 + (color_at(top + rr as i32, left + 2) & 1);
                    while cc < cc1 - 6 {
                        let indx = rr * TS + cc;

                        let uave = vcd[indx] + vcd[indx - V1] + vcd[indx - V2] + vcd[indx - V3];
                        let dave = vcd[indx] + vcd[indx + V1] + vcd[indx + V2] + vcd[indx + V3];
                        let lave = hcd[indx] + hcd[indx - 1] + hcd[indx - 2] + hcd[indx - 3];
                        let rave = hcd[indx] + hcd[indx + 1] + hcd[indx + 2] + hcd[indx + 3];

                        let dgrbvvaru = sqr(vcd[indx] - uave) + sqr(vcd[indx - V1] - uave) + sqr(vcd[indx - V2] - uave) + sqr(vcd[indx - V3] - uave);
                        let dgrbvvard = sqr(vcd[indx] - dave) + sqr(vcd[indx + V1] - dave) + sqr(vcd[indx + V2] - dave) + sqr(vcd[indx + V3] - dave);
                        let dgrbhvarl = sqr(hcd[indx] - lave) + sqr(hcd[indx - 1] - lave) + sqr(hcd[indx - 2] - lave) + sqr(hcd[indx - 3] - lave);
                        let dgrbhvarr = sqr(hcd[indx] - rave) + sqr(hcd[indx + 1] - rave) + sqr(hcd[indx + 2] - rave) + sqr(hcd[indx + 3] - rave);

                        let hwt = dirwts1[indx - 1] / (dirwts1[indx - 1] + dirwts1[indx + 1]);
                        let vwt = dirwts0[indx - V1] / (dirwts0[indx + V1] + dirwts0[indx - V1]);

                        let vcdvar = epssq + vwt * dgrbvvard + (1.0 - vwt) * dgrbvvaru;
                        let hcdvar = epssq + hwt * dgrbhvarr + (1.0 - hwt) * dgrbhvarl;

                        let dgrbvvaru1 = dgintv[indx] + dgintv[indx - V1] + dgintv[indx - V2];
                        let dgrbvvard1 = dgintv[indx] + dgintv[indx + V1] + dgintv[indx + V2];
                        let dgrbhvarl1 = dginth[indx] + dginth[indx - 1] + dginth[indx - 2];
                        let dgrbhvarr1 = dginth[indx] + dginth[indx + 1] + dginth[indx + 2];

                        let vcdvar1 = epssq + vwt * dgrbvvard1 + (1.0 - vwt) * dgrbvvaru1;
                        let hcdvar1 = epssq + hwt * dgrbhvarr1 + (1.0 - hwt) * dgrbhvarl1;

                        let varwt = hcdvar / (vcdvar + hcdvar);
                        let diffwt = hcdvar1 / (vcdvar1 + hcdvar1);

                        if (0.5 - varwt) * (0.5 - diffwt) > 0.0 && (0.5f32 - diffwt).abs() < (0.5f32 - varwt).abs() {
                            hvwt[indx >> 1] = varwt;
                        } else {
                            hvwt[indx >> 1] = diffwt;
                        }

                        cc += 2;
                    }
                }

                for rr in 6..rr1 - 6 {
                    let mut cc = 6 + (color_at(top + rr as i32, left + 2) & 1);
                    while cc < cc1 - 6 {
                        let indx = rr * TS + cc;
                        nyqutest[indx >> 1] = (gaussodd[0] * cddiffsq[indx]
                            + gaussodd[1] * (cddiffsq[indx - M1] + cddiffsq[indx - P1] + cddiffsq[indx + P1] + cddiffsq[indx + M1])
                            + gaussodd[2] * (cddiffsq[indx - V2] + cddiffsq[indx - 2] + cddiffsq[indx + 2] + cddiffsq[indx + V2])
                            + gaussodd[3] * (cddiffsq[indx - M2] + cddiffsq[indx - P2] + cddiffsq[indx + P2] + cddiffsq[indx + M2]))
                            - (gaussgrad[0] * delhvsqsum[indx]
                                + gaussgrad[1] * (delhvsqsum[indx - V1] + delhvsqsum[indx + 1] + delhvsqsum[indx - 1] + delhvsqsum[indx + V1])
                                + gaussgrad[2] * (delhvsqsum[indx - M1] + delhvsqsum[indx - P1] + delhvsqsum[indx + P1] + delhvsqsum[indx + M1])
                                + gaussgrad[3] * (delhvsqsum[indx - V2] + delhvsqsum[indx - 2] + delhvsqsum[indx + 2] + delhvsqsum[indx + V2])
                                + gaussgrad[4] * (delhvsqsum[indx - V2 - 1] + delhvsqsum[indx - V2 + 1]
                                    + delhvsqsum[indx - TS - 2] + delhvsqsum[indx - TS + 2]
                                    + delhvsqsum[indx + TS - 2] + delhvsqsum[indx + TS + 2]
                                    + delhvsqsum[indx + V2 - 1] + delhvsqsum[indx + V2 + 1])
                                + gaussgrad[5] * (delhvsqsum[indx - M2] + delhvsqsum[indx - P2] + delhvsqsum[indx + P2] + delhvsqsum[indx + M2]));
                        cc += 2;
                    }
                }

                let mut nystartrow = 0;
                let mut nyendrow = 0;
                let mut nystartcol = TS + 1;
                let mut nyendcol = 0;

                for rr in 6..rr1 - 6 {
                    let mut cc = 6 + (color_at(top + rr as i32, left + 2) & 1);
                    while cc < cc1 - 6 {
                        let indx = rr * TS + cc;
                        if nyqutest[indx >> 1] > 0.0 {
                            nyquist[indx >> 1] = 1;
                            nystartrow = if nystartrow == 0 { rr } else { nystartrow };
                            nyendrow = rr;
                            nystartcol = nystartcol.min(cc);
                            nyendcol = nyendcol.max(cc);
                        }
                        cc += 2;
                    }
                }

                let do_nyquist = nystartrow != nyendrow && nystartcol != nyendcol;

                if do_nyquist {
                    nyendrow += 1;
                    nyendcol += 1;
                    nystartcol -= nystartcol & 1;
                    nystartrow = nystartrow.max(8);
                    nyendrow = nyendrow.min(rr1 - 8);
                    nystartcol = nystartcol.max(8);
                    nyendcol = nyendcol.min(cc1 - 8);

                    nyquist2[4 * TSH .. (TS - 8) * TSH].fill(0);

                    for rr in nystartrow..nyendrow {
                        let mut indx = rr * TS + nystartcol + (color_at(top + rr as i32, left + 2) & 1);
                        while indx < rr * TS + nyendcol {
                            let nyquisttemp = nyquist[(indx - V2) >> 1] as u32
                                + nyquist[(indx - M1) >> 1] as u32
                                + nyquist[(indx - P1) >> 1] as u32 // wait! indx + p1 = indx - P1.
                                + nyquist[(indx - 2) >> 1] as u32
                                + nyquist[(indx + 2) >> 1] as u32
                                + nyquist[(indx + P1) >> 1] as u32 // wait! indx - p1 = indx + P1.
                                + nyquist[(indx + M1) >> 1] as u32
                                + nyquist[(indx + V2) >> 1] as u32;
                            nyquist2[indx >> 1] = if nyquisttemp > 4 {
                                1
                            } else if nyquisttemp < 4 {
                                0
                            } else {
                                nyquist[indx >> 1]
                            };
                            indx += 2;
                        }
                    }

                    for rr in nystartrow..nyendrow {
                        let mut indx = rr * TS + nystartcol + (color_at(top + rr as i32, left + 2) & 1);
                        while indx < rr * TS + nyendcol {
                            if nyquist2[indx >> 1] != 0 {
                                let mut sumcfa = 0.0f32;
                                let mut sumh = 0.0f32;
                                let mut sumv = 0.0f32;
                                let mut sumsqh = 0.0f32;
                                let mut sumsqv = 0.0f32;
                                let mut areawt = 0.0f32;

                                let mut i = -6i32;
                                while i < 7 {
                                    let mut indx1 = (indx as i32 + i * TS as i32 - 6) as usize;
                                    let mut j = -6i32;
                                    while j < 7 {
                                        if nyquist2[indx1 >> 1] != 0 {
                                            let cfatemp = cfa_buf[indx1];
                                            sumcfa += cfatemp;
                                            sumh += cfa_buf[indx1 - 1] + cfa_buf[indx1 + 1];
                                            sumv += cfa_buf[indx1 - V1] + cfa_buf[indx1 + V1];
                                            sumsqh += sqr(cfatemp - cfa_buf[indx1 - 1]) + sqr(cfatemp - cfa_buf[indx1 + 1]);
                                            sumsqv += sqr(cfatemp - cfa_buf[indx1 - V1]) + sqr(cfatemp - cfa_buf[indx1 + V1]);
                                            areawt += 1.0;
                                        }
                                        j += 2;
                                        indx1 += 2;
                                    }
                                    i += 2;
                                }

                                sumh = sumcfa - sumh * 0.5;
                                sumv = sumcfa - sumv * 0.5;
                                areawt = areawt * 0.5;
                                let hcdvar = epssq + (areawt * sumsqh - sumh * sumh).abs();
                                let vcdvar = epssq + (areawt * sumsqv - sumv * sumv).abs();
                                hvwt[indx >> 1] = hcdvar / (vcdvar + hcdvar);
                            }
                            indx += 2;
                        }
                    }
                }

                // Populate G at R/B sites
                for rr in 8..rr1 - 8 {
                    let mut indx = rr * TS + 8 + (color_at(top + rr as i32, left + 2) & 1);
                    while indx < rr * TS + cc1 - 8 {
                        let hvwtalt = (hvwt[(indx - M1) >> 1] + hvwt[(indx - P1) >> 1] + hvwt[(indx + P1) >> 1] + hvwt[(indx + M1) >> 1]) / 4.0;
                        hvwt[indx >> 1] = if (0.5 - hvwt[indx >> 1]).abs() < (0.5 - hvwtalt).abs() {
                            hvwtalt
                        } else {
                            hvwt[indx >> 1]
                        };

                        dgrb0[indx >> 1] = interpolatef(hvwt[indx >> 1], vcd[indx], hcd[indx]);
                        rgbgreen[indx] = cfa_buf[indx] + dgrb0[indx >> 1];

                        dgrb2_h[indx >> 1] = if nyquist2[indx >> 1] != 0 {
                            sqr(rgbgreen[indx] - (rgbgreen[indx - 1] + rgbgreen[indx + 1]) * 0.5)
                        } else {
                            0.0
                        };
                        dgrb2_v[indx >> 1] = if nyquist2[indx >> 1] != 0 {
                            sqr(rgbgreen[indx] - (rgbgreen[indx - V1] + rgbgreen[indx + V1]) * 0.5)
                        } else {
                            0.0
                        };

                        indx += 2;
                    }
                }

                if do_nyquist {
                    for rr in nystartrow..nyendrow {
                        let mut indx = rr * TS + nystartcol + (color_at(top + rr as i32, left + 2) & 1);
                        while indx < rr * TS + nyendcol {
                            if nyquist2[indx >> 1] != 0 {
                                let gvarh = epssq + (gquinc[0] * dgrb2_h[indx >> 1]
                                    + gquinc[1] * (dgrb2_h[(indx - M1) >> 1] + dgrb2_h[(indx - P1) >> 1] + dgrb2_h[(indx + P1) >> 1] + dgrb2_h[(indx + M1) >> 1])
                                    + gquinc[2] * (dgrb2_h[(indx - V2) >> 1] + dgrb2_h[(indx - 2) >> 1] + dgrb2_h[(indx + 2) >> 1] + dgrb2_h[(indx + V2) >> 1])
                                    + gquinc[3] * (dgrb2_h[(indx - M2) >> 1] + dgrb2_h[(indx - P2) >> 1] + dgrb2_h[(indx + P2) >> 1] + dgrb2_h[(indx + M2) >> 1]));
                                let gvarv = epssq + (gquinc[0] * dgrb2_v[indx >> 1]
                                    + gquinc[1] * (dgrb2_v[(indx - M1) >> 1] + dgrb2_v[(indx - P1) >> 1] + dgrb2_v[(indx + P1) >> 1] + dgrb2_v[(indx + M1) >> 1])
                                    + gquinc[2] * (dgrb2_v[(indx - V2) >> 1] + dgrb2_v[(indx - 2) >> 1] + dgrb2_v[(indx + 2) >> 1] + dgrb2_v[(indx + V2) >> 1])
                                    + gquinc[3] * (dgrb2_v[(indx - M2) >> 1] + dgrb2_v[(indx - P2) >> 1] + dgrb2_v[(indx + P2) >> 1] + dgrb2_v[(indx + M2) >> 1]));

                                dgrb0[indx >> 1] = (hcd[indx] * gvarv + vcd[indx] * gvarh) / (gvarv + gvarh);
                                rgbgreen[indx] = cfa_buf[indx] + dgrb0[indx >> 1];
                            }
                            indx += 2;
                        }
                    }
                }

                // Diagonal interpolation correction setup
                for rr in 6..rr1 - 6 {
                    let mut cc = 6;
                    let is_even_row = (color_at(top + rr as i32, left + 2) & 1) == 0;
                    while cc < cc1 - 6 {
                        let indx = rr * TS + cc;
                        if is_even_row {
                            delp[indx >> 1] = (cfa_buf[indx - P1] - cfa_buf[indx + P1]).abs();
                            delm[indx >> 1] = (cfa_buf[indx + M1] - cfa_buf[indx - M1]).abs();
                            dgrbsq1p[indx >> 1] = sqr(cfa_buf[indx + 1] - cfa_buf[indx + 1 - P1]) + sqr(cfa_buf[indx + 1] - cfa_buf[indx + 1 + P1]);
                            dgrbsq1m[indx >> 1] = sqr(cfa_buf[indx + 1] - cfa_buf[indx + 1 + M1]) + sqr(cfa_buf[indx + 1] - cfa_buf[indx + 1 - M1]);
                        } else {
                            dgrbsq1p[indx >> 1] = sqr(cfa_buf[indx] - cfa_buf[indx - P1]) + sqr(cfa_buf[indx] - cfa_buf[indx + P1]);
                            dgrbsq1m[indx >> 1] = sqr(cfa_buf[indx] - cfa_buf[indx - M1]) + sqr(cfa_buf[indx] - cfa_buf[indx + M1]);
                            delp[indx >> 1] = (cfa_buf[indx + 1 - P1] - cfa_buf[indx + 1 + P1]).abs();
                            delm[indx >> 1] = (cfa_buf[indx + 1 - M1] - cfa_buf[indx + 1 + M1]).abs();
                        }
                        cc += 2;
                    }
                }

                // Diagonal interpolation correction
                // ponytail: replaced bitwise exponent manipulation with standard multiplication for safety/portability
                for rr in 8..rr1 - 8 {
                    let mut cc = 8 + (color_at(top + rr as i32, left + 2) & 1);
                    while cc < cc1 - 8 {
                        let indx = rr * TS + cc;
                        let indx1 = indx >> 1;

                        let crse = (cfa_buf[indx + M1] * 2.0) / (eps + cfa_buf[indx] + cfa_buf[indx + M2]);
                        let crnw = (cfa_buf[indx - M1] * 2.0) / (eps + cfa_buf[indx] + cfa_buf[indx - M2]);
                        let crne = (cfa_buf[indx - P1] * 2.0) / (eps + cfa_buf[indx] + cfa_buf[indx - P2]);
                        let crsw = (cfa_buf[indx + P1] * 2.0) / (eps + cfa_buf[indx] + cfa_buf[indx + P2]);

                        let rbse = if (1.0 - crse).abs() < arthresh {
                            cfa_buf[indx] * crse
                        } else {
                            cfa_buf[indx + M1] + (cfa_buf[indx] - cfa_buf[indx + M2]) * 0.5
                        };

                        let rbnw = if (1.0 - crnw).abs() < arthresh {
                            cfa_buf[indx] * crnw
                        } else {
                            cfa_buf[indx - M1] + (cfa_buf[indx] - cfa_buf[indx - M2]) * 0.5
                        };

                        let rbne = if (1.0 - crne).abs() < arthresh {
                            cfa_buf[indx] * crne
                        } else {
                            cfa_buf[indx - P1] + (cfa_buf[indx] - cfa_buf[indx - P2]) * 0.5
                        };

                        let rbsw = if (1.0 - crsw).abs() < arthresh {
                            cfa_buf[indx] * crsw
                        } else {
                            cfa_buf[indx + P1] + (cfa_buf[indx] - cfa_buf[indx + P2]) * 0.5
                        };

                        let wtse = eps + delm[indx1] + delm[(indx + M1) >> 1] + delm[(indx + M2) >> 1];
                        let wtnw = eps + delm[indx1] + delm[(indx - M1) >> 1] + delm[(indx - M2) >> 1];
                        let wtne = eps + delp[indx1] + delp[(indx - P1) >> 1] + delp[(indx - P2) >> 1];
                        let wtsw = eps + delp[indx1] + delp[(indx + P1) >> 1] + delp[(indx + P2) >> 1];

                        rbm[indx1] = (wtse * rbnw + wtnw * rbse) / (wtse + wtnw);
                        rbp[indx1] = (wtne * rbsw + wtsw * rbne) / (wtne + wtsw);

                        let rbvarm = epssq + gausseven[0] * (dgrbsq1m[(indx - V1) >> 1] + dgrbsq1m[(indx - 1) >> 1] + dgrbsq1m[(indx + 1) >> 1] + dgrbsq1m[(indx + V1) >> 1])
                            + gausseven[1] * (dgrbsq1m[(indx - V2 - 1) >> 1] + dgrbsq1m[(indx - V2 + 1) >> 1]
                                + dgrbsq1m[(indx - 2 - V1) >> 1] + dgrbsq1m[(indx + 2 - V1) >> 1]
                                + dgrbsq1m[(indx - 2 + V1) >> 1] + dgrbsq1m[(indx + 2 + V1) >> 1]
                                + dgrbsq1m[(indx + V2 - 1) >> 1] + dgrbsq1m[(indx + V2 + 1) >> 1]);

                        let rbvarp = epssq + gausseven[0] * (dgrbsq1p[(indx - V1) >> 1] + dgrbsq1p[(indx - 1) >> 1] + dgrbsq1p[(indx + 1) >> 1] + dgrbsq1p[(indx + V1) >> 1])
                            + gausseven[1] * (dgrbsq1p[(indx - V2 - 1) >> 1] + dgrbsq1p[(indx - V2 + 1) >> 1]
                                + dgrbsq1p[(indx - 2 - V1) >> 1] + dgrbsq1p[(indx + 2 - V1) >> 1]
                                + dgrbsq1p[(indx - 2 + V1) >> 1] + dgrbsq1p[(indx + 2 + V1) >> 1]
                                + dgrbsq1p[(indx + V2 - 1) >> 1] + dgrbsq1p[(indx + V2 + 1) >> 1]);

                        pmwt[indx1] = rbvarm / (rbvarp + rbvarm);

                        if rbp[indx1] < cfa_buf[indx] {
                            if rbp[indx1] * 2.0 < cfa_buf[indx] {
                                rbp[indx1] = ulim(rbp[indx1], cfa_buf[indx - P1], cfa_buf[indx + P1]);
                            } else {
                                let pwt = (cfa_buf[indx] - rbp[indx1]) * 2.0 / (eps + rbp[indx1] + cfa_buf[indx]);
                                rbp[indx1] = pwt * rbp[indx1] + (1.0 - pwt) * ulim(rbp[indx1], cfa_buf[indx - P1], cfa_buf[indx + P1]);
                            }
                        }

                        if rbm[indx1] < cfa_buf[indx] {
                            if rbm[indx1] * 2.0 < cfa_buf[indx] {
                                rbm[indx1] = ulim(rbm[indx1], cfa_buf[indx - M1], cfa_buf[indx + M1]);
                            } else {
                                let mwt = (cfa_buf[indx] - rbm[indx1]) * 2.0 / (eps + rbm[indx1] + cfa_buf[indx]);
                                rbm[indx1] = mwt * rbm[indx1] + (1.0 - mwt) * ulim(rbm[indx1], cfa_buf[indx - M1], cfa_buf[indx + M1]);
                            }
                        }

                        if rbp[indx1] > self.clip_pt {
                            rbp[indx1] = ulim(rbp[indx1], cfa_buf[indx - P1], cfa_buf[indx + P1]);
                        }
                        if rbm[indx1] > self.clip_pt {
                            rbm[indx1] = ulim(rbm[indx1], cfa_buf[indx - M1], cfa_buf[indx + M1]);
                        }

                        cc += 2;
                    }
                }

                for rr in 10..rr1 - 10 {
                    let mut cc = 10 + (color_at(top + rr as i32, left + 2) & 1);
                    while cc < cc1 - 10 {
                        let indx = rr * TS + cc;
                        let indx1 = indx >> 1;

                        let pmwtalt = (pmwt[(indx - M1) >> 1] + pmwt[(indx - P1) >> 1] + pmwt[(indx + P1) >> 1] + pmwt[(indx + M1) >> 1]) / 4.0;
                        if (0.5 - pmwt[indx1]).abs() < (0.5 - pmwtalt).abs() {
                            pmwt[indx1] = pmwtalt;
                        }

                        rbint[indx1] = (cfa_buf[indx] + rbm[indx1] * (1.0 - pmwt[indx1]) + rbp[indx1] * pmwt[indx1]) * 0.5;
                        cc += 2;
                    }
                }

                for rr in 12..rr1 - 12 {
                    let mut cc = 12 + (color_at(top + rr as i32, left + 2) & 1);
                    while cc < cc1 - 12 {
                        let indx = rr * TS + cc;
                        let indx1 = indx >> 1;

                        if (0.5 - pmwt[indx >> 1]).abs() >= (0.5 - hvwt[indx >> 1]).abs() {
                            let cru = cfa_buf[indx - V1] * 2.0 / (eps + rbint[indx1] + rbint[indx1 - V1]);
                            let crd = cfa_buf[indx + V1] * 2.0 / (eps + rbint[indx1] + rbint[indx1 + V1]);
                            let crl = cfa_buf[indx - 1] * 2.0 / (eps + rbint[indx1] + rbint[indx1 - 1]);
                            let crr = cfa_buf[indx + 1] * 2.0 / (eps + rbint[indx1] + rbint[indx1 + 1]);

                            let gu = if (1.0 - cru).abs() < arthresh {
                                rbint[indx1] * cru
                            } else {
                                cfa_buf[indx - V1] + (rbint[indx1] - rbint[indx1 - V1]) * 0.5
                            };

                            let gd = if (1.0 - crd).abs() < arthresh {
                                rbint[indx1] * crd
                            } else {
                                cfa_buf[indx + V1] + (rbint[indx1] - rbint[indx1 + V1]) * 0.5
                            };

                            let gl = if (1.0 - crl).abs() < arthresh {
                                rbint[indx1] * crl
                            } else {
                                cfa_buf[indx - 1] + (rbint[indx1] - rbint[indx1 - 1]) * 0.5
                            };

                            let gr = if (1.0 - crr).abs() < arthresh {
                                rbint[indx1] * crr
                            } else {
                                cfa_buf[indx + 1] + (rbint[indx1] - rbint[indx1 + 1]) * 0.5
                            };

                            let mut gintv = (dirwts0[indx - V1] * gd + dirwts0[indx + V1] * gu) / (dirwts0[indx + V1] + dirwts0[indx - V1]);
                            let mut ginth = (dirwts1[indx - 1] * gr + dirwts1[indx + 1] * gl) / (dirwts1[indx - 1] + dirwts1[indx + 1]);

                            if gintv < rbint[indx1] {
                                if 2.0 * gintv < rbint[indx1] {
                                    gintv = ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]);
                                } else {
                                    let vwt = 2.0 * (rbint[indx1] - gintv) / (eps + gintv + rbint[indx1]);
                                    gintv = vwt * gintv + (1.0 - vwt) * ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]);
                                }
                            }

                            if ginth < rbint[indx1] {
                                if 2.0 * ginth < rbint[indx1] {
                                    ginth = ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]);
                                } else {
                                    let hwt = 2.0 * (rbint[indx1] - ginth) / (eps + ginth + rbint[indx1]);
                                    ginth = hwt * ginth + (1.0 - hwt) * ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]);
                                }
                            }

                            if ginth > self.clip_pt {
                                ginth = ulim(ginth, cfa_buf[indx - 1], cfa_buf[indx + 1]);
                            }
                            if gintv > self.clip_pt {
                                gintv = ulim(gintv, cfa_buf[indx - V1], cfa_buf[indx + V1]);
                            }

                            rgbgreen[indx] = ginth * (1.0 - hvwt[indx1]) + gintv * hvwt[indx1];
                            dgrb0[indx1] = rgbgreen[indx] - cfa_buf[indx];
                        }
                        cc += 2;
                    }
                }

                // Fancy chrominance interpolation
                // (ey,ex) is location of R site
                let (ex, ey) = if color_at(0, 0) == 1 {
                    if color_at(0, 1) == 0 { (1, 0) } else { (0, 1) }
                } else {
                    if color_at(0, 0) == 0 { (0, 0) } else { (1, 1) }
                };

                let mut rr = 13 - ey;
                while rr < rr1 - 12 {
                    let mut indx1 = (rr * TS + 13 - ex) >> 1;
                    let limit = (rr * TS + cc1 - 12) >> 1;
                    while indx1 < limit {
                        dgrb1[indx1] = dgrb0[indx1];
                        dgrb0[indx1] = 0.0;
                        indx1 += 1;
                    }
                    rr += 2;
                }

                for rr in 14..rr1 - 14 {
                    let mut cc = 14 + (color_at(top + rr as i32, left + 2) & 1);
                    while cc < cc1 - 14 {
                        let indx = rr * TS + cc;
                        let c = 1 - (color_at(top + rr as i32, left + cc as i32) / 2);

                        let val = {
                            let dgrb_c = if c == 0 { &dgrb0 } else { &dgrb1 };
                            let wtnw = 1.0 / (eps + (dgrb_c[(indx - M1) >> 1] - dgrb_c[(indx + M1) >> 1]).abs()
                                + (dgrb_c[(indx - M1) >> 1] - dgrb_c[(indx - M3) >> 1]).abs()
                                + (dgrb_c[(indx + M1) >> 1] - dgrb_c[(indx - M3) >> 1]).abs());
                            let wtne = 1.0 / (eps + (dgrb_c[(indx - P1) >> 1] - dgrb_c[(indx + P1) >> 1]).abs()
                                + (dgrb_c[(indx - P1) >> 1] - dgrb_c[(indx - P3) >> 1]).abs()
                                + (dgrb_c[(indx + P1) >> 1] - dgrb_c[(indx - P3) >> 1]).abs());
                            let wtsw = 1.0 / (eps + (dgrb_c[(indx + P1) >> 1] - dgrb_c[(indx - P1) >> 1]).abs()
                                + (dgrb_c[(indx + P1) >> 1] - dgrb_c[(indx + M3) >> 1]).abs()
                                + (dgrb_c[(indx - P1) >> 1] - dgrb_c[(indx + P3) >> 1]).abs());
                            let wtse = 1.0 / (eps + (dgrb_c[(indx + M1) >> 1] - dgrb_c[(indx - M1) >> 1]).abs()
                                + (dgrb_c[(indx + M1) >> 1] - dgrb_c[(indx + P3) >> 1]).abs()
                                + (dgrb_c[(indx - M1) >> 1] - dgrb_c[(indx + M3) >> 1]).abs());

                            (wtnw * (1.325 * dgrb_c[(indx - M1) >> 1] - 0.175 * dgrb_c[(indx - M3) >> 1]
                                    - 0.075 * dgrb_c[(indx - M1 - 2) >> 1] - 0.075 * dgrb_c[(indx - M1 - V2) >> 1])
                                + wtne * (1.325 * dgrb_c[(indx - P1) >> 1] - 0.175 * dgrb_c[(indx - P3) >> 1]
                                    - 0.075 * dgrb_c[(indx - P1 + 2) >> 1] - 0.075 * dgrb_c[(indx - P1 + V2) >> 1])
                                + wtsw * (1.325 * dgrb_c[(indx + P1) >> 1] - 0.175 * dgrb_c[(indx + P3) >> 1]
                                    - 0.075 * dgrb_c[(indx + P1 - 2) >> 1] - 0.075 * dgrb_c[(indx - M1) >> 1])
                                + wtse * (1.325 * dgrb_c[(indx + M1) >> 1] - 0.175 * dgrb_c[(indx + M3) >> 1]
                                    - 0.075 * dgrb_c[(indx + M1 + 2) >> 1] - 0.075 * dgrb_c[(indx + M1 + V2) >> 1]))
                                / (wtnw + wtne + wtsw + wtse)
                        };

                        let dgrb_c_mut = if c == 0 { &mut dgrb0 } else { &mut dgrb1 };
                        dgrb_c_mut[indx >> 1] = val;
                        cc += 2;
                    }
                }

                // Write outputs
                for rr in 16..rr1 - 16 {
                    let row = rr as i32 + top;
                    let mut col = left + 16;
                    let mut indx = rr * TS + 16;

                    if (color_at(top + rr as i32, left + 2) & 1) == 1 {
                        while indx < rr * TS + cc1 - 16 - (cc1 & 1) {
                            if col >= 0 && col < width as i32 && row >= 0 && row < height as i32 {
                                let temp = 1.0 / (hvwt[(indx - V1) >> 1] + 2.0 - hvwt[(indx + 1) >> 1]
                                    - hvwt[(indx - 1) >> 1] + hvwt[(indx + V1) >> 1]);

                                let r = clampnan(rgbgreen[indx]
                                    - ((hvwt[(indx - V1) >> 1]) * dgrb0[(indx - V1) >> 1]
                                        + (1.0 - hvwt[(indx + 1) >> 1]) * dgrb0[(indx + 1) >> 1]
                                        + (1.0 - hvwt[(indx - 1) >> 1]) * dgrb0[(indx - 1) >> 1]
                                        + (hvwt[(indx + V1) >> 1]) * dgrb0[(indx + V1) >> 1])
                                           * temp, 0.0, 1.0);

                                let b = clampnan(rgbgreen[indx]
                                    - ((hvwt[(indx - V1) >> 1]) * dgrb1[(indx - V1) >> 1]
                                        + (1.0 - hvwt[(indx + 1) >> 1]) * dgrb1[(indx + 1) >> 1]
                                        + (1.0 - hvwt[(indx - 1) >> 1]) * dgrb1[(indx - 1) >> 1]
                                        + (hvwt[(indx + V1) >> 1]) * dgrb1[(indx + V1) >> 1])
                                           * temp, 0.0, 1.0);

                                let g = clampnan(rgbgreen[indx], 0.0, 1.0);
                                out_slice.write_pixel((row as usize) * width + (col as usize), r, g, b);
                            }

                            indx += 1;
                            col += 1;
                            if col >= 0 && col < width as i32 && row >= 0 && row < height as i32 {
                                let r = clampnan(rgbgreen[indx] - dgrb0[indx >> 1], 0.0, 1.0);
                                let b = clampnan(rgbgreen[indx] - dgrb1[indx >> 1], 0.0, 1.0);
                                let g = clampnan(rgbgreen[indx], 0.0, 1.0);
                                out_slice.write_pixel((row as usize) * width + (col as usize), r, g, b);
                            }

                            indx += 1;
                            col += 1;
                        }

                        if (cc1 & 1) != 0 {
                            if col >= 0 && col < width as i32 && row >= 0 && row < height as i32 {
                                let temp = 1.0 / (hvwt[(indx - V1) >> 1] + 2.0 - hvwt[(indx + 1) >> 1]
                                    - hvwt[(indx - 1) >> 1] + hvwt[(indx + V1) >> 1]);

                                let r = clampnan(rgbgreen[indx]
                                    - ((hvwt[(indx - V1) >> 1]) * dgrb0[(indx - V1) >> 1]
                                        + (1.0 - hvwt[(indx + 1) >> 1]) * dgrb0[(indx + 1) >> 1]
                                        + (1.0 - hvwt[(indx - 1) >> 1]) * dgrb0[(indx - 1) >> 1]
                                        + (hvwt[(indx + V1) >> 1]) * dgrb0[(indx + V1) >> 1])
                                           * temp, 0.0, 1.0);

                                let b = clampnan(rgbgreen[indx]
                                    - ((hvwt[(indx - V1) >> 1]) * dgrb1[(indx - V1) >> 1]
                                        + (1.0 - hvwt[(indx + 1) >> 1]) * dgrb1[(indx + 1) >> 1]
                                        + (1.0 - hvwt[(indx - 1) >> 1]) * dgrb1[(indx - 1) >> 1]
                                        + (hvwt[(indx + V1) >> 1]) * dgrb1[(indx + V1) >> 1])
                                           * temp, 0.0, 1.0);

                                let g = clampnan(rgbgreen[indx], 0.0, 1.0);
                                out_slice.write_pixel((row as usize) * width + (col as usize), r, g, b);
                            }
                        }
                    } else {
                        while indx < rr * TS + cc1 - 16 - (cc1 & 1) {
                            if col >= 0 && col < width as i32 && row >= 0 && row < height as i32 {
                                let r = clampnan(rgbgreen[indx] - dgrb0[indx >> 1], 0.0, 1.0);
                                let b = clampnan(rgbgreen[indx] - dgrb1[indx >> 1], 0.0, 1.0);
                                let g = clampnan(rgbgreen[indx], 0.0, 1.0);
                                out_slice.write_pixel((row as usize) * width + (col as usize), r, g, b);
                            }

                            indx += 1;
                            col += 1;
                            if col >= 0 && col < width as i32 && row >= 0 && row < height as i32 {
                                let temp = 1.0 / (hvwt[(indx - V1) >> 1] + 2.0 - hvwt[(indx + 1) >> 1]
                                    - hvwt[(indx - 1) >> 1] + hvwt[(indx + V1) >> 1]);

                                let r = clampnan(rgbgreen[indx]
                                    - ((hvwt[(indx - V1) >> 1]) * dgrb0[(indx - V1) >> 1]
                                        + (1.0 - hvwt[(indx + 1) >> 1]) * dgrb0[(indx + 1) >> 1]
                                        + (1.0 - hvwt[(indx - 1) >> 1]) * dgrb0[(indx - 1) >> 1]
                                        + (hvwt[(indx + V1) >> 1]) * dgrb0[(indx + V1) >> 1])
                                           * temp, 0.0, 1.0);

                                let b = clampnan(rgbgreen[indx]
                                    - ((hvwt[(indx - V1) >> 1]) * dgrb1[(indx - V1) >> 1]
                                        + (1.0 - hvwt[(indx + 1) >> 1]) * dgrb1[(indx + 1) >> 1]
                                        + (1.0 - hvwt[(indx - 1) >> 1]) * dgrb1[(indx - 1) >> 1]
                                        + (hvwt[(indx + V1) >> 1]) * dgrb1[(indx + V1) >> 1])
                                           * temp, 0.0, 1.0);

                                let g = clampnan(rgbgreen[indx], 0.0, 1.0);
                                out_slice.write_pixel((row as usize) * width + (col as usize), r, g, b);
                            }

                            indx += 1;
                            col += 1;
                        }

                        if (cc1 & 1) != 0 {
                            if col >= 0 && col < width as i32 && row >= 0 && row < height as i32 {
                                let r = clampnan(rgbgreen[indx] - dgrb0[indx >> 1], 0.0, 1.0);
                                let b = clampnan(rgbgreen[indx] - dgrb1[indx >> 1], 0.0, 1.0);
                                let g = clampnan(rgbgreen[indx], 0.0, 1.0);
                                out_slice.write_pixel((row as usize) * width + (col as usize), r, g, b);
                            }
                        }
                    }
                }
            });

            let mut image_metadata = ImageMetadata::default();
            image_metadata.height = height;
            image_metadata.width = width;

            Image {
                rgb_data: rgb,
                raw_data: vec![],
                metadata: image_metadata,
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfa::CFA;

    #[test]
    fn test_fast_demosaic_uniform() {
        let width = 16;
        let height = 16;
        let cfa = CFA::new("RGGB");
        let input = vec![0.5; width * height];
        
        let image = demosaic_algorithms::Fast{}.demosaic(width, height, cfa, input);
        
        assert_eq!(image.metadata.width, width / 2);
        assert_eq!(image.metadata.height, height / 2);
        
        for pixel in &image.rgb_data {
            let diff_r = (pixel[0] - 0.5).abs();
            let diff_g = (pixel[1] - 0.5).abs();
            let diff_b = (pixel[2] - 0.5).abs();
            assert!(diff_r < 1e-4, "r is {}", pixel[0]);
            assert!(diff_g < 1e-4, "g is {}", pixel[1]);
            assert!(diff_b < 1e-4, "b is {}", pixel[2]);
        }
    }

    #[test]
    fn test_amaze_demosaic_uniform() {
        let width = 32;
        let height = 32;
        let cfa = CFA::new("RGGB");
        let input = vec![0.5; width * height];
        
        let image = demosaic_algorithms::Amaze::default().demosaic(width, height, cfa, input);
        
        assert_eq!(image.metadata.width, width);
        assert_eq!(image.metadata.height, height);
        
        for pixel in &image.rgb_data {
            let diff_r = (pixel[0] - 0.5).abs();
            let diff_g = (pixel[1] - 0.5).abs();
            let diff_b = (pixel[2] - 0.5).abs();
            assert!(diff_r < 0.2, "r is {}", pixel[0]);
            assert!(diff_g < 0.2, "g is {}", pixel[1]);
            assert!(diff_b < 0.2, "b is {}", pixel[2]);
        }
    }

    #[test]
    fn test_amaze_demosaic_tiled() {
        let width = 160;
        let height = 160;
        let cfa = CFA::new("RGGB");
        let input = vec![0.5; width * height];
        
        let image = demosaic_algorithms::Amaze::default().demosaic(width, height, cfa, input);
        
        assert_eq!(image.metadata.width, width);
        assert_eq!(image.metadata.height, height);
    }
}
