use std::usize;

use rawler::imgop;
use rawler::imgop::{Dim2, Rect};
use ndarray::{Array2, Array3};
use rawler::pixarray::RgbF32;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub fn crop(dim: Dim2, crop_rect: Rect, data: Vec<f32>) -> (Vec<f32>, usize, usize) {
    let nim = imgop::crop(&data, dim, crop_rect);
    return (nim, crop_rect.d.w, crop_rect.d.h);
}
pub fn get_cfa(cfa: rawler::CFA, crop_rect: Rect) -> rawler::CFA {
    let x = crop_rect.p.x;
    let y = crop_rect.p.y;
    let new_cfa = cfa.shift(x, y);
    return new_cfa;
}
pub struct DemosaicAlgorithms{}

impl DemosaicAlgorithms{
    pub fn passthough(
        width: usize,
        height: usize,
        black: f32,
        white: f32,
        data: Vec<u16>,
    ) -> Array2<f32> {
        let mut final_image = Array2::<f32>::zeros((height, width));
        for i in 0..height{
            for j in 0..width{
                let value = (data[(i*width)+j] as f32 - black)/(white-black);
                final_image[[i,j]] = value;
            }
        }
        return final_image
    }
    pub fn photosite(
        width: usize,
        height: usize,
        cfa: rawler::CFA,
        data: Vec<f32>,
    ) -> Array3<f32>{
        let w = width;
        let h = height;
        let mut final_image = Array3::<f32>::zeros((height, width, 3));
        for i in 0..h{
            for j in 0..w{
                let index = (i*width)+(j);
                let channel = cfa.color_at(i%cfa.height, j%cfa.height);
                final_image[[i, j, channel]] = data[index] as f32;
            }
        }
        return final_image;
    }
    pub fn linear_interpolation(
        width: usize,
        height: usize,
        cfa: rawler::CFA,
        data: Vec<f32>,
    ) -> RgbF32 {
        let w = width;
        let h = height;
        let mut final_image = RgbF32::new(w-2, h-2);
        let f = |(indx, _)| {
            let mut j: usize = indx%final_image.width;
            let mut i: usize = (indx-j)/final_image.width;
            j+=1;
            i+=1;

            let mut pixel_count = [0.0, 0.0, 0.0];
            let mut pixel = [0.0, 0.0, 0.0];

            for i2 in 0..3{
                for j2 in 0..3{
                    let index = ((i+i2-1)*width)+(j+j2-1);
                    let channel = cfa.color_at((i+i2)-1, (j+j2)-1);
                    pixel_count[channel] += 1.0;
                    pixel[channel] += data[index];
                }
            }
            [
                pixel[0]/pixel_count[0],
                pixel[1]/pixel_count[1],
                pixel[2]/pixel_count[2],
            ]

        };
        final_image.data = final_image.data.par_iter().with_min_len(final_image.width).enumerate().map(f).collect();
        return final_image;
    }

    pub fn pass(
        width: usize,
        height: usize,
        black: f32,
        white: f32,
        data: Vec<u16>,
    ) -> Array3<f32> {
        let pt = DemosaicAlgorithms::passthough(width, height, black, white, data);
        let w = width;
        let h = height;
        let mut final_image = Array3::<f32>::zeros((h, w, 3));
        for i in 0..h {
            for j in 0..w {
                final_image[[i,j,0]] = pt[[i,j]];
                final_image[[i,j,1]] = pt[[i,j]];
                final_image[[i,j,2]] = pt[[i,j]];
            }
        }
        return final_image
    }
    pub fn markesteijn(
        width: usize,
        height: usize,
        cfa: rawler::CFA,
        input: Vec<f32>,
    ) -> RgbF32 {
        // Initialize RGB buffers
        let mut rgb = vec![[0.0f32; 3]; width * height];
        rgb = rgb.par_iter().enumerate().map(|(idx, _)|{
            let col = idx % width;
            let row = (idx - col)/width;
            let idx = row * width + col;
            let color = cfa.color_at(row, col);
            let mut pix = rgb[idx];
            pix[color] = input[idx];
            pix
        }).collect();

        // Green interpolation for non-green pixels
        rgb = rgb.par_iter().enumerate().map(|(idx, _)|{
            let col = idx % width;
            let row = (idx - col)/width;
            if rgb[idx][1] != 0.0 {
                rgb[idx]
            }else{

                // Collect surrounding green values
                let mut greens = Vec::new();
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let y = row as isize + dy;
                        let x = col as isize + dx;
                        if y < 0 || y >= height as isize || x < 0 || x >= width as isize {
                            continue;
                        }
                        let y = y as usize;
                        let x = x as usize;
                        let neighbor_idx = y * width + x;
                        if cfa.color_at(y, x) == 1 {
                            greens.push(rgb[neighbor_idx][1]);
                        }
                    }
                }

                // Average the green values
                let sum: f32 = greens.iter().sum();
                [
                    rgb[idx][0],
                    sum / greens.len() as f32,
                    rgb[idx][2],
                ]
            }
        }).collect();

        // Red and Blue interpolation using the green values
        rgb = rgb.par_iter().enumerate().map(|(idx, _)|{
                let col = idx % width;
                let row = (idx - col)/width;

                let idx = row * width + col;
                let color = cfa.color_at(row, col);
                if color == 1 {
                    // Green pixel, interpolate Red and Blue
                    let mut reds = Vec::new();
                    let mut blues = Vec::new();
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let y = row as isize + dy;
                            let x = col as isize + dx;
                            if y < 0 || y >= height as isize || x < 0 || x >= width as isize {
                                continue;
                            }
                            let y = y as usize;
                            let x = x as usize;
                            let neighbor_idx = y * width + x;
                            let neighbor_color = cfa.color_at(y, x);
                            if neighbor_color == 0 {
                                reds.push(rgb[neighbor_idx][0]);
                            } else if neighbor_color == 2 {
                                blues.push(rgb[neighbor_idx][2]);
                            }
                        }
                    }
                [
                    reds.iter().sum::<f32>() / reds.len() as f32,
                    rgb[idx][1],
                    blues.iter().sum::<f32>() / blues.len() as f32,
                ]
                } else {
                    // Non-green pixel, interpolate the missing color
                    let target_color = if color == 0 { 2 } else { 0 };
                    let mut samples = Vec::new();
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let y = row as isize + dy;
                            let x = col as isize + dx;
                            if y < 0 || y >= height as isize || x < 0 || x >= width as isize {
                                continue;
                            }
                            let y = y as usize;
                            let x = x as usize;
                            let neighbor_idx = y * width + x;
                            if cfa.color_at(y, x) == target_color {
                                samples.push(rgb[neighbor_idx][target_color]);
                            }
                        }
                    }
                    let mut pix = rgb[idx];
                    pix[target_color] = samples.iter().sum::<f32>() / samples.len() as f32;
                    pix
                }
            }).collect();

        let mut output = RgbF32::new(width, height);
        output.data = rgb;

        output
    }
}
