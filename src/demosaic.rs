use std::usize;

use rawler::imgop;
use rawler::imgop::{Dim2, Rect};

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

pub mod demosaic_algorithms{
    use ndarray::{Array2, Array3};
    use rawler::pixarray::RgbF32;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
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
        let pt = passthough(width, height, black, white, data);
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
}
