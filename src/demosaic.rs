use std::usize;

use ndarray::{s, Array1, Array2, Array3};
use rawler::imgop;
use rawler::imgop::{Dim2, Rect};

pub fn crop(dim: Dim2, crop_rect: Rect, data: Vec<f32>) -> (Vec<f32>, usize, usize) {
    let nw = crop_rect.d.w - (crop_rect.d.w%6);
    let nh = crop_rect.d.h - (crop_rect.d.h%6);
    let ncrop = Rect{
        p: crop_rect.p,
        d: Dim2 { w: nw, h: nh }
    };
    let nim = imgop::crop(&data, dim, ncrop);
    println!("{:?}, {:}, {:}", ncrop.p, ncrop.d.w, ncrop.d.h);
    println!("{:}", nim.len());
    return (nim, ncrop.d.w, ncrop.d.h);
}

pub fn get_cfa(cfa: rawler::CFA) -> rawler::CFA {
    let x = 6;
    let y = 13;
    let new_cfa = cfa.shift(x, y);
    return new_cfa;
}

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
    black: f32,
    white: f32,
    cfa: rawler::CFA,
    data: Vec<f32>,
) -> Array3<f32>{
    let w = width;
    let h = height;
    let mut final_image = Array3::<f32>::zeros((height, width, 3));
    for i in (0..h){
        for j in (0..w){
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
) -> Array3<f32> {
    let cfa = get_cfa(cfa);
    let w = width;
    let h = height;
    let mut final_image = Array3::<f32>::zeros((h-2, w-2, 3));
    for i in 1..h-1 {
        for j in 1..w-1{
            let mut pixel_count = [0.0, 0.0, 0.0];
            let mut pixel = [0.0, 0.0, 0.0];

            for i2 in 0..3{
                for j2 in 0..3{
                    let index = ((i+i2-1)*width)+(j+j2-1);
                    let channel = cfa.color_at(i+(i2-1), j+(j2-1));
                    pixel_count[channel] += 1.0;
                    pixel[channel] += data[index];
                }
            }
            final_image[[i-1,j-1,0]] = pixel[0]/pixel_count[0];
            final_image[[i-1,j-1,1]] = pixel[1]/pixel_count[1];
            final_image[[i-1,j-1,2]] = pixel[2]/pixel_count[2];
        }
    }
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
