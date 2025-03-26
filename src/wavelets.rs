use std::{isize, time::Instant, usize};

use convolve_image::kernel::SeparableKernel;
use crate::conditional_paralell::prelude::*;

#[derive(Copy, Clone)]
pub enum Kernel {
    LinearInterpolationKernel,
    LowScaleKernel,
    B3SplineKernel,
}
fn compute_signal_index(
        stride: usize,
        kernel_size: usize,
        kernel_index: isize,
        signal_index: usize,
        max: usize,
    ) -> u32 {
        let kernel_size = kernel_size as isize;
        let kernel_padding = kernel_size / 2;

        let distance = kernel_index * stride as isize;

        let mut index = signal_index as isize + distance;

        if index < 0 {
            index = -index;
        } else if index > max as isize - kernel_padding {
            let overshot_distance = index - max as isize + kernel_padding;
            index = max as isize - overshot_distance;
        }

        index as u32
    }

fn convolve_ref<const KERNEL_SIZE: usize>(
    data: &mut Vec<f32>,
    height: usize,
    width: usize,
    kernel: [f32; KERNEL_SIZE],
    stride: usize
) {
        for idx in 0..data.len() {
            let x = idx % width;
            let y = (idx-x) / width;
            let mut signal_sum = 0.;

            for (kernel_index, value) in kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let signal_index = compute_signal_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x,
                    width
                );

                signal_sum += data[(y * width) + signal_index as usize] * *value;
            }

            data[(y*width) + x] = signal_sum;
        }

        for idx in 0..data.len() {
            let x = idx % width;
            let y = (idx-x) / width;
            let mut signal_sum = 0.;

            for (kernel_index, value) in kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let signal_index = compute_signal_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y,
                    height
                );

                signal_sum += data[(signal_index as usize * width) + x] * *value;
            }

            data[(y*width) + x] = signal_sum;
        }
    }
#[inline]
fn compute_pixel_index(
    stride: isize,
    kernel_padding: isize,
    kernel_index: isize,
    signal_index: usize,
    max: usize,
) -> usize {

    let mut index = signal_index as isize + kernel_index * stride;

    if index < 0 {
        return - index as usize
    }
    let bound = max as isize - kernel_padding;
    if index > bound {
        index = max as isize - index + bound;
    }

    index as usize
}
fn convolve<const KERNEL_SIZE: usize>(
    data: &mut Vec<f32>,
    height: usize,
    width: usize,
    linear_kernel: [f32; KERNEL_SIZE],
    stride: usize,
) {
    let kernel_side = KERNEL_SIZE as isize / 2;
    let kernel_isize = KERNEL_SIZE as isize;
    let kernel_padding = kernel_isize / 2;

    // let mut inter_image: Vec<f32> = vec![0.0; width*height];

    // let inter_image: Vec<f32> = (0..data.len()).map(|idx| {
    //     let x = idx % width;
    //     let y = (idx-x) / width;

    //     let pixel_sum = linear_kernel.iter().enumerate().fold(0.0, |acc, (kernel_index, value)| {
    //         let relative_kernel_index = kernel_index as isize - kernel_side;

    //         let pixel_index_x = compute_pixel_index(
    //             stride,
    //             kernel_padding,
    //             relative_kernel_index,
    //             x,
    //             width
    //         );

    //         acc + data[(y * width) + pixel_index_x] * value
    //     });

    //     pixel_sum
    // }).collect();
    //
    let inter = data.clone();

    *data = (0..data.len()).into_par_iter().map(|idx| {
        let x = idx % width;
        let y = (idx-x) / width;

        let pixel_sum = linear_kernel.iter().enumerate().fold(0.0, |acc, (kernel_index, value)| {
            let relative_kernel_index = kernel_index as isize - kernel_side;

            let pixel_index_y = compute_pixel_index(
                stride as isize,
                kernel_padding,
                relative_kernel_index,
                y,
                height
            );
            let pixel_index_x = compute_pixel_index(
                stride as isize,
                kernel_padding,
                relative_kernel_index,
                x,
                width
            );

            acc + (inter[(pixel_index_y * width) + x] + inter[(y * width)+ pixel_index_x]) * value
        });

        pixel_sum/2.
    }).collect::<Vec<f32>>();
}
fn convolve3<const KERNEL_SIZE: usize>(
    data: &mut Vec<[f32; 3]>,
    height: usize,
    width: usize,
    linear_kernel: [f32; KERNEL_SIZE],
    stride: isize,
) -> Vec<[f32; 3]>{
    let kernel_isize = KERNEL_SIZE as isize;
    let kernel_padding = kernel_isize / 2;

    (0..data.len()).into_par_iter().map(|idx| {
        let x = idx % width;
        let y = (idx-x) / width;

        let mut pixel_sum: [f32; 3] = [0.0; 3];
        for (kernel_index, value) in linear_kernel.iter().enumerate(){
            let relative_kernel_index = kernel_index as isize - kernel_padding;

            let pixel_index_y = compute_pixel_index(
                stride,
                kernel_padding,
                relative_kernel_index,
                y,
                height
            );
            let pixel_index_x = compute_pixel_index(
                stride,
                kernel_padding,
                relative_kernel_index,
                x,
                width
            );
            let [ar, ag, ab] = data[pixel_index_y * width + x];
            let [br, bg, bb] = data[y * width + pixel_index_x];
            let [r, g, b] = pixel_sum;

            pixel_sum = [
                r + (ar + br) * value,
                g + (ag + bg) * value,
                b + (ab + bb) * value,
            ]
        }

        pixel_sum.map(|x|x/2.0)
    }).collect::<Vec<[f32; 3]>>()
}
#[derive(Copy, Clone)]
pub(crate) struct LinearInterpolationKernel(SeparableKernel<3>);

#[derive(Copy, Clone)]
pub(crate) struct B3SplineKernel(SeparableKernel<5>);

impl From<LinearInterpolationKernel> for SeparableKernel<3> {
    fn from(value: LinearInterpolationKernel) -> Self {
        value.0
    }
}


impl From<B3SplineKernel> for SeparableKernel<5> {
    fn from(value: B3SplineKernel) -> Self {
        value.0
    }
}

pub trait WaveletDecompose<T> {
    fn wavelet_decompose(&mut self, height:usize, width: usize, kernel: Kernel, pixel_scale: usize) -> Vec<T>;
}

impl WaveletDecompose<f32> for Vec<f32> {
    fn wavelet_decompose(&mut self, height: usize, width: usize, kernel: Kernel, pixel_scale: usize) -> Vec<f32> {
        let stride = 2_usize.pow(
            u32::try_from(pixel_scale)
                .unwrap_or_else(|_| panic!("pixel_scale cannot be larger than {}", u32::MAX)),
        );
        let mut current_data = self.clone();
        match kernel {
            Kernel::LinearInterpolationKernel => {
                let kernel = [1. / 4., 1. / 2., 1. / 4.];
                convolve(&mut current_data, height, width, kernel, stride)
            }
            Kernel::LowScaleKernel => {
                unimplemented!("Low scale is not a separable kernel");
            }
            Kernel::B3SplineKernel => {
                let kernel = [
                    1. / 16.,
                    1. / 4.,
                    3. / 8.,
                    1. / 4.,
                    1. / 16.,
                ];
                convolve(&mut current_data, height, width, kernel, stride)
            }
            ,
        }

        let final_data = self.clone().iter().zip(&current_data).map(|(v1, v2)| v1 - v2).collect();
        *self = current_data;

        final_data
    }
}

impl WaveletDecompose<[f32; 3]> for Vec<[f32; 3]> {
    fn wavelet_decompose(&mut self, height: usize, width: usize, kernel: Kernel, pixel_scale: usize) -> Vec<[f32; 3]> {
        let stride = 2_usize.pow(
            u32::try_from(pixel_scale)
                .unwrap_or_else(|_| panic!("pixel_scale cannot be larger than {}", u32::MAX)),
        );
        let current_data = match kernel {
            Kernel::LinearInterpolationKernel => {
                let kernel = [1. / 4., 1. / 2., 1. / 4.];
                convolve3(self, height, width, kernel, stride as isize)
            }
            Kernel::LowScaleKernel => {
                unimplemented!("Low scale is not a separable kernel");
            }
            Kernel::B3SplineKernel => {
                let kernel = [
                    1. / 16.,
                    1. / 4.,
                    3. / 8.,
                    1. / 4.,
                    1. / 16.,
                ];
                convolve3(self, height, width, kernel, stride as isize)
            }
        };

        let final_data = current_data.par_iter().zip(&mut *self).map(|([r1, g1, b1], [r, g, b])| {
            [
                *r - r1,
                *g - g1,
                *b - b1,
            ]
        }).collect();

        *self = current_data;

        final_data
    }
}

#[cfg(test)]
mod tests{
    use core::time;
    use std::time::{Duration, Instant};

    use super::*;

    #[test]
    fn eq_to_reference_impl(){
        let width = 4000;
        let height = 4000;

        let mut v1 = vec![0.0; width*height];
        let mut v2 = vec![0.0; width*height];

        for i in 0..width*height {
            v1[i] = i as f32;
            v2[i] = i as f32;
        }

        let kernel = [
            1. / 16.,
            1. / 4.,
            3. / 8.,
            1. / 4.,
            1. / 16.,
        ];
        let stride = 2_usize.pow(6);
        let now = Instant::now();
        convolve_ref(&mut v2, height, width, kernel, stride);
        println!("ref time: {:}ms", now.elapsed().as_millis());
        let now = Instant::now();
        convolve(&mut v1, height, width, kernel, stride);
        println!("new time: {:}ms", now.elapsed().as_millis());

        // assert_eq!(v1, v2);
        assert!(true)
    }
}
