use convolve_image::kernel::{NonSeparableKernel, SeparableKernel};

use crate::helpers;
use rayon::prelude::*;

#[derive(Copy, Clone)]
pub enum Kernel {
    LinearInterpolationKernel,
    LowScaleKernel,
    B3SplineKernel,
}

pub trait Convolution {
    #[inline]
    fn compute_pixel_index(
        kernel_padding: isize,
        distance: isize,
        pixel_index: usize,
        max: usize,
    ) -> usize {

        let mut index = pixel_index as isize + distance;

        index = index.abs();
        let bound = max as isize - kernel_padding;
        if index > bound {
            index = max as isize - index +  bound;
        }

        index as usize
    }

    fn convolve<const KERNEL_SIZE: usize>(
        &mut self,
        height: usize,
        width: usize,
        kernel: SeparableKernel<KERNEL_SIZE>,
        stride: usize,
    );
}
impl Convolution for Vec<f32> {
    fn convolve<const KERNEL_SIZE: usize>(&mut self, height: usize, width: usize, kernel: SeparableKernel<KERNEL_SIZE>, stride: usize) {
        let linear_kernel = kernel.values();

        (0..width*height).into_iter().for_each(|idx|{
            let x = idx % width;
            let y = (idx - x) / width;
            let pixel_sum = linear_kernel.into_iter().enumerate().fold(0.0, |acc, (kernel_index, value)|{
                let kernel_side = KERNEL_SIZE as isize / 2;
                let relative_kernel_index = kernel_index as isize - kernel_side;

                let distance = relative_kernel_index * stride as isize;

                let pixel_index_x = Self::compute_pixel_index(
                    kernel_side,
                    distance,
                    x,
                    width
                );

                let pixel_index_y = Self::compute_pixel_index(
                    kernel_side,
                    distance,
                    y,
                    height
                );

                acc + ( (self[(y * width) + pixel_index_x] + self[(pixel_index_y * width) + x]) * value)
            });

            self[idx] = pixel_sum/2.;
        });
    }
}

#[derive(Copy, Clone)]
pub(crate) struct LinearInterpolationKernel(SeparableKernel<3>);

impl LinearInterpolationKernel {
    pub(crate) fn new() -> Self {
        Self(SeparableKernel::new([1. / 4., 1. / 2., 1. / 4.]))
    }
}

#[derive(Copy, Clone)]
pub(crate) struct B3SplineKernel(SeparableKernel<5>);

impl B3SplineKernel {
    pub(crate) fn new() -> Self {
        Self(SeparableKernel::new([
            1. / 16.,
            1. / 4.,
            3. / 8.,
            1. / 4.,
            1. / 16.,
        ]))
    }
}

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

pub trait WaveletDecompose {
    fn wavelet_decompose(&mut self, height:usize, width: usize, kernel: Kernel, pixel_scale: usize) -> Vec<f32>;
}

impl WaveletDecompose for Vec<f32> {
    fn wavelet_decompose(&mut self, height: usize, width: usize, kernel: Kernel, pixel_scale: usize) -> Vec<f32> {
        let stride = 2_usize.pow(
            u32::try_from(pixel_scale)
                .unwrap_or_else(|_| panic!("pixel_scale cannot be larger than {}", u32::MAX)),
        );
        let mut current_data = self.clone();
        match kernel {
            Kernel::LinearInterpolationKernel => {
                current_data.convolve(height, width, LinearInterpolationKernel::new().into(), stride);
            }
            Kernel::LowScaleKernel => {
                unimplemented!("Low scale is not a separable kernel");
            }
            Kernel::B3SplineKernel => current_data.convolve(height, width, B3SplineKernel::new().into(), stride),
        }

        let final_data = self.clone().iter().zip(&current_data).map(|(v1, v2)| v1 - v2).collect();
        *self = current_data;

        final_data
    }
}
