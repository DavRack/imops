use convolve_image::kernel::{NonSeparableKernel, SeparableKernel};
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
        stride: usize,
        kernel_size: usize,
        kernel_index: isize,
        pixel_index: usize,
        max: usize,
    ) -> u32 {
        let kernel_size = kernel_size as isize;
        let kernel_padding = kernel_size / 2;

        let distance = kernel_index * stride as isize;

        let mut index = pixel_index as isize + distance;

        if index < 0 {
            index = -index;
        } else if index > max as isize - kernel_padding {
            let overshot_distance = index - max as isize + kernel_padding;
            index = max as isize - overshot_distance;
        }

        index as u32
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

        (0..self.len()).into_iter().for_each(|idx|{
            let x = idx % width;
            let y = (idx - x)/width;

            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    x,
                    width
                );

                pixel_sum += self[(y * width) + pixel_index as usize] * *value;
            }
            self[idx] = pixel_sum;
        });

        // for (y, x) in dimensions.into_iter() {
        //     let mut pixel_sum = 0.;

        //     for (kernel_index, value) in linear_kernel.iter().enumerate() {
        //         let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
        //         let pixel_index = Self::compute_pixel_index(
        //             stride,
        //             KERNEL_SIZE,
        //             relative_kernel_index,
        //             x,
        //             width
        //         );

        //         pixel_sum += self[[y, pixel_index as usize]] * *value;
        //     }

        //     self[[y, x]] = pixel_sum;
        // }
        (0..self.len()).into_iter().for_each(|idx|{
            let x = idx % width;
            let y = (idx - x )/width;

            let mut pixel_sum = 0.;

            for (kernel_index, value) in linear_kernel.iter().enumerate() {
                let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
                let pixel_index = Self::compute_pixel_index(
                    stride,
                    KERNEL_SIZE,
                    relative_kernel_index,
                    y,
                    height
                );

                pixel_sum += self[(pixel_index as usize * width) + x] * *value;
            }
            self[idx] = pixel_sum;
        });

        // for (y, x) in dimensions.into_iter() {
        //     let mut pixel_sum = 0.;

        //     for (kernel_index, value) in linear_kernel.iter().enumerate() {
        //         let relative_kernel_index = kernel_index as isize - (KERNEL_SIZE as isize / 2);
        //         let pixel_index = Self::compute_pixel_index(
        //             stride,
        //             KERNEL_SIZE,
        //             relative_kernel_index,
        //             y,
        //             height
        //         );

        //         pixel_sum += self[[pixel_index as usize, x]] * *value;
        //     }

        //     self[[y, x]] = pixel_sum;
        // }
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
