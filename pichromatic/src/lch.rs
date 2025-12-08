use color::ColorSpaceTag::{self, Oklch};
use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;

pub fn lch(image_buffer: &mut ImageBuffer, source_cs: ColorSpaceTag, l_coef: SubPixel, c_coef: SubPixel, h_coef: SubPixel){
    image_buffer.par_iter_mut().for_each(
        |pixel| {
            let [l, c, h] = source_cs.convert(Oklch, *pixel);
            *pixel = Oklch.convert(
                Oklch,
                [l*l_coef, c*c_coef, h*h_coef]
            );
        }
    );
}
