use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;

pub fn exp(image_buffer: &mut ImageBuffer, ev: SubPixel){
    let value = (2.0 as SubPixel).powf(ev);
    image_buffer.par_iter_mut().for_each(
        |pixel|{
            *pixel = pixel.map(|sub_pixel| sub_pixel*value)
        }
    );
}
