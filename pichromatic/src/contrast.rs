use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;

pub fn contrast(image_buffer: &mut ImageBuffer, value: SubPixel){
    const MIDDLE_GRAY: SubPixel = 0.1845;
    image_buffer.par_iter_mut().for_each( |p|{
        *p = p.map(|x| MIDDLE_GRAY*(x/MIDDLE_GRAY).powf(value))
    });
}
