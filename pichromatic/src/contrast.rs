use crate::pixel::{ImageBuffer, SubPixel, MIDDLE_GRAY};
use rayon::prelude::*;

pub fn contrast(image_buffer: &mut ImageBuffer, value: SubPixel){
    image_buffer.par_iter_mut().for_each( |p|{
        *p = p.map(|x| MIDDLE_GRAY*(x/MIDDLE_GRAY).powf(value))
    });
}
