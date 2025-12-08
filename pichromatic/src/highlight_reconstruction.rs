use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;


pub fn highlight_reconstruction(image_buffer: &mut ImageBuffer, wb_coeffs: [SubPixel; 4]){
let [_, clip_g, _, _] = wb_coeffs;
        image_buffer.par_iter_mut().for_each(|pixel|{
            let [r, g, b] = *pixel;
            let factor = g/clip_g;
            let reconstructed_g = ((1.0-factor)*g) + (factor*(r+b)*(1.0/2.0));
            pixel[1] = reconstructed_g;
        });
}
