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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_reconstruction() {
        let mut pixels = vec![[0.6, 0.4, 0.8]];
        let wb_coeffs = [1.0, 1.0, 1.0, 1.0];
        highlight_reconstruction(&mut pixels, wb_coeffs);
        
        let diff_r = (pixels[0][0] - 0.6).abs();
        let diff_g = (pixels[0][1] - 0.52).abs();
        let diff_b = (pixels[0][2] - 0.8).abs();
        
        assert!(diff_r < 1e-6);
        assert!(diff_g < 1e-6, "g is {}", pixels[0][1]);
        assert!(diff_b < 1e-6);
    }
}
