use color::ColorSpaceTag::{self, Oklch};
use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;

pub fn lch(image_buffer: &mut ImageBuffer, source_cs: ColorSpaceTag, l_coef: SubPixel, c_coef: SubPixel, h_coef: SubPixel){
    image_buffer.par_iter_mut().for_each(
        |pixel| {
            let [l, c, h] = source_cs.convert(
                Oklch,
                *pixel
            );
            *pixel = Oklch.convert(
                source_cs,
                [l*l_coef, c*c_coef, h*h_coef]
            );
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lch_roundtrip_identity() {
        let mut pixels = vec![[0.4, 0.7, 0.2]];
        lch(&mut pixels, ColorSpaceTag::Srgb, 1.0, 1.0, 1.0);
        
        let diff_r = (pixels[0][0] - 0.4).abs();
        let diff_g = (pixels[0][1] - 0.7).abs();
        let diff_b = (pixels[0][2] - 0.2).abs();
        
        assert!(diff_r < 1e-4, "r diff is {}", diff_r);
        assert!(diff_g < 1e-4, "g diff is {}", diff_g);
        assert!(diff_b < 1e-4, "b diff is {}", diff_b);
    }
}
