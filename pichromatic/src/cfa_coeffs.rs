use crate::pixel::{Image, SubPixel};
use rayon::prelude::*;

pub fn cfa_coeffs(image: &mut Image, wb_coeffs: [SubPixel; 4]){
        let [rv, gv, bv, _] = wb_coeffs;
        image.data.par_iter_mut().for_each(
            |p|{
                let [r, g, b] = p;
                *p = [*r*rv, *g*gv, *b*bv]
            }
        );
}
