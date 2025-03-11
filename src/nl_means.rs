use core::panic;
use std::{f32::consts::E, usize};

use rawler::pixarray::PixF32;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct Nlmeans<'a> {
    pub h: f32,
    pub tail_size: usize,
    pub image: &'a PixF32
}
impl <'a> Nlmeans<'a> {


    pub fn denoise_image(self) -> PixF32{
        let new_data: Vec<f32> = self.image.data.par_iter().enumerate().map(|(idx, _)|{
            self.filtered_pixel(idx)
        }).collect();

        let new_img = PixF32::new_with(new_data, self.image.width, self.image.height);
        return new_img;
    }

    fn get_tail(&self, center: usize) -> Vec<(usize, f32)>{
        let col = center % self.image.width;
        let row = (center - col)/self.image.width;

        let mut tail = vec![(0, 0.0); self.tail_size.pow(2)];
        for i in 0..self.tail_size{
            for j in 0..self.tail_size{

                let image_index = (row+j).clamp(0, self.image.height-1) * self.image.width + (col+i).clamp(0, self.image.width-1);
                tail[(i*self.tail_size)+j] = (
                    image_index,
                    self.image.data[image_index]
                );
            }
        }
        return tail
    }

    fn filtered_pixel(&self, pix_indx: usize) -> f32 {
        // we asume the pixel we want to denoise is the middle one
        // image_region should be a square, odd sided 2x2 matrix
        let image_region = self.get_tail(pix_indx);
        // let matrix_side = (image_region.len() as f32).sqrt();
        // if matrix_side.fract() != 0.0 {
        //     panic!("not an squared image region")
        // }

        // let matrix_side = matrix_side as usize;
        // if matrix_side % 2 == 0 {
        //     panic!("image region should be odd")
        // }
        // as of now only 3x3 matrix is supported
        // if matrix_side != 3 {
        //     panic!("only 3x3 matrix is supported")
        // }

        let target_pixel_index = image_region.len() / 2;
        let p = image_region[target_pixel_index];

        let cp_value = self.cp(self.get_tail(p.0), self.h, &image_region);

        return image_region.iter().map(|q|q.1*&self.gaussian_weighting(
            &self.get_tail(p.0),
            &self.get_tail(q.0),
            self.h)).sum::<f32>()/cp_value
    }
    // C(p)
    fn cp(&self, p:Vec<(usize, f32)>, h: f32, image_region: &Vec<(usize, f32)>)-> f32 {
        return image_region.iter().map(|q| {
            self.gaussian_weighting(&p, &self.get_tail(q.0), h)
        }).sum()
    }


    // f(p,q) = Gaussian weighting function 
    fn gaussian_weighting(self, p: &Vec<(usize, f32)>, q: &Vec<(usize, f32)>, h: f32) -> f32 {
        let bq = self.average_surround_pixels(q);
        let bp = self.average_surround_pixels(p);
        return E.powf(-((bq.powi(2)-bp.powi(2)).abs())/h.powi(2))
    }

    // B(p) function
    fn average_surround_pixels(self, surrounding_pixels: &Vec<(usize, f32)>) -> f32{
        let target_pixel_index = surrounding_pixels.len()/2; 
        let sum: f32 = [
            &surrounding_pixels[0..target_pixel_index],
            &surrounding_pixels[target_pixel_index+1..]
        ].concat().iter().map(|(_, value)| value).sum();
        return (sum)/(surrounding_pixels.len() as f32 - 1.0)
    }
}

// #[cfg(test)]
// mod tests {
//     // Note this useful idiom: importing names from outer (for mod tests) scope.
//     use super::*;

//     #[test]
//     #[should_panic(expected = "not an squared image region")]
//     fn test_not_squared_matrix() {
//         filtered_pixel(&vec![0.0, 0.0, 0.0], 1.0);
//     }

//     #[test]
//     #[should_panic(expected = "image region should be odd")]
//     fn test_not_odd_matrix() {
//         filtered_pixel(&vec![0.0, 0.0, 0.0, 0.0], 1.0);
//     }

//     #[test]
//     fn test_average_surround_pixels() {
//         assert!(
//             average_surround_pixels(&vec![
//                 1.0, 2.0, 3.0,
//                 4.0, 5.0, 6.0,
//                 7.0, 8.0, 9.0,
//             ]) == 5.0
//         );
//     }
// }
    //
