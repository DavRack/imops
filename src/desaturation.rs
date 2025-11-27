use std::f32::consts::{PI, E};

use crate::pixels::Pixel;

fn normal_distribution(a: f32, b: f32, x: f32) -> f32 {
    (1.0/(b*(2.0*PI).powf(0.5)))*(E.powf((-0.5)*((x-a)/b).powi(2)))
}
fn normalized_normal_distribution(a: f32, b:f32, x: f32) -> f32 {
    normal_distribution(a, b, x)/normal_distribution(a, b, a)
}

pub fn desaturate_pixel(pixel: Pixel, p: f32) -> Pixel {
    let r_factor = normalized_normal_distribution(0.0, p, 1.0);
    let g_factor = normalized_normal_distribution(0.0, p, 1.0);
    let b_factor = normalized_normal_distribution(0.0, p, 1.0);

    let [r, g, b] = pixel;
    
    [
        r+r*(g_factor+b_factor),
        g+g*(r_factor+b_factor),
        b+b*(r_factor+g_factor),

    ]
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case1(){
        let result = desaturate_pixel([1.0, 1.0, 1.0], 0.5);
        let expected = [1.27067056647, 1.27067056647, 1.27067056647];

        assert_eq!(result, expected)
    }
    #[test]
    fn case2(){
        let result = desaturate_pixel([1.0, 1.0, 1.0], 0.7);
        let expected = [1.7208955772, 1.7208955772, 1.7208955772];

        assert_eq!(result, expected)
    }
    #[test]
    fn case3(){
        let result = desaturate_pixel([1.0, 1.0, 1.0], 0.00000001);
        let expected = [1.0, 1.0, 1.0];

        assert_eq!(result, expected)
    }
}
