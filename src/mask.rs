use dyn_clone::DynClone;
use rawler::RawImage;
use serde::{Deserialize, Serialize};
use crate::conditional_paralell::prelude::*;

use crate::imops::PipelineImage;
use crate::pixels::{BufferOps, PixelOps, SubPixel};

pub type MaskFn = Box<dyn Fn(&PipelineImage, &RawImage) -> Vec<SubPixel>>;

pub trait Mask: Sync + DynClone{
    fn create(&self, image: &PipelineImage, _raw_image: &RawImage) -> Vec<SubPixel>;
    fn get_name(&self) -> String;
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Constant {
    pub value: SubPixel,
}

impl Mask for Constant {
    fn create(&self, image: &PipelineImage, _raw_image: &RawImage) -> Vec<SubPixel>{
        let mask = vec![self.value; image.data.len()];
        mask
    }

    fn get_name(&self) -> String {
        "Constant".to_string()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct LuminanceGradient {
    pub r_bottom: SubPixel,
    pub r_top: SubPixel,
    pub l_bottom: SubPixel,
    pub l_top: SubPixel,
}

impl LuminanceGradient{
    fn mask(&self, x: SubPixel) -> SubPixel{
        let left_line = |x|{
            if (self.l_top - self.l_bottom) == 0.0 {
                1.0
            }else{
                (x - self.l_bottom)*(1.0/(self.l_top - self.l_bottom))
            }
        };
        let right_line = |x|{
            if (self.r_top - self.r_bottom) == 0.0 {
                1.0
            }else{
                (x - self.r_bottom)*(1.0/(self.r_top - self.r_bottom))
            }
        };
        let right_value: SubPixel = right_line(x);
        let left_value = left_line(x);

        right_value.min(left_value).clamp(0.0, 1.0)

    }
}

impl Mask for LuminanceGradient {
    fn get_name(&self) -> String {
        "LuminanceGradient".to_string()
    }
    fn create(&self, image: &PipelineImage, _raw_image: &RawImage) -> Vec<SubPixel>{

        let mut normalized_mask: Vec<SubPixel> = image.data.clone().par_iter().map(|pixel| pixel.luminance()).collect();
        let max_luminance =  normalized_mask.iter().fold(0.0, |acc: SubPixel, value: &SubPixel| acc.max(*value));
        normalized_mask.par_iter_mut().for_each(|sub_pixel|{
            let normalized_sub_pixel = *sub_pixel/max_luminance;
            *sub_pixel = self.mask(normalized_sub_pixel)
        });
        normalized_mask
    }
}

#[cfg(test)]
mod tests {
    use crate::pixels::Pixel;

    use super::*;

    fn get_raw_image() -> rawler::RawImage{
        let path: String = "./test_data/raw_sample.NEF".to_string();
        let raw_image = rawler::decode_file(path).unwrap();

        return raw_image
    }

    fn get_pipeline_image(data: Vec<Pixel>, height: usize, width: usize, ) -> PipelineImage{
        let max_raw_value = data.clone().max_luminance();
        let pipeline_image = PipelineImage {
            data,
            height,
            width,
            max_raw_value
        };
        return pipeline_image
    }

    #[test]
    fn test_constant(){
        let mask = Constant{
            value: 0.5
        };

        const HEIGHT: usize = 2;
        const WIDTH: usize = 2;

        let data: Vec<Pixel> = vec![
            [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        ];

        let pipeline_image = get_pipeline_image(data, HEIGHT, WIDTH);
        let raw_image = get_raw_image();

        let mask_value = mask.create(&pipeline_image, &raw_image);

        let expected_mask = vec![
            0.5, 0.5,
            0.5, 0.5,
        ];
        assert_eq!(mask_value, expected_mask)
    }
    #[test]
    fn test_luminance_gradient_simple(){
        let mask_cfg = LuminanceGradient{
            l_bottom: 0.0,
            l_top: 1.0,
            r_bottom: 1.0,
            r_top: 1.0,
        };
        assert_eq!(mask_cfg.mask(1.0), 1.0);
        assert_eq!(mask_cfg.mask(0.5), 0.5);
        assert_eq!(mask_cfg.mask(0.0), 0.0);
    }
    #[test]
    fn test_luminance_gradient_halve(){
        let mask_cfg = LuminanceGradient{
            l_bottom: 0.0,
            l_top: 0.25,
            r_top: 0.75,
            r_bottom: 1.0,
        };

        assert_eq!(mask_cfg.mask(0.0), 0.0);
        assert_eq!(mask_cfg.mask(0.125), 0.5);
        assert_eq!(mask_cfg.mask(0.25), 1.0);
        assert_eq!(mask_cfg.mask(0.5), 1.0);
        assert_eq!(mask_cfg.mask(0.875), 0.5);
        assert_eq!(mask_cfg.mask(1.0), 0.0);
    }
    #[test]
    fn test_luminance_gradient_constant(){
        let mask_cfg = LuminanceGradient{
            l_bottom: 0.0,
            l_top: 0.0,
            r_top: 1.0,
            r_bottom: 1.0,
        };

        assert_eq!(mask_cfg.mask(0.0),      1.0);
        assert_eq!(mask_cfg.mask(0.125),    1.0);
        assert_eq!(mask_cfg.mask(0.25),     1.0);
        assert_eq!(mask_cfg.mask(0.5),      1.0);
        assert_eq!(mask_cfg.mask(0.875),    1.0);
        assert_eq!(mask_cfg.mask(1.0),      1.0);
    }
}
