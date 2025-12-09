use color::{ColorSpaceTag};

use crate::{
    cfa_coeffs::cfa_coeffs,
    contrast::contrast,
    cst::{camera_cst, cst},
    demosaic::{self, DemosaicAlgorithm},
    exp::exp,
    highlight_reconstruction::highlight_reconstruction,
    image::ImageMetadata,
    lch::lch,
    tone_map::sigmoid
};

pub const CHANNELS_PER_PIXEL: usize = 3;

pub type SubPixel = f32;
pub type Pixel = [SubPixel; CHANNELS_PER_PIXEL];
pub type ImageBuffer = Vec<Pixel>;
pub struct Image {
    pub data: ImageBuffer,
    pub height: usize,
    pub width: usize,
    pub color_space: Option<ColorSpaceTag>
}

// pub const R_RELATIVE_LUMINANCE: SubPixel = 0.2126;
// pub const G_RELATIVE_LUMINANCE: SubPixel = 0.7152;
// pub const B_RELATIVE_LUMINANCE: SubPixel = 0.0722;
// for acescg
pub const R_RELATIVE_LUMINANCE: SubPixel = 0.2722287168;
pub const G_RELATIVE_LUMINANCE: SubPixel = 0.6740817658;
pub const B_RELATIVE_LUMINANCE: SubPixel = 0.0536895174;
pub const MIDDLE_GRAY: SubPixel = 0.185;

pub trait PixelOps {
    fn luminance(self) -> SubPixel;
    fn norm(self) -> SubPixel;
}
impl PixelOps for Pixel{
    fn luminance(self) -> SubPixel{
        let [r, g, b] = self;
        let y = R_RELATIVE_LUMINANCE*r + G_RELATIVE_LUMINANCE*g + B_RELATIVE_LUMINANCE*b;
        return y
    }
    fn norm(self) -> SubPixel{
        let [r, g, b] = self;
        let y = r.max(g).max(b);
        return y
    }
}

impl Image {
    pub fn lch(&mut self, lch_coefs: [SubPixel; 3]) -> &mut Image{
        let [l_coef, c_coef, h_coef] = lch_coefs;
        lch(
            &mut self.data,
            self.color_space.unwrap(),
            l_coef, c_coef, h_coef
        );
        return self
    }

    pub fn exp(&mut self, ev: SubPixel) -> &mut Image{
        exp( &mut self.data, ev);
        return self
    }

    pub fn cst(&mut self, target_cs: ColorSpaceTag) -> &mut Image{
        cst(
            &mut self.data,
            self.color_space.expect("Image needs to have a color space to perform a CST"),
            target_cs,
        );
        self.color_space = Some(target_cs);
        return self
    }

    pub fn camera_cst(&mut self, target_cs: ColorSpaceTag, camera_color_matrix: Vec<f32>) -> &mut Image{
        camera_cst(&mut self.data, target_cs, camera_color_matrix);
        self.color_space = Some(target_cs);
        return self
    }

    pub fn tone_map(&mut self) -> &mut Image{
        sigmoid(
            &mut self.data
        );
        return self
    }
    
    pub fn cfa_coeffs(&mut self, wb_coeffs: [SubPixel; 4]) -> &mut Image{
        cfa_coeffs(self, wb_coeffs);
        return self
    }

    pub fn contrast(&mut self, value: SubPixel) -> &mut Image{
        contrast(&mut self.data, value);
        return self
    }
    pub fn highlight_reconstruction(&mut self, wb_coeffs: [SubPixel; 4]) -> &mut Image{
        highlight_reconstruction(&mut self.data, wb_coeffs);
        return self
    }

    pub fn demosaic(
        raw_image_data: &[u16],
        image_metadata: ImageMetadata,
        demosaic_algorithm: impl DemosaicAlgorithm
    ) -> Image{
        return demosaic::demosaic(
            raw_image_data,
            image_metadata,
            demosaic_algorithm
        )
    }
}
