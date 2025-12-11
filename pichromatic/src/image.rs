use color::ColorSpaceTag;

use crate::{cfa::CFA, demosaic::Rect, pixel::SubPixel};

#[derive(Clone, Debug, Default)]
pub struct ImageMetadata {
    pub crop_area: Option<Rect>,
    pub cfa: Option<CFA>,
    pub wb_coeffs: Option<[SubPixel; 4]>,
    pub calibration_matrix_d65: Option<Vec<f32>>,
    pub black_level: Option<SubPixel>,
    pub white_level: Option<SubPixel>,
    pub color_space: Option<ColorSpaceTag>,
    pub height: usize,
    pub width: usize,
}
