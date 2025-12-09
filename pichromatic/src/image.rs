use crate::{cfa::CFA, demosaic::Rect, pixel::SubPixel};

pub struct ImageMetadata {
    pub height: usize,
    pub width: usize,
    pub black_level: Option<SubPixel>,
    pub cfa: Option<CFA>,
    pub crop_area: Option<Rect>,
    pub wb_coeffs: Option<[SubPixel; 4]>,
    pub white_level: Option<SubPixel>,
}
