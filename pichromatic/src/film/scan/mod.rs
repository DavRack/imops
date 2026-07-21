//! Scan stage orchestration.

pub mod densitometry;
pub mod invert;

use crate::film::scan::densitometry::scan_to_acescg;
use crate::film::scan::invert::invert_negative;
use crate::film::stock::FilmStock;
use crate::film::types::DyePlanes;
use crate::pixel::ImageBuffer;

pub use densitometry::normalized_dmin_acescg;
pub use invert::mean_rgb;

pub enum ScanMode {
    NegativeLinear,
    /// Invert with stock film-base Dmin and mid-gray negative (gain calibration).
    PositiveLinear {
        dmin: [f32; 3],
        mid: [f32; 3],
    },
}

pub fn scan(stock: &FilmStock, dyes: &DyePlanes, mode: ScanMode) -> ImageBuffer {
    let mut buf = scan_to_acescg(stock, dyes);
    if let ScanMode::PositiveLinear { dmin, mid } = mode {
        invert_negative(&mut buf, mid, dmin);
    }
    buf
}
