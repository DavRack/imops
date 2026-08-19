//! Scan stage orchestration.

pub mod densitometry;
pub mod invert;

use crate::film::scan::densitometry::scan_to_acescg;
use crate::film::scan::invert::invert_negative;
use crate::film::stock::FilmStock;
use crate::film::types::DyePlanes;
use crate::pixel::ImageBuffer;

pub use densitometry::{
    apply_scanner_aperture_mtf, normalized_dmin_acescg, processed_dmin_acescg,
    scanner_aperture_sigma_px, scanner_calibration_acescg, ScannerCalibration,
};
pub use invert::mean_rgb;

pub enum ScanMode {
    NegativeLinear,
    /// Invert with processed-film Dmin and a neutral mid-gray scan.
    PositiveLinear {
        dmin: [f32; 3],
        mid: [f32; 3],
    },
}

pub fn scan(
    stock: &FilmStock,
    dyes: &DyePlanes,
    mode: ScanMode,
    pixel_pitch_um: f32,
) -> ImageBuffer {
    let mut buf = scan_to_acescg(stock, dyes, pixel_pitch_um);
    if let ScanMode::PositiveLinear { dmin, mid } = mode {
        invert_negative(&mut buf, mid, dmin);
    }
    buf
}
