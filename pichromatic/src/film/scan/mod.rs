//! Scan stage orchestration.

pub mod densitometry;
pub mod invert;

use crate::film::scan::densitometry::scan_to_acescg;
use crate::film::stock::FilmStock;
use crate::film::types::DyePlanes;
use crate::pixel::ImageBuffer;

pub use densitometry::{
    apply_scanner_aperture_mtf, normalized_dmin_acescg, processed_dmin_acescg,
    scanner_aperture_sigma_px, scanner_calibration_acescg, ScannerCalibration,
};
pub use invert::{apply_scanner_scurve, invert_negative, invert_negative_inverse_hd, mean_rgb};

pub enum ScanMode {
    NegativeLinear,
    /// Invert with processed-film Dmin and a neutral mid-gray scan.
    PositiveLinear {
        dmin: [f32; 3],
        mid: [f32; 3],
    },
    /// Scene-referred HDR inverse-H&D invert with processed-film Dmin and a neutral mid-gray scan.
    PositiveInverseHd {
        dmin: [f32; 3],
        mid: [f32; 3],
        scanner_s_curve: f32,
    },
}

pub fn scan(
    stock: &FilmStock,
    dyes: &DyePlanes,
    mode: ScanMode,
    pixel_pitch_um: f32,
) -> ImageBuffer {
    let mut buf = scan_to_acescg(stock, dyes, pixel_pitch_um);
    match mode {
        ScanMode::NegativeLinear => {}
        ScanMode::PositiveLinear { dmin, mid } => {
            invert_negative(&mut buf, mid, dmin);
        }
        ScanMode::PositiveInverseHd {
            dmin,
            mid,
            scanner_s_curve,
        } => {
            invert_negative_inverse_hd(&mut buf, mid, dmin, scanner_s_curve);
        }
    }
    buf
}
