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
pub use invert::{invert_negative, mean_rgb};
#[allow(deprecated)]
pub use invert::invert_negative_inverse_hd;

pub enum ScanMode {
    NegativeLinear,
    /// Invert with processed-film Dmin and a neutral mid-gray scan to unbounded linear HDR light.
    /// `inv_gamma` is the per-channel measured inverse contrast from [`ScannerCalibration`].
    PositiveLinear {
        dmin: [f32; 3],
        mid: [f32; 3],
        inv_gamma: [f32; 3],
    },
    /// Deprecated alias for PositiveLinear.
    #[deprecated(note = "Use PositiveLinear instead")]
    PositiveInverseHd {
        dmin: [f32; 3],
        mid: [f32; 3],
        inv_gamma: [f32; 3],
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
        #[allow(deprecated)]
        ScanMode::PositiveLinear { dmin, mid, inv_gamma }
        | ScanMode::PositiveInverseHd { dmin, mid, inv_gamma } => {
            invert_negative(&mut buf, mid, dmin, inv_gamma);
        }
    }
    buf
}
