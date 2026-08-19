//! Film error types.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilmError {
    WrongColorSpace {
        expected: &'static str,
        got: String,
    },
    InvalidDimensions,
    InvalidRenderWidth,
    /// Custom physical render width is not implemented on the GPU path yet.
    UnsupportedRenderWidth,
    ScannerCalibrationDidNotConverge,
    InvalidStock(&'static str),
}

impl fmt::Display for FilmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilmError::WrongColorSpace { expected, got } => {
                write!(f, "wrong color space: expected {expected}, got {got}")
            }
            FilmError::InvalidDimensions => write!(f, "invalid image dimensions"),
            FilmError::InvalidRenderWidth => {
                write!(
                    f,
                    "invalid render width: expected a finite value greater than zero mm"
                )
            }
            FilmError::UnsupportedRenderWidth => {
                write!(
                    f,
                    "custom render_width_mm is not supported on the GPU film path; use CPU"
                )
            }
            FilmError::ScannerCalibrationDidNotConverge => {
                write!(
                    f,
                    "scanner calibration did not converge within its sample cap"
                )
            }
            FilmError::InvalidStock(msg) => write!(f, "invalid film stock: {msg}"),
        }
    }
}

impl std::error::Error for FilmError {}
