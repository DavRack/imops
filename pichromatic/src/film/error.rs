//! Film error types.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilmError {
    WrongColorSpace {
        expected: &'static str,
        got: String,
    },
    InvalidDimensions,
    InvalidStock(&'static str),
}

impl fmt::Display for FilmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilmError::WrongColorSpace { expected, got } => {
                write!(f, "wrong color space: expected {expected}, got {got}")
            }
            FilmError::InvalidDimensions => write!(f, "invalid image dimensions"),
            FilmError::InvalidStock(msg) => write!(f, "invalid film stock: {msg}"),
        }
    }
}

impl std::error::Error for FilmError {}
