//! Runtime film types: exposure metadata, planar buffers, format.

use crate::film::units::Millimeters;

/// Capture exposure metadata (camera side). Not stored on `ImageMetadata`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureMeta {
    pub shutter_seconds: f32,
    pub f_number: f32,
    pub iso: f32,
}

impl ExposureMeta {
    /// Sunny-16 equivalent for ISO 200: t=1/200, N=8, S=200.
    /// Used by all film fixtures (film-implementation.md Appendix A).
    pub const REFERENCE_ISO200: Self = Self {
        shutter_seconds: 1.0 / 200.0,
        f_number: 8.0,
        iso: 200.0,
    };
}

/// Film format → usable frame width for pixel-pitch conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilmFormat {
    /// 35 mm still: 36.0 mm usable frame width.
    Film35mm,
    /// 6×6 medium format: 56.0 mm.
    Film6x6,
    /// 4×5 sheet: 101.6 mm.
    Film4x5,
}

impl FilmFormat {
    /// Usable frame width (mm). Pitch = width_mm * 1000 / image_width_px.
    pub fn width_mm(self) -> Millimeters {
        match self {
            FilmFormat::Film35mm => Millimeters(36.0),
            FilmFormat::Film6x6 => Millimeters(56.0),
            FilmFormat::Film4x5 => Millimeters(101.6),
        }
    }

    /// Pixel pitch in micrometres given image width in pixels.
    pub fn pixel_pitch_um(self, image_width_px: usize) -> f32 {
        assert!(image_width_px > 0);
        self.width_mm().0 * 1000.0 / image_width_px as f32
    }
}

/// Post-LUT developable fraction planes (one per emulsion layer that captures).
///
/// Each inner `Vec` is a full `width * height` plane, row-major.
#[derive(Clone, Debug)]
pub struct LatentPlanes {
    pub width: usize,
    pub height: usize,
    pub layers: Vec<Vec<f32>>,
}

/// Developed dye densities. Image dye and mask dye are separate so grain never
/// modulates the orange mask / residual colored coupler.
#[derive(Clone, Debug)]
pub struct DyePlanes {
    pub width: usize,
    pub height: usize,
    /// Per emulsion layer, base-10 optical density of image-forming dye.
    pub image_dye: Vec<Vec<f32>>,
    /// Per emulsion layer (or empty), mask / residual coupler density.
    pub mask_dye: Vec<Vec<f32>>,
}
