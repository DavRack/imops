use color::ColorSpaceTag;
use serde::{Deserialize, Serialize};
use std::hash::{DefaultHasher, Hash, Hasher};
use crate::{cfa::CFA, demosaic::Rect, pixel::SubPixel};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ImageMetadata {
    pub crop_area: Option<Rect>,
    pub cfa: Option<CFA>,
    pub wb_coeffs: Option<[SubPixel; 4]>,
    pub calibration_matrix_d65: Option<Vec<f32>>,
    pub black_level: Option<SubPixel>,
    pub white_level: Option<SubPixel>,
    pub color_space: Option<ColorSpaceTag>,
    pub baseline_exposure: Option<f32>,
    /// Capture shutter time in seconds (from EXIF ExposureTime), if known.
    pub shutter_seconds: Option<f32>,
    /// Capture f-number (from EXIF FNumber), if known.
    pub f_number: Option<f32>,
    /// Capture ISO speed (from EXIF ISOSpeedRatings / ISOSpeed), if known.
    pub iso: Option<f32>,
    /// Calculated or EXIF Light Value (LV / EV_100), if known.
    pub light_value: Option<f32>,
    pub opcode_list1: Option<Vec<u8>>,
    pub opcode_list2: Option<Vec<u8>>,
    pub opcode_list3: Option<Vec<u8>>,
    pub dng_version: Option<[u8; 4]>,
    pub dng_backward_version: Option<[u8; 4]>,
    pub unique_camera_model: Option<String>,
    pub color_matrix1: Option<Vec<f32>>,
    pub color_matrix2: Option<Vec<f32>>,
    pub camera_calibration1: Option<Vec<f32>>,
    pub camera_calibration2: Option<Vec<f32>>,
    pub analog_balance: Option<Vec<f32>>,
    pub as_shot_neutral: Option<Vec<f32>>,
    pub linear_response_limit: Option<f32>,
    pub shadow_scale: Option<f32>,
    pub noise_profile: Option<Vec<f64>>,
    pub profile_name: Option<String>,
    pub profile_tone_curve: Option<Vec<f32>>,
    pub lens_info: Option<Vec<f32>>,
    pub camera_serial_number: Option<String>,
    pub height: usize,
    pub width: usize,
}

impl ImageMetadata {
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", self).hash(&mut hasher);
        return hasher.finish()
    }

    /// Retrieve or compute EXIF Light Value (LV / EV_100).
    pub fn get_light_value(&self) -> Option<f32> {
        if let Some(lv) = self.light_value {
            Some(lv)
        } else if let (Some(t), Some(n), Some(iso)) = (self.shutter_seconds, self.f_number, self.iso) {
            if t > 0.0 && n > 0.0 && iso > 0.0 {
                Some(crate::film::exposure::radiance::calculate_light_value(t as f64, n as f64, iso as f64) as f32)
            } else {
                None
            }
        } else {
            None
        }
    }
}
