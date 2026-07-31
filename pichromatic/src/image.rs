use color::ColorSpaceTag;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::{DefaultHasher, Hash, Hasher};
use crate::{cfa::CFA, demosaic::Rect, pixel::SubPixel};

/// EXIF/DNG orientation as a pure clockwise rotation (flips map to [`Normal`] for now).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageOrientation {
    #[default]
    Normal,
    Rotate90,
    Rotate180,
    Rotate270,
}

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
    /// Sensor-relative EXIF orientation (pure rotations only).
    pub orientation: ImageOrientation,
    pub height: usize,
    pub width: usize,
}

fn hash_debug<T: Debug, H: Hasher>(value: &T, hasher: &mut H) {
    format!("{value:?}").hash(hasher);
}

fn hash_option_f32<H: Hasher>(value: Option<f32>, hasher: &mut H) {
    match value {
        Some(value) => {
            true.hash(hasher);
            value.to_bits().hash(hasher);
        }
        None => false.hash(hasher),
    }
}

fn hash_option_f32s<T: AsRef<[f32]>, H: Hasher>(value: Option<&T>, hasher: &mut H) {
    match value {
        Some(values) => {
            true.hash(hasher);
            let values = values.as_ref();
            values.len().hash(hasher);
            for value in values {
                value.to_bits().hash(hasher);
            }
        }
        None => false.hash(hasher),
    }
}

fn hash_option_f64s<T: AsRef<[f64]>, H: Hasher>(value: Option<&T>, hasher: &mut H) {
    match value {
        Some(values) => {
            true.hash(hasher);
            let values = values.as_ref();
            values.len().hash(hasher);
            for value in values {
                value.to_bits().hash(hasher);
            }
        }
        None => false.hash(hasher),
    }
}

fn hash_option_bytes<T: AsRef<[u8]>, H: Hasher>(value: Option<&T>, hasher: &mut H) {
    match value {
        Some(values) => {
            true.hash(hasher);
            values.as_ref().hash(hasher);
        }
        None => false.hash(hasher),
    }
}

fn option_f32_bits_equal(left: Option<f32>, right: Option<f32>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
        (None, None) => true,
        _ => false,
    }
}

fn option_f32s_bits_equal<T: AsRef<[f32]>>(
    left: Option<&T>,
    right: Option<&T>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            let left = left.as_ref();
            let right = right.as_ref();
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        }
        (None, None) => true,
        _ => false,
    }
}

fn option_f64s_bits_equal<T: AsRef<[f64]>>(
    left: Option<&T>,
    right: Option<&T>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            let left = left.as_ref();
            let right = right.as_ref();
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        }
        (None, None) => true,
        _ => false,
    }
}

fn option_bytes_equal<T: AsRef<[u8]>>(left: Option<&T>, right: Option<&T>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.as_ref() == right.as_ref(),
        (None, None) => true,
        _ => false,
    }
}

impl Hash for ImageMetadata {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        hash_debug(&self.crop_area, hasher);
        self.cfa.hash(hasher);
        hash_option_f32s(self.wb_coeffs.as_ref(), hasher);
        hash_option_f32s(self.calibration_matrix_d65.as_ref(), hasher);
        hash_option_f32(self.black_level, hasher);
        hash_option_f32(self.white_level, hasher);
        hash_debug(&self.color_space, hasher);
        hash_option_f32(self.baseline_exposure, hasher);
        hash_option_f32(self.shutter_seconds, hasher);
        hash_option_f32(self.f_number, hasher);
        hash_option_f32(self.iso, hasher);
        hash_option_f32(self.light_value, hasher);
        hash_option_bytes(self.opcode_list1.as_ref(), hasher);
        hash_option_bytes(self.opcode_list2.as_ref(), hasher);
        hash_option_bytes(self.opcode_list3.as_ref(), hasher);
        hash_debug(&self.dng_version, hasher);
        hash_debug(&self.dng_backward_version, hasher);
        self.unique_camera_model.hash(hasher);
        hash_option_f32s(self.color_matrix1.as_ref(), hasher);
        hash_option_f32s(self.color_matrix2.as_ref(), hasher);
        hash_option_f32s(self.camera_calibration1.as_ref(), hasher);
        hash_option_f32s(self.camera_calibration2.as_ref(), hasher);
        hash_option_f32s(self.analog_balance.as_ref(), hasher);
        hash_option_f32s(self.as_shot_neutral.as_ref(), hasher);
        hash_option_f32(self.linear_response_limit, hasher);
        hash_option_f32(self.shadow_scale, hasher);
        hash_option_f64s(self.noise_profile.as_ref(), hasher);
        self.profile_name.hash(hasher);
        hash_option_f32s(self.profile_tone_curve.as_ref(), hasher);
        hash_option_f32s(self.lens_info.as_ref(), hasher);
        self.camera_serial_number.hash(hasher);
        hash_debug(&self.orientation, hasher);
        self.height.hash(hasher);
        self.width.hash(hasher);
    }
}

impl ImageMetadata {
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        self.crop_area == other.crop_area
            && self.cfa == other.cfa
            && option_f32s_bits_equal(self.wb_coeffs.as_ref(), other.wb_coeffs.as_ref())
            && option_f32s_bits_equal(
                self.calibration_matrix_d65.as_ref(),
                other.calibration_matrix_d65.as_ref(),
            )
            && option_f32_bits_equal(self.black_level, other.black_level)
            && option_f32_bits_equal(self.white_level, other.white_level)
            && self.color_space == other.color_space
            && option_f32_bits_equal(self.baseline_exposure, other.baseline_exposure)
            && option_f32_bits_equal(self.shutter_seconds, other.shutter_seconds)
            && option_f32_bits_equal(self.f_number, other.f_number)
            && option_f32_bits_equal(self.iso, other.iso)
            && option_f32_bits_equal(self.light_value, other.light_value)
            && option_bytes_equal(self.opcode_list1.as_ref(), other.opcode_list1.as_ref())
            && option_bytes_equal(self.opcode_list2.as_ref(), other.opcode_list2.as_ref())
            && option_bytes_equal(self.opcode_list3.as_ref(), other.opcode_list3.as_ref())
            && self.dng_version == other.dng_version
            && self.dng_backward_version == other.dng_backward_version
            && self.unique_camera_model == other.unique_camera_model
            && option_f32s_bits_equal(self.color_matrix1.as_ref(), other.color_matrix1.as_ref())
            && option_f32s_bits_equal(self.color_matrix2.as_ref(), other.color_matrix2.as_ref())
            && option_f32s_bits_equal(
                self.camera_calibration1.as_ref(),
                other.camera_calibration1.as_ref(),
            )
            && option_f32s_bits_equal(
                self.camera_calibration2.as_ref(),
                other.camera_calibration2.as_ref(),
            )
            && option_f32s_bits_equal(self.analog_balance.as_ref(), other.analog_balance.as_ref())
            && option_f32s_bits_equal(
                self.as_shot_neutral.as_ref(),
                other.as_shot_neutral.as_ref(),
            )
            && option_f32_bits_equal(self.linear_response_limit, other.linear_response_limit)
            && option_f32_bits_equal(self.shadow_scale, other.shadow_scale)
            && option_f64s_bits_equal(self.noise_profile.as_ref(), other.noise_profile.as_ref())
            && self.profile_name == other.profile_name
            && option_f32s_bits_equal(
                self.profile_tone_curve.as_ref(),
                other.profile_tone_curve.as_ref(),
            )
            && option_f32s_bits_equal(self.lens_info.as_ref(), other.lens_info.as_ref())
            && self.camera_serial_number == other.camera_serial_number
            && self.orientation == other.orientation
            && self.height == other.height
            && self.width == other.width
    }

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
