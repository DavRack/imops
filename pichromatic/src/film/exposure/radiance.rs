//! Digital ACEScg → radiometric / photon-fluence scaling.
//!
//! Two stages:
//! 1. **Camera relative → absolute luminance** (pipeline `BaselineExposureCompensation`):
//!    `L = K * v * N² / (t * S)` with K = 12.5 (ISO reflected-light meter constant).
//! 2. **Absolute luminance → film fluence** (film expose):
//!    `Φ ∝ L` — shutter is already folded into `L` (§5.3). Emulsion speed is the
//!    stock absorption / `capture_k` calibration, not a second ISO factor.
//!
//! Provenance: K from ISO reflected-light metering practice (film-implementation.md §5.3 / §8).

use crate::film::constants::METER_CONSTANT_K;
use crate::film::types::ExposureMeta;
use crate::film::units::IsoSpeed;

/// Camera-relative scene-linear value → absolute luminance proxy.
///
/// `L = K * v * N² / (t * S)`
pub fn relative_to_absolute_luminance(
    v: f64,
    shutter_seconds: f64,
    f_number: f64,
    iso: f64,
) -> f64 {
    METER_CONSTANT_K * v * (f_number * f_number) / (shutter_seconds * iso)
}

/// Multiplicative scale that converts a whole buffer from camera-relative to absolute luminance.
pub fn absolute_luminance_gain(shutter_seconds: f64, f_number: f64, iso: f64) -> f64 {
    relative_to_absolute_luminance(1.0, shutter_seconds, f_number, iso)
}

/// Compute EXIF Light Value (LV / EV_100) from shutter speed (seconds), f-number, and ISO.
///
/// `LV = log2( (N² / t) * (100 / S) )`
pub fn calculate_light_value(shutter_seconds: f64, f_number: f64, iso: f64) -> f64 {
    ((f_number * f_number / shutter_seconds) * (100.0 / iso)).log2()
}

/// Multiplicative scale that converts camera-relative to absolute luminance using Light Value (LV / EV_100).
///
/// `gain = 2^(LV - 3.0)` (where K = 12.5)
pub fn absolute_luminance_gain_from_lv(lv: f64) -> f64 {
    2.0_f64.powf(lv - 3.0)
}

/// Film fluence scale from absolute luminance (`Φ ∝ L`; §5.3).
pub fn film_exposure_scale(l_abs: f64) -> f64 {
    l_abs.max(0.0)
}

/// Reciprocity law failure efficiency factor η(t, p).
///
/// For long exposures t > 1.0s (LIRF), reciprocity law failure reduces quantum efficiency:
/// η = (t / 1.0)^(p - 1.0) where p ∈ (0.5, 1.0] is Schwarzschild exponent.
/// For very short exposures t < 1e-3s (HIRF), efficiency also drops.
/// Note: this relies only on total shutter time, treating the intensity profile as uniform.
pub fn reciprocity_factor(shutter_seconds: f64, p: f64) -> f64 {
    if p >= 0.9999 {
        return 1.0;
    }
    if shutter_seconds > 1.0 {
        // LIRF (Low Intensity Reciprocity Failure)
        shutter_seconds.powf(p - 1.0)
    } else if shutter_seconds < 0.001 {
        // HIRF (High Intensity Reciprocity Failure) proxy.
        // Assuming symmetric exponent for the short end as a simple MVP approximation.
        (shutter_seconds / 0.001).powf(1.0 - p)
    } else {
        1.0
    }
}

/// Sunny-16 exposure metadata for a given box ISO: t = 1/ISO, N = 8.
/// Used by Baseline/fixtures to form absolute luminance; not a Film module param.
pub fn sunny16_exposure(box_iso: IsoSpeed) -> ExposureMeta {
    let s = box_iso.0.max(1.0);
    ExposureMeta {
        shutter_seconds: 1.0 / s,
        f_number: 8.0,
        iso: s,
    }
}

/// Convert absolute-luminance scale to photon fluence density at `wavelength_nm`.
///
/// Uses relative photon energy E_rel = λ_ref / λ (λ_ref = 550 nm).
pub fn photon_fluence_density(scale: f64, wavelength_nm: f64) -> f64 {
    let e_rel = 550.0 / wavelength_nm;
    scale / e_rel
}

/// Absolute luminance → photon fluence density at λ.
pub fn absolute_to_fluence(l_abs: f64, wavelength_nm: f64) -> f64 {
    photon_fluence_density(film_exposure_scale(l_abs), wavelength_nm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposure_doubling_absolute_path() {
        // Doubling shutter in the *camera→absolute* step halves L (less light needed
        // for same relative v means darker scene). Film then sees half the luminance.
        let v = 0.185_f64;
        let l1 = relative_to_absolute_luminance(v, 1.0 / 200.0, 8.0, 200.0);
        let l2 = relative_to_absolute_luminance(v, 2.0 / 200.0, 8.0, 200.0);
        assert!((l2 / l1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn fnumber_inverse_square_absolute_path() {
        let v = 0.185_f64;
        let l1 = relative_to_absolute_luminance(v, 1.0 / 200.0, 8.0, 200.0);
        let l2 = relative_to_absolute_luminance(v, 1.0 / 200.0, 16.0, 200.0);
        assert!((l2 / l1 - 4.0).abs() < 1e-10);
    }

    #[test]
    fn iso_inverse_absolute_path() {
        let v = 0.185_f64;
        let l1 = relative_to_absolute_luminance(v, 1.0 / 200.0, 8.0, 200.0);
        let l2 = relative_to_absolute_luminance(v, 1.0 / 200.0, 8.0, 400.0);
        assert!((l2 / l1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn film_fluence_tracks_absolute_luminance() {
        assert!((film_exposure_scale(100.0) - 100.0).abs() < 1e-12);
        assert!((film_exposure_scale(200.0) / film_exposure_scale(100.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn midgray_absolute_l_is_k_v_n2_over_ts() {
        // Relative mid-gray correctly exposed at sunny-16 ISO 200 → L = K·v·N²/(t·S).
        let v = 0.185_f64;
        let cam = sunny16_exposure(IsoSpeed(200.0));
        let l = relative_to_absolute_luminance(
            v,
            cam.shutter_seconds as f64,
            cam.f_number as f64,
            cam.iso as f64,
        );
        let expected = METER_CONSTANT_K * v * 64.0; // N=8, t·S=1
        assert!(
            (l - expected).abs() < 1e-5 * expected,
            "L={l} expected={expected}"
        );
        assert!((film_exposure_scale(l) - l).abs() < 1e-12);
    }

    #[test]
    fn light_value_parity_test() {
        let t = 1.0 / 125.0;
        let n = 2.8;
        let iso = 400.0;
        let gain1 = absolute_luminance_gain(t, n, iso);
        let lv = calculate_light_value(t, n, iso);
        let gain2 = absolute_luminance_gain_from_lv(lv);
        assert!((gain1 - gain2).abs() < 1e-10, "gain1={gain1}, gain2={gain2}, lv={lv}");
        assert!((gain1 - 30.625).abs() < 1e-10);
    }
}
