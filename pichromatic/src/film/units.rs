//! Physical newtypes for the film module.
//!
//! Fluence density is stored as photons per µm² (`PhotonsPerUm2`). Never mix
//! bare `f32` for quantities that carry units in public stock definitions.

use crate::film::constants::{PLANCK_H_J_S, SPEED_OF_LIGHT_M_S};

/// Length in micrometres (µm).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Microns(pub f32);

/// Length in millimetres (mm).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Millimeters(pub f32);

/// Time in seconds.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Seconds(pub f32);

/// Photographic f-number (relative aperture N).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct FNumber(pub f32);

/// ISO arithmetic speed.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct IsoSpeed(pub f32);

/// Photon fluence density (photons / µm²). `f64` for dynamic range in integrals.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PhotonsPerUm2(pub f64);

/// Base-10 optical density (OD). Transmittance T = 10^(−D).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct OpticalDensity(pub f32);

impl Millimeters {
    /// Convert millimetres to micrometres: 1 mm = 1000 µm.
    pub fn to_microns(self) -> Microns {
        Microns(self.0 * 1000.0)
    }
}

impl Microns {
    /// Convert micrometres to millimetres: 1000 µm = 1 mm.
    pub fn to_millimeters(self) -> Millimeters {
        Millimeters(self.0 / 1000.0)
    }
}

/// Photon energy E = hc / λ at wavelength `wavelength_nm` (nanometres), in joules.
///
/// Uses CODATA exact `h` and `c`. λ is converted nm → m.
pub fn photon_energy_joules(wavelength_nm: f64) -> f64 {
    let wavelength_m = wavelength_nm * 1e-9;
    PLANCK_H_J_S * SPEED_OF_LIGHT_M_S / wavelength_m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::constants::{PLANCK_H_J_S, SPEED_OF_LIGHT_M_S};

    #[test]
    fn microns_mm_roundtrip() {
        let mm = Millimeters(1.0);
        let um = mm.to_microns();
        assert_eq!(um.0, 1000.0);
        assert_eq!(um.to_millimeters().0, 1.0);

        let um2 = Microns(1000.0);
        assert_eq!(um2.to_millimeters().0, 1.0);
        assert_eq!(um2.to_millimeters().to_microns().0, 1000.0);
    }

    #[test]
    fn photon_energy_500nm() {
        // E = hc/λ at 500 nm from CODATA exact h,c.
        // Rounded reference in film-implementation.md: 3.9728917e-19 J.
        let e = photon_energy_joules(500.0);
        let expected = PLANCK_H_J_S * SPEED_OF_LIGHT_M_S / 500e-9;
        let rel = ((e - expected) / expected).abs();
        assert!(
            rel < 1e-12,
            "photon energy at 500 nm: got {e}, expected {expected}, rel err {rel}"
        );
        // Sanity vs rounded published figure (same leading digits).
        assert!((e - 3.9728917e-19).abs() / 3.9728917e-19 < 1e-7);
    }
}
