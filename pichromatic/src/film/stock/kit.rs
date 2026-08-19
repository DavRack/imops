//! Frozen still color-negative construction kit.
//!
//! These are **class** values (Fresnel optics + Kodak C-41 T-grain patents),
//! not a Portra 400 bill of materials. Datasheet-fitted fields (ISO, layer S,
//! dyes, H&D `γ`/`d_max`/`capture_k`, scatter MTF) stay per stock.
//!
//! Citations:
//! - Acetate/air `R`: n ≈ 1.48 → Fresnel `((n−1)/(n+1))² ≈ 0.037`.
//! - AH OD band 0.3–1.0: US 6,228,569. Grey OD 0.6 is the mid of that band.
//! - T-grain plate thickness ~0.08–0.14 µm in US 5,322,766 examples.
//! - DIR coupler laydowns exist in patents; they do not map to a matrix, so
//!   DIR is off. Residual acutance is `adjacency_beta`.

use crate::film::spectrum::SpectralCurve;
use crate::film::stock::{AntihalationModel, EmulsionLayer, LayerKind};
use crate::film::units::Microns;

/// Acetate/air Fresnel reflectance. Weakly spectral; a scalar is enough.
pub const ACETATE_AIR_R: f64 = 0.037;

/// Thin T-grain plate thickness (µm). Not measured on E-4050.
pub const T_GRAIN_THICKNESS_UM: f32 = 0.10;

/// Grey antihalation optical density (pre-process, removed in C-41).
pub const AH_OD: f64 = 0.6;

pub const AH_THICKNESS_UM: f32 = 2.0;
pub const OVERCOAT_THICKNESS_UM: f32 = 1.0;

/// Support bounce model: Fresnel `R` and a placeholder wide-PSF σ (µm).
/// Do not look-tune `R` against another stock's halo; AH on/off is the lever.
pub fn backing(psf_halation_um: f32) -> AntihalationModel {
    AntihalationModel {
        reflectance: SpectralCurve::constant(ACETATE_AIR_R),
        psf_halation_um,
    }
}

/// Non-imaging layer: absorption-only. Dummy capture fields are unused.
pub fn inert(
    name: &'static str,
    depth_um: f32,
    thickness_um: f32,
    kind: LayerKind,
    absorption: SpectralCurve,
) -> EmulsionLayer {
    EmulsionLayer {
        name,
        depth_from_surface: Microns(depth_um),
        thickness: Microns(thickness_um),
        kind,
        spectral_sensitivity: Some(absorption),
        crystal_size: None,
        silver_halide_fraction: 0.0,
        coupler: None,
        gamma_contrast: 1.0,
        capture_k: 1.0,
        reciprocity_p: 1.0,
        is_reversal: false,
    }
}

pub fn overcoat(depth_um: f32) -> EmulsionLayer {
    inert(
        "overcoat",
        depth_um,
        OVERCOAT_THICKNESS_UM,
        LayerKind::Overcoat,
        SpectralCurve::constant(0.01),
    )
}

/// In-stack AH dye. `OD(λ) = AH_OD` (grey). Beer–Lambert uses `curve × t`.
pub fn antihalation_layer(depth_um: f32) -> EmulsionLayer {
    let coeff = AH_OD / AH_THICKNESS_UM as f64;
    inert(
        "antihalation",
        depth_um,
        AH_THICKNESS_UM,
        LayerKind::Antihalation,
        SpectralCurve::constant(coeff),
    )
}

pub fn yellow_filter(depth_um: f32, thickness_um: f32, absorption: SpectralCurve) -> EmulsionLayer {
    inert(
        "yellow_filter",
        depth_um,
        thickness_um,
        LayerKind::Filter,
        absorption,
    )
}
