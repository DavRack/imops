//! Beer–Lambert absorption walk through the emulsion stack.
//!
//! Layers ordered top (light-incident) → bottom. For each λ:
//!   Φ_trans = Φ_in * exp(−σ(λ) * ρ_AgX * thickness)
//!   Φ_abs   = Φ_in − Φ_trans
//! Energy conservation: Φ_abs + Φ_trans == Φ_in (within float tolerance).

use crate::film::spectrum::WavelengthGrid;
use crate::film::stock::{EmulsionLayer, LayerKind};

/// Per-layer absorbed fluence spectrum (16 λ) at one spatial sample, plus whether
/// the layer produces latent image (emulsion only).
#[derive(Clone, Debug)]
pub struct LayerAbsorption {
    pub absorbed: [f64; 16],
    pub produces_latent: bool,
}

/// Walk the stack for a single pixel given incident fluence spectrum Φ_in(λ).
///
/// `sigma_scale` folds absolute cross-section calibration (stock-level). For
/// filter / overcoat / AH layers, absorption uses the spectral curve directly as
/// optical density per µm × thickness (relative units documented in stock files).
pub fn absorb_stack(
    layers: &[EmulsionLayer],
    incident: &[f64; 16],
    sigma_scale: f64,
) -> (Vec<LayerAbsorption>, [f64; 16]) {
    let grid = WavelengthGrid::mvp();
    let mut phi = *incident;
    let mut out = Vec::with_capacity(layers.len());

    for layer in layers {
        let mut absorbed = [0.0f64; 16];
        let produces_latent = layer.kind == LayerKind::Emulsion;

        match layer.kind {
            LayerKind::Emulsion => {
                let sens = layer
                    .spectral_sensitivity
                    .as_ref()
                    .expect("emulsion has sensitivity");
                let rho = layer.silver_halide_fraction as f64;
                let thickness = layer.thickness.0 as f64;
                for (i, _) in grid.wavelengths_nm.iter().enumerate() {
                    let sigma = sens.samples[i] * sigma_scale;
                    let od = sigma * rho * thickness; // neper-style for exp
                    let trans = (-od).exp();
                    let phi_t = phi[i] * trans;
                    absorbed[i] = phi[i] - phi_t;
                    phi[i] = phi_t;
                }
            }
            LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                // Absorption-only: spectral_sensitivity holds relative absorption coeff.
                if let Some(curve) = layer.spectral_sensitivity.as_ref() {
                    let thickness = layer.thickness.0 as f64;
                    for (i, _) in grid.wavelengths_nm.iter().enumerate() {
                        let od = curve.samples[i] * thickness; // 1/µm * µm
                        let trans = (-od).exp();
                        let phi_t = phi[i] * trans;
                        absorbed[i] = phi[i] - phi_t;
                        phi[i] = phi_t;
                    }
                }
            }
            LayerKind::Support => {
                // No absorption modeled here; antihalation reflectance handled in halation.
            }
        }

        out.push(LayerAbsorption {
            absorbed,
            produces_latent,
        });
    }

    (out, phi)
}

/// Mean spectral absorbed fluence for an emulsion layer over visible spectrum (400–700 nm, photons/µm² proxy).
///
/// Computes the trapezoidal integral `∫ Φ_abs(λ) dλ` over the MVP grid
/// (400–700 nm, Δλ = 20 nm), then **divides by the 300 nm span**. The result is
/// therefore a mean spectral fluence density over wavelength — **not** a raw
/// total photon count sum over all wavelengths.
///
/// [`crate::film::constants::ABSORPTION_SIGMA_SCALE_PER_UM`] was tuned against
/// this averaged quantity; do not drop the `/300` without re-deriving that scale.
/// For the un-divided integral `∫ Φ_abs(λ) dλ`, use [`total_absorbed_fluence`].
pub fn integrated_absorbed(absorbed: &[f64; 16]) -> f64 {
    mean_absorbed_fluence(absorbed)
}

/// Mean spectral absorbed fluence: `∫ Φ_abs(λ) dλ / 300.0`.
pub fn mean_absorbed_fluence(absorbed: &[f64; 16]) -> f64 {
    total_absorbed_fluence(absorbed) / 300.0
}

/// True trapezoidal integral `∫ Φ_abs(λ) dλ` over 400–700 nm (photons · nm / µm²).
pub fn total_absorbed_fluence(absorbed: &[f64; 16]) -> f64 {
    let dlambda = 20.0;
    let mut acc = 0.0;
    for i in 0..15 {
        acc += 0.5 * (absorbed[i] + absorbed[i + 1]) * dlambda;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::spectrum::SpectralCurve;
    use crate::film::stock::{EmulsionLayer, LayerKind};
    use crate::film::units::Microns;

    fn flat_incident(v: f64) -> [f64; 16] {
        [v; 16]
    }

    #[test]
    fn absorption_energy_conservation() {
        let sens = SpectralCurve::constant(0.1);
        let layer = EmulsionLayer {
            name: "test",
            depth_from_surface: Microns(0.0),
            thickness: Microns(10.0),
            kind: LayerKind::Emulsion,
            spectral_sensitivity: Some(sens),
            crystal_size: None,
            silver_halide_fraction: 0.2,
            coupler: None,
            gamma_contrast: 1.0,
            capture_k: 1.0,
            reciprocity_p: 1.0,
            is_reversal: false,
        };
        let incident = flat_incident(1.0);
        let (layers, _transmitted) = absorb_stack(&[layer], &incident, 1.0);
        for (i, &phi_in) in incident.iter().enumerate() {
            let abs = layers[0].absorbed[i];
            assert!(abs >= -1e-12 && abs <= phi_in + 1e-12);
            let residual = (phi_in - abs) + abs - phi_in;
            assert!(residual.abs() <= 1e-6 * phi_in.max(1.0));
        }
        let (layers2, transmitted) = absorb_stack(
            &[EmulsionLayer {
                name: "test2",
                depth_from_surface: Microns(0.0),
                thickness: Microns(10.0),
                kind: LayerKind::Emulsion,
                spectral_sensitivity: Some(SpectralCurve::constant(0.05)),
                crystal_size: None,
                silver_halide_fraction: 0.2,
                coupler: None,
                gamma_contrast: 1.0,
                capture_k: 1.0,
                reciprocity_p: 1.0,
                is_reversal: false,
            }],
            &incident,
            1.0,
        );
        for i in 0..16 {
            let phi_in = incident[i];
            let sum = layers2[0].absorbed[i] + transmitted[i];
            assert!(
                (sum - phi_in).abs() <= 1e-6 * phi_in.max(1.0),
                "λ[{i}]: in={phi_in} abs+trans={sum}"
            );
        }
    }

    #[test]
    fn filter_blocks_blue() {
        let grid = WavelengthGrid::mvp();
        let samples: Vec<f64> = grid
            .wavelengths_nm
            .iter()
            .map(|&l| {
                if l < 500.0 {
                    1.0
                } else {
                    0.01
                }
            })
            .collect();
        let filter = EmulsionLayer {
            name: "yellow_filter",
            depth_from_surface: Microns(0.0),
            thickness: Microns(5.0),
            kind: LayerKind::Filter,
            spectral_sensitivity: Some(SpectralCurve::new(grid.clone(), samples)),
            crystal_size: None,
            silver_halide_fraction: 0.0,
            coupler: None,
            gamma_contrast: 1.0,
            capture_k: 1.0,
            reciprocity_p: 1.0,
            is_reversal: false,
        };
        let below = EmulsionLayer {
            name: "below",
            depth_from_surface: Microns(5.0),
            thickness: Microns(5.0),
            kind: LayerKind::Emulsion,
            spectral_sensitivity: Some(SpectralCurve::constant(0.01)),
            crystal_size: None,
            silver_halide_fraction: 0.2,
            coupler: None,
            gamma_contrast: 1.0,
            capture_k: 1.0,
            reciprocity_p: 1.0,
            is_reversal: false,
        };
        let incident = flat_incident(1.0);
        let (layers, _) = absorb_stack(&[filter, below], &incident, 1.0);
        let abs_blue = layers[1].absorbed[1];
        let abs_red = layers[1].absorbed[12];
        assert!(
            abs_blue < 0.1 * abs_red,
            "blue fluence below filter should be <10% of red: blue={abs_blue} red={abs_red}"
        );
    }

    #[test]
    fn no_emulsion_no_latent() {
        let filter = EmulsionLayer {
            name: "filter_only",
            depth_from_surface: Microns(0.0),
            thickness: Microns(5.0),
            kind: LayerKind::Filter,
            spectral_sensitivity: Some(SpectralCurve::constant(0.5)),
            crystal_size: None,
            silver_halide_fraction: 0.0,
            coupler: None,
            gamma_contrast: 1.0,
            capture_k: 1.0,
            reciprocity_p: 1.0,
            is_reversal: false,
        };
        let incident = flat_incident(1.0);
        let (layers, _) = absorb_stack(&[filter], &incident, 1.0);
        assert!(!layers[0].produces_latent);
    }

    #[test]
    fn integrated_absorbed_is_band_average_not_sum() {
        // Flat Φ_abs(λ)=2 over [400,700] → ∫ = 2·300, band average = 2.
        let absorbed = [2.0f64; 16];
        let mean = integrated_absorbed(&absorbed);
        assert!(
            (mean - 2.0).abs() < 1e-9,
            "expected band average 2.0, got {mean} (would be ~600 if left as raw integral)"
        );
    }
}
