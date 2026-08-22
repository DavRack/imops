//! Beer–Lambert absorption walk through the emulsion stack.
//!
//! Layers ordered top (light-incident) → bottom. For each λ:
//!   Φ_trans = Φ_in * exp(−σ(λ) * ρ_AgX * thickness)
//!   Φ_abs   = Φ_in − Φ_trans
//! Energy conservation: Φ_abs + Φ_trans == Φ_in (within float tolerance).
//!
//! The walk runs in f32 (WGSL has no core f64), matching the GPU `EXPOSE`
//! shader. Transmittance is precomputed in f64 and rounded to f32 exactly as
//! `gpu::bake_consts` uploads it, so CPU and GPU share the value.

use crate::film::stock::{EmulsionLayer, LayerKind};

/// Per-layer absorbed fluence spectrum (16 λ) at one spatial sample, plus whether
/// the layer produces latent image (emulsion only).
#[derive(Clone, Debug)]
pub struct LayerAbsorption {
    pub absorbed: [f32; 16],
    pub produces_latent: bool,
}

/// Walk the stack for a single pixel given incident fluence spectrum Φ_in(λ).
///
/// `sigma_scale` folds absolute cross-section calibration (stock-level). For
/// filter / overcoat / AH layers, absorption uses the spectral curve directly as
/// optical density per µm × thickness (relative units documented in stock files).
pub fn absorb_stack(
    layers: &[EmulsionLayer],
    incident: &[f32; 16],
    sigma_scale: f64,
) -> (Vec<LayerAbsorption>, [f32; 16]) {
    let mut phi = *incident;
    let mut out = Vec::with_capacity(layers.len());

    for layer in layers {
        let mut absorbed = [0.0f32; 16];
        let produces_latent = layer.kind == LayerKind::Emulsion;

        match layer.kind {
            LayerKind::Emulsion => {
                let sens = layer
                    .spectral_sensitivity
                    .as_ref()
                    .expect("emulsion has sensitivity");
                let rho = layer.silver_halide_fraction as f64;
                let thickness = layer.thickness.0 as f64;
                for i in 0..16 {
                    let od = sens.samples[i] * sigma_scale * rho * thickness;
                    let trans = (-od).exp() as f32;
                    let phi_t = phi[i] * trans;
                    absorbed[i] = phi[i] - phi_t;
                    phi[i] = phi_t;
                }
            }
            LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                // Absorption-only: spectral_sensitivity holds relative absorption coeff.
                if let Some(curve) = layer.spectral_sensitivity.as_ref() {
                    let thickness = layer.thickness.0 as f64;
                    for i in 0..16 {
                        let od = curve.samples[i] * thickness; // 1/µm * µm
                        let trans = (-od).exp() as f32;
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

/// Upward Beer–Lambert absorption walk through the emulsion stack (bottom → top).
///
/// Light reflected from the backing travels upward through the layers in reverse
/// order. For each layer, $\Phi_{\text{upward}}$ is attenuated as it passes through,
/// and absorbed fluence is recorded.
/// Returns per-layer upward absorption in the original layer order.
pub fn absorb_stack_upward(
    layers: &[EmulsionLayer],
    incident_upward: &[f32; 16],
    sigma_scale: f64,
) -> Vec<LayerAbsorption> {
    let mut phi = *incident_upward;
    let mut out = vec![
        LayerAbsorption {
            absorbed: [0.0; 16],
            produces_latent: false,
        };
        layers.len()
    ];

    for (idx, layer) in layers.iter().enumerate().rev() {
        let mut absorbed = [0.0f32; 16];
        let produces_latent = layer.kind == LayerKind::Emulsion;

        match layer.kind {
            LayerKind::Emulsion => {
                let sens = layer
                    .spectral_sensitivity
                    .as_ref()
                    .expect("emulsion has sensitivity");
                let rho = layer.silver_halide_fraction as f64;
                let thickness = layer.thickness.0 as f64;
                for i in 0..16 {
                    let od = sens.samples[i] * sigma_scale * rho * thickness;
                    let trans = (-od).exp() as f32;
                    let phi_t = phi[i] * trans;
                    absorbed[i] = phi[i] - phi_t;
                    phi[i] = phi_t;
                }
            }
            LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                if let Some(curve) = layer.spectral_sensitivity.as_ref() {
                    let thickness = layer.thickness.0 as f64;
                    for i in 0..16 {
                        let od = curve.samples[i] * thickness;
                        let trans = (-od).exp() as f32;
                        let phi_t = phi[i] * trans;
                        absorbed[i] = phi[i] - phi_t;
                        phi[i] = phi_t;
                    }
                }
            }
            LayerKind::Support => {}
        }

        out[idx] = LayerAbsorption {
            absorbed,
            produces_latent,
        };
    }

    out
}

/// Per-layer spectral transmittance exp(−OD) precomputed once per stock.
/// Bit-identical to the values computed inside `absorb_stack` /
/// `absorb_stack_upward`: same f64 OD expression, exp, then f32 round.
pub(crate) fn layer_transmittances(layers: &[EmulsionLayer], sigma_scale: f64) -> Vec<[f32; 16]> {
    layers
        .iter()
        .map(|layer| {
            let mut trans = [0.0f32; 16];
            match layer.kind {
                LayerKind::Emulsion => {
                    let sens = layer
                        .spectral_sensitivity
                        .as_ref()
                        .expect("emulsion has sensitivity");
                    let rho = layer.silver_halide_fraction as f64;
                    let thickness = layer.thickness.0 as f64;
                    for i in 0..16 {
                        let od = sens.samples[i] * sigma_scale * rho * thickness;
                        trans[i] = (-od).exp() as f32;
                    }
                }
                LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                    if let Some(curve) = layer.spectral_sensitivity.as_ref() {
                        let thickness = layer.thickness.0 as f64;
                        for i in 0..16 {
                            let od = curve.samples[i] * thickness;
                            trans[i] = (-od).exp() as f32;
                        }
                    }
                }
                LayerKind::Support => {}
            }
            trans
        })
        .collect()
}

/// Table-consuming variant of [`absorb_stack`] that reuses `out` as scratch.
/// Identical walk math (same operands, same order); only the per-tap
/// transmittance lookup differs (table vs recompute).
pub(crate) fn absorb_walk_forward_with_trans(
    layers: &[EmulsionLayer],
    incident: &[f32; 16],
    trans_table: &[[f32; 16]],
    out: &mut Vec<LayerAbsorption>,
) -> [f32; 16] {
    let mut phi = *incident;
    out.clear();

    for (layer_index, layer) in layers.iter().enumerate() {
        let mut absorbed = [0.0f32; 16];
        let produces_latent = layer.kind == LayerKind::Emulsion;

        match layer.kind {
            LayerKind::Emulsion => {
                for i in 0..16 {
                    let trans = trans_table[layer_index][i];
                    let phi_t = phi[i] * trans;
                    absorbed[i] = phi[i] - phi_t;
                    phi[i] = phi_t;
                }
            }
            LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                // Curve-less layers pass light unchanged, matching the original
                // walk's `if let Some(curve)` guard.
                if layer.spectral_sensitivity.is_some() {
                    for i in 0..16 {
                        let trans = trans_table[layer_index][i];
                        let phi_t = phi[i] * trans;
                        absorbed[i] = phi[i] - phi_t;
                        phi[i] = phi_t;
                    }
                }
            }
            LayerKind::Support => {}
        }

        out.push(LayerAbsorption {
            absorbed,
            produces_latent,
        });
    }

    phi
}

/// Table-consuming variant of [`absorb_stack_upward`] that reuses `out` as scratch.
pub(crate) fn absorb_walk_upward_with_trans(
    layers: &[EmulsionLayer],
    incident_upward: &[f32; 16],
    trans_table: &[[f32; 16]],
    out: &mut Vec<LayerAbsorption>,
) {
    let mut phi = *incident_upward;
    out.clear();
    out.resize(
        layers.len(),
        LayerAbsorption {
            absorbed: [0.0; 16],
            produces_latent: false,
        },
    );

    for (idx, layer) in layers.iter().enumerate().rev() {
        let mut absorbed = [0.0f32; 16];
        let produces_latent = layer.kind == LayerKind::Emulsion;

        match layer.kind {
            LayerKind::Emulsion => {
                for i in 0..16 {
                    let trans = trans_table[idx][i];
                    let phi_t = phi[i] * trans;
                    absorbed[i] = phi[i] - phi_t;
                    phi[i] = phi_t;
                }
            }
            LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                if layer.spectral_sensitivity.is_some() {
                    for i in 0..16 {
                        let trans = trans_table[idx][i];
                        let phi_t = phi[i] * trans;
                        absorbed[i] = phi[i] - phi_t;
                        phi[i] = phi_t;
                    }
                }
            }
            LayerKind::Support => {}
        }

        out[idx] = LayerAbsorption {
            absorbed,
            produces_latent,
        };
    }
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
pub fn integrated_absorbed(absorbed: &[f32; 16]) -> f32 {
    mean_absorbed_fluence(absorbed)
}

/// Mean spectral absorbed fluence: `∫ Φ_abs(λ) dλ / 300.0`.
pub fn mean_absorbed_fluence(absorbed: &[f32; 16]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..15 {
        acc += (absorbed[i] + absorbed[i + 1]) * (1.0 / 30.0);
    }
    acc
}

/// True trapezoidal integral `∫ Φ_abs(λ) dλ` over 400–700 nm (photons · nm / µm²).
pub fn total_absorbed_fluence(absorbed: &[f32; 16]) -> f32 {
    let dlambda = 20.0f32;
    let mut acc = 0.0f32;
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

    fn flat_incident(v: f32) -> [f32; 16] {
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
            assert!(abs >= -1e-6 && abs <= phi_in + 1e-6);
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
                (sum - phi_in).abs() <= 1e-5 * phi_in.max(1.0),
                "λ[{i}]: in={phi_in} abs+trans={sum}"
            );
        }
    }

    #[test]
    fn filter_blocks_blue() {
        use crate::film::spectrum::WavelengthGrid;
        let grid = WavelengthGrid::mvp();
        let samples: Vec<f64> = grid
            .wavelengths_nm
            .iter()
            .map(|&l| if l < 500.0 { 1.0 } else { 0.01 })
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
        let absorbed = [2.0f32; 16];
        let mean = integrated_absorbed(&absorbed);
        assert!(
            (mean - 2.0).abs() < 1e-6,
            "expected band average 2.0, got {mean} (would be ~600 if left as raw integral)"
        );
    }

    #[test]
    fn upward_absorption_energy_conservation_and_order() {
        let top = EmulsionLayer {
            name: "top_blue",
            depth_from_surface: Microns(0.0),
            thickness: Microns(5.0),
            kind: LayerKind::Emulsion,
            spectral_sensitivity: Some(SpectralCurve::constant(0.1)),
            crystal_size: None,
            silver_halide_fraction: 0.2,
            coupler: None,
            gamma_contrast: 1.0,
            capture_k: 1.0,
            reciprocity_p: 1.0,
            is_reversal: false,
        };
        let bottom = EmulsionLayer {
            name: "bottom_red",
            depth_from_surface: Microns(5.0),
            thickness: Microns(5.0),
            kind: LayerKind::Emulsion,
            spectral_sensitivity: Some(SpectralCurve::constant(0.1)),
            crystal_size: None,
            silver_halide_fraction: 0.2,
            coupler: None,
            gamma_contrast: 1.0,
            capture_k: 1.0,
            reciprocity_p: 1.0,
            is_reversal: false,
        };
        let incident_upward = flat_incident(1.0);
        let upward = absorb_stack_upward(&[top, bottom], &incident_upward, 1.0);
        assert_eq!(upward.len(), 2);
        // Bottom layer receives upward light first, so its absorption must exceed top layer's
        let abs_top = upward[0].absorbed[0];
        let abs_bottom = upward[1].absorbed[0];
        assert!(abs_bottom > abs_top, "bottom={abs_bottom} top={abs_top}");
        assert!(abs_top + abs_bottom < 1.0);
        assert!(abs_top > 0.0 && abs_bottom > 0.0);
    }
}
