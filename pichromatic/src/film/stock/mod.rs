//! Film stock definitions and validation.

pub mod bw_stub;
pub mod color_neg_200;
pub mod ektachrome_e100;
pub mod fuji_pro_400h;
pub mod portra_400;
pub mod trix_400;

use crate::film::error::FilmError;
use crate::film::exposure::capture::DevelopableFractionLut;
use crate::film::spectrum::SpectralCurve;
use crate::film::units::{IsoSpeed, Microns};

/// Parameters of ln(s) where s is equivalent crystal diameter in µm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogNormalDist {
    pub mu_ln: f64,
    pub sigma_ln: f64,
}

/// Dye coupler spectral absorptivity and max density.
#[derive(Clone, Debug)]
pub struct DyeCoupler {
    pub name: &'static str,
    /// Relative molar absorptivity ε(λ) for image dye.
    pub epsilon: SpectralCurve,
    /// Residual colored coupler (orange mask), if any.
    pub mask_epsilon: Option<SpectralCurve>,
    pub d_max: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Emulsion,
    Filter,
    Overcoat,
    Antihalation,
    Support,
}

/// One layer in the film stack (top / light-incident → bottom).
#[derive(Clone, Debug)]
pub struct EmulsionLayer {
    pub name: &'static str,
    pub depth_from_surface: Microns,
    pub thickness: Microns,
    pub kind: LayerKind,
    pub spectral_sensitivity: Option<SpectralCurve>,
    pub crystal_size: Option<LogNormalDist>,
    pub silver_halide_fraction: f32,
    pub coupler: Option<DyeCoupler>,
    pub gamma_contrast: f32,
    /// Per-layer absorption/quantum calibration `k` for the capture LUT.
    pub capture_k: f64,
    /// Schwarzschild reciprocity law exponent p ∈ (0.5, 1.0]. 1.0 = no failure.
    pub reciprocity_p: f32,
    /// True if this layer undergoes reversal E-6 development (positive dye image).
    pub is_reversal: bool,
}

impl EmulsionLayer {
    pub fn new_emulsion(
        name: &'static str,
        depth_from_surface: Microns,
        thickness: Microns,
        spectral_sensitivity: SpectralCurve,
        crystal_size: LogNormalDist,
        silver_halide_fraction: f32,
        coupler: DyeCoupler,
        gamma_contrast: f32,
        capture_k: f64,
    ) -> Self {
        Self {
            name,
            depth_from_surface,
            thickness,
            kind: LayerKind::Emulsion,
            spectral_sensitivity: Some(spectral_sensitivity),
            crystal_size: Some(crystal_size),
            silver_halide_fraction,
            coupler: Some(coupler),
            gamma_contrast,
            capture_k,
            reciprocity_p: 1.0,
            is_reversal: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AntihalationModel {
    pub reflectance: SpectralCurve,
    pub psf_local_um: f32,
    pub psf_halation_um: f32,
}

#[derive(Clone, Debug)]
pub struct FilmStock {
    pub name: &'static str,
    pub box_iso: IsoSpeed,
    /// Layers ordered top (light-incident) → bottom (support).
    pub layers: Vec<EmulsionLayer>,
    pub antihalation: AntihalationModel,
    pub developer_diffusion_length: Microns,
    pub adjacency_beta: f32,
    /// DIR (Development Inhibitor Releasing) coupler diffusion length.
    pub dir_diffusion_length: Microns,
    /// Interlayer DIR inhibition weights, `matrix[source][target]`.
    ///
    /// Emulsion indices follow top→bottom order among emulsion layers only
    /// (same order as [`FilmStock::emulsion_layers`] / dye planes). Entry
    /// `matrix[i][j]` scales inhibitor released by source emulsion `i` onto target `j`.
    /// May be non-symmetric (see Fuji Pro 400H).
    pub dir_inhibition_matrix: Vec<Vec<f32>>,
    pub scanner_light: SpectralCurve,
    /// Precomputed at load: per-layer capture LUT (None for non-emulsion).
    pub capture_luts: Vec<Option<DevelopableFractionLut>>,
    /// Per emulsion layer grain κ at a reference 1 µm pitch; scaled at runtime.
    pub grain_kappa: Vec<Option<f32>>,
}

/// Public stock identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StockId {
    BwStub,
    ColorNeg200,
    Portra400,
    FujiPro400H,
    EktachromeE100,
    TriX400,
}

impl StockId {
    pub fn load(self) -> Result<FilmStock, FilmError> {
        match self {
            StockId::BwStub => bw_stub::load(),
            StockId::ColorNeg200 => color_neg_200::load(),
            StockId::Portra400 => portra_400::load(),
            StockId::FujiPro400H => fuji_pro_400h::load(),
            StockId::EktachromeE100 => ektachrome_e100::load(),
            StockId::TriX400 => trix_400::load(),
        }
    }
}

impl FilmStock {
    /// Validate structural invariants and build capture LUTs / grain κ.
    pub fn finalize(mut self) -> Result<Self, FilmError> {
        if self.layers.is_empty() {
            return Err(FilmError::InvalidStock("stock has zero layers"));
        }
        let mut prev_depth = f32::NEG_INFINITY;
        for layer in &self.layers {
            if layer.depth_from_surface.0 + 1e-6 < prev_depth {
                return Err(FilmError::InvalidStock(
                    "layer depth_from_surface must be nondecreasing top→bottom",
                ));
            }
            prev_depth = layer.depth_from_surface.0;
            if layer.kind == LayerKind::Emulsion {
                if layer.thickness.0 <= 0.0 {
                    return Err(FilmError::InvalidStock("emulsion thickness must be > 0"));
                }
                if layer.spectral_sensitivity.is_none() {
                    return Err(FilmError::InvalidStock("emulsion missing spectral sensitivity"));
                }
                if layer.crystal_size.is_none() {
                    return Err(FilmError::InvalidStock("emulsion missing crystal_size"));
                }
            }
        }

        self.capture_luts = self
            .layers
            .iter()
            .map(|layer| {
                if layer.kind == LayerKind::Emulsion {
                    let dist = layer.crystal_size.unwrap();
                    Some(DevelopableFractionLut::build(&dist, layer.capture_k, 64))
                } else {
                    None
                }
            })
            .collect();

        // Areal grain density ρ [µm⁻²] = (packing · thickness) / ⟨crystal volume⟩.
        // packing is volumetric AgX fraction; thickness is layer depth (µm);
        // ⟨V⟩ ≈ (4/3)π r³ for equivalent-sphere diameter s=2r from the lognormal.
        self.grain_kappa = self
            .layers
            .iter()
            .map(|layer| {
                if layer.kind == LayerKind::Emulsion {
                    let dist = layer.crystal_size.unwrap();
                    let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
                    let r = (mean_s * 0.5).max(1e-6);
                    let volume = std::f64::consts::FRAC_PI_3 * 4.0 * r * r * r; // (4/3)π r³
                    let packing = layer.silver_halide_fraction as f64;
                    let thickness = layer.thickness.0 as f64;
                    let rho_areal = packing * thickness / volume.max(1e-18);
                    // κ at 1 µm pitch: fluctuation scale ~ 1/√(ρ · A_pixel); A=1 µm² → 1/√ρ.
                    Some((1.0 / rho_areal.sqrt()) as f32)
                } else {
                    None
                }
            })
            .collect();

        Ok(self)
    }

    pub fn emulsion_layers(&self) -> impl Iterator<Item = (usize, &EmulsionLayer)> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.kind == LayerKind::Emulsion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::spectrum::SpectralCurve;
    use crate::film::units::Microns;

    #[test]
    fn bw_stub_validates() {
        let stock = StockId::BwStub.load().expect("bw stub loads");
        let emulsions: Vec<_> = stock.emulsion_layers().collect();
        assert_eq!(emulsions.len(), 1);
        let layer = emulsions[0].1;
        assert!(layer.thickness.0 > 0.0);
        let sens = layer.spectral_sensitivity.as_ref().unwrap();
        assert!(sens.integrate() > 0.0);
    }

    #[test]
    fn bw_stub_layer_order() {
        let stock = StockId::BwStub.load().unwrap();
        let mut prev = f32::NEG_INFINITY;
        for layer in &stock.layers {
            assert!(layer.depth_from_surface.0 >= prev);
            prev = layer.depth_from_surface.0;
        }
    }

    #[test]
    fn stock_rejects_empty_layers() {
        let stock = FilmStock {
            name: "empty",
            box_iso: IsoSpeed(100.0),
            layers: vec![],
            antihalation: AntihalationModel {
                reflectance: SpectralCurve::constant(0.0),
                psf_local_um: 1.0,
                psf_halation_um: 50.0,
            },
            developer_diffusion_length: Microns(5.0),
            adjacency_beta: 0.0,
            dir_diffusion_length: Microns(5.0),
            dir_inhibition_matrix: vec![],
            scanner_light: SpectralCurve::constant(1.0),
            capture_luts: vec![],
            grain_kappa: vec![],
        };
        let err = stock.finalize().unwrap_err();
        assert!(matches!(err, FilmError::InvalidStock(_)));
    }
}
