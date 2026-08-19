//! Film stock definitions and validation.

pub mod bw_stub;
pub mod cinestill_50d;
pub mod color_neg_200;
pub mod ektachrome_e100;
pub mod ektar_100;
pub mod fuji_pro_400h;
pub mod fujichrome_velvia_100;
pub mod kit;
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
    /// Support / air-interface reflectance `R(λ)`. CPU bounce is `T_AH² · R`.
    pub reflectance: SpectralCurve,
    /// Gaussian σ (µm) of the wide backing-bounce PSF.
    pub psf_halation_um: f32,
}

/// Effective stock-specific component of a joint record-response fit.
///
/// Its decomposition from cloud formation and adjacency is not independently
/// identified irradiation physics, nor an exact Status-M calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrradiationResponse {
    /// Gaussian core sigma in microns.
    pub core_sigma_um: f32,
    /// Exponential-tail decay length in microns.
    pub tail_decay_um: f32,
    /// Tail mixture weights for the blue, green, and red records. Each weight
    /// is shared by that record's fast and slow emulsion layers.
    pub tail_weight_bgr: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct FilmStock {
    pub name: &'static str,
    pub box_iso: IsoSpeed,
    /// Layers ordered top (light-incident) → bottom (support).
    pub layers: Vec<EmulsionLayer>,
    pub antihalation: AntihalationModel,
    /// Optional pre-capture component of an effective joint record-response
    /// fit; cloud formation and adjacency remain separate model components.
    pub irradiation_response: Option<IrradiationResponse>,
    pub developer_diffusion_length: Microns,
    pub adjacency_beta: f32,
    pub adjacency_beta_record: f32,
    pub adjacency_beta_cross: f32,
    pub scanner_light: SpectralCurve,
    /// Precomputed at load: per-layer capture LUT (None for non-emulsion).
    pub capture_luts: Vec<Option<DevelopableFractionLut>>,
    /// Per emulsion layer grain coefficient κ_ref = 1/√ρ. The inverse square
    /// supplies the physical crystal areal density used by the particle grain
    /// realization; the render pitch is applied when deriving crystals/pixel.
    pub grain_kappa: Vec<Option<f32>>,
    /// Some(t) = emulsion crystals are thin circular plates of thickness t µm
    /// (tabular T-grain morphology); None = equivalent-sphere (equant)
    /// morphology. Plate volume V = π·(d/2)²·t with d = crystal diameter.
    pub tabular_grain_thickness_um: Option<f32>,
}

/// Public stock identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StockId {
    BwStub,
    ColorNeg200,
    Portra400,
    Ektar100,
    FujiPro400H,
    FujichromeVelvia100,
    EktachromeE100,
    TriX400,
    CineStill50D,
}

impl StockId {
    pub fn load(self) -> Result<FilmStock, FilmError> {
        match self {
            StockId::BwStub => bw_stub::load(),
            StockId::ColorNeg200 => color_neg_200::load(),
            StockId::Portra400 => portra_400::load(),
            StockId::Ektar100 => ektar_100::load(),
            StockId::FujiPro400H => fuji_pro_400h::load(),
            StockId::FujichromeVelvia100 => fujichrome_velvia_100::load(),
            StockId::EktachromeE100 => ektachrome_e100::load(),
            StockId::TriX400 => trix_400::load(),
            StockId::CineStill50D => cinestill_50d::load(),
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
                    return Err(FilmError::InvalidStock(
                        "emulsion missing spectral sensitivity",
                    ));
                }
                if layer.crystal_size.is_none() {
                    return Err(FilmError::InvalidStock("emulsion missing crystal_size"));
                }
                if layer.coupler.is_none() {
                    return Err(FilmError::InvalidStock("emulsion missing coupler"));
                }
            }
        }

        if let Some(response) = self.irradiation_response {
            if !response.core_sigma_um.is_finite() || response.core_sigma_um <= 0.0 {
                return Err(FilmError::InvalidStock(
                    "irradiation core sigma must be finite and > 0",
                ));
            }
            if !response.tail_decay_um.is_finite() || response.tail_decay_um <= 0.0 {
                return Err(FilmError::InvalidStock(
                    "irradiation tail decay must be finite and > 0",
                ));
            }
            if response
                .tail_weight_bgr
                .iter()
                .any(|weight| !weight.is_finite() || !(0.0..=1.0).contains(weight))
            {
                return Err(FilmError::InvalidStock(
                    "irradiation tail weights must be finite and in [0, 1]",
                ));
            }
            if self.emulsion_layers().count() != 6 {
                return Err(FilmError::InvalidStock(
                    "B/G/R irradiation response requires six fast/slow emulsion layers",
                ));
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
        // For lognormal diameter d, E[d^k] = exp(kμ + k²σ²/2).
        // ⟨V⟩ = (π/4)E[d²]t for tabular grains, or (π/6)E[d³]
        // for equivalent spheres. Powers of E[d] are not these moments.
        let tabular_t = self.tabular_grain_thickness_um.map(|t| t as f64);
        self.grain_kappa = self
            .layers
            .iter()
            .map(|layer| {
                if layer.kind == LayerKind::Emulsion {
                    let dist = layer.crystal_size.unwrap();
                    let volume = match tabular_t {
                        Some(t) => {
                            std::f64::consts::FRAC_PI_4
                                * (2.0 * dist.mu_ln + 2.0 * dist.sigma_ln.powi(2)).exp()
                                * t
                        }
                        None => {
                            std::f64::consts::PI / 6.0
                                * (3.0 * dist.mu_ln + 4.5 * dist.sigma_ln.powi(2)).exp()
                        }
                    };
                    let packing = layer.silver_halide_fraction as f64;
                    let thickness = layer.thickness.0 as f64;
                    let rho_areal = packing * thickness / volume.max(1e-18);
                    // Selwyn coefficient κ_ref = 1/√ρ. The particle renderer
                    // recovers ρ and multiplies it by the physical pixel area
                    // at render time, so no image-resolution-specific grain
                    // amount is baked into the stock.
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

    /// Build the NxN cross-layer adjacency coupling matrix M for the emulsion layers.
    ///
    /// - Diagonal M_ii = adjacency_beta (beta_self)
    /// - Same color record off-diagonal M_ij = adjacency_beta_record (e.g. fast/slow of same coupler)
    /// - Cross color record off-diagonal M_ij = adjacency_beta_cross (e.g. Y <-> M <-> C)
    pub fn adjacency_matrix(&self) -> Vec<Vec<f32>> {
        let emulsions: Vec<_> = self.emulsion_layers().map(|(_, l)| l).collect();
        let n = emulsions.len();
        let mut m = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            let coupler_i = emulsions[i].coupler.as_ref().map(|c| c.name);
            for j in 0..n {
                if i == j {
                    m[i][j] = self.adjacency_beta;
                } else if coupler_i.is_some() && coupler_i == emulsions[j].coupler.as_ref().map(|c| c.name) {
                    m[i][j] = self.adjacency_beta_record;
                } else {
                    m[i][j] = self.adjacency_beta_cross;
                }
            }
        }
        m
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
    fn spherical_grain_kappa_uses_third_lognormal_moment() {
        let stock = StockId::BwStub.load().unwrap();
        let (layer_idx, layer) = stock.emulsion_layers().next().unwrap();
        let dist = layer.crystal_size.unwrap();
        assert!(dist.sigma_ln > 0.0);
        let mean_d3 = (3.0 * dist.mu_ln + 4.5 * dist.sigma_ln.powi(2)).exp();
        let volume = std::f64::consts::PI / 6.0 * mean_d3;
        let expected = (volume / (layer.silver_halide_fraction as f64 * layer.thickness.0 as f64))
            .sqrt() as f32;
        let actual = stock.grain_kappa[layer_idx].unwrap();
        assert!((actual - expected).abs() <= expected * 1e-6);

        let mean_d = (dist.mu_ln + 0.5 * dist.sigma_ln.powi(2)).exp();
        let wrong = (std::f64::consts::PI / 6.0 * mean_d.powi(3)
            / (layer.silver_halide_fraction as f64 * layer.thickness.0 as f64))
            .sqrt() as f32;
        assert!((actual - wrong).abs() > expected * 1e-3);
    }

    #[test]
    fn tabular_grain_kappa_uses_second_lognormal_moment() {
        let stock = StockId::Portra400.load().unwrap();
        let (layer_idx, layer) = stock.emulsion_layers().next().unwrap();
        let dist = layer.crystal_size.unwrap();
        let plate_t = stock.tabular_grain_thickness_um.unwrap() as f64;
        assert!(dist.sigma_ln > 0.0);
        let mean_d2 = (2.0 * dist.mu_ln + 2.0 * dist.sigma_ln.powi(2)).exp();
        let volume = std::f64::consts::FRAC_PI_4 * mean_d2 * plate_t;
        let expected = (volume / (layer.silver_halide_fraction as f64 * layer.thickness.0 as f64))
            .sqrt() as f32;
        let actual = stock.grain_kappa[layer_idx].unwrap();
        assert!((actual - expected).abs() <= expected * 1e-6);

        let mean_d = (dist.mu_ln + 0.5 * dist.sigma_ln.powi(2)).exp();
        let wrong = (std::f64::consts::FRAC_PI_4 * mean_d.powi(2) * plate_t
            / (layer.silver_halide_fraction as f64 * layer.thickness.0 as f64))
            .sqrt() as f32;
        assert!((actual - wrong).abs() > expected * 1e-3);
    }

    #[test]
    fn stock_rejects_empty_layers() {
        let stock = FilmStock {
            name: "empty",
            box_iso: IsoSpeed(100.0),
            layers: vec![],
            antihalation: AntihalationModel {
                reflectance: SpectralCurve::constant(0.0),
                psf_halation_um: 50.0,
            },
            irradiation_response: None,
            developer_diffusion_length: Microns(5.0),
            adjacency_beta: 0.0,
            adjacency_beta_record: 0.0,
            adjacency_beta_cross: 0.0,
            scanner_light: SpectralCurve::constant(1.0),
            capture_luts: vec![],
            grain_kappa: vec![],
            tabular_grain_thickness_um: None,
        };
        let err = stock.finalize().unwrap_err();
        assert!(matches!(err, FilmError::InvalidStock(_)));
    }
}
