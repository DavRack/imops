//! Shared physical constants for the film module.
//!
//! Every literal here must cite CODATA, a published photographic quantity,
//! or a derivation from those sources (see film-implementation.md §8).

/// Speed of light in vacuum (m/s). CODATA 2018 exact value.
pub const SPEED_OF_LIGHT_M_S: f64 = 299_792_458.0;

/// Planck constant (J·s). CODATA 2018 exact value.
pub const PLANCK_H_J_S: f64 = 6.626_070_15e-34;

/// ISO reflected-light meter calibration constant K.
/// Common value used with reflected-light metering (ISO 2720 / photographic practice).
pub const METER_CONSTANT_K: f64 = 12.5;

/// MVP visible wavelength range start (nm).
pub const WAVELENGTH_MIN_NM: f64 = 400.0;

/// MVP visible wavelength range end (nm).
pub const WAVELENGTH_MAX_NM: f64 = 700.0;

/// MVP wavelength sampling step (nm) → 16 samples on [400, 700].
pub const WAVELENGTH_STEP_NM: f64 = 20.0;

/// Number of MVP wavelength samples: (700−400)/20 + 1 = 16.
pub const WAVELENGTH_SAMPLE_COUNT: usize = 16;

/// Developability threshold: silver atoms at a sensitivity speck for a developable
/// latent-image speck. Standard AgX photographic-science assumption (T = 4).
pub const DEVELOPABILITY_THRESHOLD_ATOMS: u32 = 4;

/// Typical chromogenic dye-cloud correlation length (µm).
/// Order-of-magnitude from published chromogenic emulsion surveys.
pub const DYE_CLOUD_CORRELATION_UM: f32 = 3.0;

/// Scales silver-count κ (`1/√(ρ_areal·A)`) down to chromogenic *dye-cloud* granularity.
///
/// After fixing areal density to `(packing·thickness)/⟨V_crystal⟩`, silver-only κ still
/// over-predicts scanned color-neg RMS (each dye cloud averages many grains). Factor
/// 0.15 targets ~0.015–0.03 D mid-density RMS at ~9–12 µm/px. Empirically fit — not
/// a published constant.
pub const CHROMOGENIC_DYE_GRAIN_SCALE: f32 = 0.15;

/// Relative absorption cross-section scale (1/µm) for Beer–Lambert in
/// [`crate::film::exposure::absorption::absorb_stack`].
///
/// **Not** a measured AgX molar absorptivity. Tuned so absolute mid-gray under
/// sunny-16 / box ISO lands near developable fraction ≈ 0.3–0.4 when fluence
/// comes from `upsample_acescg` directly (no RGB-mean / luminance re-scale).
/// Single global MVP calibration — not per-stock physics.
pub const ABSORPTION_SIGMA_SCALE_PER_UM: f64 = 2.0;

/// Fraction of each emulsion's absorbed fluence mixed with a local gelatin-scatter
/// PSF (`AntihalationModel::psf_local_um`). Energy-conserving:
/// `Φ' = (1−f)·Φ + f·(Φ ⊛ PSF_local)`. Order-of-magnitude in-emulsion scatter.
pub const LOCAL_SCATTER_MIX: f32 = 0.18;

/// Relative acceptance of support-bounce (wide halation) by emulsion index
/// top→bottom (blue, green, red). Bounce arrives from below, so the deepest
/// emulsion takes the most; shallower layers get a red-biased bleed (colored halo).
/// Empirically chosen MVP weights — not measured interlayer coupling.
pub const HALATION_BLEED_WEIGHTS_BGR: [f32; 3] = [0.20, 0.45, 1.0];

/// Empirically fit residual colored-coupler (orange mask) density as a fraction
/// of `d_max` at undeveloped (f=0). Shared by reduction and Dmin densitometry —
/// keep a single site so white-point calibration cannot desync from mask formation.
pub const MASK_DENSITY_FRACTION_OF_DMAX: f32 = 0.4;
