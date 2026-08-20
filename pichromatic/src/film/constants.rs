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

/// Physical chromogenic dye cloud radius (µm).
/// Matches Kodak C-41 oxidized developer diffusion radius in gelatin (5.0 µm diameter).
pub const DYE_CLOUD_RADIUS_UM: f32 = 2.5;

/// Typical chromogenic dye-cloud physical extent / diameter (µm).
/// Order-of-magnitude from published chromogenic emulsion surveys. The CPU
/// particle path treats this as an effective cloud diameter (2 * DYE_CLOUD_RADIUS_UM = 5.0 µm)
/// and converts it to a Gaussian footprint from the circular-area second moment.
pub const DYE_CLOUD_CORRELATION_UM: f32 = 2.0 * DYE_CLOUD_RADIUS_UM;

/// Relative absorption cross-section scale (1/µm) for Beer–Lambert in
/// [`crate::film::exposure::absorption::absorb_stack`].
///
/// **Not** a measured AgX molar absorptivity. Tuned so absolute mid-gray under
/// sunny-16 / box ISO lands near developable fraction ≈ 0.3–0.4 when fluence
/// comes from `upsample_acescg` directly (no RGB-mean / luminance re-scale).
/// Single global MVP calibration — not per-stock physics.
pub const ABSORPTION_SIGMA_SCALE_PER_UM: f64 = 2.0;

/// Absolute radiometric anchor. Converts the relative upsampled proxy spectrum
/// into absolute photon fluence to correctly anchor the T=4 crystal threshold model.
pub const RADIOMETRIC_SCALE: f64 = 6.0;

/// Legacy GPU scatter amount `s` in the in-emulsion irradiation mix
/// `scattered = (1−w)·G(σ_core)*Φ + w·Exp(λ)*Φ`,
/// `Φ' = (1−s)·Φ + s·scattered`.
///
/// Retained only while the GPU is frozen under the CPU-first workflow. The CPU
/// does not use these legacy constants: Kodak E-4050 publishes distinct
/// processed-film B/G/R responses with an acutance lobe above 100%, which a
/// nonnegative normalized kernel cannot represent by itself. The CPU uses the
/// optional stock irradiation component jointly calibrated with cloud formation
/// and post-realization adjacency.
// Source context: Kodak E-4050, page 4, Daylight exposure / Process C-41.
pub const SCATTER_AMOUNT: f32 = 1.0;

/// Legacy GPU Gaussian core σ (µm); not used by the corrected CPU path.
pub const SCATTER_CORE_UM: f32 = 2.5;

/// Legacy GPU exponential-tail decay length λ (µm); not used by the CPU path.
pub const SCATTER_TAIL_UM: f32 = 2.5;

/// Legacy GPU exponential-tail weight; not used by the corrected CPU path.
pub const SCATTER_TAIL_WEIGHT: f32 = 0.75;

/// Empirically fit residual colored-coupler (orange mask) density as a fraction
/// of `d_max` at undeveloped (f=0). Shared by reduction and Dmin densitometry —
/// keep a single site so white-point calibration cannot desync from mask formation.
pub const MASK_DENSITY_FRACTION_OF_DMAX: f32 = 0.4;

/// Substrate fog optical density (OD) floor above reference Dmin (~0.005).
/// Shared by scanner invert and chemical-fog developable fraction in reduction:
/// random developable crystals at zero exposure yield linear particle density
/// ≈ `FOG_OFFSET` when overwrite uses `D ≈ d_max · f`.
pub const FOG_OFFSET: f32 = 0.005;

/// Scanner detector / reconstruction Gaussian PSF sampling floor standard deviation (pixels).
///
/// In physical film scanners and digital camera digitization rigs, the sensor optical
/// low-pass filter (OLPF), pixel aperture fill-factor diffusion, and optical reconstruction
/// yield an effective detector sampling floor on the order of 0.6–0.7 px.
pub const SCANNER_SENSOR_SIGMA_PX: f32 = 0.65;

/// Scanner optical lens Gaussian PSF standard deviation (µm).
///
/// Physical film scanners and digital camera scanning lenses have a finite optical transfer
/// function (diffraction, aberrations, focus depth) on the order of 1.5–2.5 µm.
pub const SCANNER_OPTICAL_SIGMA_UM: f32 = 2.0;



