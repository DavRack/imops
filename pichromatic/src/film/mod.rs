//! Physically-based analog film simulation.
//!
//! Checkpointed implementation; see `film-implementation.md`.
//!
//! # Honest scope (MVP)
//! Stage *shapes* follow photographic physics (Beer–Lambert, crystal-population
//! LUT, dye-cloud grain, adjacency, mask-aware invert). Absolute *scale* still
//! depends on named empirical constants in [`constants`]:
//! `ABSORPTION_SIGMA_SCALE_PER_UM`, `MASK_DENSITY_FRACTION_OF_DMAX`, plus per-layer `capture_k`
//! and `adjacency_beta`. Construction (R, AH OD, T-grain plate thickness) is the
//! shared C-41 kit in [`stock::kit`], not a per-stock look slider.

pub mod blur;
pub mod colorimetry;
pub mod constants;
pub mod development;
pub mod error;
pub mod exposure;
pub mod fixtures;
pub mod gpu;
pub mod scan;
pub mod spectrum;
pub mod stock;
pub mod types;
pub mod units;

pub use error::FilmError;
pub use gpu::process_gpu;
pub use stock::StockId;
pub use types::{ExposureMeta, FilmFormat, FilmRenderGeometry};

use crate::color::ColorSpaceTag;
use crate::film::development::develop;
use crate::film::exposure::{expose_with_pitch_and_shutter, expose_with_pitch_shutter_and_scale};
use crate::film::scan::{scan, ScanMode};
use crate::pixel::Image;

/// Module version string for linkage / checkpoint tracking.
pub fn film_version() -> &'static str {
    "0.1.0"
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilmOutput {
    /// Densitometric ACEScg, Dmin normalized ~1 (scanned negative).
    NegativeLinear,
    /// Bounded scanner invert from processed Dmin and a neutral mid-gray scan.
    PositiveLinear,
    /// Scene-referred HDR inverse-H&D reconstruction from processed Dmin and neutral mid-gray scan.
    PositiveInverseHd,
}

#[derive(Clone, Debug)]
pub struct FilmParams {
    pub stock: StockId,
    pub film_format: FilmFormat,
    /// Optional physical width of the source image on film, in millimetres.
    /// `None` uses the selected [`FilmFormat`] width.
    pub render_width_mm: Option<f32>,
    /// RNG seed for grain (deterministic shot noise).
    pub seed: u64,
    pub output: FilmOutput,
    /// Enable the wide backing halation (reflectance-driven bounce). `false`
    /// zeroes the AH reflectance so `max_r <= 0` gates the wide bounce.
    pub enable_halation: bool,
    /// Normalize exposure across stocks to the capture ISO (scene-relative
    /// fluence `∝ v`, independent of stock box speed). `false` keeps the raw
    /// box-speed difference: a faster stock receives `box_iso/capture_iso`
    /// more fluence for the same input — the "shot with the same camera
    /// settings" look.
    pub compensate_box_speed: bool,
    /// Contrast tuning factor for the scanner S-curve (0.0 = linear HDR, 1.0 = standard S-curve).
    pub scanner_s_curve: f32,
}

impl FilmParams {
    /// Resolve and validate the effective physical width for this render.
    pub fn effective_width_mm(&self) -> Result<f32, FilmError> {
        let width_mm = self
            .render_width_mm
            .unwrap_or_else(|| self.film_format.width_mm().0);
        if width_mm.is_finite() && width_mm > 0.0 {
            Ok(width_mm)
        } else {
            Err(FilmError::InvalidRenderWidth)
        }
    }

    /// Validate image dimensions and derive the physical pixel pitch once.
    pub fn render_geometry(
        &self,
        width: usize,
        height: usize,
    ) -> Result<FilmRenderGeometry, FilmError> {
        if width == 0 || height == 0 {
            return Err(FilmError::InvalidDimensions);
        }
        let pixel_count = width
            .checked_mul(height)
            .ok_or(FilmError::InvalidDimensions)?;
        let width_mm = self.effective_width_mm()?;
        let pixel_pitch_um = (f64::from(width_mm) * 1000.0 / width as f64) as f32;
        if !pixel_pitch_um.is_finite() || pixel_pitch_um <= 0.0 {
            return Err(FilmError::InvalidRenderWidth);
        }
        Ok(FilmRenderGeometry {
            pixel_count,
            width_mm,
            pixel_pitch_um,
        })
    }
}

impl Default for FilmParams {
    fn default() -> Self {
        Self {
            stock: StockId::BwStub,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 0,
            output: FilmOutput::NegativeLinear,
            enable_halation: true,
            compensate_box_speed: true,
            scanner_s_curve: 0.0,
        }
    }
}

/// Run the film simulation in-place on an absolute-luminance ACEScg [`Image`].
///
/// Input must already be absolute luminance (pipeline: `BaselineExposureCompensation`).
/// Film fluence is `Φ ∝ L`; emulsion speed is stock absorption / `capture_k` calibration.
pub fn process(image: &mut Image, params: &FilmParams) -> Result<(), FilmError> {
    let cs = image.metadata.color_space;
    match cs {
        Some(ColorSpaceTag::AcesCg) => {}
        other => {
            return Err(FilmError::WrongColorSpace {
                expected: "ACEScg",
                got: format!("{other:?}"),
            });
        }
    }

    let width = image.metadata.width;
    let height = image.metadata.height;
    let geometry = params.render_geometry(width, height)?;
    if image.rgb_data.len() != geometry.pixel_count {
        return Err(FilmError::InvalidDimensions);
    }

    let mut stock = params.stock.load()?;
    if !params.enable_halation {
        stock.antihalation.reflectance = crate::film::spectrum::SpectralCurve::constant(0.0);
    }
    let pitch = geometry.pixel_pitch_um;
    let shutter = image
        .metadata
        .shutter_seconds
        .unwrap_or(1.0 / stock.box_iso.0);

    let capture_scale = camera_capture_scale(
        &image.metadata,
        stock.box_iso.0,
        params.compensate_box_speed,
    );
    let latent = expose_with_pitch_shutter_and_scale(
        &image.rgb_data,
        width,
        height,
        &stock,
        pitch,
        shutter,
        capture_scale,
    );
    let dyes = develop(&stock, &latent, params.seed, pitch);

    let is_reversal = stock.layers.iter().any(|l| l.is_reversal);
    image.rgb_data = if is_reversal || params.output == FilmOutput::NegativeLinear {
        scan(&stock, &dyes, ScanMode::NegativeLinear, pitch)
    } else {
        let calibration = crate::film::scan::scanner_calibration_acescg(&stock, pitch, shutter)?;
        let mode = match params.output {
            FilmOutput::PositiveInverseHd => ScanMode::PositiveInverseHd {
                dmin: calibration.dmin,
                mid: calibration.mid,
                scanner_s_curve: params.scanner_s_curve,
            },
            FilmOutput::PositiveLinear => ScanMode::PositiveLinear {
                dmin: calibration.dmin,
                mid: calibration.mid,
            },
            FilmOutput::NegativeLinear => ScanMode::NegativeLinear,
        };
        scan(&stock, &dyes, mode, pitch)
    };
    image.metadata.color_space = Some(ColorSpaceTag::AcesCg);
    Ok(())
}

/// Legacy N32 mid-gray reference retained only for the stale GPU constants.
///
/// CPU PositiveLinear calls the joint adaptive scanner calibration directly.
pub(crate) fn mid_negative_acescg(
    stock: &crate::film::stock::FilmStock,
    pitch_um: f32,
    shutter_seconds: f32,
) -> [f32; 3] {
    use crate::pixel::MIDDLE_GRAY;
    const N: usize = 32;
    let g = relative_to_absolute_y(MIDDLE_GRAY, stock.box_iso.0);
    let rgb = vec![[g, g, g]; N * N];
    let latent = expose_with_pitch_and_shutter(&rgb, N, N, stock, pitch_um, shutter_seconds);
    let dyes = develop(stock, &latent, 0, pitch_um);
    let buf = scan(stock, &dyes, ScanMode::NegativeLinear, pitch_um);
    crate::film::scan::mean_rgb(&buf)
}

pub(crate) fn relative_to_absolute_y(y_rel: f32, box_iso: f32) -> f32 {
    use crate::film::exposure::radiance::{relative_to_absolute_luminance, sunny16_exposure};
    use crate::film::units::IsoSpeed;
    let e = sunny16_exposure(IsoSpeed(box_iso));
    relative_to_absolute_luminance(
        y_rel as f64,
        e.shutter_seconds as f64,
        e.f_number as f64,
        e.iso as f64,
    ) as f32
}

/// Reapply the camera exposure after `BaselineExposureCompensation` made the
/// input absolute. Normalized to Sunny-16 at the stock box ISO, preserving the
/// existing stock calibration; invalid metadata keeps the legacy fallback path.
///
/// Two explicit steps:
/// 1. **Print exposure** `64·iso·t/N²` — the capture ISO cancels the
///    absolute-luminance decode (`L·scale = 64·K·v`), so the film input is the
///    image's own scene-relative brightness.
/// 2. **Box-speed compensation** — `1.0` when compensated (box speed
///    neutralized; mid-gray stays mid-gray, swapping stocks shifts nothing) or
///    `box_iso/iso` when not (the film is exposed at its box speed and the raw
///    box-speed step between stocks shows).
pub(crate) fn camera_capture_scale(
    meta: &crate::image::ImageMetadata,
    box_iso: f32,
    compensate_box_speed: bool,
) -> f32 {
    match (meta.shutter_seconds, meta.f_number, meta.iso) {
        (Some(t), Some(n), Some(iso))
            if t.is_finite()
                && n.is_finite()
                && iso.is_finite()
                && t > 0.0
                && n > 0.0
                && iso > 0.0 =>
        {
            let print_scale = 64.0 * iso * t / (n * n);
            let box_speed_comp = if compensate_box_speed {
                1.0
            } else {
                box_iso / iso
            };
            print_scale * box_speed_comp
        }
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::ColorSpaceTag;
    use crate::image::ImageMetadata;
    use crate::pixel::{PixelOps, MIDDLE_GRAY};

    fn to_absolute_rgb(rgb: [f32; 3], box_iso: f32) -> [f32; 3] {
        let g = relative_to_absolute_y(1.0, box_iso);
        [rgb[0] * g, rgb[1] * g, rgb[2] * g]
    }

    #[test]
    fn camera_reexposure_keeps_equivalent_capture_film_input_equal() {
        let a = ImageMetadata {
            shutter_seconds: Some(1.0 / 100.0),
            f_number: Some(4.0),
            iso: Some(200.0),
            ..Default::default()
        };
        let b = ImageMetadata {
            shutter_seconds: Some(1.0 / 25.0),
            f_number: Some(8.0),
            iso: Some(200.0),
            ..Default::default()
        };
        let input_a = crate::film::exposure::radiance::absolute_luminance_gain(
            a.shutter_seconds.unwrap() as f64,
            a.f_number.unwrap() as f64,
            a.iso.unwrap() as f64,
        ) as f32
            * camera_capture_scale(&a, 400.0, false);
        let input_b = crate::film::exposure::radiance::absolute_luminance_gain(
            b.shutter_seconds.unwrap() as f64,
            b.f_number.unwrap() as f64,
            b.iso.unwrap() as f64,
        ) as f32
            * camera_capture_scale(&b, 400.0, false);
        assert!((input_a - input_b).abs() < 1e-5, "{input_a} != {input_b}");
        assert!((input_a - 1600.0).abs() < 1e-4);
        // Compensated mode must preserve the equivalence too.
        let comp_a = crate::film::exposure::radiance::absolute_luminance_gain(
            a.shutter_seconds.unwrap() as f64,
            a.f_number.unwrap() as f64,
            a.iso.unwrap() as f64,
        ) as f32
            * camera_capture_scale(&a, 400.0, true);
        let comp_b = crate::film::exposure::radiance::absolute_luminance_gain(
            b.shutter_seconds.unwrap() as f64,
            b.f_number.unwrap() as f64,
            b.iso.unwrap() as f64,
        ) as f32
            * camera_capture_scale(&b, 400.0, true);
        assert!((comp_a - comp_b).abs() < 1e-5, "{comp_a} != {comp_b}");
        assert!((comp_a - 800.0).abs() < 1e-4, "compensated input={comp_a}");
    }

    #[test]
    fn camera_reexposure_preserves_stock_to_camera_iso_ratio() {
        let meta = ImageMetadata {
            shutter_seconds: Some(1.0 / 100.0),
            f_number: Some(4.0),
            iso: Some(200.0),
            ..Default::default()
        };
        let input =
            crate::film::exposure::radiance::absolute_luminance_gain(1.0 / 100.0, 4.0, 200.0)
                as f32
                * camera_capture_scale(&meta, 400.0, false);
        assert!((input - 800.0 * 400.0 / 200.0).abs() < 1e-4);
    }

    #[test]
    fn compensated_capture_scale_drops_box_speed() {
        let meta = ImageMetadata {
            shutter_seconds: Some(1.0 / 100.0),
            f_number: Some(4.0),
            iso: Some(200.0),
            ..Default::default()
        };
        // Compensated: scale depends on the capture ISO only — box ISO is inert.
        let s100 = camera_capture_scale(&meta, 100.0, true);
        let s400 = camera_capture_scale(&meta, 400.0, true);
        assert!((s100 - s400).abs() < 1e-6, "{s100} != {s400}");
        assert!((s400 - 8.0).abs() < 1e-5, "scale={s400} (64·iso·t/N²)");
        // Uncompensated: box ISO drives the full step (100 → 400 = 2 stops).
        let u100 = camera_capture_scale(&meta, 100.0, false);
        let u400 = camera_capture_scale(&meta, 400.0, false);
        assert!((u400 / u100 - 4.0).abs() < 1e-6);
        // Compensated total fluence factor is scene-relative: L·scale = 64·K·v.
        let gain = crate::film::exposure::radiance::absolute_luminance_gain(1.0 / 100.0, 4.0, 200.0)
            as f32;
        let expected = 64.0 * crate::film::constants::METER_CONSTANT_K as f32;
        assert!(
            (gain * s400 - expected).abs() < 1e-3,
            "{} != {expected}",
            gain * s400
        );
    }

    #[test]
    fn compensated_keeps_film_input_exif_independent() {
        // Calibrated stock + mid-gray keeps the response on the steep LUT
        // region, so exposure differences are actually visible in the output.
        // Input is the pipeline-consistent absolute luminance for the same
        // scene-relative mid-gray under each capture ISO.
        let params = color_params(FilmOutput::PositiveLinear);
        let mk = |iso: f32| {
            let l = crate::film::exposure::radiance::relative_to_absolute_luminance(
                0.185_f64,
                1.0 / 100.0,
                4.0,
                iso as f64,
            ) as f32;
            Image {
                rgb_data: vec![[l, l, l]; 16],
                raw_data: std::sync::Arc::from([]),
                metadata: ImageMetadata {
                    width: 4,
                    height: 4,
                    color_space: Some(ColorSpaceTag::AcesCg),
                    shutter_seconds: Some(1.0 / 100.0),
                    f_number: Some(4.0),
                    iso: Some(iso),
                    ..Default::default()
                },
            }
        };
        let mut a = mk(100.0);
        let mut b = mk(400.0);
        process(&mut a, &params).unwrap();
        process(&mut b, &params).unwrap();
        let max_diff = a
            .rgb_data
            .iter()
            .zip(b.rgb_data.iter())
            .flat_map(|(pa, pb)| pa.iter().zip(pb.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "compensated film input must not depend on capture ISO (max diff {max_diff})"
        );
        // Uncompensated keeps the box-speed step: ColorNeg200 box 200 vs capture
        // ISO 100 → 2× fluence, vs capture 400 → 0.5×, so outputs must differ.
        let mut off = params.clone();
        off.compensate_box_speed = false;
        let mut c = mk(100.0);
        let mut d = mk(400.0);
        process(&mut c, &off).unwrap();
        process(&mut d, &off).unwrap();
        assert_ne!(
            c.rgb_data, d.rgb_data,
            "uncompensated must keep the box-speed step"
        );
    }

    fn make_image(width: usize, height: usize, fill_rel: [f32; 3], box_iso: f32) -> Image {
        let fill = to_absolute_rgb(fill_rel, box_iso);
        let e = crate::film::exposure::radiance::sunny16_exposure(crate::film::units::IsoSpeed(
            box_iso,
        ));
        Image {
            rgb_data: vec![fill; width * height],
            raw_data: std::sync::Arc::from([]),
            metadata: ImageMetadata {
                width,
                height,
                color_space: Some(ColorSpaceTag::AcesCg),
                shutter_seconds: Some(e.shutter_seconds),
                f_number: Some(e.f_number),
                iso: Some(e.iso),
                ..Default::default()
            },
        }
    }

    fn bw_params() -> FilmParams {
        FilmParams {
            stock: StockId::BwStub,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 1,
            output: FilmOutput::NegativeLinear,
            enable_halation: true,
            compensate_box_speed: true,
            scanner_s_curve: 0.0,
        }
    }

    fn color_params(output: FilmOutput) -> FilmParams {
        FilmParams {
            stock: StockId::ColorNeg200,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 1,
            output,
            enable_halation: true,
            compensate_box_speed: true,
            scanner_s_curve: 0.0,
        }
    }

    #[test]
    fn render_geometry_uses_format_or_override_width() {
        let params = color_params(FilmOutput::NegativeLinear);
        let normal = params.render_geometry(1000, 2).unwrap();
        assert_eq!(normal.pixel_count, 2000);
        assert_eq!(normal.width_mm, 36.0);
        assert_eq!(normal.pixel_pitch_um, 36.0);

        let mut super8 = params.clone();
        super8.film_format = FilmFormat::FilmSuper8;
        let super8_g = super8.render_geometry(1000, 2).unwrap();
        assert_eq!(super8_g.width_mm, 5.79);
        assert_eq!(super8_g.pixel_pitch_um, 5.79);

        let mut standard8 = params.clone();
        standard8.film_format = FilmFormat::FilmStandard8;
        let standard8_g = standard8.render_geometry(1000, 2).unwrap();
        assert_eq!(standard8_g.width_mm, 4.90);
        assert_eq!(standard8_g.pixel_pitch_um, 4.90);

        for (width_mm, expected_pitch) in [(1.0, 1.0), (0.5, 0.5)] {
            let mut custom = params.clone();
            custom.render_width_mm = Some(width_mm);
            let geometry = custom.render_geometry(1000, 2).unwrap();
            assert_eq!(geometry.pixel_count, 2000);
            assert_eq!(geometry.width_mm, width_mm);
            assert_eq!(geometry.pixel_pitch_um, expected_pitch);
        }

        let mut alias = params.clone();
        alias.film_format = FilmFormat::Film1mmDebug;
        let mut generic = params;
        generic.render_width_mm = Some(1.0);
        assert_eq!(
            alias.render_geometry(1000, 2),
            generic.render_geometry(1000, 2)
        );
    }

    #[test]
    fn render_geometry_rejects_invalid_widths_and_overflow() {
        let params = color_params(FilmOutput::NegativeLinear);
        for width_mm in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut invalid = params.clone();
            invalid.render_width_mm = Some(width_mm);
            assert_eq!(
                invalid.render_geometry(8, 8),
                Err(FilmError::InvalidRenderWidth)
            );
        }
        assert_eq!(
            params.render_geometry(usize::MAX, 2),
            Err(FilmError::InvalidDimensions)
        );
    }

    #[test]
    fn physical_width_outputs_are_deterministic_finite_and_same_size() {
        for stock in [StockId::ColorNeg200, StockId::Portra400] {
            for width_mm in [None, Some(1.0), Some(0.5)] {
                let mut params = color_params(FilmOutput::NegativeLinear);
                params.stock = stock;
                params.render_width_mm = width_mm;
                let mut first = make_image(16, 12, [0.2, 0.15, 0.1], 200.0);
                let mut second = first.clone();
                process(&mut first, &params).unwrap();
                process(&mut second, &params).unwrap();

                assert_eq!(first.metadata.width, 16);
                assert_eq!(first.metadata.height, 12);
                assert_eq!(first.rgb_data.len(), 16 * 12);
                assert_eq!(first.rgb_data, second.rgb_data);
                assert!(first
                    .rgb_data
                    .iter()
                    .flat_map(|px| px.iter())
                    .all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn film_module_is_linked() {
        assert!(!film_version().is_empty());
    }

    #[test]
    fn all_stocks_load_and_validate() {
        for id in [
            StockId::BwStub,
            StockId::ColorNeg200,
            StockId::Portra400,
            StockId::Ektar100,
            StockId::FujiPro400H,
            StockId::EktachromeE100,
            StockId::TriX400,
            StockId::CineStill50D,
        ] {
            id.load().unwrap_or_else(|e| panic!("{id:?}: {e}"));
        }
    }

    #[test]
    fn hd_curve_monotonic_patches() {
        let params = bw_params();
        let levels = [0.02, 0.05, 0.1, 0.185, 0.3, 0.5, 0.8, 1.2];
        let mut scanned_y = Vec::new();
        for &y in &levels {
            let mut img = make_image(8, 8, [y, y, y], 100.0);
            process(&mut img, &params).unwrap();
            let mean_y: f32 =
                img.rgb_data.iter().map(|p| p.luminance()).sum::<f32>() / img.rgb_data.len() as f32;
            scanned_y.push(mean_y);
        }
        for i in 1..scanned_y.len() {
            assert!(
                scanned_y[i] <= scanned_y[i - 1] + 1e-4,
                "H&D not monotonic for negative at i={i}: {:?}",
                scanned_y
            );
        }
    }

    #[test]
    fn unexposed_is_dmin() {
        let params = bw_params();
        let mut img = make_image(8, 8, [0.0, 0.0, 0.0], 100.0);
        process(&mut img, &params).unwrap();
        let peak = img
            .rgb_data
            .iter()
            .map(|p| p[0].max(p[1]).max(p[2]))
            .sum::<f32>()
            / img.rgb_data.len() as f32;
        assert!(
            (peak - 1.0).abs() < 0.05,
            "unexposed peak should be ~1.0, got {peak}"
        );
    }

    #[test]
    fn clipping_free_midgray() {
        let params = bw_params();
        let mut img = make_image(8, 8, [MIDDLE_GRAY, MIDDLE_GRAY, MIDDLE_GRAY], 100.0);
        process(&mut img, &params).unwrap();
        for px in &img.rgb_data {
            for &c in px {
                assert!(c.is_finite(), "NaN/Inf in mid-gray output");
                assert!(c > 0.0 && c < 1.5, "mid-gray channel out of (0,1.5): {c}");
            }
        }
    }

    #[test]
    fn acescg_tag_preserved() {
        let params = bw_params();
        let mut img = make_image(4, 4, [0.185, 0.185, 0.185], 100.0);
        process(&mut img, &params).unwrap();
        assert_eq!(img.metadata.color_space, Some(ColorSpaceTag::AcesCg));
    }

    #[test]
    fn neutral_stays_near_neutral_positive() {
        let params = color_params(FilmOutput::PositiveLinear);
        let g = MIDDLE_GRAY;
        let mut img = make_image(16, 16, [g, g, g], 200.0);
        process(&mut img, &params).unwrap();
        let mean = crate::film::scan::mean_rgb(&img.rgb_data);
        let lab = crate::film::colorimetry::acescg_to_lab(mean);
        assert!(
            lab[1].abs() < 5.0 && lab[2].abs() < 5.0,
            "neutral Lab a,b should be <5, got a={} b={}",
            lab[1],
            lab[2]
        );
    }

    #[test]
    fn saturated_red_affects_cyan_dye() {
        let stock = StockId::ColorNeg200.load().unwrap();
        let rgb_rel = [0.6f32, 0.05, 0.02];
        let rgb = vec![to_absolute_rgb(rgb_rel, 200.0); 16];
        let latent = crate::film::exposure::expose(&rgb, 4, 4, &stock);
        let dyes = crate::film::development::reduction::reduce(&stock, &latent);
        let mean = |plane: &[f32]| plane.iter().sum::<f32>() / plane.len() as f32;
        let d_yellow = mean(&dyes.image_dye[0]);
        let d_cyan = mean(&dyes.image_dye[2]);
        assert!(
            d_cyan > d_yellow,
            "red scene should drive cyan dye > yellow dye: cyan={d_cyan} yellow={d_yellow}"
        );
    }

    #[test]
    fn determinism_color_path() {
        let params = color_params(FilmOutput::NegativeLinear);
        let mut a = make_image(8, 8, [0.185, 0.1, 0.05], 200.0);
        let mut b = make_image(8, 8, [0.185, 0.1, 0.05], 200.0);
        process(&mut a, &params).unwrap();
        process(&mut b, &params).unwrap();
        assert_eq!(a.rgb_data, b.rgb_data);
    }

    #[test]
    fn grain_shadow_chroma_after_invert() {
        let params = FilmParams {
            stock: StockId::ColorNeg200,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 99,
            output: FilmOutput::PositiveLinear,
            enable_halation: true,
            compensate_box_speed: true,
            scanner_s_curve: 0.0,
        };
        let mut img = make_image(64, 64, [0.0, 0.0, 0.0], 200.0);
        process(&mut img, &params).unwrap();
        let mean = crate::film::scan::mean_rgb(&img.rgb_data);
        let lab = crate::film::colorimetry::acescg_to_lab(mean);
        let c = crate::film::colorimetry::chroma_ab(lab);
        assert!(c < 4.0, "shadow chroma after grain+invert C*ab={c}");
    }

    #[test]
    fn full_pipeline_smoke() {
        let mut img = make_image(64, 64, [0.2, 0.15, 0.1], 200.0);
        for (i, px) in img.rgb_data.iter_mut().enumerate() {
            let v = ((i * 1103515245 + 12345) % 1000) as f32 / 1000.0;
            *px = to_absolute_rgb([v * 0.5, v * 0.4, v * 0.3], 200.0);
        }
        for output in [
            FilmOutput::NegativeLinear,
            FilmOutput::PositiveLinear,
            FilmOutput::PositiveInverseHd,
        ] {
            let params = FilmParams {
                stock: StockId::ColorNeg200,
                film_format: FilmFormat::Film35mm,
                render_width_mm: None,
                seed: 1,
                output,
                enable_halation: true,
                compensate_box_speed: true,
                scanner_s_curve: 0.0,
            };
            let mut copy = img.clone();
            process(&mut copy, &params).unwrap();
            for px in &copy.rgb_data {
                for &c in px {
                    assert!(c.is_finite());
                }
            }
        }
    }

    #[test]
    fn physics_effects_always_on_smoke() {
        let params = FilmParams {
            stock: StockId::ColorNeg200,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 2,
            output: FilmOutput::NegativeLinear,
            enable_halation: true,
            compensate_box_speed: true,
            scanner_s_curve: 0.0,
        };
        let mut img = make_image(32, 32, [0.185, 0.185, 0.185], 200.0);
        process(&mut img, &params).unwrap();
        for px in &img.rgb_data {
            for &c in px {
                assert!(c.is_finite());
            }
        }
    }

    #[test]
    fn enable_halation_off_matches_zero_reflectance_stock() {
        let mut params = color_params(FilmOutput::PositiveLinear);
        params.film_format = FilmFormat::Film1mmDebug;
        let mut off_params = params.clone();
        off_params.enable_halation = false;

        let base = to_absolute_rgb([0.02, 0.02, 0.02], 200.0);
        let patch = to_absolute_rgb([9.0, 9.0, 9.0], 200.0);
        let mut data = vec![base; 32 * 32];
        for y in 12..20 {
            for x in 12..20 {
                data[y * 32 + x] = patch;
            }
        }
        let e =
            crate::film::exposure::radiance::sunny16_exposure(crate::film::units::IsoSpeed(200.0));
        let mk = |data: Vec<[f32; 3]>| Image {
            rgb_data: data,
            raw_data: std::sync::Arc::from([]),
            metadata: ImageMetadata {
                width: 32,
                height: 32,
                color_space: Some(ColorSpaceTag::AcesCg),
                shutter_seconds: Some(e.shutter_seconds),
                f_number: Some(e.f_number),
                iso: Some(e.iso),
                ..Default::default()
            },
        };

        let mut with = mk(data.clone());
        let mut without = mk(data);
        process(&mut with, &params).unwrap();
        process(&mut without, &off_params).unwrap();
        assert_ne!(
            with.rgb_data, without.rgb_data,
            "halation must change the output on a specular patch"
        );
        let ring_diff = with
            .rgb_data
            .iter()
            .zip(without.rgb_data.iter())
            .enumerate()
            .filter(|(i, _)| !(12..20).contains(&(*i / 32)) || !(12..20).contains(&(*i % 32)))
            .flat_map(|(_, (a, b))| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0.0f32, f32::max);
        assert!(
            ring_diff > 1e-4,
            "halation must add signal around the patch (ring diff {ring_diff})"
        );
        for px in with.rgb_data.iter().chain(without.rgb_data.iter()) {
            for &c in px {
                assert!(c.is_finite(), "NaN/Inf in halation toggle output");
            }
        }
    }

    #[test]
    fn image_method_wrapper() {
        let mut img = make_image(16, 16, [0.185, 0.185, 0.185], 200.0);
        let params = color_params(FilmOutput::NegativeLinear);
        img.film(&params).unwrap();
        assert_eq!(img.metadata.color_space, Some(ColorSpaceTag::AcesCg));
    }

    #[test]
    fn wrong_colorspace_errors() {
        let mut img = make_image(8, 8, [0.1, 0.1, 0.1], 200.0);
        img.metadata.color_space = Some(ColorSpaceTag::Srgb);
        let params = color_params(FilmOutput::NegativeLinear);
        let err = process(&mut img, &params).unwrap_err();
        assert!(matches!(err, FilmError::WrongColorSpace { .. }));
    }

    #[test]
    fn colorchecker_runs() {
        let (mut img, _) = crate::film::fixtures::colorchecker_image(8);
        for px in &mut img.rgb_data {
            *px = to_absolute_rgb(*px, 200.0);
        }
        let params = color_params(FilmOutput::PositiveLinear);
        process(&mut img, &params).unwrap();
        for px in &img.rgb_data {
            for &c in px {
                assert!(c.is_finite(), "NaN/Inf in ColorChecker output");
            }
        }
    }

    #[test]
    fn gray_ramp_monotonic() {
        let patch = 8;
        let (mut img, _) = crate::film::fixtures::colorchecker_image(patch);
        for px in &mut img.rgb_data {
            *px = to_absolute_rgb(*px, 200.0);
        }
        let params = color_params(FilmOutput::PositiveLinear);
        process(&mut img, &params).unwrap();
        let means = crate::film::fixtures::sample_patch_means(&img, patch);
        let ys: Vec<f32> = (18..24).rev().map(|i| means[i].luminance()).collect();
        for i in 1..ys.len() {
            assert!(
                ys[i] + 1e-4 >= ys[i - 1],
                "gray ramp not increasing: {:?}",
                ys
            );
        }
    }

    #[test]
    fn neutrals_low_chroma() {
        let patch = 8;
        let (mut img, _) = crate::film::fixtures::colorchecker_image(patch);
        for px in &mut img.rgb_data {
            *px = to_absolute_rgb(*px, 200.0);
        }
        let params = color_params(FilmOutput::PositiveLinear);
        process(&mut img, &params).unwrap();
        let means = crate::film::fixtures::sample_patch_means(&img, patch);
        let mut c_sum = 0.0f64;
        for i in 18..24 {
            let lab = crate::film::colorimetry::acescg_to_lab(means[i]);
            c_sum += crate::film::colorimetry::chroma_ab(lab);
        }
        let mean_c = c_sum / 6.0;
        // The bounded processed-Dmin invert only pins neutrality at the calibration mid;
        // darker/lighter neutrals pick up H&D channel imbalance (no gray-ramp).
        // Mid-gray neutrality is covered by `neutral_stays_near_neutral_positive`.
        assert!(mean_c < 55.0, "mean C*ab of neutrals = {mean_c}");
    }

    #[test]
    fn colorchecker_roundtrip_delta_e() {
        let patch = 8;
        let (mut img, refs) = crate::film::fixtures::colorchecker_image(patch);
        for px in &mut img.rgb_data {
            *px = to_absolute_rgb(*px, 200.0);
        }
        let params = color_params(FilmOutput::PositiveLinear);
        process(&mut img, &params).unwrap();
        let means = crate::film::fixtures::sample_patch_means(&img, patch);
        let mut deltas = Vec::new();
        for i in 0..24 {
            let expected = refs[i];
            let lab_ref = crate::film::colorimetry::acescg_to_lab(expected);
            let lab_out = crate::film::colorimetry::acescg_to_lab(means[i]);
            let d = crate::film::colorimetry::ciede2000(lab_ref, lab_out);
            deltas.push(d);
        }
        deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = deltas[deltas.len() / 2];
        assert!(
            median < 25.0,
            "median round-trip ΔE00={median} (scene-linear gate < 25)"
        );
    }

    #[test]
    fn midgray_developable_fraction_near_straight_line() {
        let stock = StockId::ColorNeg200.load().unwrap();
        let g = relative_to_absolute_y(MIDDLE_GRAY, 200.0);
        let rgb = vec![[g, g, g]; 16];
        let latent = crate::film::exposure::expose(&rgb, 4, 4, &stock);
        let mean_f: f32 = latent.layers[1].iter().sum::<f32>() / latent.layers[1].len() as f32;
        assert!(
            mean_f > 0.2 && mean_f < 0.7,
            "mid-gray developable fraction should be mid-scale, got {mean_f}"
        );
    }

    #[test]
    fn reversal_ektachrome_positive_slide() {
        let params = FilmParams {
            stock: StockId::EktachromeE100,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 1,
            output: FilmOutput::PositiveLinear, // ignored for reversal
            enable_halation: true,
            compensate_box_speed: true,
            scanner_s_curve: 0.0,
        };
        let mut img = make_image(16, 16, [MIDDLE_GRAY, MIDDLE_GRAY, MIDDLE_GRAY], 100.0);
        process(&mut img, &params).unwrap();
        let mean = crate::film::scan::mean_rgb(&img.rgb_data);
        assert!(mean[0].is_finite() && mean[1].is_finite() && mean[2].is_finite());
    }

    #[test]
    fn processed_dmin_includes_chemical_fog() {
        let stock = StockId::ColorNeg200.load().unwrap();
        let base = crate::film::scan::normalized_dmin_acescg(&stock);
        let peak = base[0].max(base[1]).max(base[2]);
        assert!(
            (peak - 1.0).abs() < 1e-5,
            "normalized base peak must be 1, got {base:?}"
        );
        // Orange mask: R > G > B on the film base.
        assert!(
            base[0] > base[1] && base[1] > base[2],
            "expected orange film base {base:?}"
        );

        let dmin = crate::film::scan::processed_dmin_acescg(
            &stock,
            FilmFormat::Film35mm.pixel_pitch_um(1000),
            1.0 / stock.box_iso.0,
        )
        .unwrap();
        assert!(
            dmin.luminance() < base.luminance(),
            "developed unexposed Dmin must include absorbing chemical fog: base={base:?}, dmin={dmin:?}"
        );
    }

    #[test]
    fn positive_half_mm_pre_rotation_width_halation_off_converges() {
        // Film precedes final portrait rotation, so vale_lago reaches this stage
        // at width 4032. Height 1 preserves the exact pitch without a 12 MP fixture.
        let mut image = make_image(4032, 1, [MIDDLE_GRAY, MIDDLE_GRAY, MIDDLE_GRAY], 121.0);
        image.metadata.shutter_seconds = Some(1.0 / 121.0);
        let params = FilmParams {
            stock: StockId::Portra400,
            film_format: FilmFormat::Film35mm,
            render_width_mm: Some(0.5),
            seed: 1,
            output: FilmOutput::PositiveLinear,
            enable_halation: false,
            compensate_box_speed: true,
            scanner_s_curve: 0.0,
        };

        process(&mut image, &params).unwrap();
        assert!(
            image
                .rgb_data
                .iter()
                .flatten()
                .all(|value| value.is_finite()),
            "production-state PositiveLinear output must remain finite"
        );
    }

    #[test]
    fn positive_inverse_hd_scanner_s_curve_pipeline() {
        let mut image = make_image(16, 16, [0.185, 0.185, 0.185], 200.0);
        let mut params = color_params(FilmOutput::PositiveInverseHd);
        params.scanner_s_curve = 1.0;
        process(&mut image, &params).unwrap();
        for px in &image.rgb_data {
            for &c in px {
                assert!(c.is_finite() && c >= 0.0);
            }
        }
    }
}
