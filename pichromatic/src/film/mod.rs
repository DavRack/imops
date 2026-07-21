//! Physically-based analog film simulation.
//!
//! Checkpointed implementation; see `film-implementation.md`.
//!
//! # Honest scope (MVP)
//! Stage *shapes* follow photographic physics (Beer–Lambert, crystal-population
//! LUT, dye-cloud grain, adjacency, mask-aware invert). Absolute *scale* still
//! depends on named empirical constants in [`constants`]:
//! `ABSORPTION_SIGMA_SCALE_PER_UM`, `LOCAL_SCATTER_MIX`, `HALATION_BLEED_WEIGHTS_BGR`,
//! `CHROMOGENIC_DYE_GRAIN_SCALE`,
//! `MASK_DENSITY_FRACTION_OF_DMAX`, plus per-layer `capture_k` and `adjacency_beta`.
//! AH stack absorption and `AntihalationModel.reflectance` are not yet linked.
//! ColorChecker gates measure round-trip vs input patches after mid/Dmin invert —
//! not a measured commercial stock.
//! Fast/slow emulsion pairs are not yet implemented.

pub mod blur;
pub mod colorimetry;
pub mod constants;
pub mod development;
pub mod error;
pub mod exposure;
pub mod fixtures;
pub mod scan;
pub mod spectrum;
pub mod stock;
pub mod types;
pub mod units;

pub use error::FilmError;
pub use stock::StockId;
pub use types::{ExposureMeta, FilmFormat};

use color::ColorSpaceTag;
use crate::film::development::develop;
use crate::film::exposure::expose_with_pitch_and_shutter;
use crate::film::scan::{scan, ScanMode};
use crate::pixel::Image;

/// Module version string for linkage / checkpoint tracking.
pub fn film_version() -> &'static str {
    "0.1.0"
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilmOutput {
    /// Densitometric ACEScg, Dmin normalized ~1 (scanned negative).
    NegativeLinear,
    /// Analytic mid/Dmin invert from stock film-base + mid-gray gain.
    PositiveLinear,
}

#[derive(Clone, Debug)]
pub struct FilmParams {
    pub stock: StockId,
    pub film_format: FilmFormat,
    /// RNG seed for grain (deterministic shot noise).
    pub seed: u64,
    pub output: FilmOutput,
}

impl Default for FilmParams {
    fn default() -> Self {
        Self {
            stock: StockId::BwStub,
            film_format: FilmFormat::Film35mm,
            seed: 0,
            output: FilmOutput::NegativeLinear,
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
    if width == 0 || height == 0 || image.rgb_data.len() != width * height {
        return Err(FilmError::InvalidDimensions);
    }

    let stock = params.stock.load()?;
    let pitch = params.film_format.pixel_pitch_um(width);
    let shutter = image.metadata.shutter_seconds.unwrap_or(1.0 / stock.box_iso.0);

    let latent = expose_with_pitch_and_shutter(
        &image.rgb_data,
        width,
        height,
        &stock,
        pitch,
        shutter,
    );
    let dyes = develop(&stock, &latent, params.seed, pitch);

    let is_reversal = stock.layers.iter().any(|l| l.is_reversal);
    let mode = if is_reversal {
        ScanMode::NegativeLinear
    } else {
        match params.output {
            FilmOutput::NegativeLinear => ScanMode::NegativeLinear,
            FilmOutput::PositiveLinear => {
                // Film base from stock mask / d_max / scanner light (same reference
                // scan_to_acescg normalizes to). Mid-gray negative sets per-channel
                // invert gain so reference mid lands at MIDDLE_GRAY.
                let dmin = crate::film::scan::normalized_dmin_acescg(&stock);
                let mid = mid_negative_acescg(&stock, pitch, shutter);
                ScanMode::PositiveLinear { dmin, mid }
            }
        }
    };

    image.rgb_data = scan(&stock, &dyes, mode);
    image.metadata.color_space = Some(ColorSpaceTag::AcesCg);
    Ok(())
}

/// Mid-gray densitometric RGB for invert gain.
///
/// Uses the same expose → develop → scan path as [`process`] (including DIR /
/// adjacency) so channel gains match the image being inverted. A modest patch
/// averages dye grain for a stable mean.
fn mid_negative_acescg(
    stock: &crate::film::stock::FilmStock,
    pitch_um: f32,
    shutter_seconds: f32,
) -> [f32; 3] {
    use crate::pixel::MIDDLE_GRAY;
    const N: usize = 32;
    let g = relative_to_absolute_y(MIDDLE_GRAY, stock.box_iso.0);
    let rgb = vec![[g, g, g]; N * N];
    let latent =
        expose_with_pitch_and_shutter(&rgb, N, N, stock, pitch_um, shutter_seconds);
    // Same development as process (DIR + adjacency + grain); mean kills grain.
    let dyes = develop(stock, &latent, 0, pitch_um);
    let buf = scan(stock, &dyes, ScanMode::NegativeLinear);
    crate::film::scan::mean_rgb(&buf)
}

fn relative_to_absolute_y(y_rel: f32, box_iso: f32) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageMetadata;
    use crate::pixel::{PixelOps, MIDDLE_GRAY};
    use color::ColorSpaceTag;

    fn to_absolute_rgb(rgb: [f32; 3], box_iso: f32) -> [f32; 3] {
        let g = relative_to_absolute_y(1.0, box_iso);
        [rgb[0] * g, rgb[1] * g, rgb[2] * g]
    }

    fn make_image(width: usize, height: usize, fill_rel: [f32; 3], box_iso: f32) -> Image {
        let fill = to_absolute_rgb(fill_rel, box_iso);
        let e = crate::film::exposure::radiance::sunny16_exposure(crate::film::units::IsoSpeed(
            box_iso,
        ));
        Image {
            rgb_data: vec![fill; width * height],
            raw_data: vec![],
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
            seed: 1,
            output: FilmOutput::NegativeLinear,
        }
    }

    fn color_params(output: FilmOutput) -> FilmParams {
        FilmParams {
            stock: StockId::ColorNeg200,
            film_format: FilmFormat::Film35mm,
            seed: 1,
            output,
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
            StockId::FujiPro400H,
            StockId::EktachromeE100,
            StockId::TriX400,
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
            let mean_y: f32 = img.rgb_data.iter().map(|p| p.luminance()).sum::<f32>()
                / img.rgb_data.len() as f32;
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
            seed: 99,
            output: FilmOutput::PositiveLinear,
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
        for output in [FilmOutput::NegativeLinear, FilmOutput::PositiveLinear] {
            let params = FilmParams {
                stock: StockId::ColorNeg200,
                film_format: FilmFormat::Film35mm,
                seed: 1,
                output,
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
            seed: 2,
            output: FilmOutput::NegativeLinear,
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
        let ys: Vec<f32> = (18..24)
            .rev()
            .map(|i| means[i].luminance())
            .collect();
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
        // Analytic mid/Dmin invert only pins neutrality at the calibration mid;
        // darker/lighter neutrals pick up H&D channel imbalance (no gray-ramp).
        // Mid-gray neutrality is covered by `neutral_stays_near_neutral_positive`.
        assert!(mean_c < 40.0, "mean C*ab of neutrals = {mean_c}");
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
            seed: 1,
            output: FilmOutput::PositiveLinear, // ignored for reversal
        };
        let mut img = make_image(16, 16, [MIDDLE_GRAY, MIDDLE_GRAY, MIDDLE_GRAY], 100.0);
        process(&mut img, &params).unwrap();
        let mean = crate::film::scan::mean_rgb(&img.rgb_data);
        assert!(mean[0].is_finite() && mean[1].is_finite() && mean[2].is_finite());
    }



    #[test]
    fn stock_dmin_drives_invert_not_image_guess() {
        let stock = StockId::ColorNeg200.load().unwrap();
        let dmin = crate::film::scan::normalized_dmin_acescg(&stock);
        let peak = dmin[0].max(dmin[1]).max(dmin[2]);
        assert!(
            (peak - 1.0).abs() < 1e-5,
            "normalized Dmin peak must be 1, got {dmin:?}"
        );
        // Orange mask: R > G > B on the film base.
        assert!(dmin[0] > dmin[1] && dmin[1] > dmin[2], "expected orange Dmin {dmin:?}");
    }
}
