//! Highlight reconstruction (ISP algorithm #4): luma-preserving, hue-
//! preserving desaturation knee applied to white-balanced camera-linear RGB.
//!
//! Must run AFTER `CFACoeffs` (white balance): only in the post-WB signal is
//! "sensor clipped" well defined — the loader scales raws so full well = 1.0,
//! and WB gains push already-clipped channels above that. Channels clip one
//! at a time, so blown regions otherwise keep a cast from whichever channel
//! saturated last; real saturated emulsion/sensor response trends toward
//! neutral instead. Above the knee the chroma of each pixel fades smoothly
//! toward the neutral gray of its own luminance; below the knee pixels stay
//! bitwise untouched. Known pointwise tradeoff: legitimately saturated colors
//! within the knee lose part of their chroma — distinguishing them from
//! clipped casts requires spatial reconstruction, which this stage has not.

use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::ImageBuffer;
use rayon::prelude::*;

/// Post-WB full-well level: the loader normalizes raws to [0, 1]; WB gains
/// can carry clipped channels above it.
const CLIP: f32 = 1.0;

/// Knee onset as a fraction of clip. Conventional ISP highlight-knee
/// placement: gradual saturation starts well below full-well, while normally
/// exposed mid-grays (~18%) are never touched.
const KNEE_START_FRACTION: f32 = 0.6;

pub fn highlight_reconstruction(image_buffer: &mut ImageBuffer) {
    let threshold = KNEE_START_FRACTION * CLIP;
    let inv_range = 1.0 / (CLIP - threshold);
    image_buffer.par_iter_mut().for_each(|pixel| {
        let [r, g, b] = *pixel;
        if !(r.is_finite() && g.is_finite() && b.is_finite()) {
            return;
        }
        // Key on the hottest channel: single-channel clipping is what casts
        // highlights, and post-WB it can only be seen there.
        let hottest = r.max(g).max(b);
        let t_raw = ((hottest - threshold) * inv_range).clamp(0.0, 1.0);
        if t_raw <= 0.0 {
            return;
        }
        let t = t_raw * t_raw * (3.0 - 2.0 * t_raw);
        // Sensor-space luminance approximation using the shared scene weights.
        let y = crate::pixel::R_RELATIVE_LUMINANCE * r
            + crate::pixel::G_RELATIVE_LUMINANCE * g
            + crate::pixel::B_RELATIVE_LUMINANCE * b;
        let keep = 1.0 - t;
        *pixel = [
            y + (r - y) * keep,
            y + (g - y) * keep,
            y + (b - y) * keep,
        ];
    });
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HighlightReconstructionParamsGpu {
    threshold: f32,
    inv_range: f32,
    width: u32,
    height: u32,
}

pub fn highlight_reconstruction_gpu(
    ctx: &GpuContext,
    storage_buffer: &GpuImageBuffer,
) {
    let params = HighlightReconstructionParamsGpu {
        threshold: KNEE_START_FRACTION * CLIP,
        inv_range: 1.0 / (CLIP - KNEE_START_FRACTION * CLIP),
        width: storage_buffer.width as u32,
        height: storage_buffer.height as u32,
    };
    let shader_source = r#"
        struct Params {
            threshold: f32,
            inv_range: f32,
            width: u32,
            height: u32,
        };

        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<uniform> params: Params;

        const LUMA_R: f32 = 0.2722287168;
        const LUMA_G: f32 = 0.6740817658;
        const LUMA_B: f32 = 0.0536895174;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = global_id.x;
            let y = global_id.y;
            if (x >= params.width || y >= params.height) {
                return;
            }
            let index = y * params.width + x;
            let p = pixels[index];
            // Non-finite pass-through (matches CPU): v-v is NaN for both NaN
            // and +-Inf inputs, so this catches them all.
            if (!((p.r - p.r == 0.0) && (p.g - p.g == 0.0) && (p.b - p.b == 0.0))) {
                return;
            }
            let hottest = max(p.r, max(p.g, p.b));
            var t = clamp((hottest - params.threshold) * params.inv_range, 0.0, 1.0);
            if (t <= 0.0) {
                return;
            }
            t = t * t * (3.0 - 2.0 * t);
            let lum = LUMA_R * p.r + LUMA_G * p.g + LUMA_B * p.b;
            let keep = 1.0 - t;
            pixels[index] = vec4<f32>(
                lum + (p.r - lum) * keep,
                lum + (p.g - lum) * keep,
                lum + (p.b - lum) * keep,
                p.a
            );
        }
    "#;

    ctx.dispatch_compute_shader_2d(
        "highlight_reconstruction",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luma(px: [f32; 3]) -> f32 {
        crate::pixel::R_RELATIVE_LUMINANCE * px[0]
            + crate::pixel::G_RELATIVE_LUMINANCE * px[1]
            + crate::pixel::B_RELATIVE_LUMINANCE * px[2]
    }

    #[test]
    fn below_knee_is_bitwise_unchanged() {
        // Hottest channel under the knee onset stays exact.
        let mut pixels = vec![[0.5f32, 0.4, 0.55], [0.55, 0.55, 0.55], [0.59, 0.1, 0.3]];
        let before = pixels.clone();
        highlight_reconstruction(&mut pixels);
        assert_eq!(pixels, before);
    }

    #[test]
    fn clipped_single_channel_cast_fully_neutralizes() {
        // The artifact this stage exists for: one channel blown hard while
        // others lag. Max-channel keying must fully neutralize these.
        let mut pixels = vec![[1.6f32, 0.2, 0.9], [0.3, 1.4, 0.4], [0.9, 0.8, 1.05]];
        highlight_reconstruction(&mut pixels);
        for px in &pixels {
            assert!(
                px.iter().all(|&v| (v - luma(*px)).abs() < 1e-4),
                "clipped pixel {px:?} not neutral"
            );
        }
    }

    #[test]
    fn bright_neutral_stays_bright_neutral() {
        let mut px = vec![[1.5f32, 1.45, 1.4]];
        highlight_reconstruction(&mut px);
        assert!(px[0].iter().all(|&v| (v - luma(px[0])).abs() < 1e-4));
    }

    #[test]
    fn luminance_is_preserved_across_the_knee() {
        let mut pixels: Vec<[f32; 3]> = (0..24)
            .map(|i| {
                let s = 0.3 + i as f32 * 0.12;
                [s * 1.2, s, s * 0.7]
            })
            .collect();
        let luma_in: Vec<f32> = pixels.iter().map(|p| luma(*p)).collect();
        highlight_reconstruction(&mut pixels);
        for (px, y_in) in pixels.iter().zip(luma_in) {
            assert!(
                (luma(*px) - y_in).abs() < 1e-4,
                "luma drift {y_in} -> {} at {px:?}",
                luma(*px)
            );
        }
    }

    #[test]
    fn hue_direction_is_preserved_inside_the_knee() {
        // Chroma scales about the gray axis: offset ratios between channels
        // stay invariant for a pixel partially inside the knee.
        let base = [0.85f32, 0.5, 0.35];
        assert!(base[0] > 0.6 && base[0] < 1.0);
        let y_base = luma(base);
        for k in [1.0f32, 0.95] {
            let mut px = vec![base.map(|c| c * k)];
            highlight_reconstruction(&mut px);
            let out = px[0];
            let y = luma(out);
            let ratio_out = (out[0] - y) / (out[2] - y);
            let ratio_in = (base[0] - y_base) / (base[2] - y_base);
            assert!(
                (ratio_out - ratio_in).abs() < 1e-3,
                "hue shifted: {ratio_out} vs {ratio_in} at k={k}"
            );
        }
    }

    #[test]
    fn chroma_suppression_ratio_is_monotonic_in_hottest_channel() {
        // Suppressed-chroma fraction of the input chroma never increases as
        // the hottest channel climbs (constant below the knee, 0 at clip).
        let suppression_at = |h: f32| {
            // Fixed hue, ramping only the hottest channel through the knee.
            let input = [h, h * 0.65, h * 0.4];
            let mut px = vec![input];
            highlight_reconstruction(&mut px);
            let p = px[0];
            let y_in = luma(input);
            let y_out = luma(p);
            let chroma_in = ((input[0] - y_in).powi(2)
                + (input[1] - y_in).powi(2)
                + (input[2] - y_in).powi(2))
            .sqrt();
            let chroma_out =
                ((p[0] - y_out).powi(2) + (p[1] - y_out).powi(2) + (p[2] - y_out).powi(2)).sqrt();
            chroma_out / chroma_in
        };
        let mut prev = suppression_at(0.61);
        for i in 1..20 {
            let c = suppression_at(0.61 + i as f32 * 0.02);
            assert!(c <= prev + 1e-5, "suppression ratio rose at step {i}: {prev} -> {c}");
            prev = c;
        }
    }

    #[test]
    fn non_finite_pixels_pass_through_untouched() {
        let mut pixels = vec![
            [f32::NAN, 0.5, 0.5],
            [f32::INFINITY, 0.5, 0.5],
            [0.5, f32::NEG_INFINITY, 0.5],
        ];
        let before: Vec<[u32; 3]> = pixels.iter().map(|p| p.map(|v| v.to_bits())).collect();
        highlight_reconstruction(&mut pixels);
        let after: Vec<[u32; 3]> = pixels.iter().map(|p| p.map(|v| v.to_bits())).collect();
        assert_eq!(after, before);
    }
}
