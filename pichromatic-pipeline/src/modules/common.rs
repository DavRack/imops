use pichromatic::cst::ColorSpaceTag;
use pichromatic::gpu::GpuContext;
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::{Image, Pixel, SubPixel};
use crate::backend::{Backend, PipelineImage};
use super::PipelineModule;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// CPU-vs-GPU comparison tolerance, relative to the larger of the two compared
/// values (floored at 1.0 so near-zero values keep a tight absolute gate).
/// The CPU and GPU film pipelines both run f32 but may differ by a few ULPs
/// from hardware transcendentals and FMA contraction; 1e-4 (0.01%) leaves
/// comfortable headroom across GPUs while still catching real divergence.
pub const CPU_GPU_ABS_TOLERANCE: f32 = 32.0 * f32::EPSILON;

/// Generates a deterministic 128x128 RGB test image using a fixed seed.
pub fn generate_test_image_512x512(seed: u64) -> Image {
    let width = 128;
    let height = 128;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut rgb_data: Vec<Pixel> = Vec::with_capacity(width * height);

    for _ in 0..(width * height) {
        rgb_data.push([
            rng.gen::<SubPixel>(),
            rng.gen::<SubPixel>(),
            rng.gen::<SubPixel>(),
        ]);
    }

    let mut raw_data: Vec<SubPixel> = Vec::with_capacity(width * height);
    for _ in 0..(width * height) {
        raw_data.push(rng.gen::<SubPixel>());
    }

    Image {
        rgb_data,
        raw_data: raw_data.into(),
        metadata: ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::Srgb),
            ..Default::default()
        },
    }
}

/// Runs a PipelineModule on both CPU and WGPU GPU and compares their outputs
/// with the shared CPU/GPU floating-point precision threshold.
pub fn test_pipeline_module_cpu_vs_gpu(module: &dyn PipelineModule, seed: u64) {
    let ctx = GpuContext::new_sync();
    let seed_image = generate_test_image_512x512(seed);

    // 1. CPU Processing
    let mut cpu_pipeline_img = PipelineImage::Cpu(seed_image.clone());
    module.process(&Backend::Cpu, &mut cpu_pipeline_img);
    let cpu_out = cpu_pipeline_img.to_cpu(None);

    // 2. GPU Processing
    let mut gpu_pipeline_img = PipelineImage::new_gpu(&ctx, &seed_image);
    module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_pipeline_img);
    let gpu_out = gpu_pipeline_img.to_cpu(Some(&ctx));

    // Compare every output subpixel using the shared threshold.
    assert_images_equal_abs_tol(&cpu_out, &gpu_out);
}

pub fn assert_images_equal_abs_tol(cpu_image: &Image, gpu_image: &Image) {
    assert_images_equal_abs_tol_with(cpu_image, gpu_image, CPU_GPU_ABS_TOLERANCE);
}

/// Same as [`assert_images_equal_abs_tol`] with an explicit relative tolerance
/// (scaled by `max(1.0, |cpu|, |gpu|)` per channel).
pub fn assert_images_equal_abs_tol_with(cpu_image: &Image, gpu_image: &Image, tolerance: f32) {
    assert_eq!(
        cpu_image.rgb_data.len(),
        gpu_image.rgb_data.len(),
        "Image pixel buffer lengths do not match! CPU={}, GPU={}",
        cpu_image.rgb_data.len(),
        gpu_image.rgb_data.len()
    );

    let mut max_diff = 0.0f32;
    let mut max_location = (0usize, 0usize);
    for (idx, (c_pixel, g_pixel)) in cpu_image
        .rgb_data
        .iter()
        .zip(gpu_image.rgb_data.iter())
        .enumerate()
    {
        for ch in 0..3 {
            let cpu_bits = c_pixel[ch].to_bits();
            let gpu_bits = g_pixel[ch].to_bits();
            let diff = (c_pixel[ch] - g_pixel[ch]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_location = (idx, ch);
            }
            let tol = tolerance * c_pixel[ch].abs().max(g_pixel[ch].abs()).max(1.0);
            assert!(
                diff.is_finite() && diff <= tol,
                "RGB mismatch at index {} (x={}, y={}), channel {}: CPU={:?} ({:#010x}), GPU={:?} ({:#010x}), diff={}, tolerance={}",
                idx,
                idx % cpu_image.metadata.width,
                idx / cpu_image.metadata.width,
                ch,
                c_pixel[ch],
                cpu_bits,
                g_pixel[ch],
                gpu_bits,
                diff,
                tolerance
            );
        }
    }

    assert_eq!(
        cpu_image.raw_data.len(),
        gpu_image.raw_data.len(),
        "Image raw_data lengths do not match! CPU={}, GPU={}",
        cpu_image.raw_data.len(),
        gpu_image.raw_data.len()
    );
    for (idx, (c_subpixel, g_subpixel)) in cpu_image
        .raw_data
        .iter()
        .zip(gpu_image.raw_data.iter())
        .enumerate()
    {
        assert_eq!(
            c_subpixel.to_bits(),
            g_subpixel.to_bits(),
            "Raw subpixel mismatch at index {}: CPU={:?} ({:#010x}), GPU={:?} ({:#010x})",
            idx,
            c_subpixel,
            c_subpixel.to_bits(),
            g_subpixel,
            g_subpixel.to_bits()
        );
    }

    assert!(
        cpu_image.metadata.bitwise_eq(&gpu_image.metadata),
        "Metadata mismatch between CPU and GPU output!"
    );

    println!(
        "CPU/GPU max RGB difference: {} at pixel {}, channel {}",
        max_diff, max_location.0, max_location.1
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_image_512x512() {
        let img1 = generate_test_image_512x512(42);
        assert_eq!(img1.rgb_data.len(), 128 * 128);
        assert_eq!(img1.raw_data.len(), 128 * 128);

        let img2 = generate_test_image_512x512(42);
        assert_eq!(img1.rgb_data, img2.rgb_data);
        assert_eq!(img1.raw_data, img2.raw_data);
        assert_eq!(img1.metadata, img2.metadata);
    }
}
