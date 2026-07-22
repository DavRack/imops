use pichromatic::cst::ColorSpaceTag;
use pichromatic::gpu::GpuContext;
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::{Image, Pixel, SubPixel};
use crate::backend::{Backend, PipelineImage};
use super::PipelineModule;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
        raw_data,
        metadata: ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::Srgb),
            ..Default::default()
        },
    }
}

/// Runs a PipelineModule on both CPU and WGPU GPU and asserts their pixel outputs match identically.
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

    // 3. Compare outputs for exact equality
    assert_images_equal(&cpu_out, &gpu_out);
}

/// Compares CPU ground truth image against GPU output image pixel by pixel.
pub fn assert_images_equal(cpu_image: &Image, gpu_image: &Image) {
    assert_eq!(
        cpu_image.rgb_data.len(),
        gpu_image.rgb_data.len(),
        "Image pixel buffer lengths do not match! CPU={}, GPU={}",
        cpu_image.rgb_data.len(),
        gpu_image.rgb_data.len()
    );

    let mut mismatch_count = 0;

    for (idx, (c_pixel, g_pixel)) in cpu_image
        .rgb_data
        .iter()
        .zip(gpu_image.rgb_data.iter())
        .enumerate()
    {
        for ch in 0..3 {
            let diff = (c_pixel[ch] - g_pixel[ch]).abs();
            if diff > 1e-4 {
                mismatch_count += 1;
                if mismatch_count <= 10 {
                    eprintln!(
                        "Pixel mismatch at index {} (x={}, y={}), channel {}: CPU={:.9}, GPU={:.9}, diff={:.9}",
                        idx,
                        idx % cpu_image.metadata.width,
                        idx / cpu_image.metadata.width,
                        ch,
                        c_pixel[ch],
                        g_pixel[ch],
                        diff
                    );
                }
            }
        }
    }

    assert!(
        mismatch_count == 0,
        "CPU and GPU images differ! Total mismatches: {}",
        mismatch_count
    );

    assert_eq!(
        cpu_image.raw_data.len(),
        gpu_image.raw_data.len(),
        "Image raw_data lengths do not match! CPU={}, GPU={}",
        cpu_image.raw_data.len(),
        gpu_image.raw_data.len()
    );

    let mut raw_mismatch_count = 0;

    for (idx, (c_subpixel, g_subpixel)) in cpu_image
        .raw_data
        .iter()
        .zip(gpu_image.raw_data.iter())
        .enumerate()
    {
        let diff = (c_subpixel - g_subpixel).abs();
        if diff > 1e-4 {
            raw_mismatch_count += 1;
            if raw_mismatch_count <= 10 {
                eprintln!(
                    "Raw subpixel mismatch at index {}: CPU={:.9}, GPU={:.9}, diff={:.9}",
                    idx, c_subpixel, g_subpixel, diff
                );
            }
        }
    }

    assert!(
        raw_mismatch_count == 0,
        "CPU and GPU raw_data differ! Total mismatches: {}",
        raw_mismatch_count
    );

    assert_eq!(
        cpu_image.metadata, gpu_image.metadata,
        "Metadata mismatch between CPU and GPU output!"
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
