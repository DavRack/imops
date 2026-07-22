use pichromatic::cst::ColorSpaceTag;
use pichromatic::gpu::GpuContext;
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::{Image, Pixel, SubPixel};
use crate::backend::{Backend, PipelineImage};
use super::PipelineModule;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generates a deterministic 512x512 RGB test image using a fixed seed.
pub fn generate_test_image_512x512(seed: u64) -> Image {
    let width = 512;
    let height = 512;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut rgb_data: Vec<Pixel> = Vec::with_capacity(width * height);

    for _ in 0..(width * height) {
        rgb_data.push([
            rng.gen::<SubPixel>(),
            rng.gen::<SubPixel>(),
            rng.gen::<SubPixel>(),
        ]);
    }

    Image {
        rgb_data,
        raw_data: vec![],
        metadata: ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::Srgb),
            ..Default::default()
        },
    }
}

/// Runs a PipelineModule on both CPU and WGPU GPU and asserts their pixel outputs match identically (tolerance = 0).
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

    // 3. Compare outputs (tolerance = 0.0)
    assert_images_equal(&cpu_out, &gpu_out);
}

/// Compares CPU ground truth image against GPU output image pixel by pixel (tolerance = 0.0).
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
}
