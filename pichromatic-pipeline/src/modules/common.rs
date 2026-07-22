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

/// Compares CPU ground truth image against GPU output image for exact equality (rgb_data, raw_data, and metadata).
pub fn assert_images_equal(cpu_image: &Image, gpu_image: &Image) {
    assert_eq!(
        cpu_image, gpu_image,
        "CPU and GPU images are not exactly equal!"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_test_image_512x512() {
        let img1 = generate_test_image_512x512(42);
        assert_eq!(img1.rgb_data.len(), 512 * 512);
        assert_eq!(img1.raw_data.len(), 512 * 512);

        let img2 = generate_test_image_512x512(42);
        assert_eq!(img1.rgb_data, img2.rgb_data);
        assert_eq!(img1.raw_data, img2.raw_data);
        assert_eq!(img1.metadata, img2.metadata);
    }

    #[test]
    fn test_assert_images_equal_pass() {
        let img1 = generate_test_image_512x512(42);
        let img2 = img1.clone();
        assert_images_equal(&img1, &img2);
    }

    #[test]
    #[should_panic]
    fn test_assert_images_equal_rgb_mismatch() {
        let img1 = generate_test_image_512x512(42);
        let mut img2 = img1.clone();
        img2.rgb_data[0][0] += 0.1;
        assert_images_equal(&img1, &img2);
    }

    #[test]
    #[should_panic]
    fn test_assert_images_equal_raw_len_mismatch() {
        let img1 = generate_test_image_512x512(42);
        let mut img2 = img1.clone();
        img2.raw_data.pop();
        assert_images_equal(&img1, &img2);
    }

    #[test]
    #[should_panic]
    fn test_assert_images_equal_raw_value_mismatch() {
        let img1 = generate_test_image_512x512(42);
        let mut img2 = img1.clone();
        img2.raw_data[0] += 0.1;
        assert_images_equal(&img1, &img2);
    }

    #[test]
    #[should_panic]
    fn test_assert_images_equal_metadata_mismatch() {
        let img1 = generate_test_image_512x512(42);
        let mut img2 = img1.clone();
        img2.metadata.width = 256;
        assert_images_equal(&img1, &img2);
    }

    #[test]
    #[should_panic]
    fn test_assert_images_equal_nan_in_rgb() {
        let img1 = generate_test_image_512x512(42);
        let mut img2 = img1.clone();
        img2.rgb_data[0][0] = f32::NAN;
        assert_images_equal(&img1, &img2);
    }

    #[test]
    #[should_panic]
    fn test_assert_images_equal_nan_in_raw() {
        let img1 = generate_test_image_512x512(42);
        let mut img2 = img1.clone();
        img2.raw_data[0] = f32::NAN;
        assert_images_equal(&img1, &img2);
    }
}
