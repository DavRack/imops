use pichromatic::gpu::GpuContext;
use pichromatic::pixel::Image;
use pichromatic_pipeline::backend::Backend;
use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::extern_pipeline::get_raw_img_internal;
use pichromatic_pipeline::pipeline::run_pixel_pipeline_with_backend;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const FILM_PIPELINE_TOML: &str = r#"
[[pipeline_modules]]
name = "Demosaic"
algorithm = "markesteijn"

[[pipeline_modules]]
name = "Vignette"
strength = 1.0

[[pipeline_modules]]
name = "CFACoeffs"

[[pipeline_modules]]
name = "LumaGuidedChromaDenoise"
radius = 2
epsilon = 0.01

[[pipeline_modules]]
name = "HighlightReconstruction"

[[pipeline_modules]]
name = "BaselineExposureCompensation"

[[pipeline_modules]]
name = "Exp"
ev = 0.0

[[pipeline_modules]]
name = "CST"
target_color_space = "AcesCg"

[[pipeline_modules]]
name = "Film"
stock = "Ektar100"
film_format = "Film35mm"
seed = 1
output = "PositiveLinear"

[[pipeline_modules]]
name = "SigmoidToneMap"

[[pipeline_modules]]
name = "CST"
target_color_space = "Srgb"

[[pipeline_modules]]
name = "Rotation"
angle = "auto"
"#;

const EXPECTED_IMAGE_HASH: u64 = 0x6a9b_fe36_92aa_6831;

fn image_hash(image: &Image) -> u64 {
    let mut hasher = DefaultHasher::new();
    Hash::hash(image, &mut hasher);
    hasher.finish()
}

#[test]
fn maya_film_pipeline_cpu_and_wgpu_have_stable_whole_image_hash() {
    let dng_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/maya.dng");
    let dng_bytes = std::fs::read(dng_path).expect("read maya.dng");
    let source = get_raw_img_internal(&dng_bytes);

    let gpu_context =
        GpuContext::try_new_sync().expect("WGPU is required for the explicit CPU/WGPU parity test");
    let mut cpu_image = source.clone();
    let mut wgpu_image = source;
    let mut cpu_pipeline = parse_config(FILM_PIPELINE_TOML.to_owned());
    let mut wgpu_pipeline = parse_config(FILM_PIPELINE_TOML.to_owned());

    run_pixel_pipeline_with_backend(&mut cpu_image, &mut cpu_pipeline, &Backend::Cpu);
    run_pixel_pipeline_with_backend(
        &mut wgpu_image,
        &mut wgpu_pipeline,
        &Backend::Wgpu(gpu_context),
    );

    let cpu_hash = image_hash(&cpu_image);
    let wgpu_hash = image_hash(&wgpu_image);
    println!("CPU whole-image hash: {cpu_hash:016x}");
    println!("WGPU whole-image hash: {wgpu_hash:016x}");

    assert!(
        cpu_image.bitwise_eq(&wgpu_image),
        "explicit CPU and WGPU images differ bit-for-bit"
    );
    assert_eq!(
        cpu_hash, wgpu_hash,
        "explicit CPU and WGPU whole-image hashes differ"
    );
    assert_eq!(
        cpu_hash, EXPECTED_IMAGE_HASH,
        "stable whole-image hash changed"
    );
}
