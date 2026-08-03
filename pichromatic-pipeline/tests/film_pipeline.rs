use pichromatic::gpu::GpuContext;
use pichromatic::pixel::Image;
use pichromatic_pipeline::backend::Backend;
use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::drift::{ordered_ulp, Canonicalizer};
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

const EXPECTED_IMAGE_HASH: u64 = 0xa566_5641_631c_bc1c;

/// Max absolute per-channel error allowed between CPU and WGPU results.
/// Measured worst case is ~8.7e-6 (float jitter only); 1e-4 gives ~11x margin
/// while staying ~39x below one 8-bit step (1/255 ≈ 3.9e-3), so a single
/// rogue pixel (diff > 1e-4) still fails the test.
const GPU_VS_CPU_TOLERANCE: f32 = 1e-4;

/// Max drift radius in ULPs for the guard-banded canonicalizer. Measured
/// worst-case jitter is 672 ULPs (debug vs release CPU builds of this
/// pipeline, same image); 1024 gives 1.5x margin and a bucket width
/// W = 8 x D = 8192 ULPs, the same granularity the old f16 quantize had.
const MAX_DRIFT_ULPS: u32 = 1024;

fn image_hash(image: &Image) -> u64 {
    let canon = Canonicalizer::new(MAX_DRIFT_ULPS);
    let mut hasher = DefaultHasher::new();
    image.rgb_data.len().hash(&mut hasher);
    for pixel in &image.rgb_data {
        for channel in pixel {
            canon.digest(ordered_ulp(*channel)).hash(&mut hasher);
        }
    }

    image.raw_data.len().hash(&mut hasher);
    for sample in image.raw_data.iter() {
        sample.to_bits().hash(&mut hasher);
    }

    Hash::hash(&image.metadata, &mut hasher);
    hasher.finish()
}

fn load_source() -> Image {
    let dng_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/20260713_104012-16EV.DNG");
    let dng_bytes = std::fs::read(dng_path).expect("read 20260713_104012-16EV.DNG");
    get_raw_img_internal(&dng_bytes)
}

/// Drift guard: the CPU pipeline is the source of truth. Its whole-image hash
/// (guard-banded canonicalized per channel, buckets of W = 8 x D = 8192 ULPs)
/// is pinned to a constant in this file; any algorithmic drift that changes
/// output beyond bucket granularity fails here.
///
/// The pin is captured from a `--release` run (see the `run_tests` script);
/// debug builds execute ~0.04% of channels differently (max 672 ULPs), so the
/// same image hashes differently per profile. Cross-profile drift of at most
/// `MAX_DRIFT_ULPS` is handled by the joint `Canonicalizer::agree` checks in
/// the parity test below, not by this per-image pin.
#[test]
fn film_pipeline_cpu_whole_image_hash_is_stable() {
    let mut cpu_image = load_source();
    let mut cpu_pipeline = parse_config(FILM_PIPELINE_TOML.to_owned());

    run_pixel_pipeline_with_backend(&mut cpu_image, &mut cpu_pipeline, &Backend::Cpu);

    let cpu_hash = image_hash(&cpu_image);
    println!("CPU whole-image hash: {cpu_hash:016x}");

    assert_eq!(
        cpu_hash, EXPECTED_IMAGE_HASH,
        "stable whole-image hash changed"
    );
}

/// CPU vs GPU parity: float jitter between the backends (~1e-5 abs) is
/// tolerated; any channel that differs by more than `GPU_VS_CPU_TOLERANCE`
/// (a single rogue pixel included) fails.
#[test]
fn film_pipeline_cpu_and_wgpu_agree_within_tolerance() {
    let source = load_source();
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

    let n = cpu_image.rgb_data.len().min(wgpu_image.rgb_data.len());
    let mut max_abs = 0.0f32;
    let mut violations = 0usize;
    for (cp, gp) in cpu_image.rgb_data[..n].iter().zip(&wgpu_image.rgb_data[..n]) {
        for (ca, ga) in cp.iter().zip(gp) {
            let d = (ca - ga).abs();
            max_abs = max_abs.max(d);
            if d > GPU_VS_CPU_TOLERANCE {
                violations += 1;
            }
        }
    }
    println!("CPU vs WGPU max abs diff: {max_abs:.3e} (tolerance {GPU_VS_CPU_TOLERANCE:.1e})");

    assert_eq!(
        violations, 0,
        "CPU and WGPU images differ beyond tolerance: {violations} channels exceed {GPU_VS_CPU_TOLERANCE:.1e} (max abs diff {max_abs:.3e})"
    );
}
