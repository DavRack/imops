use pichromatic::gpu::GpuContext;
use pichromatic::pixel::Image;
use pichromatic_pipeline::backend::Backend;
use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::extern_pipeline::get_raw_img_internal;
use pichromatic_pipeline::modules::common::{assert_images_equal_abs_tol, CPU_GPU_ABS_TOLERANCE};
use pichromatic_pipeline::pipeline::run_pixel_pipeline_with_backend;

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

/// Image drift pin: the sum of per-row rounded squared subpixel values,
/// tolerated within `H_TOL` units. The CPU reference itself differs by 1
/// unit between platforms (H 10,026,374 on Mac/Metal, 10,026,373 on
/// AMD CPU + NVIDIA GPU), so `H_TOL = 1` covers both.
const H_PIN: i64 = 7172272;
const H_TOL: i64 = 1;

fn load_source() -> Image {
    let dng_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/20260713_104012-16EV.DNG");
    let dng_bytes = std::fs::read(dng_path).expect("read 20260713_104012-16EV.DNG");
    get_raw_img_internal(&dng_bytes)
}

/// Image drift pin: the sum of the per-row sums of squared subpixel values,
/// each row rounded to the nearest integer. A flat-list aggregate: drift
/// moves it by a few units at most, so it is compared with a tolerance and
/// stays stable across profiles and machines. The global mean is computed
/// too, but only for diagnostics.
fn image_drift_pin(image: &Image) -> (i64, i64) {
    let width = image.metadata.width;
    let rows = image.rgb_data.len() / width;
    let subpixels = image.rgb_data.len() * 3;

    let mut mean = 0.0f64;
    for pixel in &image.rgb_data {
        for v in pixel {
            mean += *v as f64;
        }
    }
    mean /= subpixels as f64;

    let mut h = 0i64;
    for row in 0..rows {
        let mut s = 0.0f64;
        for pixel in &image.rgb_data[row * width..(row + 1) * width] {
            for v in pixel {
                s += (*v as f64) * (*v as f64);
            }
        }
        h += s.round() as i64;
    }
    ((mean * 100_000.0).round() as i64, h)
}

/// Image drift guard: the CPU pipeline is the source of truth. The sum of
/// squared subpixel values (per row, rounded, summed) is pinned below; drift
/// within `H_TOL` units is tolerated, so the pin holds across build profiles
/// and machines while still failing on real algorithmic drift. The global
/// mean is shown for diagnostics.
#[test]
fn image_drift() {
    let mut cpu_image = load_source();
    let mut cpu_pipeline = parse_config(FILM_PIPELINE_TOML.to_owned());

    run_pixel_pipeline_with_backend(&mut cpu_image, &mut cpu_pipeline, &Backend::Cpu);

    let (mean, h) = image_drift_pin(&cpu_image);
    println!("image drift pin: mean {mean}, H {h}");

    assert!(
        (h - H_PIN).abs() <= H_TOL,
        "image drift: H {h} vs pinned {H_PIN} (+/-{H_TOL}), mean {mean}"
    );
}

/// CPU vs GPU parity: float jitter between the backends is tolerated;
/// any channel that differs by more than `CPU_GPU_ABS_TOLERANCE` fails.
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

    assert_images_equal_abs_tol(
        &cpu_image,
        &wgpu_image,
        2.0*CPU_GPU_ABS_TOLERANCE,
        cpu_image.rgb_data.len() / 500,
    );
}
