use pichromatic::gpu::GpuContext;
use pichromatic::pixel::Image;
use pichromatic_pipeline::backend::Backend;
use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::drift::ordered_ulp;
use pichromatic_pipeline::extern_pipeline::get_raw_img_internal;
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
const H_PIN: i64 = 10030164;
const H_TOL: i64 = 1;

/// Per-channel CPU-vs-GPU tolerance, relative to the larger of the two
/// compared values (floored at 1.0 so near-zero values keep a tight absolute
/// gate): `tol = K * EPSILON * max(|cpu|, |gpu|, 1.0)` = K ULPs at the
/// value's own magnitude (same rule as `CPU_GPU_ABS_TOLERANCE`).
/// K = 512 (6.10e-5): covers the shadow-jitter floor on both platforms
/// (measured 2.264e-5 on Mac/Metal, 3.628e-5 on AMD CPU + NVIDIA GPU)
/// with comfortable margin, without chasing bit-exact GPU math across
/// backends (CPU reference itself differs per platform, see image_drift).
const GPU_VS_CPU_TOLERANCE: f32 = 512.0 * f32::EPSILON;

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
    let mut max_rel = 0.0f32;
    let mut max_ulp = 0u32;
    let mut violations = 0usize;
    for (cp, gp) in cpu_image.rgb_data[..n].iter().zip(&wgpu_image.rgb_data[..n]) {
        for (ca, ga) in cp.iter().zip(gp) {
            let d = (ca - ga).abs();
            let peak = ca.abs().max(ga.abs());
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(if peak > 0.0 { d / peak } else { d });
            max_ulp = max_ulp.max(ordered_ulp(*ca).abs_diff(ordered_ulp(*ga)));
            let tol = GPU_VS_CPU_TOLERANCE * peak.max(1.0);
            if d > tol {
                violations += 1;
            }
        }
    }
    println!(
        "CPU vs WGPU max diff: abs {max_abs:.3e}, rel {max_rel:.3e}, {max_ulp} ULPs (tolerance {GPU_VS_CPU_TOLERANCE:.1e})"
    );

    assert_eq!(
        violations, 0,
        "CPU and WGPU images differ beyond tolerance: {violations} channels exceed {GPU_VS_CPU_TOLERANCE:.1e} (max abs diff {max_abs:.3e}, max rel {max_rel:.3e}, max {max_ulp} ULPs)"
    );
}
