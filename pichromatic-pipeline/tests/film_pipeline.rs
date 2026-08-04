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

/// Image drift pins, captured from a `--release` run (identical in debug on
/// this machine). `MEAN_PIN` is the global mean of all subpixels in 1e-5
/// units, tolerated +/-2; `H_PIN` is the sum of rounded per-row squared
/// deviations, tolerated within `H_TOL` (measured cross-profile drift: 0).
const MEAN_PIN: i64 = 47031;
const H_PIN: i64 = 3640005;
const H_TOL: i64 = 512;

/// Per-channel CPU-vs-GPU tolerance, relative to the larger of the two
/// compared values (floored at 1.0 so near-zero values keep a tight absolute
/// gate): `tol = K * EPSILON * max(|cpu|, |gpu|, 1.0)` = K ULPs at the
/// value's own magnitude (same rule as `CPU_GPU_ABS_TOLERANCE`).
/// Calibrated "barely" against the dark-pixel drift floor: the worst
/// channels are shadows (values 0.02-0.1) where the film curve amplifies
/// jitter to ~0.12% relative (~12000 ULPs, abs 2.264e-5, measured on this
/// machine; 1.948e-5 on AMD/NVIDIA). K = 200 (2.38e-5) passes both with ~5%
/// margin; K = 128 (1.53e-5) is below the floor and fails both.
const GPU_VS_CPU_TOLERANCE: f32 = 200.0 * f32::EPSILON;

fn load_source() -> Image {
    let dng_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/20260713_104012-16EV.DNG");
    let dng_bytes = std::fs::read(dng_path).expect("read 20260713_104012-16EV.DNG");
    get_raw_img_internal(&dng_bytes)
}

/// Image drift pin: global mean over all subpixels (in 1e-5 units) plus the
/// sum of the per-row sums of squared deviations from that mean, each row
/// rounded to the nearest integer. A flat-list aggregate: drift moves it by
/// a few units at most, so it is compared with a tolerance and stays stable
/// across profiles and machines.
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
                let d = *v as f64 - mean;
                s += d * d;
            }
        }
        h += s.round() as i64;
    }
    ((mean * 100_000.0).round() as i64, h)
}

/// Image drift guard: the CPU pipeline is the source of truth. Its global
/// mean and the sum of per-row squared deviations are pinned below; drift
/// within `H_TOL` units and +/-2 in the mean is tolerated, so the pin holds
/// across build profiles and machines while still failing on real
/// algorithmic drift.
#[test]
fn image_drift() {
    let mut cpu_image = load_source();
    let mut cpu_pipeline = parse_config(FILM_PIPELINE_TOML.to_owned());

    run_pixel_pipeline_with_backend(&mut cpu_image, &mut cpu_pipeline, &Backend::Cpu);

    let (mean, h) = image_drift_pin(&cpu_image);
    println!("image drift pin: mean {mean}, H {h}");

    assert!(
        (mean - MEAN_PIN).abs() <= 2 && (h - H_PIN).abs() <= H_TOL,
        "image drift: mean {mean} vs pinned {MEAN_PIN} (+/-2), H {h} vs pinned {H_PIN} (+/-{H_TOL})"
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
