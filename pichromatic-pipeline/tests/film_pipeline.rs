use pichromatic::gpu::GpuContext;
use pichromatic::pixel::Image;
use pichromatic_pipeline::backend::Backend;
use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::drift::{ordered_ulp, Canonicalizer};
use pichromatic_pipeline::extern_pipeline::get_raw_img_internal;
use pichromatic_pipeline::pipeline::run_pixel_pipeline_with_backend;
use sha2::{Digest, Sha256};

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

/// Pinned SHA-256 of the guard-banded canonicalized CPU output, captured
/// from a `--release` run (see `run_tests`). Debug builds execute ~0.04% of
/// channels differently (max 672 ULPs), so they hash differently and are not
/// pinned.
const RELEASE_PIN: [u8; 32] = [
    0xb9, 0x54, 0x9c, 0x28, 0x07, 0xdc, 0x80, 0x04, 0xef, 0x9a, 0x31, 0xeb, 0x62, 0x55, 0xd0, 0x15,
    0xc0, 0x17, 0x1e, 0xdf, 0x74, 0x7e, 0x64, 0x2b, 0xd8, 0x59, 0x9c, 0x17, 0xad, 0xf4, 0x81, 0x8c,
];

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

/// Max drift radius in ULPs for the guard-banded canonicalizer used by the
/// pinned hash: buckets of W = 8 x D = 2048 ULPs (~4x finer than the old
/// f16 quantize, so smaller algorithmic changes flip the pin). Measured
/// worst-case cross-profile jitter is 672 ULPs (debug vs release, same
/// image), so D = 256 is well below the drift: the pin is strictly
/// per-release-build, and any other profile or machine hashes differently.
const MAX_DRIFT_ULPS: u32 = 256;

fn load_source() -> Image {
    let dng_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/20260713_104012-16EV.DNG");
    let dng_bytes = std::fs::read(dng_path).expect("read 20260713_104012-16EV.DNG");
    get_raw_img_internal(&dng_bytes)
}

/// Guard-banded quantization + crypto hash: every channel is mapped into the
/// ordered ULP space and quantized with the `Canonicalizer` (buckets of
/// W = 8 x D = 8192 ULPs, guard-band exceptions keep the pair of adjacent
/// bucket IDs near edges), then the digest stream is hashed with SHA-256.
fn image_sha256(image: &Image) -> [u8; 32] {
    let canon = Canonicalizer::new(MAX_DRIFT_ULPS);
    let mut hasher = Sha256::new();
    hasher.update((image.rgb_data.len() as u64).to_le_bytes());
    for pixel in &image.rgb_data {
        for channel in pixel {
            hasher.update(canon.digest(ordered_ulp(*channel)).to_le_bytes());
        }
    }
    hasher.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Drift guard: the CPU pipeline is the source of truth. Its guard-banded,
/// SHA-256-hashed whole-image digest is pinned to a constant in this file;
/// any algorithmic drift that changes output beyond bucket granularity
/// (W = 8192 ULPs) fails here.
///
/// The pin is captured from a `--release` run (see the `run_tests` script);
/// debug builds execute ~0.04% of channels differently (max 672 ULPs), which
/// flips a handful of guard-band digests, so the same image hashes
/// differently per profile. Other machines/compilers may need their own pin.
#[test]
fn film_pipeline_cpu_whole_image_hash_is_stable() {
    let mut cpu_image = load_source();
    let mut cpu_pipeline = parse_config(FILM_PIPELINE_TOML.to_owned());

    run_pixel_pipeline_with_backend(&mut cpu_image, &mut cpu_pipeline, &Backend::Cpu);

    let digest = image_sha256(&cpu_image);
    println!("CPU whole-image SHA-256: {}", hex(&digest));

    assert_eq!(
        digest, RELEASE_PIN,
        "stable whole-image hash changed (run in --release and update the pin)"
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
