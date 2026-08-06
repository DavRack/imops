use pichromatic::gpu::GpuContext;
use pichromatic::pixel::Image;
use pichromatic_pipeline::backend::{Backend, PipelineImage};
use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::drift::ordered_ulp;
use pichromatic_pipeline::extern_pipeline::get_raw_img_internal;

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

fn stats(cpu: &Image, gpu: &Image, label: &str) {
    let n = cpu.rgb_data.len().min(gpu.rgb_data.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_ulp = 0u32;
    let mut violations = 0usize;
    let mut worst = (0usize, 0usize, 0.0f32, 0.0f32, 0.0f32);
    let mut peak_max = 0.0f32;
    for (i, (cp, gp)) in cpu.rgb_data[..n].iter().zip(&gpu.rgb_data[..n]).enumerate() {
        for (c, (ca, ga)) in cp.iter().zip(gp).enumerate() {
            let d = (ca - ga).abs();
            let peak = ca.abs().max(ga.abs());
            peak_max = peak_max.max(peak);
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(if peak > 0.0 { d / peak } else { d });
            let u = ordered_ulp(*ca).abs_diff(ordered_ulp(*ga));
            max_ulp = max_ulp.max(u);
            let tol = 200.0 * f32::EPSILON * peak.max(1.0);
            if d > tol {
                violations += 1;
                if worst.4 == 0.0 || d > worst.4 {
                    worst = (i, c, *ca, *ga, d);
                }
            }
        }
    }
    println!(
        "{label:>45}: max_abs {:9.3e}  max_rel {:9.3e}  max_ulp {:7}  viol {:7}  (peak {:.3e}, worst px {} ch {} cpu {:.5e} gpu {:.5e} diff {:.3e})",
        max_abs, max_rel, max_ulp, violations, peak_max, worst.0, worst.1, worst.2, worst.3, worst.4
    );
}

#[test]
fn zz_stage_diag() {
    let dng_path =
        concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/20260713_104012-16EV.DNG");
    let dng_bytes = std::fs::read(dng_path).expect("read DNG");
    let source = get_raw_img_internal(&dng_bytes);
    let ctx = GpuContext::try_new_sync().expect("wgpu");
    let mut pipeline = parse_config(FILM_PIPELINE_TOML.to_owned());
    let names: Vec<String> = pipeline
        .pipeline_modules
        .iter()
        .map(|m| m.schema().name)
        .collect();
    println!("modules: {:?}", names);

    // ── Phase 1: cumulative prefixes (pipeline-parallel CPU vs GPU) ──
    println!("\n-- cumulative prefixes --");
    for i in 1..=pipeline.pipeline_modules.len() {
        let mut cpu_pipe = PipelineImage::new_cpu(source.clone());
        let mut gpu_pipe = PipelineImage::new_gpu(&ctx, &source);

        for m in pipeline.pipeline_modules.iter().take(i) {
            m.process(&Backend::Cpu, &mut cpu_pipe);
            m.process(&Backend::Wgpu(ctx.clone()), &mut gpu_pipe);
        }
        let gpu_img = gpu_pipe.to_cpu(Some(&ctx));
        let label = names[..i].join("+");
        if let PipelineImage::Cpu(cpu_img) = &cpu_pipe {
            stats(cpu_img, &gpu_img, &label);
        }
        gpu_pipe.recycle_gpu_buffer(&ctx);
        cpu_pipe.recycle_gpu_buffer(&ctx);
    }

    // ── Phase 2: per-module isolation (identical input fed to CPU & GPU) ──
    println!("\n-- per-module isolation (same input to both backends) --");
    let mut cpu_img = source.clone();
    for i in 0..pipeline.pipeline_modules.len() {
        let m = &pipeline.pipeline_modules[i];
        let input = cpu_img.clone();
        let mut cpu_pipe = PipelineImage::Cpu(cpu_img.clone());
        m.process(&Backend::Cpu, &mut cpu_pipe);

        let mut gpu_pipe = PipelineImage::new_gpu(&ctx, &input);
        m.process(&Backend::Wgpu(ctx.clone()), &mut gpu_pipe);
        let gpu_img = gpu_pipe.to_cpu(Some(&ctx));
        if let PipelineImage::Cpu(cpu_out) = &cpu_pipe {
            cpu_img = cpu_out.clone();
            stats(cpu_out, &gpu_img, &names[i]);
        }
        gpu_pipe.recycle_gpu_buffer(&ctx);
    }
}
