//! Repeatable timing harness for the GPU film simulation.
//!
//! The pixel pipeline as a whole is a poor benchmark for this stage: decode,
//! demosaic and denoise dominate the wall clock, and a single run measures a GPU
//! that has not yet clocked up. This runs the film stage alone, back to back, and
//! reports best, warmed median and p95.
//!
//! ```text
//! cargo run --release -p pichromatic --example film_bench -- [megapixels] [iterations] [stock]
//! PICHROMATIC_FILM_TIMING=1 cargo run --release -p pichromatic --example film_bench
//! ```

use std::time::Instant;

use pichromatic::film::gpu::{film_roi_memory_breakdown, film_workspace_requested_bytes};
use pichromatic::film::stock::StockId;
use pichromatic::film::types::FilmFormat;
use pichromatic::film::{FilmOutput, FilmParams};
use pichromatic::gpu::GpuContext;
use pichromatic::image::ImageMetadata;

/// Streaming-copy kernels used to establish what the device can actually sustain,
/// so stage timings can be read as "fraction of achievable bandwidth".
const COPY_SCALAR: &str = r#"
struct U { n:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    dst[i] = src[i] * 1.0001;
}
"#;

const COPY_VEC4: &str = r#"
struct U { n:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    dst[i] = src[i] * 1.0001;
}
"#;

/// Same shape as the per-pixel planar stages: one thread touches all E planes.
const COPY_PLANES: &str = r#"
struct U { n:u32, num_emul:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    for (var e = 0u; e < u.num_emul; e = e + 1u) {
        dst[e * u.n + i] = src[e * u.n + i] * 1.0001;
    }
}
"#;

fn plane_probe(ctx: &GpuContext, n: usize, num_emul: usize) {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct U {
        n: u32,
        num_emul: u32,
        p: [u32; 2],
    }
    let src = ctx.create_f32_buffer(n * num_emul, "bench_planes_src");
    let dst = ctx.create_f32_buffer(n * num_emul, "bench_planes_dst");
    let u = U {
        n: n as u32,
        num_emul: num_emul as u32,
        p: [0; 2],
    };
    let bytes = (n * num_emul * 4 * 2) as f64;
    let mut best = f64::INFINITY;
    for _ in 0..8 {
        ctx.poll_wait();
        let start = Instant::now();
        ctx.dispatch_compute_shader_multi(
            "bench_copy_planes",
            COPY_PLANES,
            &[&src, &dst],
            bytemuck::bytes_of(&u),
            ((n as u32) + 255) / 256,
        );
        ctx.poll_wait();
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!(
        "  copy {num_emul} planes: {:6.2} ms  → {:6.1} GB/s",
        best * 1e3,
        bytes / best / 1e9
    );
}

/// Cost of one separable-blur half-pass at a given radius, against the 2·n floats
/// it must move. Anything well above the streaming figure is redundant-read cost.
fn blur_probe(ctx: &GpuContext, width: usize, height: usize, radii: &[u32]) {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct BlurU {
        width: u32,
        height: u32,
        n: u32,
        radius: u32,
        src_off: u32,
        dst_off: u32,
        a: f32,
        b: f32,
        num_planes: u32,
        _p: [u32; 3],
    }
    let n = width * height;
    let src = ctx.create_f32_buffer(n, "bench_blur_src");
    let dst = ctx.create_f32_buffer(n, "bench_blur_dst");
    let bytes = (n * 4 * 2) as f64;
    for &radius in radii {
        let kernel = vec![1.0f32 / (2 * radius + 1) as f32; (2 * radius + 1) as usize];
        let kbuf = ctx.create_f32_buffer_init(&kernel, "bench_blur_k");
        let u = BlurU {
            width: width as u32,
            height: height as u32,
            n: n as u32,
            radius,
            src_off: 0,
            dst_off: 0,
            a: 0.0,
            b: 0.0,
            num_planes: 1,
            _p: [0; 3],
        };
        let wg = ((n as u32) + 255) / 256;
        let variants: [(&str, &str, u32, u32); 4] = [
            (
                "h  tiled",
                pichromatic::film::gpu::shaders::BLUR_H_TILED,
                ((width as u32) + 255) / 256,
                height as u32,
            ),
            ("h untiled", pichromatic::film::gpu::shaders::BLUR_H, wg, 0),
            (
                "v  tiled",
                pichromatic::film::gpu::shaders::BLUR_V_TILED,
                width as u32,
                ((height as u32) + 255) / 256,
            ),
            ("v untiled", pichromatic::film::gpu::shaders::BLUR_V, wg, 0),
        ];
        for (label, source, gx, gy) in variants {
            let mut best = f64::INFINITY;
            for _ in 0..6 {
                ctx.poll_wait();
                let start = Instant::now();
                ctx.dispatch_compute_passes(
                    label,
                    &[pichromatic::gpu::ComputePassDesc {
                        label,
                        wgsl_source: source,
                        storage_buffers: &[&src, &dst, &kbuf],
                        uniform_bytes: bytemuck::bytes_of(&u),
                        workgroups_x: gx,
                        workgroups_y: gy,
                    }],
                );
                ctx.poll_wait();
                best = best.min(start.elapsed().as_secs_f64());
            }
            println!(
                "  blur r={radius:<3} {label}: {:6.2} ms → {:6.1} GB/s effective",
                best * 1e3,
                bytes / best / 1e9
            );
        }
    }
}

fn bandwidth_probe(ctx: &GpuContext, floats: usize) {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct U {
        n: u32,
        p: [u32; 3],
    }
    let src = ctx.create_f32_buffer(floats, "bench_src");
    let dst = ctx.create_f32_buffer(floats, "bench_dst");
    let bytes = (floats * 4 * 2) as f64; // one read + one write
    for (label, shader, lanes) in [
        ("scalar f32", COPY_SCALAR, floats),
        ("vec4  f32", COPY_VEC4, floats / 4),
    ] {
        let u = U {
            n: lanes as u32,
            p: [0; 3],
        };
        let mut best = f64::INFINITY;
        for _ in 0..8 {
            ctx.poll_wait();
            let start = Instant::now();
            ctx.dispatch_compute_shader_multi(
                label,
                shader,
                &[&src, &dst],
                bytemuck::bytes_of(&u),
                ((lanes as u32) + 255) / 256,
            );
            ctx.poll_wait();
            best = best.min(start.elapsed().as_secs_f64());
        }
        println!(
            "  copy {label}: {:6.2} ms  → {:6.1} GB/s",
            best * 1e3,
            bytes / best / 1e9
        );
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let megapixels: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(60.0);
    let iterations: usize = args
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10)
        .max(2);
    let stock = match args.next().as_deref() {
        None | Some("Ektar100") => StockId::Ektar100,
        Some("Portra400") => StockId::Portra400,
        Some("FujiPro400H") => StockId::FujiPro400H,
        Some("EktachromeE100") => StockId::EktachromeE100,
        Some("TriX400") => StockId::TriX400,
        Some("ColorNeg200") => StockId::ColorNeg200,
        Some(other) => panic!("unknown film stock {other}"),
    };

    // 3:2 frame with the requested pixel count.
    let height = ((megapixels * 1.0e6 / 1.5).sqrt()).round() as usize;
    let width = (height * 3) / 2;
    let n = width * height;

    let ctx = GpuContext::new_sync();

    // Deterministic mid-key content with enough local variation that no stage can
    // short-circuit on flat input.
    let mut rgb = vec![[0.0f32; 4]; n];
    for (i, px) in rgb.iter_mut().enumerate() {
        let x = (i % width) as f32 / width as f32;
        let y = (i / width) as f32 / height as f32;
        let t = (x * 37.0).sin() * (y * 23.0).cos();
        *px = [
            0.18 * (1.0 + 0.8 * t),
            0.18 * (1.0 + 0.6 * (t + 0.3).sin()),
            0.18 * (1.0 + 0.7 * (t - 0.4).cos()),
            1.0,
        ];
    }
    let flat: Vec<f32> = rgb.iter().flat_map(|p| p.iter().copied()).collect();

    let params = FilmParams {
        stock,
        film_format: FilmFormat::Film35mm,
        seed: 1,
        output: FilmOutput::PositiveLinear,
        enable_halation: true,
        compensate_box_speed: true,
    };
    let meta = ImageMetadata {
        width,
        height,
        color_space: Some(pichromatic::cst::ColorSpaceTag::AcesCg),
        ..Default::default()
    };

    let gpu_buf = ctx.create_output_buffer(width, height);
    println!(
        "film_bench: {width}×{height} ({:.1} MP), stock {stock:?}, {iterations} iterations",
        n as f64 / 1.0e6
    );

    println!("device streaming bandwidth (same footprint as one emulsion plane):");
    bandwidth_probe(&ctx, n);
    plane_probe(&ctx, n, 6);
    blur_probe(&ctx, width, height, &[3, 6, 13, 56]);

    let mut durations = Vec::with_capacity(iterations);
    for iter in 0..iterations {
        ctx.queue
            .write_buffer(&gpu_buf.buffer, 0, bytemuck::cast_slice(&flat));
        ctx.poll_wait();
        let start = Instant::now();
        pollster::block_on(pichromatic::film::process_gpu(
            &ctx, &gpu_buf, &meta, &params,
        ))
        .expect("film process_gpu");
        ctx.poll_wait();
        let ms = start.elapsed().as_secs_f64() * 1.0e3;
        durations.push(ms);
        println!("  iter {iter:>2}: {ms:8.2} ms");
    }

    let best = durations.iter().cloned().fold(f64::INFINITY, f64::min);
    let warmed: Vec<f64> = durations[1..].to_vec();
    let mut sorted = warmed.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        let mid = sorted.len() / 2;
        (sorted[mid - 1] + sorted[mid]) * 0.5
    };
    let p95_idx = ((sorted.len() as f64) * 0.95).ceil() as usize - 1;
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];

    let num_emul = params
        .stock
        .load()
        .expect("film stock")
        .emulsion_layers()
        .count();
    let roi_breakdown = film_roi_memory_breakdown(1094, 1094, width, height, num_emul).ok();
    let ws_bytes = film_workspace_requested_bytes(width, height, num_emul);

    println!(
        "best iteration: {best:.2} ms  ({:.1} MP/s, Film-only synthetic)",
        n as f64 / 1.0e6 / (best / 1.0e3)
    );
    println!(
        "warmed (skip 1): median {median:.2} ms  p95 {p95:.2} ms  ({} iterations)",
        warmed.len()
    );
    if let Some(b) = roi_breakdown {
        println!(
            "film ROI workspace allocated: {:.1} MiB  (arena: {:.1} MiB, spill: {:.1} MiB, output: {:.1} MiB)",
            b.total_film_owned_bytes as f64 / (1024.0 * 1024.0),
            b.arena_bytes as f64 / (1024.0 * 1024.0),
            b.grain_spill_bytes as f64 / (1024.0 * 1024.0),
            b.output_bytes as f64 / (1024.0 * 1024.0),
        );
    } else {
        println!(
            "film workspace requested: {:.1} MiB  ({num_emul} emulsions, {} planes)",
            ws_bytes as f64 / (1024.0 * 1024.0),
            4 * num_emul + 3
        );
    }
}
