use rayon::prelude::*;

use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::{R_RELATIVE_LUMINANCE, G_RELATIVE_LUMINANCE, B_RELATIVE_LUMINANCE};

/// Luma-guided chroma denoising via the guided filter (He et al. 2013).
///
/// Converts the image to a luminance–chroma difference representation
/// (`C_R = R - Y`, `C_B = B - Y`), smooths each chroma plane with a
/// guided filter using the luminance as the edge-preserving guide,
/// then reconstructs the RGB output.
///
/// `radius` controls the spatial extent of the filter (typical: 2–8).
/// `epsilon` controls edge sensitivity (typical: 0.01–0.1 for linear data).
pub fn chroma_denoise(
    rgb_data: &mut Vec<[f32; 3]>,
    width: usize,
    height: usize,
    radius: usize,
    epsilon: f32,
) {
    let luma = |p: &[f32; 3]| -> f32 {
        R_RELATIVE_LUMINANCE * p[0] + G_RELATIVE_LUMINANCE * p[1] + B_RELATIVE_LUMINANCE * p[2]
    };

    let y: Vec<f32> = rgb_data.par_iter().map(luma).collect();
    let cr: Vec<f32> = rgb_data.par_iter().zip(&y).map(|(p, &yv)| p[0] - yv).collect();
    let cb: Vec<f32> = rgb_data.par_iter().zip(&y).map(|(p, &yv)| p[2] - yv).collect();

    let cr_smooth = guided_filter(&y, &cr, width, height, radius, epsilon);
    let cb_smooth = guided_filter(&y, &cb, width, height, radius, epsilon);

    let wr = R_RELATIVE_LUMINANCE;
    let wg = G_RELATIVE_LUMINANCE;
    let wb = B_RELATIVE_LUMINANCE;

    rgb_data.par_iter_mut().enumerate().for_each(|(i, p)| {
        let yv = luma(p);
        p[0] = yv + cr_smooth[i];
        p[1] = yv - (wr / wg) * cr_smooth[i] - (wb / wg) * cb_smooth[i];
        p[2] = yv + cb_smooth[i];
    });
}

/// Guided filter (single-channel).
fn guided_filter(
    guide: &[f32],
    source: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mean_g = box_filter(guide, width, height, radius);
    let mean_p = box_filter(source, width, height, radius);

    let guide_sq: Vec<f32> = guide.par_iter().map(|&v| v * v).collect();
    let guide_p: Vec<f32> = guide.par_iter().zip(source).map(|(&g, &s)| g * s).collect();

    let mean_gg = box_filter(&guide_sq, width, height, radius);
    let mean_gp = box_filter(&guide_p, width, height, radius);

    let var_g: Vec<f32> = mean_gg
        .par_iter()
        .zip(&mean_g)
        .map(|(&gg, &mg)| gg - mg * mg)
        .collect();
    let cov_gp: Vec<f32> = mean_gp
        .par_iter()
        .zip(&mean_g)
        .zip(&mean_p)
        .map(|((&gp, &mg), &mp)| gp - mg * mp)
        .collect();

    let a: Vec<f32> = var_g
        .par_iter()
        .zip(&cov_gp)
        .map(|(&v, &c)| c / (v + epsilon))
        .collect();
    let b: Vec<f32> = mean_p
        .par_iter()
        .zip(&a)
        .zip(&mean_g)
        .map(|((&mp, &a_val), &mg)| mp - a_val * mg)
        .collect();

    let mean_a = box_filter(&a, width, height, radius);
    let mean_b = box_filter(&b, width, height, radius);

    mean_a
        .par_iter()
        .zip(&mean_b)
        .zip(guide)
        .map(|((&ma, &mb), &gv)| ma * gv + mb)
        .collect()
}

/// Separable box filter (axis-aligned sliding window) with correct border
/// handling.  Each pixel is the mean over the largest window that fits inside
/// the image bounds at that location.
fn box_filter(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut tmp = vec![0.0; data.len()];

    // Horizontal pass
    for y in 0..height {
        let row = y * width;
        let mut sum = 0.0;
        for dx in 0..=radius.min(width - 1) {
            sum += data[row + dx];
        }
        tmp[row] = sum;
        for x in 1..width {
            if x > radius {
                sum -= data[row + x - radius - 1];
            }
            if x + radius < width {
                sum += data[row + x + radius];
            }
            tmp[row + x] = sum;
        }
    }

    let mut result = vec![0.0; data.len()];

    // Vertical pass
    for x in 0..width {
        let mut sum = 0.0;
        for dy in 0..=radius.min(height - 1) {
            sum += tmp[dy * width + x];
        }
        result[x] = sum;
        for y in 1..height {
            if y > radius {
                sum -= tmp[(y - radius - 1) * width + x];
            }
            if y + radius < height {
                sum += tmp[(y + radius) * width + x];
            }
            result[y * width + x] = sum;
        }
    }

    // Normalize by the actual window size at each position
    for y in 0..height {
        let cy = (radius.min(y) + radius.min(height - 1 - y) + 1) as f32;
        for x in 0..width {
            let cx = (radius.min(x) + radius.min(width - 1 - x) + 1) as f32;
            result[y * width + x] /= cx * cy;
        }
    }

    result
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ChromaDenoiseParamsGpu {
    width: u32,
    height: u32,
    radius: u32,
    epsilon: f32,
    wr: f32,
    wg: f32,
    wb: f32,
    _pad: f32,
}

/// Native GPU luma-guided chroma denoise (guided filter). No CPU fallback.
pub fn chroma_denoise_gpu(
    ctx: &GpuContext,
    storage_buffer: &GpuImageBuffer,
    radius: usize,
    epsilon: f32,
) {
    let width = storage_buffer.width;
    let height = storage_buffer.height;
    let n = width * height;
    if n == 0 {
        return;
    }

    let params = ChromaDenoiseParamsGpu {
        width: width as u32,
        height: height as u32,
        radius: radius as u32,
        epsilon,
        wr: R_RELATIVE_LUMINANCE,
        wg: G_RELATIVE_LUMINANCE,
        wb: B_RELATIVE_LUMINANCE,
        _pad: 0.0,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let workgroups = ((n as u32) + 255) / 256;

    let y = ctx.create_f32_buffer(n, "chroma_denoise_y");
    let cr = ctx.create_f32_buffer(n, "chroma_denoise_cr");
    let cb = ctx.create_f32_buffer(n, "chroma_denoise_cb");
    let tmp = ctx.create_f32_buffer(n, "chroma_denoise_tmp");
    let mean_g = ctx.create_f32_buffer(n, "chroma_denoise_mean_g");
    let mean_gg = ctx.create_f32_buffer(n, "chroma_denoise_mean_gg");
    let var_g = ctx.create_f32_buffer(n, "chroma_denoise_var_g");
    let mean_p = ctx.create_f32_buffer(n, "chroma_denoise_mean_p");
    let prod = ctx.create_f32_buffer(n, "chroma_denoise_prod");
    let mean_gp = ctx.create_f32_buffer(n, "chroma_denoise_mean_gp");
    let a_buf = ctx.create_f32_buffer(n, "chroma_denoise_a");
    let b_buf = ctx.create_f32_buffer(n, "chroma_denoise_b");
    let mean_a = ctx.create_f32_buffer(n, "chroma_denoise_mean_a");
    let mean_b = ctx.create_f32_buffer(n, "chroma_denoise_mean_b");
    let out_plane = ctx.create_f32_buffer(n, "chroma_denoise_out_plane");

    // Extract Y, Cr, Cb from RGB.
    let extract_shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read_write> y: array<f32>;
        @group(0) @binding(2) var<storage, read_write> cr: array<f32>;
        @group(0) @binding(3) var<storage, read_write> cb: array<f32>;
        @group(0) @binding(4) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let p = pixels[i];
            let yv = params.wr * p.r + params.wg * p.g + params.wb * p.b;
            y[i] = yv;
            cr[i] = p.r - yv;
            cb[i] = p.b - yv;
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "chroma_denoise_extract",
        extract_shader,
        &[&storage_buffer.buffer, &y, &cr, &cb],
        params_bytes,
        workgroups,
    );

    // Shared guide statistics: mean_g, mean_gg, var_g.
    box_filter_gpu(ctx, &y, &mean_g, &tmp, &params, workgroups);
    mul_plane_gpu(ctx, &y, &y, &prod, &params, workgroups);
    box_filter_gpu(ctx, &prod, &mean_gg, &tmp, &params, workgroups);
    var_from_means_gpu(ctx, &mean_gg, &mean_g, &var_g, &params, workgroups);

    guided_filter_channel_gpu(
        ctx,
        &y,
        &cr,
        &mean_g,
        &var_g,
        &mean_p,
        &prod,
        &mean_gp,
        &a_buf,
        &b_buf,
        &mean_a,
        &mean_b,
        &tmp,
        &out_plane,
        &params,
        workgroups,
    );
    // out_plane -> cr
    copy_plane_gpu(ctx, &out_plane, &cr, &params, workgroups);

    guided_filter_channel_gpu(
        ctx,
        &y,
        &cb,
        &mean_g,
        &var_g,
        &mean_p,
        &prod,
        &mean_gp,
        &a_buf,
        &b_buf,
        &mean_a,
        &mean_b,
        &tmp,
        &out_plane,
        &params,
        workgroups,
    );
    copy_plane_gpu(ctx, &out_plane, &cb, &params, workgroups);

    // Reconstruct RGB from smoothed chroma (luma recomputed from original RGB).
    let reconstruct_shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read_write> cr: array<f32>;
        @group(0) @binding(2) var<storage, read_write> cb: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let p = pixels[i];
            let yv = params.wr * p.r + params.wg * p.g + params.wb * p.b;
            let crs = cr[i];
            let cbs = cb[i];
            let r = yv + crs;
            let g = yv - (params.wr / params.wg) * crs - (params.wb / params.wg) * cbs;
            let b = yv + cbs;
            pixels[i] = vec4<f32>(r, g, b, p.a);
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "chroma_denoise_reconstruct",
        reconstruct_shader,
        &[&storage_buffer.buffer, &cr, &cb],
        params_bytes,
        workgroups,
    );
}

fn box_filter_gpu(
    ctx: &GpuContext,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    tmp: &wgpu::Buffer,
    params: &ChromaDenoiseParamsGpu,
    _workgroups: u32,
) {
    let params_bytes = bytemuck::bytes_of(params);

    // Horizontal pass: unnormalized sliding-window sums (matches CPU).
    // One workgroup item per row.
    let h_shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read_write> input_data: array<f32>;
        @group(0) @binding(1) var<storage, read_write> tmp: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let y = global_id.x;
            if (y >= params.height) { return; }
            let row = y * params.width;
            let r = params.radius;
            let w = params.width;

            var sum = 0.0;
            let dx_max = min(r, w - 1u);
            for (var dx = 0u; dx <= dx_max; dx++) {
                sum += input_data[row + dx];
            }
            tmp[row] = sum;

            for (var x = 1u; x < w; x++) {
                if (x > r) {
                    sum -= input_data[row + x - r - 1u];
                }
                if (x + r < w) {
                    sum += input_data[row + x + r];
                }
                tmp[row + x] = sum;
            }
        }
    "#;
    let h_workgroups = (params.height + 63) / 64;
    ctx.dispatch_compute_shader_multi(
        "box_filter_h",
        h_shader,
        &[input, tmp],
        params_bytes,
        h_workgroups,
    );

    // Vertical pass + normalize by clipped window size.
    let v_shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read_write> tmp: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = global_id.x;
            if (x >= params.width) { return; }
            let r = params.radius;
            let w = params.width;
            let h = params.height;

            var sum = 0.0;
            let dy_max = min(r, h - 1u);
            for (var dy = 0u; dy <= dy_max; dy++) {
                sum += tmp[dy * w + x];
            }
            // Store unnormalized; normalize below in same loop over y.
            // First pixel y=0
            {
                let cy = f32(min(r, 0u) + min(r, h - 1u) + 1u);
                let cx = f32(min(r, x) + min(r, w - 1u - x) + 1u);
                output_data[x] = sum / (cx * cy);
            }

            for (var y = 1u; y < h; y++) {
                if (y > r) {
                    sum -= tmp[(y - r - 1u) * w + x];
                }
                if (y + r < h) {
                    sum += tmp[(y + r) * w + x];
                }
                let cy = f32(min(r, y) + min(r, h - 1u - y) + 1u);
                let cx = f32(min(r, x) + min(r, w - 1u - x) + 1u);
                output_data[y * w + x] = sum / (cx * cy);
            }
        }
    "#;
    let v_workgroups = (params.width + 63) / 64;
    ctx.dispatch_compute_shader_multi(
        "box_filter_v",
        v_shader,
        &[tmp, output],
        params_bytes,
        v_workgroups,
    );
}

fn mul_plane_gpu(
    ctx: &GpuContext,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    out: &wgpu::Buffer,
    params: &ChromaDenoiseParamsGpu,
    workgroups: u32,
) {
    let shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out_data: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            out_data[i] = a[i] * b[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "mul_plane",
        shader,
        &[a, b, out],
        bytemuck::bytes_of(params),
        workgroups,
    );
}

fn var_from_means_gpu(
    ctx: &GpuContext,
    mean_gg: &wgpu::Buffer,
    mean_g: &wgpu::Buffer,
    var_g: &wgpu::Buffer,
    params: &ChromaDenoiseParamsGpu,
    workgroups: u32,
) {
    let shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read> mean_gg: array<f32>;
        @group(0) @binding(1) var<storage, read> mean_g: array<f32>;
        @group(0) @binding(2) var<storage, read_write> var_g: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let mg = mean_g[i];
            var_g[i] = mean_gg[i] - mg * mg;
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "var_from_means",
        shader,
        &[mean_gg, mean_g, var_g],
        bytemuck::bytes_of(params),
        workgroups,
    );
}

fn copy_plane_gpu(
    ctx: &GpuContext,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    params: &ChromaDenoiseParamsGpu,
    workgroups: u32,
) {
    let shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read> src: array<f32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            dst[i] = src[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "copy_plane",
        shader,
        &[src, dst],
        bytemuck::bytes_of(params),
        workgroups,
    );
}

fn guided_filter_channel_gpu(
    ctx: &GpuContext,
    guide: &wgpu::Buffer,
    source: &wgpu::Buffer,
    mean_g: &wgpu::Buffer,
    var_g: &wgpu::Buffer,
    mean_p: &wgpu::Buffer,
    prod: &wgpu::Buffer,
    mean_gp: &wgpu::Buffer,
    a_buf: &wgpu::Buffer,
    b_buf: &wgpu::Buffer,
    mean_a: &wgpu::Buffer,
    mean_b: &wgpu::Buffer,
    tmp: &wgpu::Buffer,
    out: &wgpu::Buffer,
    params: &ChromaDenoiseParamsGpu,
    workgroups: u32,
) {
    let params_bytes = bytemuck::bytes_of(params);

    box_filter_gpu(ctx, source, mean_p, tmp, params, workgroups);
    mul_plane_gpu(ctx, guide, source, prod, params, workgroups);
    box_filter_gpu(ctx, prod, mean_gp, tmp, params, workgroups);

    // a = cov / (var + eps), b = mean_p - a * mean_g
    let ab_shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read> mean_g: array<f32>;
        @group(0) @binding(1) var<storage, read> mean_p: array<f32>;
        @group(0) @binding(2) var<storage, read> mean_gp: array<f32>;
        @group(0) @binding(3) var<storage, read> var_g: array<f32>;
        @group(0) @binding(4) var<storage, read_write> a_out: array<f32>;
        @group(0) @binding(5) var<storage, read_write> b_out: array<f32>;
        @group(0) @binding(6) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let mg = mean_g[i];
            let mp = mean_p[i];
            let cov = mean_gp[i] - mg * mp;
            let a_val = cov / (var_g[i] + params.epsilon);
            a_out[i] = a_val;
            b_out[i] = mp - a_val * mg;
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "guided_ab",
        ab_shader,
        &[mean_g, mean_p, mean_gp, var_g, a_buf, b_buf],
        params_bytes,
        workgroups,
    );

    box_filter_gpu(ctx, a_buf, mean_a, tmp, params, workgroups);
    box_filter_gpu(ctx, b_buf, mean_b, tmp, params, workgroups);

    let apply_shader = r#"
        struct Params {
            width: u32,
            height: u32,
            radius: u32,
            epsilon: f32,
            wr: f32,
            wg: f32,
            wb: f32,
            _pad: f32,
        };
        @group(0) @binding(0) var<storage, read> mean_a: array<f32>;
        @group(0) @binding(1) var<storage, read> mean_b: array<f32>;
        @group(0) @binding(2) var<storage, read> guide: array<f32>;
        @group(0) @binding(3) var<storage, read_write> out_data: array<f32>;
        @group(0) @binding(4) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            out_data[i] = mean_a[i] * guide[i] + mean_b[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "guided_apply",
        apply_shader,
        &[mean_a, mean_b, guide, out],
        params_bytes,
        workgroups,
    );
}
