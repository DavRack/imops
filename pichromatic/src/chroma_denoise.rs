use rayon::prelude::*;

use crate::gpu::{GpuContext, GpuImageBuffer, GpuSend};
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
    let n = rgb_data.len();
    if n == 0 || width == 0 || height == 0 {
        return;
    }

    let wr = R_RELATIVE_LUMINANCE;
    let wg = G_RELATIVE_LUMINANCE;
    let wb = B_RELATIVE_LUMINANCE;
    let luma = |p: &[f32; 3]| -> f32 { wr * p[0] + wg * p[1] + wb * p[2] };

    let mut y = vec![0.0; n];
    let mut cr = vec![0.0; n];
    let mut cb = vec![0.0; n];
    rgb_data
        .par_iter()
        .zip(&mut y)
        .zip(&mut cr)
        .zip(&mut cb)
        .for_each(|(((p, yv), c_r), c_b)| {
            let l = luma(p);
            *yv = l;
            *c_r = p[0] - l;
            *c_b = p[2] - l;
        });

    // Guide (luma) statistics are shared by both chroma channels.
    let mut mean_g = vec![0.0; n];
    let mut var_g = vec![0.0; n];
    let mut scratch = vec![0.0; n];
    let mut box_tmp = vec![0.0; n];
    box_filter(&y, &mut mean_g, &mut box_tmp, width, height, radius);
    y.par_iter()
        .zip(&mut scratch)
        .for_each(|(&g, s)| *s = g * g);
    box_filter(&scratch, &mut var_g, &mut box_tmp, width, height, radius);
    var_g.par_iter_mut()
        .zip(&mean_g)
        .for_each(|(v, &mg)| *v -= mg * mg);

    let cr_smooth = guided_filter_channel(
        &y, &mean_g, &var_g, &cr, &mut scratch, &mut box_tmp, width, height, radius, epsilon,
    );
    let cb_smooth = guided_filter_channel(
        &y, &mean_g, &var_g, &cb, &mut scratch, &mut box_tmp, width, height, radius, epsilon,
    );

    rgb_data
        .par_iter_mut()
        .enumerate()
        .zip(&y)
        .zip(&cr_smooth)
        .zip(&cb_smooth)
        .for_each(|((((_, p), &yv), &crs), &cbs)| {
            p[0] = yv + crs;
            p[1] = yv - (wr / wg) * crs - (wb / wg) * cbs;
            p[2] = yv + cbs;
        });
}

/// Guided filter for a single channel, reusing the shared guide statistics
/// (mean and variance of the guide) computed once in [`chroma_denoise`].
fn guided_filter_channel(
    guide: &[f32],
    mean_g: &[f32],
    var_g: &[f32],
    source: &[f32],
    scratch: &mut [f32],
    box_tmp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    epsilon: f32,
) -> Vec<f32> {
    let n = source.len();

    let mut mean_p = vec![0.0; n];
    let mut b = vec![0.0; n];
    box_filter(source, &mut mean_p, box_tmp, width, height, radius);
    guide
        .par_iter()
        .zip(source)
        .zip(scratch.par_iter_mut())
        .for_each(|((&g, &s), out)| *out = g * s);
    box_filter(&scratch, &mut b, box_tmp, width, height, radius);

    mean_p
        .par_iter_mut()
        .zip(&mut b)
        .zip(mean_g)
        .zip(var_g)
        .for_each(|(((mp, mgp), &mg), &vg)| {
            let a_val = (*mgp - mg * *mp) / (vg + epsilon);
            let b_val = *mp - a_val * mg;
            *mp = a_val;
            *mgp = b_val;
        });

    let mut result = vec![0.0; n];
    box_filter(&mean_p, scratch, box_tmp, width, height, radius);
    box_filter(&b, &mut mean_p, box_tmp, width, height, radius);
    scratch
        .par_iter()
        .zip(&mean_p)
        .zip(guide)
        .zip(&mut result)
        .for_each(|(((ma, mb), &gv), out)| *out = ma * gv + mb);
    result
}

/// Width of a column block in the vertical pass of [`box_filter`]. Chosen so
/// that each row of a block is a small run of contiguous cache lines.
const VERTICAL_BLOCK: usize = 32;

/// Raw pointer shared with rayon workers. Sound only when every worker writes
/// a disjoint region of the target slice, which the vertical pass of
/// [`box_filter`] guarantees (each worker owns a distinct column range).
#[derive(Copy, Clone)]
struct RawPtr(std::ptr::NonNull<f32>);
unsafe impl Send for RawPtr {}
unsafe impl Sync for RawPtr {}

impl RawPtr {
    unsafe fn write(&self, i: usize, v: f32) {
        *self.0.as_ptr().add(i) = v;
    }
}

/// Separable box filter (axis-aligned sliding window) with correct border
/// handling. Each pixel is the mean over the largest window that fits inside
/// the image bounds at that location.
///
/// Horizontal pass: one row per worker, sequential sliding window.
/// Vertical pass: workers own a block of `VERTICAL_BLOCK` columns and keep the
/// running window sums of all its columns in a small stack array while sliding
/// down the rows, so the filter stays within a few cache lines per row instead
/// of striding across the full image width. Normalization is fused into the
/// vertical pass. Every pixel is the sum of exactly the same values, in the
/// same order, as a plain per-column accumulation, so results are bit-identical.
fn box_filter(
    data: &[f32],
    out: &mut [f32],
    tmp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let n = data.len();
    if n == 0 || width == 0 || height == 0 {
        return;
    }
    debug_assert!(
        width * height <= out.len() && width * height <= tmp.len(),
        "box_filter: buffers must be at least width*height"
    );

    // Horizontal pass: unnormalized sliding-window sums, one worker per row.
    data.par_chunks_exact(width)
        .zip(tmp.par_chunks_exact_mut(width))
        .for_each(|(row, trow)| {
            let mut sum = 0.0;
            let dx_max = radius.min(width - 1);
            for dx in 0..=dx_max {
                sum += row[dx];
            }
            trow[0] = sum;
            for x in 1..width {
                if x > radius {
                    sum -= row[x - radius - 1];
                }
                if x + radius < width {
                    sum += row[x + radius];
                }
                trow[x] = sum;
            }
        });

    // Vertical pass: each worker owns a contiguous block of columns and slides
    // all of the block's window sums down the rows together. Writes go through
    // a raw pointer: rayon's `for_each` needs `Fn` closures, and each block
    // writes a disjoint column range of `out`, so the writes never alias.
    let out_ptr = RawPtr(std::ptr::NonNull::new(out.as_mut_ptr()).unwrap());
    let dy_max = radius.min(height - 1);
    let n_blocks = (width + VERTICAL_BLOCK - 1) / VERTICAL_BLOCK;
    (0..n_blocks).into_par_iter().for_each(move |block| {
        let xb = block * VERTICAL_BLOCK;
        let xe = (xb + VERTICAL_BLOCK).min(width);
        let block_w = xe - xb;
        let mut sums = [0.0f32; VERTICAL_BLOCK];
        for dy in 0..=dy_max {
            let row = dy * width + xb;
            for (x, s) in sums.iter_mut().take(block_w).enumerate() {
                *s += tmp[row + x];
            }
        }
        let cy0 = (radius.min(0) + radius.min(height - 1) + 1) as f32;
        for x in 0..block_w {
            let cx = (radius.min(xb + x) + radius.min(width - 1 - (xb + x)) + 1) as f32;
            unsafe { out_ptr.write(xb + x, sums[x] / (cx * cy0)) };
        }
        for y in 1..height {
            if y > radius {
                let row = (y - radius - 1) * width + xb;
                for (x, s) in sums.iter_mut().take(block_w).enumerate() {
                    *s -= tmp[row + x];
                }
            }
            if y + radius < height {
                let row = (y + radius) * width + xb;
                for (x, s) in sums.iter_mut().take(block_w).enumerate() {
                    *s += tmp[row + x];
                }
            }
            let cy = (radius.min(y) + radius.min(height - 1 - y) + 1) as f32;
            let base = y * width + xb;
            for x in 0..block_w {
                let cx = (radius.min(xb + x) + radius.min(width - 1 - (xb + x)) + 1) as f32;
                unsafe { out_ptr.write(base + x, sums[x] / (cx * cy)) };
            }
        }
    });
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

struct ChromaDenoiseScratch {
    n: usize,
    y: wgpu::Buffer,
    cr: wgpu::Buffer,
    cb: wgpu::Buffer,
    tmp: wgpu::Buffer,
    mean_g: wgpu::Buffer,
    var_g: wgpu::Buffer,
    work0: wgpu::Buffer,
    work1: wgpu::Buffer,
    work2: wgpu::Buffer,
    work3: wgpu::Buffer,
}

impl ChromaDenoiseScratch {
    fn allocate(ctx: &GpuContext, n: usize) -> Self {
        Self {
            n,
            y: ctx.create_f32_buffer(n, "chroma_denoise_y"),
            cr: ctx.create_f32_buffer(n, "chroma_denoise_cr"),
            cb: ctx.create_f32_buffer(n, "chroma_denoise_cb"),
            tmp: ctx.create_f32_buffer(n, "chroma_denoise_tmp"),
            mean_g: ctx.create_f32_buffer(n, "chroma_denoise_mean_g"),
            var_g: ctx.create_f32_buffer(n, "chroma_denoise_var_g"),
            work0: ctx.create_f32_buffer(n, "chroma_denoise_work0"),
            work1: ctx.create_f32_buffer(n, "chroma_denoise_work1"),
            work2: ctx.create_f32_buffer(n, "chroma_denoise_work2"),
            work3: ctx.create_f32_buffer(n, "chroma_denoise_work3"),
        }
    }
}

static CHROMA_SCRATCH: std::sync::Mutex<Option<(u64, GpuSend<ChromaDenoiseScratch>)>> =
    std::sync::Mutex::new(None);

const EXTRACT_SHADER: &str = r#"
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

const RECONSTRUCT_SHADER: &str = r#"
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

const BOX_FILTER_H_SHADER: &str = r#"
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

const BOX_FILTER_V_SHADER: &str = r#"
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

const MUL_PLANE_SHADER: &str = r#"
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

const VAR_FROM_MEANS_SHADER: &str = r#"
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

const GUIDED_AB_SHADER: &str = r#"
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

const GUIDED_APPLY_SHADER: &str = r#"
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

    let scratch = {
        let mut guard = CHROMA_SCRATCH.lock().unwrap();
        if let Some((gen, s)) = guard.take() {
            if gen == ctx.generation && s.get().n >= n {
                s
            } else {
                GpuSend::new(ChromaDenoiseScratch::allocate(ctx, n))
            }
        } else {
            GpuSend::new(ChromaDenoiseScratch::allocate(ctx, n))
        }
    }
    .into_inner();

    let mut encoder = ctx.create_command_encoder("chroma_denoise");
    let mut keep: Vec<(wgpu::BindGroup, Option<wgpu::Buffer>)> = Vec::with_capacity(32);

    // Extract Y, Cr, Cb from RGB.
    keep.push(ctx.encode_compute_shader_multi(
        &mut encoder,
        "chroma_denoise_extract",
        EXTRACT_SHADER,
        &[&storage_buffer.buffer, &scratch.y, &scratch.cr, &scratch.cb],
        params_bytes,
        workgroups,
    ));

    // Shared guide statistics: mean_g, mean_gg, var_g.
    encode_box_filter_gpu(ctx, &mut encoder, &mut keep, &scratch.y, &scratch.mean_g, &scratch.tmp, &params);
    keep.push(ctx.encode_compute_shader_multi(
        &mut encoder,
        "mul_plane_gg",
        MUL_PLANE_SHADER,
        &[&scratch.y, &scratch.y, &scratch.work0],
        params_bytes,
        workgroups,
    ));
    encode_box_filter_gpu(ctx, &mut encoder, &mut keep, &scratch.work0, &scratch.work1, &scratch.tmp, &params);
    keep.push(ctx.encode_compute_shader_multi(
        &mut encoder,
        "var_from_means",
        VAR_FROM_MEANS_SHADER,
        &[&scratch.work1, &scratch.mean_g, &scratch.var_g],
        params_bytes,
        workgroups,
    ));

    encode_guided_filter_channel_gpu(
        ctx,
        &mut encoder,
        &mut keep,
        &scratch.y,
        &scratch.cr,
        &scratch.mean_g,
        &scratch.var_g,
        [&scratch.work0, &scratch.work1, &scratch.work2, &scratch.work3],
        &scratch.tmp,
        &params,
        workgroups,
    );

    encode_guided_filter_channel_gpu(
        ctx,
        &mut encoder,
        &mut keep,
        &scratch.y,
        &scratch.cb,
        &scratch.mean_g,
        &scratch.var_g,
        [&scratch.work0, &scratch.work1, &scratch.work2, &scratch.work3],
        &scratch.tmp,
        &params,
        workgroups,
    );

    // Reconstruct RGB from smoothed chroma (luma recomputed from original RGB).
    keep.push(ctx.encode_compute_shader_multi(
        &mut encoder,
        "chroma_denoise_reconstruct",
        RECONSTRUCT_SHADER,
        &[&storage_buffer.buffer, &scratch.cr, &scratch.cb],
        params_bytes,
        workgroups,
    ));

    ctx.queue.submit(Some(encoder.finish()));
    #[cfg(not(target_arch = "wasm32"))]
    ctx.poll_wait();
    drop(keep);

    // Return scratch to pool
    let mut guard = CHROMA_SCRATCH.lock().unwrap();
    *guard = Some((ctx.generation, GpuSend::new(scratch)));
}

fn encode_box_filter_gpu(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    keep: &mut Vec<(wgpu::BindGroup, Option<wgpu::Buffer>)>,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    tmp: &wgpu::Buffer,
    params: &ChromaDenoiseParamsGpu,
) {
    let params_bytes = bytemuck::bytes_of(params);
    let h_workgroups = (params.height + 63) / 64;
    keep.push(ctx.encode_compute_shader_multi(
        encoder,
        "box_filter_h",
        BOX_FILTER_H_SHADER,
        &[input, tmp],
        params_bytes,
        h_workgroups,
    ));

    let v_workgroups = (params.width + 63) / 64;
    keep.push(ctx.encode_compute_shader_multi(
        encoder,
        "box_filter_v",
        BOX_FILTER_V_SHADER,
        &[tmp, output],
        params_bytes,
        v_workgroups,
    ));
}

fn encode_guided_filter_channel_gpu(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    keep: &mut Vec<(wgpu::BindGroup, Option<wgpu::Buffer>)>,
    guide: &wgpu::Buffer,
    source: &wgpu::Buffer,
    mean_g: &wgpu::Buffer,
    var_g: &wgpu::Buffer,
    [work0, work1, work2, work3]: [&wgpu::Buffer; 4],
    tmp: &wgpu::Buffer,
    params: &ChromaDenoiseParamsGpu,
    workgroups: u32,
) {
    let params_bytes = bytemuck::bytes_of(params);

    encode_box_filter_gpu(ctx, encoder, keep, source, work0, tmp, params); // mean_p
    keep.push(ctx.encode_compute_shader_multi(
        encoder,
        "mul_gp",
        MUL_PLANE_SHADER,
        &[guide, source, work1],
        params_bytes,
        workgroups,
    ));
    encode_box_filter_gpu(ctx, encoder, keep, work1, work2, tmp, params); // mean_gp

    keep.push(ctx.encode_compute_shader_multi(
        encoder,
        "guided_ab",
        GUIDED_AB_SHADER,
        &[mean_g, work0, work2, var_g, work1, work3],
        params_bytes,
        workgroups,
    ));

    encode_box_filter_gpu(ctx, encoder, keep, work1, work2, tmp, params); // mean_a
    encode_box_filter_gpu(ctx, encoder, keep, work3, work0, tmp, params); // mean_b

    keep.push(ctx.encode_compute_shader_multi(
        encoder,
        "guided_apply",
        GUIDED_APPLY_SHADER,
        &[work2, work0, guide, source],
        params_bytes,
        workgroups,
    ));
}
