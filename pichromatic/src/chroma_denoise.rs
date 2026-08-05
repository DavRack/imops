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
    let var_g = ctx.create_f32_buffer(n, "chroma_denoise_var_g");
    let work0 = ctx.create_f32_buffer(n, "chroma_denoise_work0");
    let work1 = ctx.create_f32_buffer(n, "chroma_denoise_work1");
    let work2 = ctx.create_f32_buffer(n, "chroma_denoise_work2");
    let work3 = ctx.create_f32_buffer(n, "chroma_denoise_work3");
    let rec_products = ctx.create_f32_buffer(4 * n, "chroma_denoise_rec_products");

    // Extract Y, Cr, Cb from RGB.
    //
    // Two passes so the luma computation is bit-identical to the CPU
    // (`wr*r + wg*g + wb*b` with plain f32 rounding at every step): pass 1
    // stores the three weighted products in a storage buffer, pass 2 sums
    // them with plain adds. A single-pass expression gets contracted to an
    // FMA chain by the Metal compiler, which shifts the result by up to 2
    // ULP; that tiny luma jitter is amplified enormously by the guided
    // filter's box-filter cancellation, so it must be avoided.
    let extract_products_shader = r#"
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
        @group(0) @binding(1) var<storage, read_write> prod: array<vec4<f32>>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let p = pixels[i];
            prod[i] = vec4<f32>(params.wr * p.r, params.wg * p.g, params.wb * p.b, 0.0);
        }
    "#;
    let prod = ctx.create_f32_buffer(4 * n, "chroma_denoise_prod");
    ctx.dispatch_compute_shader_multi(
        "chroma_denoise_extract_products",
        extract_products_shader,
        &[&storage_buffer.buffer, &prod],
        params_bytes,
        workgroups,
    );

    let extract_luma_shader = r#"
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
        @group(0) @binding(1) var<storage, read> prod: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read_write> y: array<f32>;
        @group(0) @binding(3) var<storage, read_write> cr: array<f32>;
        @group(0) @binding(4) var<storage, read_write> cb: array<f32>;
        @group(0) @binding(5) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let p = pixels[i];
            let pr = prod[i];
            let yv = (pr.x + pr.y) + pr.z;
            y[i] = yv;
            cr[i] = p.r - yv;
            cb[i] = p.b - yv;
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "chroma_denoise_extract_luma",
        extract_luma_shader,
        &[&storage_buffer.buffer, &prod, &y, &cr, &cb],
        params_bytes,
        workgroups,
    );

    // Shared guide statistics: mean_g, mean_gg, var_g.
    box_filter_gpu(ctx, &y, &mean_g, &tmp, &params, workgroups);
    mul_plane_gpu(ctx, &y, &y, &work0, &params, workgroups);
    box_filter_gpu(ctx, &work0, &work1, &tmp, &params, workgroups);
    var_from_means_gpu(ctx, &work1, &mean_g, &var_g, &params, workgroups);

    guided_filter_channel_gpu(
        ctx,
        &y,
        &cr,
        &mean_g,
        &var_g,
        [&work0, &work1, &work2, &work3],
        &tmp,
        &params,
        workgroups,
    );

    guided_filter_channel_gpu(
        ctx,
        &y,
        &cb,
        &mean_g,
        &var_g,
        [&work0, &work1, &work2, &work3],
        &tmp,
        &params,
        workgroups,
    );

    // Reconstruct RGB from smoothed chroma (luma recomputed from original RGB).
    //
    // Two passes: `yv - k1*crs - k2*cbs` must round k1*crs and k2*cbs before
    // the subtractions, like the CPU; the metal compiler otherwise contracts
    // the products into the subtractions via fma and rounds `wr/wg` as a fast
    // reciprocal multiply. All values pass through storage memory so every
    // op is plain.
    let reconstruct_products_shader = r#"
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
        @group(0) @binding(1) var<storage, read> cr: array<f32>;
        @group(0) @binding(2) var<storage, read> cb: array<f32>;
        @group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
        @group(0) @binding(4) var<uniform> params: Params;

        fn div_exact(a: f32, b: f32) -> f32 {
            let inv = 1.0 / b;
            let q0 = a * inv;
            let e = fma(-b, q0, a);
            return fma(e, inv, q0);
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let p = pixels[i];
            let crs = cr[i];
            let cbs = cb[i];
            out[i] = vec4<f32>(div_exact(params.wr, params.wg) * crs, div_exact(params.wb, params.wg) * cbs, 0.0, p.a);
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "chroma_denoise_reconstruct_products",
        reconstruct_products_shader,
        &[&storage_buffer.buffer, &cr, &cb, &rec_products],
        params_bytes,
        workgroups,
    );

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
        @group(0) @binding(1) var<storage, read> cr: array<f32>;
        @group(0) @binding(2) var<storage, read> cb: array<f32>;
        @group(0) @binding(3) var<storage, read> prod: array<vec4<f32>>;
        @group(0) @binding(4) var<storage, read> out: array<vec4<f32>>;
        @group(0) @binding(5) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let p = pixels[i];
            let pr = prod[i];
            let yv = (pr.x + pr.y) + pr.z;
            let crs = cr[i];
            let cbs = cb[i];
            let op = out[i];
            let r = yv + crs;
            let g = (yv - op.x) - op.y;
            let b = yv + cbs;
            pixels[i] = vec4<f32>(r, g, b, p.a);
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "chroma_denoise_reconstruct",
        reconstruct_shader,
        &[&storage_buffer.buffer, &cr, &cb, &prod, &rec_products],
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

        fn div_exact(a: f32, b: f32) -> f32 {
            let inv = 1.0 / b;
            let q0 = a * inv;
            let e = fma(-b, q0, a);
            return fma(e, inv, q0);
        }

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
                output_data[x] = div_exact(sum, cx * cy);
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
                output_data[y * w + x] = div_exact(sum, cx * cy);
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
    let params_bytes = bytemuck::bytes_of(params);
    // Two passes so the subtraction is plain, like the CPU's `mean_gg - mg*mq`
    // with f32 rounding at every step: pass 1 stores the squared mean, pass 2
    // subtracts. A single-pass `mean_gg - mg * mg` gets contracted to
    // `fma(-mg, mg, mean_gg)` by the Metal compiler; the exact product leaves
    // a denormal residual where the CPU result cancels to exactly zero, and
    // that residual is amplified catastrophically by the guided filter's
    // small-denominator divisions.
    let products_shader = r#"
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
        @group(0) @binding(1) var<storage, read_write> sq: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let mg = mean_g[i];
            sq[i] = mg * mg;
        }
    "#;
    let sq = ctx.create_f32_buffer(
        (params.width * params.height) as usize,
        "var_g_sq",
    );
    ctx.dispatch_compute_shader_multi(
        "var_sq",
        products_shader,
        &[mean_g, &sq],
        params_bytes,
        workgroups,
    );

    let subtract_shader = r#"
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
        @group(0) @binding(1) var<storage, read> sq: array<f32>;
        @group(0) @binding(2) var<storage, read_write> var_g: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            var_g[i] = mean_gg[i] - sq[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "var_sub",
        subtract_shader,
        &[mean_gg, &sq, var_g],
        params_bytes,
        workgroups,
    );
}

fn guided_filter_channel_gpu(
    ctx: &GpuContext,
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

    // Reuse four full-frame planes as each intermediate becomes dead.
    box_filter_gpu(ctx, source, work0, tmp, params, workgroups); // mean_p
    mul_plane_gpu(ctx, guide, source, work1, params, workgroups); // prod
    box_filter_gpu(ctx, work1, work2, tmp, params, workgroups); // mean_gp

    // a = cov / (var + eps), b = mean_p - a * mean_g
    //
    // Three passes so every multiply is rounded before its use, exactly like
    // the CPU's plain f32 ops: the Metal compiler otherwise contracts
    // `mean_gp - mg*mp` and `mp - a_val*mg` into FMA chains whose exact
    // products leave denormal residuals where the CPU cancels to zero.
    // mean_gp and amg are overwritten in place (same-index read-then-write),
    // so every buffer appears at exactly one binding.
    let ab_products_shader = r#"
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
        @group(0) @binding(0) var<storage, read> mean_p: array<f32>;
        @group(0) @binding(1) var<storage, read> mean_g: array<f32>;
        @group(0) @binding(2) var<storage, read_write> prod: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            prod[i] = mean_p[i] * mean_g[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "guided_ab_products",
        ab_products_shader,
        &[work0, mean_g, work1],
        params_bytes,
        workgroups,
    );

    let ab_a_shader = r#"
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
        @group(0) @binding(1) var<storage, read_write> mean_gp: array<f32>;
        @group(0) @binding(2) var<storage, read> var_g: array<f32>;
        @group(0) @binding(3) var<storage, read> prod: array<f32>;
        @group(0) @binding(4) var<storage, read_write> amg: array<f32>;
        @group(0) @binding(5) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            let mg = mean_g[i];
            let cov = mean_gp[i] - prod[i];
            let denom = var_g[i] + params.epsilon;
            let inv = 1.0 / denom;
            let q0 = cov * inv;
            let e = fma(-denom, q0, cov);
            let a_val = fma(e, inv, q0);
            mean_gp[i] = a_val;
            amg[i] = a_val * mg;
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "guided_ab_a",
        ab_a_shader,
        &[mean_g, work2, var_g, work1, work3],
        params_bytes,
        workgroups,
    );

    let ab_b_shader = r#"
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
        @group(0) @binding(0) var<storage, read> mean_p: array<f32>;
        @group(0) @binding(1) var<storage, read_write> amg: array<f32>;
        @group(0) @binding(2) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            amg[i] = mean_p[i] - amg[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "guided_ab_b",
        ab_b_shader,
        &[work0, work3],
        params_bytes,
        workgroups,
    );

    box_filter_gpu(ctx, work2, work1, tmp, params, workgroups); // mean_a
    box_filter_gpu(ctx, work3, work0, tmp, params, workgroups); // mean_b

    // out = mean_a * guide + mean_b
    //
    // Two passes so the multiply is rounded before the add, like the CPU's
    // plain `ma * gv + mb`; a fused `fma(ma, gv, mb)` differs by 1 ULP.
    let apply_products_shader = r#"
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
        @group(0) @binding(1) var<storage, read> guide: array<f32>;
        @group(0) @binding(2) var<storage, read_write> prod: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            prod[i] = mean_a[i] * guide[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "guided_apply_products",
        apply_products_shader,
        &[work1, guide, tmp],
        params_bytes,
        workgroups,
    );

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
        @group(0) @binding(0) var<storage, read> prod: array<f32>;
        @group(0) @binding(1) var<storage, read> mean_b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> out_data: array<f32>;
        @group(0) @binding(3) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let i = global_id.x + global_id.y * 16776960u;
            let n = params.width * params.height;
            if (i >= n) { return; }
            out_data[i] = prod[i] + mean_b[i];
        }
    "#;
    ctx.dispatch_compute_shader_multi(
        "guided_apply",
        apply_shader,
        &[tmp, work0, source],
        params_bytes,
        workgroups,
    );
}
