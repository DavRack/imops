use crate::pixel::{ImageBuffer};

pub fn apply_vignette_radial_correction(
    rgb_data: &mut ImageBuffer,
    width: usize,
    height: usize,
    opcode_list3: &[u8],
    strength: f32,
) {
    if opcode_list3.len() < 4 {
        return;
    }
    // All opcodes lists are stored in big-endian byte order
    let count = u32::from_be_bytes([opcode_list3[0], opcode_list3[1], opcode_list3[2], opcode_list3[3]]) as usize;
    let mut offset = 4;

    for _ in 0..count {
        if offset + 16 > opcode_list3.len() {
            break;
        }
        let opcode_id = u32::from_be_bytes([opcode_list3[offset], opcode_list3[offset+1], opcode_list3[offset+2], opcode_list3[offset+3]]);
        let _version = u32::from_be_bytes([opcode_list3[offset+4], opcode_list3[offset+5], opcode_list3[offset+6], opcode_list3[offset+7]]);
        let _flags = u32::from_be_bytes([opcode_list3[offset+8], opcode_list3[offset+9], opcode_list3[offset+10], opcode_list3[offset+11]]);
        let parameter_size = u32::from_be_bytes([opcode_list3[offset+12], opcode_list3[offset+13], opcode_list3[offset+14], opcode_list3[offset+15]]) as usize;
        
        offset += 16;
        if offset + parameter_size > opcode_list3.len() {
            break;
        }

        let params = &opcode_list3[offset..offset + parameter_size];
        offset += parameter_size;

        if opcode_id == 3 { // FixVignetteRadial
            if parameter_size < 56 {
                continue;
            }
            // Parse 7 doubles (f64), stored in big-endian
            let read_double = |idx: usize| -> f64 {
                let bytes = [
                    params[idx], params[idx+1], params[idx+2], params[idx+3],
                    params[idx+4], params[idx+5], params[idx+6], params[idx+7]
                ];
                f64::from_be_bytes(bytes)
            };

            let k0 = read_double(0) as f32;
            let k1 = read_double(8) as f32;
            let k2 = read_double(16) as f32;
            let k3 = read_double(24) as f32;
            let k4 = read_double(32) as f32;
            let cx = read_double(40) as f32;
            let cy = read_double(48) as f32;

            // Apply vignette correction in parallel
            use rayon::prelude::*;

            let w_f32 = (width - 1) as f32;
            let h_f32 = (height - 1) as f32;

            // Furthest corner distance calculation
            // Corners: (0,0), (1,0), (0,1), (1,1) in normalized coords
            let d_00 = (cx * cx + cy * cy).sqrt();
            let d_10 = ((1.0 - cx).powi(2) + cy * cy).sqrt();
            let d_01 = (cx * cx + (1.0 - cy).powi(2)).sqrt();
            let d_11 = ((1.0 - cx).powi(2) + (1.0 - cy).powi(2)).sqrt();
            let d_max = d_00.max(d_10).max(d_01).max(d_11);

            if d_max > 0.0 {
                rgb_data.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
                    let py = (idx / width) as f32;
                    let px = (idx % width) as f32;

                    let u = px / w_f32;
                    let v = py / h_f32;

                    let du = u - cx;
                    let dv = v - cy;
                    let d = (du * du + dv * dv).sqrt();
                    let r = d / d_max;

                    let r2 = r * r;
                    let r4 = r2 * r2;
                    let r6 = r4 * r2;
                    let r8 = r4 * r4;
                    let r10 = r8 * r2;

                    let correction = k0 * r2 + k1 * r4 + k2 * r6 + k3 * r8 + k4 * r10;
                    let gain = 1.0 + strength * correction;
                    let gain_clamped = gain.max(0.0);

                    pixel[0] *= gain_clamped;
                    pixel[1] *= gain_clamped;
                    pixel[2] *= gain_clamped;
                });
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VignetteParamsGpu {
    pub k0: f32,
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub k4: f32,
    pub cx: f32,
    pub cy: f32,
    pub strength: f32,
    pub d_max: f32,
    pub width: u32,
    pub height: u32,
    pub _pad: u32,
}

pub fn apply_vignette_radial_correction_gpu(
    ctx: &crate::gpu::GpuContext,
    storage_buffer: &crate::gpu::GpuImageBuffer,
    opcode_list3: &[u8],
    strength: f32,
) {
    if opcode_list3.len() < 4 {
        return;
    }
    let count = u32::from_be_bytes([opcode_list3[0], opcode_list3[1], opcode_list3[2], opcode_list3[3]]) as usize;
    let mut offset = 4;

    for _ in 0..count {
        if offset + 16 > opcode_list3.len() {
            break;
        }
        let opcode_id = u32::from_be_bytes([opcode_list3[offset], opcode_list3[offset+1], opcode_list3[offset+2], opcode_list3[offset+3]]);
        let _version = u32::from_be_bytes([opcode_list3[offset+4], opcode_list3[offset+5], opcode_list3[offset+6], opcode_list3[offset+7]]);
        let _flags = u32::from_be_bytes([opcode_list3[offset+8], opcode_list3[offset+9], opcode_list3[offset+10], opcode_list3[offset+11]]);
        let parameter_size = u32::from_be_bytes([opcode_list3[offset+12], opcode_list3[offset+13], opcode_list3[offset+14], opcode_list3[offset+15]]) as usize;
        
        offset += 16;
        if offset + parameter_size > opcode_list3.len() {
            break;
        }

        let params = &opcode_list3[offset..offset + parameter_size];
        offset += parameter_size;

        if opcode_id == 3 && parameter_size >= 56 {
            let read_double = |idx: usize| -> f64 {
                let bytes = [
                    params[idx], params[idx+1], params[idx+2], params[idx+3],
                    params[idx+4], params[idx+5], params[idx+6], params[idx+7]
                ];
                f64::from_be_bytes(bytes)
            };

            let k0 = read_double(0) as f32;
            let k1 = read_double(8) as f32;
            let k2 = read_double(16) as f32;
            let k3 = read_double(24) as f32;
            let k4 = read_double(32) as f32;
            let cx = read_double(40) as f32;
            let cy = read_double(48) as f32;

            let d_00 = (cx * cx + cy * cy).sqrt();
            let d_10 = ((1.0 - cx).powi(2) + cy * cy).sqrt();
            let d_01 = (cx * cx + (1.0 - cy).powi(2)).sqrt();
            let d_11 = ((1.0 - cx).powi(2) + (1.0 - cy).powi(2)).sqrt();
            let d_max = d_00.max(d_10).max(d_01).max(d_11);

            if d_max > 0.0 {
                let gpu_params = VignetteParamsGpu {
                    k0, k1, k2, k3, k4, cx, cy, strength, d_max,
                    width: storage_buffer.width as u32,
                    height: storage_buffer.height as u32,
                    _pad: 0,
                };

                // Four passes so every float op matches the CPU bit-for-bit.
                // A single-pass shader gets the correction polynomial and the
                // `1.0 + strength*correction` fused into FMA chains by the
                // Metal compiler; the polynomial has large intermediate terms
                // (~6.0) that cancel down to ~1.6, so the fusion shifts the
                // gain by several ULP. Passing intermediates through storage
                // memory (products -> distance -> scaled correction -> gain)
                // makes the adds plain, exactly like the CPU.
                let uvdr_shader = r#"
                    struct Params {
                        k0: f32,
                        k1: f32,
                        k2: f32,
                        k3: f32,
                        k4: f32,
                        cx: f32,
                        cy: f32,
                        strength: f32,
                        d_max: f32,
                        width: u32,
                        height: u32,
                        pad: u32,
                    };

                    @group(0) @binding(0) var<storage, read_write> sqd: array<vec4<f32>>;
                    @group(0) @binding(1) var<uniform> params: Params;

                    fn div_exact(a: f32, b: f32) -> f32 {
                        let inv = 1.0 / b;
                        let q0 = a * inv;
                        let e = fma(-b, q0, a);
                        return fma(e, inv, q0);
                    }

                    @compute @workgroup_size(16, 16)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let px = f32(global_id.x);
                        let py = f32(global_id.y);
                        if (global_id.x >= params.width || global_id.y >= params.height) {
                            return;
                        }
                        let index = global_id.y * params.width + global_id.x;

                        let w_f32 = f32(params.width - 1u);
                        let h_f32 = f32(params.height - 1u);

                        // Correctly-rounded division (the compiler lowers `/`
                        // to a reciprocal multiply, which is 1-2 ULP off the
                        // CPU's IEEE division; one Newton step recovers the
                        // exact quotient).
                        let u = div_exact(px, w_f32);
                        let v = div_exact(py, h_f32);

                        let du = u - params.cx;
                        let dv = v - params.cy;
                        sqd[index] = vec4<f32>(du * du, dv * dv, 0.0, 0.0);
                    }
                "#;
                let drterms_shader = r#"
                    struct Params {
                        k0: f32,
                        k1: f32,
                        k2: f32,
                        k3: f32,
                        k4: f32,
                        cx: f32,
                        cy: f32,
                        strength: f32,
                        d_max: f32,
                        width: u32,
                        height: u32,
                        pad: u32,
                    };

                    @group(0) @binding(0) var<storage, read_write> sqd: array<vec4<f32>>;
                    @group(0) @binding(1) var<storage, read_write> terms0: array<vec4<f32>>;
                    @group(0) @binding(2) var<storage, read_write> terms1: array<vec4<f32>>;
                    @group(0) @binding(3) var<uniform> params: Params;

                    fn div_exact(a: f32, b: f32) -> f32 {
                        let inv = 1.0 / b;
                        let q0 = a * inv;
                        let e = fma(-b, q0, a);
                        return fma(e, inv, q0);
                    }

                    // Correctly-rounded sqrt: the builtin `sqrt` is up to 2 ULP
                    // off the CPU's IEEE sqrt (measured over the vignette's
                    // distance range); one residual-corrected Newton step on
                    // `inverseSqrt` is bit-exact there.
                    fn sqrt_exact(x: f32) -> f32 {
                        if (x == 0.0) { return 0.0; }
                        let r = inverseSqrt(x);
                        let y = x * r;
                        let e = fma(-y, y, x);
                        return fma(e, 0.5 * r, y);
                    }

                    @compute @workgroup_size(16, 16)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        if (global_id.x >= params.width || global_id.y >= params.height) {
                            return;
                        }
                        let index = global_id.y * params.width + global_id.x;

                        let s = sqd[index];
                        let d = sqrt_exact(s.x + s.y);
                        let r = div_exact(d, params.d_max);

                        let r2 = r * r;
                        let r4 = r2 * r2;
                        let r6 = r4 * r2;
                        let r8 = r4 * r4;
                        let r10 = r8 * r2;

                        sqd[index] = vec4<f32>(d, r, 0.0, 0.0);
                        terms0[index] = vec4<f32>(params.k0 * r2, params.k1 * r4, params.k2 * r6, params.k3 * r8);
                        terms1[index] = vec4<f32>(params.k4 * r10, 0.0, 0.0, 0.0);
                    }
                "#;
                let gain_shader = r#"
                    struct Params {
                        k0: f32,
                        k1: f32,
                        k2: f32,
                        k3: f32,
                        k4: f32,
                        cx: f32,
                        cy: f32,
                        strength: f32,
                        d_max: f32,
                        width: u32,
                        height: u32,
                        pad: u32,
                    };

                    @group(0) @binding(0) var<storage, read> terms0: array<vec4<f32>>;
                    @group(0) @binding(1) var<storage, read> terms1: array<vec4<f32>>;
                    @group(0) @binding(2) var<storage, read_write> gain: array<f32>;
                    @group(0) @binding(3) var<uniform> params: Params;

                    @compute @workgroup_size(16, 16)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        if (global_id.x >= params.width || global_id.y >= params.height) {
                            return;
                        }
                        let index = global_id.y * params.width + global_id.x;

                        let t = terms0[index];
                        let t4 = terms1[index].x;
                        let correction = (((t.x + t.y) + t.z) + t.w) + t4;
                        gain[index] = params.strength * correction;
                    }
                "#;
                let apply_shader = r#"
                    struct Params {
                        k0: f32,
                        k1: f32,
                        k2: f32,
                        k3: f32,
                        k4: f32,
                        cx: f32,
                        cy: f32,
                        strength: f32,
                        d_max: f32,
                        width: u32,
                        height: u32,
                        pad: u32,
                    };

                    @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
                    @group(0) @binding(1) var<storage, read> gain: array<f32>;
                    @group(0) @binding(2) var<uniform> params: Params;

                    @compute @workgroup_size(16, 16)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        if (global_id.x >= params.width || global_id.y >= params.height) {
                            return;
                        }
                        let index = global_id.y * params.width + global_id.x;

                        let gain_clamped = max(0.0, 1.0 + gain[index]);
                        let p = pixels[index];
                        pixels[index] = vec4<f32>(p.rgb * gain_clamped, p.a);
                    }
                "#;

                let count = storage_buffer.width * storage_buffer.height;
                let sqd_buf = ctx.create_f32_buffer(4 * count, "vignette_sqd");
                let terms0_buf = ctx.create_f32_buffer(4 * count, "vignette_terms0");
                let terms1_buf = ctx.create_f32_buffer(4 * count, "vignette_terms1");
                let gain_buf = ctx.create_f32_buffer(count, "vignette_gain");
                let gx = (storage_buffer.width as u32 + 15) / 16;
                let gy = (storage_buffer.height as u32 + 15) / 16;
                ctx.dispatch_compute_shader_2d_multi(
                    "vignette_uvdr",
                    uvdr_shader,
                    &[&sqd_buf],
                    bytemuck::bytes_of(&gpu_params),
                    gx,
                    gy,
                );
                ctx.dispatch_compute_shader_2d_multi(
                    "vignette_drterms",
                    drterms_shader,
                    &[&sqd_buf, &terms0_buf, &terms1_buf],
                    bytemuck::bytes_of(&gpu_params),
                    gx,
                    gy,
                );
                ctx.dispatch_compute_shader_2d_multi(
                    "vignette_gain",
                    gain_shader,
                    &[&terms0_buf, &terms1_buf, &gain_buf],
                    bytemuck::bytes_of(&gpu_params),
                    gx,
                    gy,
                );
                ctx.dispatch_compute_shader_2d_multi(
                    "vignette_apply",
                    apply_shader,
                    &[&storage_buffer.buffer, &gain_buf],
                    bytemuck::bytes_of(&gpu_params),
                    gx,
                    gy,
                );
            }
        }
    }
}
