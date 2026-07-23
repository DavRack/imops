use color::ColorSpaceTag::{self, LinearSrgb, Oklch};
use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;

pub fn lch(image_buffer: &mut ImageBuffer, source_cs: ColorSpaceTag, l_coef: SubPixel, c_coef: SubPixel, h_coef: SubPixel){
    image_buffer.par_iter_mut().for_each(
        |pixel| {
            let [l, c, h] = source_cs.convert(
                Oklch,
                *pixel
            );
            *pixel = Oklch.convert(
                source_cs,
                [l*l_coef, c*c_coef, h*h_coef]
            );
        }
    );
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LchParamsGpu {
    to_lin_r: [f32; 4],
    to_lin_g: [f32; 4],
    to_lin_b: [f32; 4],
    from_lin_r: [f32; 4],
    from_lin_g: [f32; 4],
    from_lin_b: [f32; 4],
    coeffs: [f32; 4], // l, c, h, pad
    flags: [u32; 4],  // is_srgb, pad...
}

/// Native GPU LCH adjust: source → Oklch, scale L/C/H, convert back.
/// Matches CPU `lch` (via the `color` crate) within ~1e-4.
pub fn lch_gpu(
    ctx: &GpuContext,
    storage_buffer: &GpuImageBuffer,
    source_cs: ColorSpaceTag,
    l_coef: SubPixel,
    c_coef: SubPixel,
    h_coef: SubPixel,
) {
    let is_srgb = matches!(source_cs, ColorSpaceTag::Srgb);
    let intermediate = if is_srgb {
        LinearSrgb
    } else {
        source_cs
    };

    // Bake linear matrices: intermediate ↔ LinearSrgb (identity when already LinearSrgb / after EOTF).
    let to0 = intermediate.convert(LinearSrgb, [1.0, 0.0, 0.0]);
    let to1 = intermediate.convert(LinearSrgb, [0.0, 1.0, 0.0]);
    let to2 = intermediate.convert(LinearSrgb, [0.0, 0.0, 1.0]);
    let from0 = LinearSrgb.convert(intermediate, [1.0, 0.0, 0.0]);
    let from1 = LinearSrgb.convert(intermediate, [0.0, 1.0, 0.0]);
    let from2 = LinearSrgb.convert(intermediate, [0.0, 0.0, 1.0]);

    let params = LchParamsGpu {
        to_lin_r: [to0[0], to1[0], to2[0], 0.0],
        to_lin_g: [to0[1], to1[1], to2[1], 0.0],
        to_lin_b: [to0[2], to1[2], to2[2], 0.0],
        from_lin_r: [from0[0], from1[0], from2[0], 0.0],
        from_lin_g: [from0[1], from1[1], from2[1], 0.0],
        from_lin_b: [from0[2], from1[2], from2[2], 0.0],
        coeffs: [l_coef, c_coef, h_coef, 0.0],
        flags: [
            if is_srgb { 1 } else { 0 },
            0,
            storage_buffer.width as u32,
            storage_buffer.height as u32,
        ],
    };

    // Oklab matrices from color-0.3.x (Björn Ottosson), precision reduced to f32.
    let shader_source = r#"
        struct Params {
            to_lin_r: vec4<f32>,
            to_lin_g: vec4<f32>,
            to_lin_b: vec4<f32>,
            from_lin_r: vec4<f32>,
            from_lin_g: vec4<f32>,
            from_lin_b: vec4<f32>,
            coeffs: vec4<f32>,
            flags: vec4<u32>,
        };

        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<uniform> params: Params;

        fn srgb_eotf(c: f32) -> f32 {
            let abs_c = abs(c);
            let decoded = select(
                pow((abs_c + 0.055) / 1.055, 2.4),
                abs_c / 12.92,
                abs_c <= 0.04045
            );
            return select(-decoded, decoded, c >= 0.0);
        }

        fn srgb_oetf(c: f32) -> f32 {
            let abs_c = abs(c);
            let encoded = select(
                1.055 * pow(abs_c, 1.0 / 2.4) - 0.055,
                12.92 * abs_c,
                abs_c <= 0.0031308
            );
            return select(-encoded, encoded, c >= 0.0);
        }

        fn cbrt_signed(x: f32) -> f32 {
            return sign(x) * pow(abs(x), 1.0 / 3.0);
        }

        fn mat3_mul_rows(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
            return vec3<f32>(dot(r0, v), dot(r1, v), dot(r2, v));
        }

        // Linear sRGB → Oklab (color crate OKLAB_SRGB_TO_LMS + cbrt + OKLAB_LMS_TO_LAB)
        fn linear_srgb_to_oklab(rgb: vec3<f32>) -> vec3<f32> {
            let lms = mat3_mul_rows(
                vec3<f32>(0.41222146, 0.53633255, 0.051445995),
                vec3<f32>(0.2119035, 0.6806995, 0.10739696),
                vec3<f32>(0.08830246, 0.28171885, 0.6299787),
                rgb
            );
            let lms_c = vec3<f32>(cbrt_signed(lms.x), cbrt_signed(lms.y), cbrt_signed(lms.z));
            return mat3_mul_rows(
                vec3<f32>(0.21045426, 0.7936178, -0.004072047),
                vec3<f32>(1.9779985, -2.4285922, 0.4505937),
                vec3<f32>(0.025904037, 0.78277177, -0.80867577),
                lms_c
            );
        }

        // Oklab → Linear sRGB
        fn oklab_to_linear_srgb(lab: vec3<f32>) -> vec3<f32> {
            let lms = mat3_mul_rows(
                vec3<f32>(1.0, 0.39633778, 0.21580376),
                vec3<f32>(1.0, -0.105561346, -0.06385417),
                vec3<f32>(1.0, -0.08948418, -1.2914855),
                lab
            );
            let lms3 = lms * lms * lms;
            return mat3_mul_rows(
                vec3<f32>(4.0767417, -3.3077116, 0.23096994),
                vec3<f32>(-1.268438, 2.6097574, -0.34131938),
                vec3<f32>(-0.0041960863, -0.7034186, 1.7076147),
                lms3
            );
        }

        // Oklab [L,a,b] → Oklch [L,C,h°]
        fn oklab_to_oklch(lab: vec3<f32>) -> vec3<f32> {
            var h = degrees(atan2(lab.z, lab.y));
            if (h < 0.0) {
                h += 360.0;
            }
            let c = length(lab.yz);
            return vec3<f32>(lab.x, c, h);
        }

        fn oklch_to_oklab(lch: vec3<f32>) -> vec3<f32> {
            let h_rad = radians(lch.z);
            let a = lch.y * cos(h_rad);
            let b = lch.y * sin(h_rad);
            return vec3<f32>(lch.x, a, b);
        }

        fn source_to_linear_srgb(src: vec3<f32>) -> vec3<f32> {
            var v = src;
            if (params.flags.x == 1u) {
                v = vec3<f32>(srgb_eotf(v.x), srgb_eotf(v.y), srgb_eotf(v.z));
            }
            return mat3_mul_rows(
                params.to_lin_r.xyz,
                params.to_lin_g.xyz,
                params.to_lin_b.xyz,
                v
            );
        }

        fn linear_srgb_to_source(lin: vec3<f32>) -> vec3<f32> {
            var v = mat3_mul_rows(
                params.from_lin_r.xyz,
                params.from_lin_g.xyz,
                params.from_lin_b.xyz,
                lin
            );
            if (params.flags.x == 1u) {
                v = vec3<f32>(srgb_oetf(v.x), srgb_oetf(v.y), srgb_oetf(v.z));
            }
            return v;
        }

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = global_id.x;
            let y = global_id.y;
            if (x >= params.flags.z || y >= params.flags.w) {
                return;
            }
            let index = y * params.flags.z + x;
            let p = pixels[index];
            let lin = source_to_linear_srgb(p.rgb);
            let lab = linear_srgb_to_oklab(lin);
            var lch = oklab_to_oklch(lab);
            lch = vec3<f32>(
                lch.x * params.coeffs.x,
                lch.y * params.coeffs.y,
                lch.z * params.coeffs.z
            );
            let lab2 = oklch_to_oklab(lch);
            let lin2 = oklab_to_linear_srgb(lab2);
            let out_rgb = linear_srgb_to_source(lin2);
            pixels[index] = vec4<f32>(out_rgb, p.a);
        }
    "#;

    ctx.dispatch_compute_shader_2d(
        "lch",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lch_roundtrip_identity() {
        let mut pixels = vec![[0.4, 0.7, 0.2]];
        lch(&mut pixels, ColorSpaceTag::Srgb, 1.0, 1.0, 1.0);
        
        let diff_r = (pixels[0][0] - 0.4).abs();
        let diff_g = (pixels[0][1] - 0.7).abs();
        let diff_b = (pixels[0][2] - 0.2).abs();
        
        assert!(diff_r < 1e-4, "r diff is {}", diff_r);
        assert!(diff_g < 1e-4, "g diff is {}", diff_g);
        assert!(diff_b < 1e-4, "b diff is {}", diff_b);
    }
}
