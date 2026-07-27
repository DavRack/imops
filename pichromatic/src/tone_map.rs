use crate::pixel::{ImageBuffer, PixelOps};
use crate::gpu::{GpuContext, GpuImageBuffer};
use color::ColorSpaceTag::{Oklch, AcesCg};
use rayon::prelude::*;

#[inline(always)]
pub fn square_sigmoid(x: f32) -> f32 {
    /// Constant multiplier to ensure 18% middle gray in maps to 18% gray out:
    /// c = 1.0 / (1.0 - 0.18) = 1.219512
    const C: f32 = 1.219512;
    if x <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 + (1.0 / (C * x)))
}
pub fn sigmoid(image_buffer: &mut ImageBuffer){
    let params = &RgcParams::default();
    image_buffer.par_iter_mut().for_each(|pixel|{
        let gamut_compressed_pixel = gamut_compress_pixel(*pixel, params);
        // let s = square_sigmoid(m);
        // let factor = s/m;
        // println!("{} --- {}", s, m);
        let [_, _ ,h] = AcesCg.convert(Oklch, gamut_compressed_pixel);
        let p = gamut_compressed_pixel.map(|subp| square_sigmoid(subp));
        let k = 5.0;
        let s = p.luminance();
        let m = 1.0-s.powf(k);
        let [l, c, _] = AcesCg.convert(Oklch, p);
        *pixel = Oklch.convert(AcesCg, [l, c*m, h]);
    });
}

/// Native GPU sigmoid tone map matching CPU `sigmoid` (ACEScg assumed, no metadata check).
pub fn sigmoid_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer) {
    // Bake ACEScg ↔ LinearSrgb matrices from the color crate (chromatically adapted).
    let to0 = AcesCg.convert(color::ColorSpaceTag::LinearSrgb, [1.0, 0.0, 0.0]);
    let to1 = AcesCg.convert(color::ColorSpaceTag::LinearSrgb, [0.0, 1.0, 0.0]);
    let to2 = AcesCg.convert(color::ColorSpaceTag::LinearSrgb, [0.0, 0.0, 1.0]);
    let from0 = color::ColorSpaceTag::LinearSrgb.convert(AcesCg, [1.0, 0.0, 0.0]);
    let from1 = color::ColorSpaceTag::LinearSrgb.convert(AcesCg, [0.0, 1.0, 0.0]);
    let from2 = color::ColorSpaceTag::LinearSrgb.convert(AcesCg, [0.0, 0.0, 1.0]);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SigmoidParamsGpu {
        to_lin_r: [f32; 4],
        to_lin_g: [f32; 4],
        to_lin_b: [f32; 4],
        from_lin_r: [f32; 4],
        from_lin_g: [f32; 4],
        from_lin_b: [f32; 4],
        width: u32,
        height: u32,
        _pad: [u32; 2],
    }

    let params = SigmoidParamsGpu {
        to_lin_r: [to0[0], to1[0], to2[0], 0.0],
        to_lin_g: [to0[1], to1[1], to2[1], 0.0],
        to_lin_b: [to0[2], to1[2], to2[2], 0.0],
        from_lin_r: [from0[0], from1[0], from2[0], 0.0],
        from_lin_g: [from0[1], from1[1], from2[1], 0.0],
        from_lin_b: [from0[2], from1[2], from2[2], 0.0],
        width: storage_buffer.width as u32,
        height: storage_buffer.height as u32,
        _pad: [0; 2],
    };

    let shader_source = r#"
        struct Params {
            to_lin_r: vec4<f32>,
            to_lin_g: vec4<f32>,
            to_lin_b: vec4<f32>,
            from_lin_r: vec4<f32>,
            from_lin_g: vec4<f32>,
            from_lin_b: vec4<f32>,
            width: u32,
            height: u32,
            _pad0: u32,
            _pad1: u32,
        };

        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<uniform> params: Params;

        // ACEScg luminance (matches pichromatic::pixel)
        const LUMA_R: f32 = 0.2722287168;
        const LUMA_G: f32 = 0.6740817658;
        const LUMA_B: f32 = 0.0536895174;

        const POWER: f32 = 1.2;
        const SIGMOID_C: f32 = 1.219512;
        const CHROMA_K: f32 = 5.0;

        fn cbrt_signed(x: f32) -> f32 {
            return sign(x) * pow(abs(x), 1.0 / 3.0);
        }

        fn mat3_mul_rows(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
            return vec3<f32>(dot(r0, v), dot(r1, v), dot(r2, v));
        }

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

        fn oklab_to_oklch(lab: vec3<f32>) -> vec3<f32> {
            var h = degrees(atan2(lab.z, lab.y));
            if (h < 0.0) {
                h = h + 360.0;
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

        fn acescg_to_oklch(aces: vec3<f32>) -> vec3<f32> {
            let lin = mat3_mul_rows(
                params.to_lin_r.xyz,
                params.to_lin_g.xyz,
                params.to_lin_b.xyz,
                aces
            );
            return oklab_to_oklch(linear_srgb_to_oklab(lin));
        }

        fn oklch_to_acescg(lch: vec3<f32>) -> vec3<f32> {
            let lin = oklab_to_linear_srgb(oklch_to_oklab(lch));
            return mat3_mul_rows(
                params.from_lin_r.xyz,
                params.from_lin_g.xyz,
                params.from_lin_b.xyz,
                lin
            );
        }

        fn square_sigmoid(x: f32) -> f32 {
            if (x <= 0.0) {
                return 0.0;
            }
            return 1.0 / (1.0 + (1.0 / (SIGMOID_C * x)));
        }

        fn compress_value(dist: f32, lim: f32, thr: f32) -> f32 {
            if (dist < thr) {
                return dist;
            }
            let base_inner = (1.0 - thr) / (lim - thr);
            if (base_inner <= 0.0 || abs(lim - thr) < 1e-6) {
                return dist;
            }
            let denom_inner = pow(base_inner, -POWER) - 1.0;
            if (denom_inner <= 0.0) {
                return dist;
            }
            let s = (lim - thr) / pow(denom_inner, 1.0 / POWER);
            let dist_norm = (dist - thr) / s;
            let denominator = pow(1.0 + pow(dist_norm, POWER), 1.0 / POWER);
            return thr + s * dist_norm / denominator;
        }

        fn gamut_compress(rgb: vec3<f32>) -> vec3<f32> {
            let THR = vec3<f32>(0.815, 0.803, 0.88);
            let LIM = vec3<f32>(1.147, 1.264, 1.312);
            let ach = max(rgb.r, max(rgb.g, rgb.b));
            if (ach == 0.0) {
                return vec3<f32>(0.0);
            }
            let abs_ach = abs(ach);
            let dist = (vec3<f32>(ach) - rgb) / abs_ach;
            let cdist = vec3<f32>(
                compress_value(dist.x, LIM.x, THR.x),
                compress_value(dist.y, LIM.y, THR.y),
                compress_value(dist.z, LIM.z, THR.z)
            );
            return vec3<f32>(ach) - cdist * abs_ach;
        }

        fn luminance(rgb: vec3<f32>) -> f32 {
            return LUMA_R * rgb.r + LUMA_G * rgb.g + LUMA_B * rgb.b;
        }

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = global_id.x;
            let y = global_id.y;
            if (x >= params.width || y >= params.height) {
                return;
            }
            let index = y * params.width + x;
            let alpha = pixels[index].a;
            let gc = gamut_compress(pixels[index].rgb);
            let h = acescg_to_oklch(gc).z;
            let p = vec3<f32>(
                square_sigmoid(gc.r),
                square_sigmoid(gc.g),
                square_sigmoid(gc.b)
            );
            let s = luminance(p);
            let m = 1.0 - pow(s, CHROMA_K);
            let lc = acescg_to_oklch(p);
            let out_rgb = oklch_to_acescg(vec3<f32>(lc.x, lc.y * m, h));
            pixels[index] = vec4<f32>(out_rgb, alpha);
        }
    "#;

    ctx.dispatch_compute_shader_2d(
        "sigmoid",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}

/// Power-law display encode: `out = max(0, linear)^(1/γ)`.
///
/// Default γ = 2.2 approximates the sRGB OETF; γ = 2.4 matches BT.1886 /
/// pure “gamma 2.4” display encoding. Apply on linear display-referred RGB
/// (e.g. after CST to sRGB / Rec.709 / Rec.2020).
pub fn gamma_encode(image_buffer: &mut ImageBuffer, gamma: f32) {
    let inv = 1.0 / gamma.max(1e-6);
    image_buffer.par_iter_mut().for_each(|pixel| {
        *pixel = pixel.map(|c| {
            if c <= 0.0 {
                0.0
            } else {
                c.powf(inv)
            }
        });
    });
}

pub fn gamma_encode_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, gamma: f32) {
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GammaParamsGpu {
        inv_gamma: f32,
        width: u32,
        height: u32,
        _pad: u32,
    }

    let params = GammaParamsGpu {
        inv_gamma: 1.0 / gamma.max(1e-6),
        width: storage_buffer.width as u32,
        height: storage_buffer.height as u32,
        _pad: 0,
    };

    let shader_source = r#"
        struct Params {
            inv_gamma: f32,
            width: u32,
            height: u32,
            _pad: u32,
        };

        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<uniform> params: Params;

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let x = global_id.x;
            let y = global_id.y;
            if (x >= params.width || y >= params.height) {
                return;
            }
            let index = y * params.width + x;
            let p = pixels[index];
            let r = select(pow(max(p.r, 0.0), params.inv_gamma), 0.0, p.r <= 0.0);
            let g = select(pow(max(p.g, 0.0), params.inv_gamma), 0.0, p.g <= 0.0);
            let b = select(pow(max(p.b, 0.0), params.inv_gamma), 0.0, p.b <= 0.0);
            pixels[index] = vec4<f32>(r, g, b, p.a);
        }
    "#;

    ctx.dispatch_compute_shader_2d("gamma_encode", shader_source, storage_buffer, bytemuck::bytes_of(&params));
}

/// Configuration parameters for the Reference Gamut Compression.
/// Defaults match the ACES 1.3 LMT implementation.
#[derive(Debug, Clone)]
pub struct RgcParams {
    pub threshold: [f32; 3], // [Cyan, Magenta, Yellow] direction thresholds
    pub limits: [f32; 3],    // [Cyan, Magenta, Yellow] direction limits
    pub power: f32,
    pub invert: bool,
}

impl Default for RgcParams {
    fn default() -> Self {
        // Defaults from the Python 'main' function arguments
        // limit = value + 1.0 (as per the python code: cyan+1, etc.)
        Self {
            threshold: [0.815, 0.803, 0.88],
            limits: [1.147, 1.264, 1.312], // 0.147+1, 0.264+1, 0.312+1
            power: 1.2,
            invert: false,
        }
    }
}

#[inline]
fn compress_value(dist: f32, lim: f32, thr: f32, power: f32, invert: bool) -> f32 {
    // 1. If distance is below threshold, no compression needed.
    // The Python code does this via masking: `cdist[dist < thr] = dist[dist < thr]`
    if dist < thr {
        return dist;
    }

    // 2. Calculate scale factor 's'
    // Python: s = (lim-thr)/np.power(np.power((1-thr)/(lim-thr),-power)-1,1/power)
    // This calculates the y=1 intersect to ensure smooth continuity.
    let base_inner = (1.0 - thr) / (lim - thr);
    
    // Safety check for div by zero or invalid pow inputs
    if base_inner <= 0.0 || (lim - thr).abs() < 1e-6 {
        return dist; 
    }

    let denom_inner = base_inner.powf(-power) - 1.0;
    
    // Safety check for complex numbers result
    if denom_inner <= 0.0 {
         return dist;
    }

    let s = (lim - thr) / denom_inner.powf(1.0 / power);

    // 3. Apply Curve
    let dist_norm = (dist - thr) / s; // (dist-thr)/s

    if !invert {
        // Forward Compression
        // y = thr + s * ( ((dist-thr)/s) / (1 + ((dist-thr)/s)^p)^(1/p) )
        let denominator = (1.0 + dist_norm.powf(power)).powf(1.0 / power);
        thr + s * (dist_norm / denominator)
    } else {
        // Inverse (Un-compression)
        // y = thr + s * ( - ( x^p / (x^p - 1) ) )^(1/p)
        // Note: The python math for inverse is simpler to write as:
        // inner = dist_norm^p / (dist_norm^p - 1)
        // result = thr + s * (-inner)^(1/p)
        
        let dn_pow = dist_norm.powf(power);
        let inner = dn_pow / (dn_pow - 1.0);
        
        // Safety for inverse: if inner is positive (which implies -inner is neg),
        // we can't root it without complex numbers.
        if inner >= 0.0 {
             dist 
        } else {
            thr + s * (-inner).powf(1.0 / power)
        }
    }
}

/// Applies ACES Reference Gamut Compression to a single pixel.
/// 
/// Input: ACEScg RGB [f32; 3]
/// Output: Compressed RGB [f32; 3]
pub fn gamut_compress_pixel(rgb: [f32; 3], params: &RgcParams) -> [f32; 3] {
    // 1. Calculate Achromatic axis (Max(R, G, B))
    // Python: ach = np.max(rgb, axis=-1)
    let ach = rgb[0].max(rgb[1]).max(rgb[2]);

    // Handle pure black to avoid division by zero
    if ach == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    // 2. Calculate Distance
    // Python: dist = np.where(ach == 0.0, 0.0, (ach-rgb)/np.abs(ach))
    // Since we handled ach=0 above, we just divide.
    let abs_ach = ach.abs();
    
    // We compute this per channel
    let dist = [
        (ach - rgb[0]) / abs_ach,
        (ach - rgb[1]) / abs_ach,
        (ach - rgb[2]) / abs_ach,
    ];

    // 3. Compress Distance
    // The Python code maps lim/thr arrays to the R, G, B channels respectively.
    let cdist = [
        compress_value(dist[0], params.limits[0], params.threshold[0], params.power, params.invert),
        compress_value(dist[1], params.limits[1], params.threshold[1], params.power, params.invert),
        compress_value(dist[2], params.limits[2], params.threshold[2], params.power, params.invert),
    ];

    // 4. Reconstruct RGB
    // Python: crgb = ach - cdist * np.abs(ach)
    [
        ach - cdist[0] * abs_ach,
        ach - cdist[1] * abs_ach,
        ach - cdist[2] * abs_ach,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_preservation() {
        let input = 0.18_f32;
        let output = square_sigmoid(input);
        assert!(
            (output - input).abs() < 1e-5,
            "18% gray shifted from {} to {}",
            input,
            output
        );
    }

    #[test]
    fn gamma_encode_22_roundtrip_mid() {
        let mut buf = vec![[0.18, 0.18, 0.18]];
        gamma_encode(&mut buf, 2.2);
        let e = buf[0][0];
        let back = e.powf(2.2);
        assert!((back - 0.18).abs() < 1e-5, "γ2.2 roundtrip {back}");
    }
}

