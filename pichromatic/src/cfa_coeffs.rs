use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::{Image, SubPixel};
use rayon::prelude::*;

pub fn cfa_coeffs(image: &mut Image, wb_coeffs: [SubPixel; 4]){
        let [rv, gv, bv, _] = wb_coeffs;
        image.rgb_data.par_iter_mut().for_each(
            |p|{
                let [r, g, b] = p;
                *p = [*r*rv, *g*gv, *b*bv]
            }
        );
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CfaCoeffsParamsGpu {
    pub coeffs: [f32; 4],
}

pub fn cfa_coeffs_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, wb_coeffs: [SubPixel; 4]) {
    let params = CfaCoeffsParamsGpu { coeffs: wb_coeffs };
    let shader_source = r#"
        struct Params {
            coeffs: vec4<f32>,
        };

        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&pixels)) {
                return;
            }
            let p = pixels[index];
            let rgb = p.rgb * params.coeffs.rgb;
            pixels[index] = vec4<f32>(rgb, p.a);
        }
    "#;

    ctx.dispatch_compute_shader(
        "cfa_coeffs",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}

