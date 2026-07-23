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
    pub width: u32,
    pub height: u32,
    pub _pad: [u32; 2],
}

pub fn cfa_coeffs_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, wb_coeffs: [SubPixel; 4]) {
    let params = CfaCoeffsParamsGpu {
        coeffs: wb_coeffs,
        width: storage_buffer.width as u32,
        height: storage_buffer.height as u32,
        _pad: [0; 2],
    };
    let shader_source = r#"
        struct Params {
            coeffs: vec4<f32>,
            width: u32,
            height: u32,
            _pad0: u32,
            _pad1: u32,
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
            let rgb = p.rgb * params.coeffs.rgb;
            pixels[index] = vec4<f32>(rgb, p.a);
        }
    "#;

    ctx.dispatch_compute_shader_2d(
        "cfa_coeffs",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}
