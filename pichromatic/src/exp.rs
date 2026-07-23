use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;

pub fn exp(image_buffer: &mut ImageBuffer, ev: SubPixel) {
    let value = (2.0 as SubPixel).powf(ev);
    image_buffer.par_iter_mut().for_each(|pixel| {
        *pixel = pixel.map(|sub_pixel| sub_pixel * value)
    });
}

pub fn exp_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, ev: SubPixel) {
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ExpParamsGpu {
        scale: f32,
        width: u32,
        height: u32,
        _pad: u32,
    }

    let params = ExpParamsGpu {
        scale: 2.0f32.powf(ev),
        width: storage_buffer.width as u32,
        height: storage_buffer.height as u32,
        _pad: 0,
    };

    let shader_source = r#"
        struct Params {
            scale: f32,
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
            pixels[index] = vec4<f32>(p.rgb * params.scale, p.a);
        }
    "#;

    ctx.dispatch_compute_shader_2d("exp", shader_source, storage_buffer, bytemuck::bytes_of(&params));
}
