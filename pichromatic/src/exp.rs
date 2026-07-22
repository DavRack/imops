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
    let scale = 2.0f32.powf(ev);
    let shader_source = r#"
        struct Params {
            scale: f32,
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
            pixels[index] = vec4<f32>(p.rgb * params.scale, p.a);
        }
    "#;

    ctx.dispatch_compute_shader("exp", shader_source, storage_buffer, bytemuck::bytes_of(&scale));
}
