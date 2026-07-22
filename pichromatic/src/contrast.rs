use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::{ImageBuffer, SubPixel, MIDDLE_GRAY};
use rayon::prelude::*;

pub fn contrast(image_buffer: &mut ImageBuffer, value: SubPixel) {
    image_buffer.par_iter_mut().for_each(|p| {
        *p = p.map(|x| MIDDLE_GRAY * (x / MIDDLE_GRAY).powf(value));
    });
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContrastParamsGpu {
    pub contrast: f32,
    pub pivot: f32,
    pub _pad: [f32; 2],
}

pub fn contrast_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, c: SubPixel) {
    let params = ContrastParamsGpu {
        contrast: c,
        pivot: MIDDLE_GRAY,
        _pad: [0.0; 2],
    };

    let shader_source = r#"
        struct Params {
            contrast: f32,
            pivot: f32,
            _pad: vec2<f32>,
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
            let rgb = params.pivot * pow(p.rgb / params.pivot, vec3<f32>(params.contrast));
            pixels[index] = vec4<f32>(rgb, p.a);
        }
    "#;

    ctx.dispatch_compute_shader(
        "contrast",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}
