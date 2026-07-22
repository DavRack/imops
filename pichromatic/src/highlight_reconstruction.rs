use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::{ImageBuffer, SubPixel};
use rayon::prelude::*;


pub fn highlight_reconstruction(image_buffer: &mut ImageBuffer, wb_coeffs: [SubPixel; 4]){
let [_, clip_g, _, _] = wb_coeffs;
        image_buffer.par_iter_mut().for_each(|pixel|{
            let [r, g, b] = *pixel;
            let factor = g/clip_g;
            let reconstructed_g = ((1.0-factor)*g) + (factor*(r+b)*(1.0/2.0));
            pixel[1] = reconstructed_g;
        });
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HighlightReconstructionParamsGpu {
    clip_g: f32,
    _pad: [f32; 3],
}

pub fn highlight_reconstruction_gpu(
    ctx: &GpuContext,
    storage_buffer: &GpuImageBuffer,
    wb_coeffs: [SubPixel; 4],
) {
    let params = HighlightReconstructionParamsGpu {
        clip_g: wb_coeffs[1],
        _pad: [0.0; 3],
    };
    let shader_source = r#"
        struct Params {
            clip_g: f32,
            _pad0: f32,
            _pad1: f32,
            _pad2: f32,
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
            let factor = p.g / params.clip_g;
            let reconstructed_g = ((1.0 - factor) * p.g) + (factor * (p.r + p.b) * 0.5);
            pixels[index] = vec4<f32>(p.r, reconstructed_g, p.b, p.a);
        }
    "#;

    ctx.dispatch_compute_shader(
        "highlight_reconstruction",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_reconstruction() {
        let mut pixels = vec![[0.6, 0.4, 0.8]];
        let wb_coeffs = [1.0, 1.0, 1.0, 1.0];
        highlight_reconstruction(&mut pixels, wb_coeffs);
        
        let diff_r = (pixels[0][0] - 0.6).abs();
        let diff_g = (pixels[0][1] - 0.52).abs();
        let diff_b = (pixels[0][2] - 0.8).abs();
        
        assert!(diff_r < 1e-6);
        assert!(diff_g < 1e-6, "g is {}", pixels[0][1]);
        assert!(diff_b < 1e-6);
    }
}
