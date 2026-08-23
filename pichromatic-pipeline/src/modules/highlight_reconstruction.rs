use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{Module, ModuleSchema, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct HighlightReconstruction {
}

impl PipelineModule for Module<HighlightReconstruction> {
    fn process_cpu(&self, image: &mut Image) {
        // Must run after CFACoeffs: reconstruction keys on post-WB channel
        // values against the normalized full-well clip of 1.0.
        image.highlight_reconstruction();
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        pichromatic::highlight_reconstruction::highlight_reconstruction_gpu(ctx, gpu_buf);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "HighlightReconstruction".to_string(),
            description: "Rebuild clipped channels from neighboring unclipped ratios (LibRaw-style); keeps recovered highlights colored.".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<HighlightReconstruction> {
            name: self.schema().name,
            cache: None,
            config: HighlightReconstruction {},
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, PipelineImage};
    use crate::modules::common::{
        assert_images_equal_abs_tol, generate_test_image_512x512, CPU_GPU_ABS_TOLERANCE,
    };
    use pichromatic::gpu::GpuContext;

    #[test]
    fn test_highlight_reconstruction_module_cpu_vs_gpu() {
        let highlight_reconstruction_module = Module::<HighlightReconstruction> {
            name: "HighlightReconstruction".to_string(),
            cache: None,
            config: HighlightReconstruction {},
        };

        let seed_image = generate_test_image_512x512(333);

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        highlight_reconstruction_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        highlight_reconstruction_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal_abs_tol(&cpu_out, &gpu_out, CPU_GPU_ABS_TOLERANCE, 0);
    }

    /// The uniform random fixture above never clips (values < 1.0), so this
    /// test exercises the actual reconstruction path on CPU and GPU: a
    /// gradient field with hot spots clipping 1 and 2 channels plus an
    /// all-channels-clipped block.
    #[test]
    fn test_highlight_reconstruction_module_cpu_vs_gpu_with_clipped_pixels() {
        let highlight_reconstruction_module = Module::<HighlightReconstruction> {
            name: "HighlightReconstruction".to_string(),
            cache: None,
            config: HighlightReconstruction {},
        };

        let width = 64usize;
        let height = 64usize;
        let mut rgb_data = vec![[0.0f32; 3]; width * height];
        for y in 0..height {
            for x in 0..width {
                let s = (x + y) as f32 / ((width + height) as f32);
                rgb_data[y * width + x] = [s * 0.9 + 0.02, s * 0.7 + 0.02, s * 0.5 + 0.02];
            }
        }
        // Hot spots: single-channel clip in warm area, two-channel clip,
        // three-channel blown block (interior must stay untouched on both
        // backends identically).
        rgb_data[10 * width + 10] = [1.35, 0.62, 0.4];
        rgb_data[30 * width + 40] = [0.8, 1.2, 1.1];
        for y in 50..54 {
            for x in 20..24 {
                rgb_data[y * width + x] = [1.6, 1.5, 1.45];
            }
        }
        let seed_image = pichromatic::pixel::Image {
            rgb_data,
            raw_data: std::sync::Arc::from([]),
            metadata: pichromatic::image::ImageMetadata {
                width,
                height,
                ..Default::default()
            },
        };

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        highlight_reconstruction_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        highlight_reconstruction_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal_abs_tol(&cpu_out, &gpu_out, CPU_GPU_ABS_TOLERANCE, 0);
    }
}
