use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{Module, ModuleSchema, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct HighlightReconstruction {
}

impl PipelineModule for Module<HighlightReconstruction> {
    fn process_cpu(&self, image: &mut Image) {
        // Must run after CFACoeffs: the knee keys on post-WB channel values
        // against the normalized full-well clip of 1.0.
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
            description: "Desaturate clipped highlights toward neutral (post-WB, keys on hottest channel vs full well).".to_string(),
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
}
