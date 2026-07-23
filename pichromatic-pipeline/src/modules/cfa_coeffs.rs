use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{Module, ModuleSchema, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CFACoeffs {
}

impl PipelineModule for Module<CFACoeffs> {
    fn process_cpu(&self, image: &mut Image) {
        if let Some(wb_coeffs) = image.metadata.wb_coeffs {
            image.cfa_coeffs(wb_coeffs);
        }
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        if let Some(wb_coeffs) = meta.wb_coeffs {
            pichromatic::cfa_coeffs::cfa_coeffs_gpu(ctx, gpu_buf, wb_coeffs);
        }
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "CFACoeffs".to_string(),
            description: "Apply Color Filter Array (CFA) white balance coefficients.".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<CFACoeffs> {
            name: self.schema().name,
            cache: None,
            config: CFACoeffs {},
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, PipelineImage};
    use crate::modules::common::{assert_images_equal, generate_test_image_512x512};
    use pichromatic::gpu::GpuContext;

    #[test]
    fn test_cfa_coeffs_module_cpu_vs_gpu() {
        let cfa_coeffs_module = Module::<CFACoeffs> {
            name: "CFACoeffs".to_string(),
            cache: None,
            config: CFACoeffs {},
        };

        let mut seed_image = generate_test_image_512x512(333);
        seed_image.metadata.wb_coeffs = Some([2.0, 1.0, 1.5, 1.0]);

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        cfa_coeffs_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        cfa_coeffs_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal(&cpu_out, &gpu_out);
    }
}
