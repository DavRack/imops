use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{Module, ModuleSchema, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BaselineExposureCompensation {
}

/// Standard film baseline calibration scale factor in EV (+9.643856 EV = log2(800.0)).
/// Maps camera raw middle-gray (v ≈ 0.18) to calibrated film mid-gray luminance (144.0 cd/m²).
pub const FILM_BASELINE_SCALE_EV: f32 = 9.643856;

impl PipelineModule for Module<BaselineExposureCompensation> {
    fn process_cpu(&self, image: &mut Image) {
        let total_ev = image.metadata.baseline_exposure.unwrap_or(0.0) + FILM_BASELINE_SCALE_EV;
        image.exp(total_ev);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        let total_ev = meta.baseline_exposure.unwrap_or(0.0) + FILM_BASELINE_SCALE_EV;
        pichromatic::exp::exp_gpu(ctx, gpu_buf, total_ev);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "BaselineExposureCompensation".to_string(),
            description: "Apply DNG BaselineExposure EV compensation plus film baseline calibration scale factor (+9.64 EV).".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<BaselineExposureCompensation> {
            name: self.schema().name,
            cache: None,
            config: BaselineExposureCompensation {},
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
    fn test_baseline_exp_module_cpu_vs_gpu() {
        let baseline_module = Module::<BaselineExposureCompensation> {
            name: "BaselineExposureCompensation".to_string(),
            cache: None,
            config: BaselineExposureCompensation {},
        };

        // Nonsense 512×512 RGB is fine — what matters is EXIF fields so the
        // module actually runs (otherwise both backends no-op and the test lies).
        let mut seed_image = generate_test_image_512x512(1);
        seed_image.metadata.baseline_exposure = Some(0.5);
        seed_image.metadata.shutter_seconds = Some(1.0 / 125.0);
        seed_image.metadata.f_number = Some(2.8);
        seed_image.metadata.iso = Some(400.0);

        let ctx = GpuContext::new_sync();

        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        baseline_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        baseline_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal(&cpu_out, &gpu_out);
    }
}
