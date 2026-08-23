use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use pichromatic::image::ExposureGain;
use super::{Module, ModuleSchema, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct SigmoidToneMap {}

impl PipelineModule for Module<SigmoidToneMap> {
    fn process_cpu(&self, image: &mut Image) {
        let gain = image
            .metadata
            .extensions
            .get::<ExposureGain>()
            .map(|g| g.0)
            .unwrap_or(1.0);
        image.sigmoid_tone_map_with_gain(gain);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        let gain = meta
            .extensions
            .get::<ExposureGain>()
            .map(|g| g.0)
            .unwrap_or(1.0);
        pichromatic::tone_map::sigmoid_gpu_with_gain(ctx, gpu_buf, gain);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "SigmoidToneMap".to_string(),
            description: "Sigmoid display tone map with ACES reference gamut compression (ACEScg).".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<SigmoidToneMap> {
            name: self.schema().name,
            cache: None,
            config: SigmoidToneMap {},
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, PipelineImage};
    use crate::modules::common::{
        assert_images_equal_abs_tol, generate_test_image_512x512, test_pipeline_module_cpu_vs_gpu,
        CPU_GPU_ABS_TOLERANCE,
    };
    use pichromatic::gpu::GpuContext;

    #[test]
    fn test_sigmoid_module_cpu_exposure_gain() {
        let module = Module::<SigmoidToneMap>::default();
        let mut image = Image::default();
        let gain = 400.0;
        let mid = pichromatic::pixel::MIDDLE_GRAY;
        image.rgb_data = vec![[mid * gain, mid * gain, mid * gain]];
        image.metadata.extensions.insert(ExposureGain(gain));

        module.process_cpu(&mut image);

        for c in 0..3 {
            assert!(
                (image.rgb_data[0][c] - mid).abs() < 1e-4,
                "SigmoidToneMap with ExposureGain {gain} should map midgray {c} to {mid}, got {}",
                image.rgb_data[0][c]
            );
        }
    }

    #[test]
    fn test_sigmoid_module_cpu_vs_gpu() {
        let sigmoid_module = Module::<SigmoidToneMap> {
            name: "SigmoidToneMap".to_string(),
            cache: None,
            config: SigmoidToneMap {},
        };

        test_pipeline_module_cpu_vs_gpu(&sigmoid_module, 888);
    }

    #[test]
    fn test_sigmoid_module_cpu_vs_gpu_with_exposure_gain() {
        let sigmoid_module = Module::<SigmoidToneMap> {
            name: "SigmoidToneMap".to_string(),
            cache: None,
            config: SigmoidToneMap {},
        };

        let ctx = GpuContext::new_sync();
        let mut seed_image = generate_test_image_512x512(888);
        let gain = 400.0;
        seed_image.metadata.extensions.insert(ExposureGain(gain));
        for p in &mut seed_image.rgb_data {
            p[0] *= gain;
            p[1] *= gain;
            p[2] *= gain;
        }

        let mut cpu_pipeline_img = PipelineImage::Cpu(seed_image.clone());
        sigmoid_module.process(&Backend::Cpu, &mut cpu_pipeline_img);
        let cpu_out = cpu_pipeline_img.to_cpu(None);

        let mut gpu_pipeline_img = PipelineImage::new_gpu(&ctx, &seed_image);
        sigmoid_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_pipeline_img);
        let gpu_out = gpu_pipeline_img.to_cpu(Some(&ctx));

        assert_images_equal_abs_tol(&cpu_out, &gpu_out, CPU_GPU_ABS_TOLERANCE, 0);
    }
}
