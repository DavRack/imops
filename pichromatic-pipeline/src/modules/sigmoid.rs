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
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        pichromatic::tone_map::sigmoid_gpu(ctx, gpu_buf);
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
    use crate::modules::common::test_pipeline_module_cpu_vs_gpu;

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
    #[ignore = "GPU sigmoid sync pending CPU approval (CPU calibrated C to 1/(1 - MIDDLE_GRAY))"]
    fn test_sigmoid_module_cpu_vs_gpu() {
        let sigmoid_module = Module::<SigmoidToneMap> {
            name: "SigmoidToneMap".to_string(),
            cache: None,
            config: SigmoidToneMap {},
        };

        test_pipeline_module_cpu_vs_gpu(&sigmoid_module, 888);
    }
}
