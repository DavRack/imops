use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{Module, ModuleSchema, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct SigmoidToneMap {}

impl PipelineModule for Module<SigmoidToneMap> {
    fn process_cpu(&self, image: &mut Image) {
        image.sigmoid_tone_map();
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
    fn test_sigmoid_module_cpu_vs_gpu() {
        let sigmoid_module = Module::<SigmoidToneMap> {
            name: "SigmoidToneMap".to_string(),
            cache: None,
            config: SigmoidToneMap {},
        };

        test_pipeline_module_cpu_vs_gpu(&sigmoid_module, 888);
    }
}
