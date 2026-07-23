use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct LumaGuidedChromaDenoise {
    pub radius: Parameter<usize>,
    pub epsilon: Parameter<f32>,
}

impl Default for LumaGuidedChromaDenoise {
    fn default() -> Self {
        Self {
            radius: Parameter::new(4, "Guided filter radius in pixels."),
            epsilon: Parameter::new(0.01, "Guided filter edge-preservation epsilon (linear light)."),
        }
    }
}

impl PipelineModule for Module<LumaGuidedChromaDenoise> {
    fn process_cpu(&self, image: &mut Image) {
        image.chroma_denoise(self.config.radius.value, self.config.epsilon.value);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        pichromatic::chroma_denoise::chroma_denoise_gpu(
            ctx,
            gpu_buf,
            self.config.radius.value,
            self.config.epsilon.value,
        );
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "LumaGuidedChromaDenoise".to_string(),
            description: "Apply luma-guided chroma denoising.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: LumaGuidedChromaDenoise = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid LumaGuidedChromaDenoise config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::common::test_pipeline_module_cpu_vs_gpu;

    #[test]
    fn test_luma_guided_chroma_denoise_module_cpu_vs_gpu() {
        let luma_guided_chroma_denoise_module = Module::<LumaGuidedChromaDenoise> {
            name: "LumaGuidedChromaDenoise".to_string(),
            cache: None,
            config: LumaGuidedChromaDenoise::default(),
        };

        test_pipeline_module_cpu_vs_gpu(&luma_guided_chroma_denoise_module, 333);
    }
}

