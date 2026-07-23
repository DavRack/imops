use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Contrast {
    pub c: Parameter<SubPixel>,
}

impl Default for Contrast {
    fn default() -> Self {
        Self {
            c: Parameter::new(1.0, "Contrast adjustment factor."),
        }
    }
}

impl PipelineModule for Module<Contrast> {
    fn process_cpu(&self, image: &mut Image) {
        pichromatic::contrast::contrast(&mut image.rgb_data, self.config.c.value);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        pichromatic::contrast::contrast_gpu(ctx, gpu_buf, self.config.c.value);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Contrast".to_string(),
            description: "Adjust image contrast.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: Contrast = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Contrast config");
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
    fn test_contrast_module_cpu_vs_gpu() {
        let contrast_module = Module::<Contrast> {
            name: "Contrast".to_string(),
            cache: None,
            config: Contrast {
                c: Parameter::new(1.8, "Contrast adjustment factor."),
            },
        };

        test_pipeline_module_cpu_vs_gpu(&contrast_module, 999);
    }
}
