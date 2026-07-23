use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LCH {
    pub lc: Parameter<SubPixel>,
    pub cc: Parameter<SubPixel>,
    pub hc: Parameter<SubPixel>,
}

impl Default for LCH {
    fn default() -> Self {
        Self {
            lc: Parameter::new(1.0, "Lightness coefficient multiplier."),
            cc: Parameter::new(1.0, "Chroma (saturation) coefficient multiplier."),
            hc: Parameter::new(1.0, "Hue coefficient multiplier."),
        }
    }
}

impl PipelineModule for Module<LCH> {
    fn process_cpu(&self, image: &mut Image) {
        image.lch([
            self.config.lc.value,
            self.config.cc.value,
            self.config.hc.value,
        ]);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        let source_cs = meta
            .color_space
            .expect("LCH requires image metadata color_space");
        pichromatic::lch::lch_gpu(
            ctx,
            gpu_buf,
            source_cs,
            self.config.lc.value,
            self.config.cc.value,
            self.config.hc.value,
        );
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "LCH".to_string(),
            description: "Adjust lightness, chroma, and hue coefficients.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: LCH = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid LCH config");
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
    fn test_lch_module_cpu_vs_gpu() {
        let lch_module = Module::<LCH> {
            name: "LCH".to_string(),
            cache: None,
            config: LCH {
                lc: Parameter::new(1.1, "Lightness"),
                cc: Parameter::new(1.2, "Chroma"),
                hc: Parameter::new(1.0, "Hue"),
            },
        };

        test_pipeline_module_cpu_vs_gpu(&lch_module, 777);
    }
}
