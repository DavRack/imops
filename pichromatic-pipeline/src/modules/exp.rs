use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Exp {
    pub ev: Parameter<SubPixel>,
}

impl Default for Exp {
    fn default() -> Self {
        Self {
            ev: Parameter::new_ranged(0.0, -6.0, 6.0, "Exposure compensation value in EV."),
        }
    }
}

impl PipelineModule for Module<Exp> {
    fn process_cpu(&self, image: &mut Image) {
        image.exp(self.config.ev.value);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        pichromatic::exp::exp_gpu(ctx, gpu_buf, self.config.ev.value);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Exp".to_string(),
            description: "Apply manual exposure compensation in EV.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: Exp = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Exp config");
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
    fn test_exp_module_cpu_vs_gpu() {
        let exp_module = Module::<Exp> {
            name: "Exp".to_string(),
            cache: None,
            config: Exp {
                ev: Parameter::new(1.5, "Exposure compensation value in EV."),
            },
        };

        test_pipeline_module_cpu_vs_gpu(&exp_module, 42);
    }
}
