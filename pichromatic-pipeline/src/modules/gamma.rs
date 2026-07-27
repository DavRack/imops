use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

/// Power-law display encode (sRGB ≈ γ 2.2, BT.1886 = γ 2.4).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Gamma {
    pub gamma: Parameter<SubPixel>,
}

impl Default for Gamma {
    fn default() -> Self {
        Self {
            gamma: Parameter::new_ranged(2.2, 0.1, 4.0, "Display gamma (2.2 ≈ sRGB, 2.4 = BT.1886)."),
        }
    }
}

impl PipelineModule for Module<Gamma> {
    fn process_cpu(&self, image: &mut Image) {
        image.gamma(self.config.gamma.value);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        pichromatic::tone_map::gamma_encode_gpu(ctx, gpu_buf, self.config.gamma.value);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Gamma".to_string(),
            description: "Power-law display encode: out = linear^(1/γ). Use γ=2.2 for sRGB-like, γ=2.4 for BT.1886. Apply after CST to a linear display space.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: Gamma = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Gamma config");
        Box::new(Module::<Gamma> {
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
    fn test_gamma_module_cpu_vs_gpu() {
        let gamma_module = Module::<Gamma> {
            name: "Gamma".to_string(),
            cache: None,
            config: Gamma {
                gamma: Parameter::new(2.2, "Display gamma."),
            },
        };

        test_pipeline_module_cpu_vs_gpu(&gamma_module, 12345);
    }
}
