use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BM3D {
    pub intensity: Parameter<SubPixel>,
}

impl Default for BM3D {
    fn default() -> Self {
        Self {
            intensity: Parameter::new_ranged(0.0, 0.0, 1.0, "Denoising intensity factor."),
        }
    }
}

impl PipelineModule for Module<BM3D> {
    fn process_cpu(&self, image: &mut Image) {
        image.bm3d(self.config.intensity.value);
    }

    fn process_gpu(
        &self,
        _ctx: &pichromatic::gpu::GpuContext,
        _gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        let intensity = self.config.intensity.value;
        if intensity <= 0.0 {
            return;
        }
        unimplemented!("GPU BM3D is not implemented yet — use Backend::Cpu");
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "BM3D".to_string(),
            description: "Apply BM3D image denoising.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: BM3D = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid BM3D config");
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
    #[ignore = "to implement"]
    fn test_bm3d_module_cpu_vs_gpu() {
        let bm3d_module = Module::<BM3D> {
            name: "BM3D".to_string(),
            cache: None,
            config: BM3D {
                intensity: Parameter::new(0.2, "Denoising intensity factor."),
            },
        };

        test_pipeline_module_cpu_vs_gpu(&bm3d_module, 333);
    }
}

