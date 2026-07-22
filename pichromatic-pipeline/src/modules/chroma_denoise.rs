use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChromaDenoise {
    pub intensity: Parameter<SubPixel>,
}

impl Default for ChromaDenoise {
    fn default() -> Self {
        Self {
            intensity: Parameter::new(0.0, "Chroma denoising intensity factor."),
        }
    }
}

impl PipelineModule for Module<ChromaDenoise> {
    fn process_cpu(&self, image: &mut Image) {
        image.chroma_bm3d(self.config.intensity.value);
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
        unimplemented!("GPU chroma BM3D is not implemented yet — use Backend::Cpu");
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "ChromaDenoise".to_string(),
            description: "Apply chroma-only BM3D denoising.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: ChromaDenoise = module.try_into().expect("Invalid ChromaDenoise config");
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
    #[ignore = "GPU chroma BM3D not implemented; Default intensity=0 made this a false green no-op"]
    fn test_chroma_denoise_module_cpu_vs_gpu() {
        let chroma_denoise_module = Module::<ChromaDenoise> {
            name: "ChromaDenoise".to_string(),
            cache: None,
            config: ChromaDenoise {
                intensity: Parameter::new(0.2, "Chroma denoising intensity factor."),
            },
        };

        test_pipeline_module_cpu_vs_gpu(&chroma_denoise_module, 333);
    }
}

