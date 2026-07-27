use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Vignette {
    pub strength: Parameter<f32>,
}

impl Default for Vignette {
    fn default() -> Self {
        Self {
            strength: Parameter::new_ranged(1.0, 0.0, 3.0, "Correction strength modifier."),
        }
    }
}

impl PipelineModule for Module<Vignette> {
    fn process_cpu(&self, image: &mut Image) {
        if let Some(opcode_list3) = &image.metadata.opcode_list3 {
            pichromatic::vignette::apply_vignette_radial_correction(
                &mut image.rgb_data,
                image.metadata.width,
                image.metadata.height,
                opcode_list3,
                self.config.strength.value,
            );
        }
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        if let Some(opcode_list3) = &meta.opcode_list3 {
            pichromatic::vignette::apply_vignette_radial_correction_gpu(
                ctx,
                gpu_buf,
                opcode_list3,
                self.config.strength.value,
            );
        }
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Vignette".to_string(),
            description: "Apply radial vignette correction.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: Vignette = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Vignette config");
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
    use crate::backend::{Backend, PipelineImage};
    use crate::modules::common::{assert_images_equal, generate_test_image_512x512};
    use pichromatic::gpu::GpuContext;

    /// Minimal DNG OpcodeList3 with one FixVignetteRadial (id=3) opcode.
    fn fake_fix_vignette_radial_opcode() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_be_bytes()); // count
        bytes.extend_from_slice(&3u32.to_be_bytes()); // opcode_id = FixVignetteRadial
        bytes.extend_from_slice(&1u32.to_be_bytes()); // version
        bytes.extend_from_slice(&0u32.to_be_bytes()); // flags
        bytes.extend_from_slice(&56u32.to_be_bytes()); // parameter_size (7×f64)
        for v in [0.35f64, 0.1, 0.0, 0.0, 0.0, 0.5, 0.5] {
            bytes.extend_from_slice(&v.to_be_bytes());
        }
        bytes
    }

    #[test]
    fn test_vignette_module_cpu_vs_gpu() {
        let vignette_module = Module::<Vignette> {
            name: "Vignette".to_string(),
            cache: None,
            config: Vignette {
                strength: Parameter::new(1.0, "Correction strength modifier."),
            },
        };

        let mut seed_image = generate_test_image_512x512(111);
        seed_image.metadata.opcode_list3 = Some(fake_fix_vignette_radial_opcode());

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        vignette_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        vignette_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal(&cpu_out, &gpu_out);
    }
}
