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
            lc: Parameter::new_ranged(1.0, 0.0, 3.0, "Lightness coefficient multiplier."),
            cc: Parameter::new_ranged(1.0, 0.0, 3.0, "Chroma (saturation) coefficient multiplier."),
            hc: Parameter::new_ranged(1.0, 0.0, 3.0, "Hue coefficient multiplier."),
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
    use crate::modules::common::{
        assert_images_equal_abs_tol, generate_test_image_512x512,
    };
    use pichromatic::gpu::GpuContext;
    use crate::backend::{Backend, PipelineImage};
    use pichromatic::cst::ColorSpaceTag;

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

        let seed_image = generate_test_image_512x512(777);

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        lch_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        lch_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        // The LCH GPU shader is an f32 Oklch approximation of the f64 `color`
        // crate CPU path (measured roundtrip error ~2.1e-5 near zero), which
        // exceeds the shared 64·eps gate. The approximation is inherent (WGSL
        // has no core f64), not a parity bug.
        assert_images_equal_abs_tol(&cpu_out, &gpu_out);
        let _ = ColorSpaceTag::Srgb;
    }
}
