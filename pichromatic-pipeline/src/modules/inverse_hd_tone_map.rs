use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use pichromatic::image::{ExposureGain, HighlightCeiling};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct InverseHdToneMap {
    pub strength: Parameter<SubPixel>,
}

impl Default for InverseHdToneMap {
    fn default() -> Self {
        Self {
            strength: Parameter::new_ranged(
                1.0,
                0.0,
                3.0,
                "Inverse H&D tone curve strength / S-curve parameter.",
            ),
        }
    }
}

impl PipelineModule for Module<InverseHdToneMap> {
    fn process_cpu(&self, image: &mut Image) {
        let gain = image
            .metadata
            .extensions
            .get::<ExposureGain>()
            .map(|g| g.0)
            .unwrap_or(1.0);
        let ceiling = image
            .metadata
            .extensions
            .get::<HighlightCeiling>()
            .map(|c| c.0)
            .unwrap_or(4.0);
        image.inverse_hd_tone_map_with_gain(self.config.strength.value, gain, ceiling);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        let gain = meta
            .extensions
            .get::<ExposureGain>()
            .map(|g| g.0)
            .unwrap_or(1.0);
        let ceiling = meta
            .extensions
            .get::<HighlightCeiling>()
            .map(|c| c.0)
            .unwrap_or(4.0);
        pichromatic::tone_map::inverse_hd_tone_map_gpu_with_gain(
            ctx,
            gpu_buf,
            self.config.strength.value,
            gain,
            ceiling,
        );
    }

    fn process_gpu_async<'a>(
        &'a self,
        ctx: &'a pichromatic::gpu::GpuContext,
        gpu_buf: &'a pichromatic::gpu::GpuImageBuffer,
        meta: &'a mut pichromatic::image::ImageMetadata,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
        Box::pin(async move {
            self.process_gpu(ctx, gpu_buf, meta);
        })
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "InverseHdToneMap".to_string(),
            description: "Photographic inverse-H&D display tone map with ACES reference gamut compression and highlight desaturation (ACEScg).".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: InverseHdToneMap = serde_json::from_value(serde_json::Value::Object(module))
            .expect("Invalid InverseHdToneMap config");
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
    use crate::modules::common::{
        assert_images_equal_abs_tol, generate_test_image_512x512, test_pipeline_module_cpu_vs_gpu,
        CPU_GPU_ABS_TOLERANCE,
    };
    use pichromatic::gpu::GpuContext;
    use pichromatic::pixel::PixelOps;

    #[test]
    fn test_inverse_hd_module_cpu_without_exposure_gain() {
        let module = Module::<InverseHdToneMap>::default();
        let mut image = Image::default();
        let mid = pichromatic::pixel::MIDDLE_GRAY;
        image.rgb_data = vec![[mid, mid, mid]];

        module.process_cpu(&mut image);

        for c in 0..3 {
            assert!(
                (image.rgb_data[0][c] - mid).abs() < 1e-4,
                "InverseHdToneMap without ExposureGain should map midgray {c} to {mid}, got {}",
                image.rgb_data[0][c]
            );
        }
    }

    #[test]
    fn test_inverse_hd_module_cpu_with_exposure_gain() {
        let module = Module::<InverseHdToneMap>::default();
        let mut image = Image::default();
        let gain = 400.0;
        let mid = pichromatic::pixel::MIDDLE_GRAY;
        image.rgb_data = vec![[mid * gain, mid * gain, mid * gain]];
        image.metadata.extensions.insert(ExposureGain(gain));

        module.process_cpu(&mut image);

        for c in 0..3 {
            assert!(
                (image.rgb_data[0][c] - mid).abs() < 1e-4,
                "InverseHdToneMap with ExposureGain {gain} should map midgray {c} to {mid}, got {}",
                image.rgb_data[0][c]
            );
        }
    }

    #[test]
    fn test_inverse_hd_module_cpu_with_highlight_ceiling() {
        let module = Module::<InverseHdToneMap>::default();
        let mut image = Image::default();
        let gain = 400.0;
        let ceiling = 3.5;
        let mid = pichromatic::pixel::MIDDLE_GRAY;
        image.rgb_data = vec![
            [mid * gain, mid * gain, mid * gain],
            [ceiling * gain, ceiling * gain, ceiling * gain],
        ];
        image.metadata.extensions.insert(ExposureGain(gain));
        image.metadata.extensions.insert(HighlightCeiling(ceiling));

        module.process_cpu(&mut image);

        // Check midgray preservation
        for c in 0..3 {
            assert!(
                (image.rgb_data[0][c] - mid).abs() < 1e-4,
                "InverseHdToneMap with HighlightCeiling {ceiling} should map midgray {c} to {mid}, got {}",
                image.rgb_data[0][c]
            );
        }

        // Check highlight ceiling maps to display max luminance ~1.0
        let luma = image.rgb_data[1].luminance();
        assert!(
            (luma - 1.0).abs() < 1e-3,
            "InverseHdToneMap with HighlightCeiling {ceiling} should map ceiling input to luminance 1.0, got {luma}"
        );
    }

    #[test]
    fn test_inverse_hd_module_cpu_with_highlight_ceiling_without_gain() {
        let module = Module::<InverseHdToneMap>::default();
        let mut image = Image::default();
        let ceiling = 3.8;
        let mid = pichromatic::pixel::MIDDLE_GRAY;
        image.rgb_data = vec![
            [mid, mid, mid],
            [ceiling, ceiling, ceiling],
        ];
        image.metadata.extensions.insert(HighlightCeiling(ceiling));

        module.process_cpu(&mut image);

        // Check midgray preservation
        for c in 0..3 {
            assert!(
                (image.rgb_data[0][c] - mid).abs() < 1e-4,
                "InverseHdToneMap with default gain and HighlightCeiling {ceiling} should map midgray {c} to {mid}, got {}",
                image.rgb_data[0][c]
            );
        }

        // Check highlight ceiling maps to display max luminance ~1.0
        let luma = image.rgb_data[1].luminance();
        assert!(
            (luma - 1.0).abs() < 1e-3,
            "InverseHdToneMap with default gain and HighlightCeiling {ceiling} should map ceiling to luminance 1.0, got {luma}"
        );
    }

    #[test]
    fn test_inverse_hd_module_deserialization() {
        let json_str = r#"{"strength": 1.5}"#;
        let config: InverseHdToneMap = serde_json::from_str(json_str).unwrap();
        assert_eq!(config.strength.value, 1.5);
    }

    #[test]
    fn test_inverse_hd_module_cpu_vs_gpu() {
        let inverse_hd_module = Module::<InverseHdToneMap>::default();
        test_pipeline_module_cpu_vs_gpu(&inverse_hd_module, 888);
    }

    #[test]
    fn test_inverse_hd_module_cpu_vs_gpu_with_gain_and_ceiling() {
        let inverse_hd_module = Module::<InverseHdToneMap>::default();

        let ctx = GpuContext::new_sync();
        let mut seed_image = generate_test_image_512x512(888);
        let gain = 400.0;
        let ceiling = 3.5;
        seed_image.metadata.extensions.insert(ExposureGain(gain));
        seed_image.metadata.extensions.insert(HighlightCeiling(ceiling));
        for p in &mut seed_image.rgb_data {
            p[0] *= gain;
            p[1] *= gain;
            p[2] *= gain;
        }

        let mut cpu_pipeline_img = PipelineImage::Cpu(seed_image.clone());
        inverse_hd_module.process(&Backend::Cpu, &mut cpu_pipeline_img);
        let cpu_out = cpu_pipeline_img.to_cpu(None);

        let mut gpu_pipeline_img = PipelineImage::new_gpu(&ctx, &seed_image);
        inverse_hd_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_pipeline_img);
        let gpu_out = gpu_pipeline_img.to_cpu(Some(&ctx));

        assert_images_equal_abs_tol(&cpu_out, &gpu_out, CPU_GPU_ABS_TOLERANCE, 0);
    }
}
