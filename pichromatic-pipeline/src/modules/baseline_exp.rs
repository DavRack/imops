use super::{Module, ModuleSchema, PipelineModule};
use pichromatic::pixel::Image;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BaselineExposureCompensation {}

/// Used when shutter time, f-number, or ISO metadata is absent, non-finite, or non-positive.
/// This preserves the previous 800× calibration for inputs without usable camera metadata.
const FALLBACK_LUMINANCE_GAIN: f32 = 800.0;

fn exposure_ev(metadata: &pichromatic::image::ImageMetadata) -> f32 {
    let gain = match (metadata.shutter_seconds, metadata.f_number, metadata.iso) {
        (Some(t), Some(n), Some(iso))
            if t.is_finite()
                && n.is_finite()
                && iso.is_finite()
                && t > 0.0
                && n > 0.0
                && iso > 0.0 =>
        {
            pichromatic::film::exposure::radiance::absolute_luminance_gain(
                t as f64, n as f64, iso as f64,
            ) as f32
        }
        _ => FALLBACK_LUMINANCE_GAIN,
    };
    gain.log2()
        + metadata
            .baseline_exposure
            .filter(|ev| ev.is_finite())
            .unwrap_or(0.0)
}

impl PipelineModule for Module<BaselineExposureCompensation> {
    fn process_cpu(&self, image: &mut Image) {
        let ev = exposure_ev(&image.metadata);
        let gain = 2.0_f32.powf(ev);
        image
            .metadata
            .extensions
            .insert(pichromatic::image::ExposureGain(gain));
        image.exp(ev);
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        let ev = exposure_ev(meta);
        let gain = 2.0_f32.powf(ev);
        meta.extensions
            .insert(pichromatic::image::ExposureGain(gain));
        pichromatic::exp::exp_gpu(ctx, gpu_buf, ev);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "BaselineExposureCompensation".to_string(),
            description: "Convert camera-relative values to absolute luminance using exposure metadata and DNG BaselineExposure; use the documented 800× fallback when metadata is invalid.".to_string(),
            fields: vec![],
        }
    }

    fn create(
        &self,
        _module: serde_json::Map<String, serde_json::Value>,
    ) -> Box<dyn PipelineModule> {
        Box::new(Module::<BaselineExposureCompensation> {
            name: self.schema().name,
            cache: None,
            config: BaselineExposureCompensation {},
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, PipelineImage};
    use crate::modules::common::{
        assert_images_equal_abs_tol, generate_test_image_512x512, CPU_GPU_ABS_TOLERANCE,
    };
    use pichromatic::gpu::GpuContext;

    #[test]
    fn test_baseline_exp_module_cpu_vs_gpu() {
        let baseline_module = Module::<BaselineExposureCompensation> {
            name: "BaselineExposureCompensation".to_string(),
            cache: None,
            config: BaselineExposureCompensation {},
        };

        // Nonsense 512×512 RGB is fine — what matters is EXIF fields so the
        // module actually runs (otherwise both backends no-op and the test lies).
        let mut seed_image = generate_test_image_512x512(1);
        seed_image.metadata.baseline_exposure = Some(0.5);
        seed_image.metadata.shutter_seconds = Some(1.0 / 125.0);
        seed_image.metadata.f_number = Some(2.8);
        seed_image.metadata.iso = Some(400.0);

        let ctx = GpuContext::new_sync();

        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        baseline_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        baseline_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal_abs_tol(&cpu_out, &gpu_out, CPU_GPU_ABS_TOLERANCE, 0);
    }

    #[test]
    fn cpu_uses_camera_exposure_metadata_and_baseline_exposure() {
        let module = Module::<BaselineExposureCompensation>::default();
        let mut image = generate_test_image_512x512(1);
        let input = image.rgb_data[0];
        image.metadata.shutter_seconds = Some(1.0 / 125.0);
        image.metadata.f_number = Some(2.8);
        image.metadata.iso = Some(400.0);
        image.metadata.baseline_exposure = Some(0.5);

        module.process_cpu(&mut image);

        let gain = 12.5 * 2.8_f32.powi(2) / ((1.0 / 125.0) * 400.0) * 2.0_f32.powf(0.5);
        for (actual, expected) in image.rgb_data[0]
            .iter()
            .zip(input.map(|channel| channel * gain))
        {
            assert!((actual - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn invalid_camera_exposure_metadata_uses_800x_fallback() {
        let module = Module::<BaselineExposureCompensation>::default();
        let mut image = generate_test_image_512x512(1);
        let input = image.rgb_data[0];
        image.metadata.shutter_seconds = Some(0.0);
        image.metadata.f_number = Some(2.8);
        image.metadata.iso = Some(400.0);

        module.process_cpu(&mut image);

        for (actual, expected) in image.rgb_data[0]
            .iter()
            .zip(input.map(|channel| channel * FALLBACK_LUMINANCE_GAIN))
        {
            assert!((actual - expected).abs() < 1e-4);
        }
    }
}
