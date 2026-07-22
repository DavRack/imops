use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use super::{Module, ModuleSchema, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BaselineExposureCompensation {
}

impl PipelineModule for Module<BaselineExposureCompensation> {
    fn process_cpu(&self, image: &mut Image) {
        use pichromatic::film::exposure::radiance::absolute_luminance_gain;
        use pichromatic::pixel::MIDDLE_GRAY;
        use rayon::prelude::*;

        // 1) DNG BaselineExposure EV (relative scale).
        let ev = image.metadata.baseline_exposure.unwrap_or(0.0);
        if ev.abs() > 1e-6 {
            image.exp(ev);
        }

        // 2) Camera-relative → absolute luminance via reflected-light meter equation:
        //    L = K * v * N² / (t * S)
        // Requires shutter / f-number / ISO from EXIF (filled by DNG metadata parse).
        let t = image.metadata.shutter_seconds;
        let n = image.metadata.f_number;
        let s = image.metadata.iso;
        match (t, n, s) {
            (Some(t), Some(n), Some(iso)) if t > 0.0 && n > 0.0 && iso > 0.0 => {
                let gain = absolute_luminance_gain(t as f64, n as f64, iso as f64) as f32;
                image.rgb_data.par_iter_mut().for_each(|px| {
                    *px = px.map(|c| c * gain);
                });
                let _ = MIDDLE_GRAY;
            }
            _ => {
                eprintln!(
                    "BaselineExposureCompensation: missing shutter/f-number/ISO; \
                     skipping absolute luminance conversion"
                );
            }
        }
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        use pichromatic::film::exposure::radiance::absolute_luminance_gain;

        let ev = meta.baseline_exposure.unwrap_or(0.0);
        let t = meta.shutter_seconds;
        let n = meta.f_number;
        let s = meta.iso;

        let gain = match (t, n, s) {
            (Some(t), Some(n), Some(iso)) if t > 0.0 && n > 0.0 && iso > 0.0 => {
                absolute_luminance_gain(t as f64, n as f64, iso as f64) as f32
            }
            _ => 1.0,
        };

        let total_ev = ev + gain.log2();
        if total_ev.abs() > 1e-6 {
            pichromatic::exp::exp_gpu(ctx, gpu_buf, total_ev);
        }
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "BaselineExposureCompensation".to_string(),
            description: "Apply DNG BaselineExposure EV, then convert camera-relative linear values to absolute luminance using EXIF shutter/f-number/ISO (L = K·v·N²/(t·S)).".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
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
    use crate::modules::common::{assert_images_equal, generate_test_image_512x512};
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

        assert_images_equal(&cpu_out, &gpu_out);
    }
}
