use serde::{Deserialize, Serialize};
use pichromatic::cst::ColorSpaceTag;
use pichromatic::pixel::Image;
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule, SUPPORTED_COLOR_SPACES};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CST {
    pub target_color_space: Parameter<String>,
}

impl Default for CST {
    fn default() -> Self {
        Self {
            target_color_space: Parameter::new_with_choices(
                "".to_string(),
                "Target color space to convert to.",
                SUPPORTED_COLOR_SPACES.iter().map(|cs| format!("{:?}", cs)).collect(),
            ),
        }
    }
}

impl PipelineModule for Module<CST> {
    fn process_cpu(&self, image: &mut Image) {
        let target_color_space: ColorSpaceTag = serde_plain::from_str(
            self.config.target_color_space.value.trim()
        ).expect(
            &format!("Color space not recognized: {}", self.config.target_color_space.value)
        );
        match image.metadata.color_space {
            Some(_) => { image.cst(target_color_space); },
            None => {
                image.camera_cst(
                    target_color_space,
                    &image.metadata.calibration_matrix_d65.clone().expect(
                        "A calibration matrix must be set to perform a camera cst transform"
                    )
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
        let target_color_space: ColorSpaceTag = serde_plain::from_str(
            self.config.target_color_space.value.trim()
        ).expect(
            &format!("Color space not recognized: {}", self.config.target_color_space.value)
        );
        match meta.color_space {
            Some(source_cs) => {
                pichromatic::cst::cst_gpu(ctx, gpu_buf, source_cs, target_color_space);
            }
            None => {
                if let Some(cal_matrix) = &meta.calibration_matrix_d65 {
                    pichromatic::cst::camera_cst_gpu(ctx, gpu_buf, target_color_space, cal_matrix);
                }
            }
        }
        // Must match CPU Image::cst / camera_cst — Film and later CSTs key off this.
        meta.color_space = Some(target_color_space);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "CST".to_string(),
            description: "Perform Color Space Transform.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: CST = module.try_into().expect("Invalid CST config");
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

    #[test]
    fn test_cst_module_cpu_vs_gpu() {
        let cst_module = Module::<CST> {
            name: "CST".to_string(),
            cache: None,
            config: CST {
                target_color_space: Parameter::new("AcesCg".to_string(), "Target CS"),
            },
        };

        let seed_image = generate_test_image_512x512(222);
        let ctx = GpuContext::new_sync();

        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        cst_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        cst_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal(&cpu_out, &gpu_out);
        assert_eq!(
            cpu_out.metadata.color_space,
            Some(ColorSpaceTag::AcesCg),
            "CPU CST must update metadata.color_space"
        );
        assert_eq!(
            gpu_out.metadata.color_space,
            Some(ColorSpaceTag::AcesCg),
            "GPU CST must update metadata.color_space (Film / later CSTs depend on it)"
        );
    }

    #[test]
    fn test_cst_module_cpu_vs_gpu_linear_srgb() {
        let cst_module = Module::<CST> {
            name: "CST".to_string(),
            cache: None,
            config: CST {
                target_color_space: Parameter::new("LinearSrgb".to_string(), "Target CS"),
            },
        };

        // Film pipeline ends with AcesCg → LinearSrgb before gamma encode.
        let mut seed_image = generate_test_image_512x512(223);
        seed_image.metadata.color_space = Some(ColorSpaceTag::AcesCg);
        let ctx = GpuContext::new_sync();

        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        cst_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        cst_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal(&cpu_out, &gpu_out);
        assert_eq!(
            cpu_out.metadata.color_space,
            Some(ColorSpaceTag::LinearSrgb),
            "CPU CST must update metadata.color_space"
        );
        assert_eq!(
            gpu_out.metadata.color_space,
            Some(ColorSpaceTag::LinearSrgb),
            "GPU CST must update metadata.color_space"
        );
    }

    #[test]
    fn test_cst_gpu_updates_color_space_for_camera_rgb() {
        // After demosaic, color_space is None and camera_cst is used.
        // GPU must still tag the buffer as the target space — otherwise the next
        // CST re-runs camera_cst and Film skips ACEScg.
        let cst_module = Module::<CST> {
            name: "CST".to_string(),
            cache: None,
            config: CST {
                target_color_space: Parameter::new("AcesCg".to_string(), "Target CS"),
            },
        };

        let mut seed_image = generate_test_image_512x512(7);
        seed_image.metadata.color_space = None;
        seed_image.metadata.calibration_matrix_d65 = Some(vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);

        let ctx = GpuContext::new_sync();
        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        cst_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));
        assert_eq!(gpu_out.metadata.color_space, Some(ColorSpaceTag::AcesCg));
    }
}
