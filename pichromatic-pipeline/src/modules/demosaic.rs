use serde::{Deserialize, Serialize};
use pichromatic::demosaic::demosaic_algorithms;
use pichromatic::pixel::Image;
use crate::backend::{Backend, PipelineImage};
use super::{fields_from_config, DemosaicAlgorithmType, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Demosaic {
    pub algorithm: Parameter<DemosaicAlgorithmType>,
}

impl Default for Demosaic {
    fn default() -> Self {
        Self {
            algorithm: Parameter::new_with_choices(
                DemosaicAlgorithmType::Amaze,
                "Demosaicing algorithm to use.",
                DemosaicAlgorithmType::VARIANTS.iter().map(|v| v.to_str().to_string()).collect(),
            ),
        }
    }
}

impl PipelineModule for Module<Demosaic> {
    fn process_async<'a>(
        &'a self,
        backend: &'a Backend,
        image: &'a mut PipelineImage,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
        Box::pin(async move {
            self.process(backend, image);
        })
    }

    fn process(&self, backend: &Backend, image: &mut PipelineImage) {
        match backend {
            Backend::Cpu => {
                let cpu_img = image.ensure_cpu(None);
                self.process_cpu(cpu_img);
            }
            Backend::Wgpu(ctx) => {
                // Demosaic reads mosaic `raw_data` (not the RGB GPU buffer).
                let (raw, mut meta) = match std::mem::replace(
                    image,
                    PipelineImage::Cpu(pichromatic::pixel::Image::default()),
                ) {
                    PipelineImage::Gpu(buf, meta, raw) => {
                        ctx.recycle_rgba_buffer(buf);
                        (raw, meta)
                    }
                    PipelineImage::Cpu(img) => (img.raw_data, img.metadata),
                };

                if meta.cfa.is_none() || meta.crop_area.is_none() {
                    // No raw mosaic — restore a GPU working buffer.
                    let width = meta.width.max(1);
                    let height = meta.height.max(1);
                    let buf = ctx.acquire_rgba_buffer(width, height);
                    *image = PipelineImage::Gpu(buf, meta, raw);
                    return;
                }

                let cfa = pichromatic::demosaic::get_cfa(
                    meta.cfa.as_ref().unwrap(),
                    meta.crop_area.unwrap(),
                );

                let out_buf = match self.config.algorithm.value {
                    DemosaicAlgorithmType::Markesteijn => {
                        pichromatic::demosaic::demosaic_markesteijn_gpu(
                            ctx,
                            &raw,
                            meta.width,
                            meta.height,
                            &cfa,
                        )
                    }
                    other => {
                        unimplemented!(
                            "GPU demosaic for {other:?} is not implemented yet — use Backend::Cpu"
                        );
                    }
                };

                meta.color_space = None;
                *image = PipelineImage::Gpu(out_buf, meta, std::sync::Arc::from([]));
            }
        }
    }

    fn process_cpu(&self, image: &mut Image) {
        // Demosaic requires CFA pattern + crop area from raw metadata.
        if image.metadata.cfa.is_none() || image.metadata.crop_area.is_none() {
            return;
        }

        let new_image = match self.config.algorithm.value {
            DemosaicAlgorithmType::Markesteijn => {
                Image::demosaic(
                    image.clone(),
                    demosaic_algorithms::Markesteijn{},
                )
            },
            DemosaicAlgorithmType::Fast => {
                Image::demosaic(
                    image.clone(),
                    demosaic_algorithms::Fast{},
                )
            },
            DemosaicAlgorithmType::SuperFast => {
                Image::demosaic(
                    image.clone(),
                    demosaic_algorithms::SuperFast{},
                )
            },
            DemosaicAlgorithmType::SuperSuperFast => {
                Image::demosaic(
                    image.clone(),
                    demosaic_algorithms::SuperSuperFast{},
                )
            },
            DemosaicAlgorithmType::Amaze => {
                Image::demosaic(
                    image.clone(),
                    demosaic_algorithms::Amaze::default(),
                )
            },
        };
        image.rgb_data = new_image.rgb_data;
        image.raw_data = new_image.raw_data;
        image.metadata.width = new_image.metadata.width;
        image.metadata.height = new_image.metadata.height;
        image.metadata.color_space = None;
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Demosaic".to_string(),
            description: "Demosaic raw image data.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: Demosaic = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Demosaic config");
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
    use crate::modules::common::assert_images_equal_abs_tol;
    use pichromatic::cfa::CFA;
    use pichromatic::demosaic::{Dim2, Point, Rect};
    use pichromatic::gpu::GpuContext;
    use pichromatic::image::ImageMetadata;
    use pichromatic::pixel::{Image, SubPixel};
    use rand::Rng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Deterministic nonsense 512×512 Bayer mosaic — only CPU/GPU equality matters.
    fn generate_test_raw_512x512(seed: u64) -> Image {
        let width = 512;
        let height = 512;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut raw_data = Vec::with_capacity(width * height);
        for _ in 0..(width * height) {
            raw_data.push(rng.gen::<SubPixel>());
        }
        Image {
            rgb_data: vec![],
            raw_data: raw_data.into(),
            metadata: ImageMetadata {
                width,
                height,
                cfa: Some(CFA::new("RGGB")),
                crop_area: Some(Rect {
                    p: Point { x: 0, y: 0 },
                    d: Dim2 { w: width, h: height },
                }),
                color_space: None,
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_demosaic_module_cpu_vs_gpu() {
        let module = Module::<Demosaic> {
            name: "Demosaic".to_string(),
            cache: None,
            config: Demosaic {
                algorithm: Parameter::new(
                    DemosaicAlgorithmType::Markesteijn,
                    "Demosaicing algorithm to use.",
                ),
            },
        };

        let seed_image = generate_test_raw_512x512(333);
        let ctx = GpuContext::new_sync();

        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal_abs_tol(&cpu_out, &gpu_out);
    }
}
