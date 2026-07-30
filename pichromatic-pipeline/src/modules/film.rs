use serde::{Deserialize, Serialize};
use pichromatic::film::{FilmFormat, FilmOutput, FilmParams, StockId};
use pichromatic::pixel::Image;
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Film {
    pub stock: Parameter<String>,
    pub film_format: Parameter<String>,
    pub seed: Parameter<u64>,
    pub output: Parameter<String>,
}

fn parse_stock(s: &str) -> Option<StockId> {
    Some(match s {
        "BwStub" => StockId::BwStub,
        "ColorNeg200" => StockId::ColorNeg200,
        "Portra400" => StockId::Portra400,
        "Ektar100" => StockId::Ektar100,
        "FujiPro400H" => StockId::FujiPro400H,
        "FujichromeVelvia100" => StockId::FujichromeVelvia100,
        "EktachromeE100" => StockId::EktachromeE100,
        "TriX400" => StockId::TriX400,
        _ => return None,
    })
}

fn parse_film_format(s: &str) -> Option<FilmFormat> {
    Some(match s {
        "Film35mm" => FilmFormat::Film35mm,
        "Film6x6" => FilmFormat::Film6x6,
        "Film4x5" => FilmFormat::Film4x5,
        _ => return None,
    })
}

fn parse_output(s: &str) -> Option<FilmOutput> {
    Some(match s {
        "NegativeLinear" => FilmOutput::NegativeLinear,
        "PositiveLinear" => FilmOutput::PositiveLinear,
        _ => return None,
    })
}

fn film_params_from_config(config: &Film) -> Option<FilmParams> {
    let stock = match parse_stock(config.stock.value.as_str()) {
        Some(s) => s,
        None => {
            web_sys_warn(&format!("Unknown film stock: {}", config.stock.value));
            return None;
        }
    };
    let film_format = match parse_film_format(config.film_format.value.as_str()) {
        Some(f) => f,
        None => {
            web_sys_warn(&format!("Unknown film format: {}", config.film_format.value));
            return None;
        }
    };
    let output = match parse_output(config.output.value.as_str()) {
        Some(o) => o,
        None => {
            web_sys_warn(&format!("Unknown film output: {}", config.output.value));
            return None;
        }
    };
    Some(FilmParams {
        stock,
        film_format,
        seed: config.seed.value,
        output,
    })
}

fn web_sys_warn(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::warn_1(&wasm_bindgen::JsValue::from_str(msg));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        eprintln!("{msg}");
    }
}

impl Default for Film {
    fn default() -> Self {
        Self {
            stock: Parameter::new_with_choices(
                "ColorNeg200".to_string(),
                "Film stock identifier.",
                vec![
                    "BwStub".to_string(),
                    "ColorNeg200".to_string(),
                    "Portra400".to_string(),
                    "Ektar100".to_string(),
                    "FujiPro400H".to_string(),
                    "FujichromeVelvia100".to_string(),
                    "EktachromeE100".to_string(),
                    "TriX400".to_string(),
                ],
            ),
            film_format: Parameter::new_with_choices(
                "Film35mm".to_string(),
                "Film format (sets pixel pitch from frame width).",
                vec![
                    "Film35mm".to_string(),
                    "Film6x6".to_string(),
                    "Film4x5".to_string(),
                ],
            ),
            seed: Parameter::new_ranged(1, 1, 100000, "RNG seed for grain (deterministic)."),
            output: Parameter::new_with_choices(
                "NegativeLinear".to_string(),
                "NegativeLinear: densitometric scanned negative. PositiveLinear: mid/Dmin invert from stock film base + mid-gray gain.",
                vec!["NegativeLinear".to_string(), "PositiveLinear".to_string()],
            ),
        }
    }
}

fn film_require_acescg(cs: Option<pichromatic::cst::ColorSpaceTag>, where_: &str) {
    use pichromatic::cst::ColorSpaceTag;
    if matches!(cs, Some(ColorSpaceTag::AcesCg)) {
        return;
    }
    // Skipping Film after BaselineExposureCompensation leaves absolute-luminance
    // values (~10²) that SigmoidToneMap compresses to ~1 → pure white on present.
    let msg = format!(
        "Film ({where_}): requires ACEScg after CST, got {cs:?}. \
         Skipping Film produces a near-white image."
    );
    web_sys_warn(&msg);
    panic!("{msg}");
}

impl PipelineModule for Module<Film> {
    fn process_cpu(&self, image: &mut Image) {
        film_require_acescg(image.metadata.color_space, "cpu");

        let Some(params) = film_params_from_config(&self.config) else {
            panic!("Film (cpu): invalid stock/format/output config");
        };

        pichromatic::film::process(image, &params).expect("Film process failed");
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        film_require_acescg(meta.color_space, "gpu");

        let Some(params) = film_params_from_config(&self.config) else {
            panic!("Film (gpu): invalid stock/format/output config");
        };

        pollster::block_on(pichromatic::film::process_gpu(ctx, gpu_buf, meta, &params))
            .expect("GPU film process failed");
    }

    fn process_gpu_async<'a>(
        &'a self,
        ctx: &'a pichromatic::gpu::GpuContext,
        gpu_buf: &'a pichromatic::gpu::GpuImageBuffer,
        meta: &'a mut pichromatic::image::ImageMetadata,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
        Box::pin(async move {
            film_require_acescg(meta.color_space, "gpu-async");

            let Some(params) = film_params_from_config(&self.config) else {
                panic!("Film (gpu-async): invalid stock/format/output config");
            };

            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(
                &format!(
                    "[Film] start stock={:?} format={:?} output={:?} {}x{}",
                    params.stock,
                    params.film_format,
                    params.output,
                    gpu_buf.width,
                    gpu_buf.height
                )
                .into(),
            );

            pichromatic::film::process_gpu(ctx, gpu_buf, meta, &params)
                .await
                .expect("GPU film process failed");

            // Ensure Film's queue submits are visible before Sigmoid/CST/present.
            // Without this, wasm can tone-map the pre-Film absolute-luminance
            // buffer (~white after sigmoid).
            #[cfg(target_arch = "wasm32")]
            {
                if let Err(e) = ctx.end_of_render_fence_async(&gpu_buf.buffer).await {
                    panic!("Film (gpu-async): post-film fence failed: {e}");
                }
                web_sys::console::log_1(&"[Film] done (fence ok)".into());
            }
        })
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Film".to_string(),
            description: "Physically-based analog film simulation. NegativeLinear exports the densitometric scan; PositiveLinear inverts with the stock film-base Dmin and mid-gray gain.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: Film = serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Film config");
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
    use pichromatic::cst::ColorSpaceTag;
    use pichromatic::gpu::GpuContext;

    #[test]
    fn test_film_module_cpu_vs_gpu() {
        let film_module = Module::<Film> {
            name: "Film".to_string(),
            cache: None,
            config: Film::default(),
        };

        let mut seed_image = generate_test_image_512x512(333);
        // Film requires ACEScg (pipeline: CST → Film).
        seed_image.cst(ColorSpaceTag::AcesCg);

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        film_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        film_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        assert_images_equal(&cpu_out, &gpu_out);
    }

    #[test]
    fn test_film_module_cpu_vs_gpu_positive_linear() {
        let mut film_config = Film::default();
        film_config.output.value = "PositiveLinear".to_string();
        let film_module = Module::<Film> {
            name: "Film".to_string(),
            cache: None,
            config: film_config,
        };

        let mut seed_image = generate_test_image_512x512(333);
        for (i, px) in seed_image.rgb_data.iter_mut().enumerate() {
            let scale = if i % 2 == 0 { 50.0 } else { 0.001 };
            px[0] *= scale;
            px[1] *= scale;
            px[2] *= scale;
        }
        seed_image.cst(ColorSpaceTag::AcesCg);

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        film_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        film_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        use crate::modules::common::assert_images_equal_tol;
        assert_images_equal_tol(&cpu_out, &gpu_out, 0.31);
    }

    #[test]
    fn test_film_deserialization() {
        let json_str = r#"{"stock": "Portra400", "film_format": "Film35mm", "seed": 1, "output": "PositiveLinear"}"#;
        let map: toml::Table = serde_json::from_str(json_str).unwrap();
        let _film: Film = map.try_into().unwrap();
    }
}
