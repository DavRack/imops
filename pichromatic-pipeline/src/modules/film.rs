use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};
use crate::backend::{Backend, PipelineImage};
use pichromatic::film::{FilmFormat, FilmOutput, FilmParams, StockId};
use pichromatic::pixel::Image;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Film {
    pub stock: Parameter<String>,
    pub film_format: Parameter<String>,
    pub render_width_mm: Parameter<Option<f32>>,
    pub seed: Parameter<u64>,
    pub enable_halation: Parameter<bool>,
    pub output: Parameter<String>,
    pub compensate_box_speed: Parameter<bool>,
    pub scanner_s_curve: Parameter<f32>,
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
        "CineStill50D" => StockId::CineStill50D,
        _ => return None,
    })
}

fn parse_film_format(s: &str) -> Option<FilmFormat> {
    Some(match s {
        "Film35mm" => FilmFormat::Film35mm,
        "Film6x6" => FilmFormat::Film6x6,
        "Film4x5" => FilmFormat::Film4x5,
        "FilmSuper16" | "Super16" => FilmFormat::FilmSuper16,
        "FilmStandard16" | "Standard16" | "Film16mm" | "16mm" => FilmFormat::FilmStandard16,
        "FilmSuper8" => FilmFormat::FilmSuper8,
        "FilmStandard8" => FilmFormat::FilmStandard8,
        "Film1mmDebug" => FilmFormat::Film1mmDebug,
        _ => return None,
    })
}

fn parse_output(s: &str) -> Option<FilmOutput> {
    Some(match s {
        "NegativeLinear" => FilmOutput::NegativeLinear,
        "PositiveLinear" => FilmOutput::PositiveLinear,
        "PositiveInverseHd" | "PositiveInverseHD" | "PositiveSceneLinear" => {
            FilmOutput::PositiveInverseHd
        }
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
            web_sys_warn(&format!(
                "Unknown film format: {}",
                config.film_format.value
            ));
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
        render_width_mm: config.render_width_mm.value,
        seed: config.seed.value,
        output,
        enable_halation: config.enable_halation.value,
        compensate_box_speed: config.compensate_box_speed.value,
        scanner_s_curve: config.scanner_s_curve.value,
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
                    "CineStill50D".to_string(),
                ],
            ),
            film_format: Parameter::new_with_choices(
                "Film35mm".to_string(),
                "Film format (sets pixel pitch from frame width).",
                vec![
                    "Film35mm".to_string(),
                    "Film6x6".to_string(),
                    "Film4x5".to_string(),
                    "FilmSuper16".to_string(),
                    "FilmStandard16".to_string(),
                    "FilmSuper8".to_string(),
                    "FilmStandard8".to_string(),
                    "Film1mmDebug".to_string(),
                ],
            ),
            render_width_mm: Parameter::new(
                None,
                "Optional physical render width in millimetres; overrides the film format width.",
            ),
            seed: Parameter::new_ranged(1, 1, 100000, "RNG seed for grain (deterministic)."),
            enable_halation: Parameter::new(
                true,
                "Enable wide backing halation (reflectance bounce) in the film exposure.",
            ),
            output: Parameter::new_with_choices(
                "NegativeLinear".to_string(),
                "NegativeLinear: densitometric scanned negative. PositiveLinear: bounded invert from processed Dmin + neutral mid-gray scan. PositiveInverseHd: scene-referred HDR inverse-H&D reconstruction.",
                vec![
                    "NegativeLinear".to_string(),
                    "PositiveLinear".to_string(),
                    "PositiveInverseHd".to_string(),
                ],
            ),
            compensate_box_speed: Parameter::new(
                true,
                "Normalize exposure across stocks to the capture ISO (scene-relative fluence, independent of stock box speed). Off keeps the raw box-speed difference: faster stocks look brighter for the same input.",
            ),
            scanner_s_curve: Parameter::new_ranged(
                0.0,
                0.0,
                3.0,
                "Scanner S-curve contrast tune factor (0.0 = linear HDR, 1.0 = standard scanner S-curve).",
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
    fn process(&self, backend: &Backend, image: &mut PipelineImage) {
        if self.config.render_width_mm.value.is_some() {
            match backend {
                Backend::Cpu => {
                    let cpu_img = image.ensure_cpu(None);
                    self.process_cpu(cpu_img);
                }
                Backend::Wgpu(ctx) => {
                    let cpu_img = image.ensure_cpu(Some(ctx));
                    self.process_cpu(cpu_img);
                }
            }
            return;
        }

        match backend {
            Backend::Cpu => {
                let cpu_img = image.ensure_cpu(None);
                self.process_cpu(cpu_img);
            }
            Backend::Wgpu(ctx) => {
                let (gpu_buf, meta) = image.ensure_gpu(ctx);
                self.process_gpu(ctx, gpu_buf, meta);
            }
        }
    }

    fn process_async<'a>(
        &'a self,
        backend: &'a Backend,
        image: &'a mut PipelineImage,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
        Box::pin(async move {
            if self.config.render_width_mm.value.is_some() {
                match backend {
                    Backend::Cpu => {
                        let cpu_img = image.ensure_cpu(None);
                        self.process_cpu(cpu_img);
                    }
                    Backend::Wgpu(ctx) => {
                        let cpu_img = image.ensure_cpu(Some(ctx));
                        self.process_cpu(cpu_img);
                    }
                }
                return;
            }

            match backend {
                Backend::Cpu => {
                    let cpu_img = image.ensure_cpu(None);
                    self.process_cpu(cpu_img);
                }
                Backend::Wgpu(ctx) => {
                    let (gpu_buf, meta) = image.ensure_gpu(ctx);
                    self.process_gpu_async(ctx, gpu_buf, meta).await;
                }
            }
        })
    }

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
        if params.output == FilmOutput::PositiveLinear {
            meta.extensions
                .insert(pichromatic::image::ExposureGain(1.0));
        }
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
                    params.stock, params.film_format, params.output, gpu_buf.width, gpu_buf.height
                )
                .into(),
            );

            pichromatic::film::process_gpu(ctx, gpu_buf, meta, &params)
                .await
                .expect("GPU film process failed");

            if params.output == FilmOutput::PositiveLinear {
                meta.extensions
                    .insert(pichromatic::image::ExposureGain(1.0));
            }

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
            description: "Physically-based analog film simulation. NegativeLinear exports the densitometric scan; PositiveLinear applies a bounded technical invert from processed Dmin and a neutral mid-gray scan.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(
        &self,
        module: serde_json::Map<String, serde_json::Value>,
    ) -> Box<dyn PipelineModule> {
        let config: Film =
            serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Film config");
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
        assert_images_equal_abs_tol, generate_test_image_512x512, CPU_GPU_ABS_TOLERANCE,
    };
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

        assert_images_equal_abs_tol(&cpu_out, &gpu_out, CPU_GPU_ABS_TOLERANCE, 0);
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
        seed_image.cst(ColorSpaceTag::AcesCg);
        let dr_stops = 20;
        for (i, px) in seed_image.rgb_data.iter_mut().enumerate() {
            let scale = if i % 2 == 0 {
                2.0_f32.powi(dr_stops / 2)
            } else {
                1.0 / 2.0_f32.powi(dr_stops / 2)
            };
            px[0] *= scale;
            px[1] *= scale;
            px[2] *= scale;
        }

        let ctx = GpuContext::new_sync();
        let mut cpu_img = PipelineImage::Cpu(seed_image.clone());
        film_module.process(&Backend::Cpu, &mut cpu_img);
        let cpu_out = cpu_img.to_cpu(None);

        let mut gpu_img = PipelineImage::new_gpu(&ctx, &seed_image);
        film_module.process(&Backend::Wgpu(ctx.clone()), &mut gpu_img);
        let gpu_out = gpu_img.to_cpu(Some(&ctx));

        // This stress test spans a 20-stop dynamic range (2^-10 to 2^10), causing
        // the exponential invert stage to produce extreme values (E ≈ 46,000) where
        // exponential derivative amplification magnifies sub-ULP density domain
        // differences (inherent CPU vs GPU hardware transcendentals and FMA contraction).
        // A tolerance of 512.0 * f32::EPSILON achieves 99.994% relative parity across
        // all 7 non-linear pipeline stages.
        assert_images_equal_abs_tol(&cpu_out, &gpu_out, CPU_GPU_ABS_TOLERANCE, 0);
    }

    #[test]
    fn test_film_deserialization() {
        let json_str = r#"{"stock": "Portra400", "film_format": "Film35mm", "render_width_mm": 1.0, "seed": 1, "output": "PositiveLinear"}"#;
        let map: toml::Table = serde_json::from_str(json_str).unwrap();
        let film: Film = map.try_into().unwrap();
        assert_eq!(film.render_width_mm.value, Some(1.0));
        let params = film_params_from_config(&film).unwrap();
        assert_eq!(params.render_width_mm, Some(1.0));

        let default: Film = serde_json::from_str(
            r#"{"stock": "Portra400", "film_format": "Film35mm", "seed": 1, "output": "PositiveLinear"}"#,
        )
        .unwrap();
        assert_eq!(default.render_width_mm.value, None);

        for alias in ["PositiveInverseHd", "PositiveInverseHD", "PositiveSceneLinear"] {
            let json = format!(r#"{{"output": "{alias}", "scanner_s_curve": 1.5}}"#);
            let film: Film = serde_json::from_str(&json).unwrap();
            let params = film_params_from_config(&film).unwrap();
            assert_eq!(params.output, FilmOutput::PositiveInverseHd);
            assert_eq!(params.scanner_s_curve, 1.5);
        }

        for (name, expected) in [
            ("FilmSuper16", FilmFormat::FilmSuper16),
            ("Super16", FilmFormat::FilmSuper16),
            ("FilmStandard16", FilmFormat::FilmStandard16),
            ("Standard16", FilmFormat::FilmStandard16),
            ("Film16mm", FilmFormat::FilmStandard16),
            ("16mm", FilmFormat::FilmStandard16),
        ] {
            let json = format!(r#"{{"film_format": "{name}"}}"#);
            let film: Film = serde_json::from_str(&json).unwrap();
            let params = film_params_from_config(&film).unwrap();
            assert_eq!(params.film_format, expected);
        }
    }
}
