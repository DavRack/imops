use serde::{Deserialize, Serialize};
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
            seed: Parameter::new(1, "RNG seed for grain (deterministic)."),
            output: Parameter::new_with_choices(
                "NegativeLinear".to_string(),
                "NegativeLinear: densitometric scanned negative. PositiveLinear: mid/Dmin invert from stock film base + mid-gray gain.",
                vec!["NegativeLinear".to_string(), "PositiveLinear".to_string()],
            ),
        }
    }
}

impl PipelineModule for Module<Film> {
    fn process_cpu(&self, image: &mut Image) {
        use pichromatic::cst::ColorSpaceTag;
        use pichromatic::film::{FilmFormat, FilmOutput, FilmParams, StockId};

        // Film requires scene-linear ACEScg (pipeline: CST → Film).
        if !matches!(image.metadata.color_space, Some(ColorSpaceTag::AcesCg)) {
            return;
        }

        let stock = match self.config.stock.value.as_str() {
            "BwStub" => StockId::BwStub,
            "ColorNeg200" => StockId::ColorNeg200,
            "Portra400" => StockId::Portra400,
            "Ektar100" => StockId::Ektar100,
            "FujiPro400H" => StockId::FujiPro400H,
            "EktachromeE100" => StockId::EktachromeE100,
            "TriX400" => StockId::TriX400,
            other => panic!("Unknown film stock: {other}"),
        };
        let film_format = match self.config.film_format.value.as_str() {
            "Film35mm" => FilmFormat::Film35mm,
            "Film6x6" => FilmFormat::Film6x6,
            "Film4x5" => FilmFormat::Film4x5,
            other => panic!("Unknown film format: {other}"),
        };
        let output = match self.config.output.value.as_str() {
            "NegativeLinear" => FilmOutput::NegativeLinear,
            "PositiveLinear" => FilmOutput::PositiveLinear,
            other => panic!("Unknown film output: {other}"),
        };

        let params = FilmParams {
            stock,
            film_format,
            seed: self.config.seed.value,
            output,
        };

        pichromatic::film::process(image, &params).expect("Film process failed");
    }

    fn process_gpu(
        &self,
        ctx: &pichromatic::gpu::GpuContext,
        gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        meta: &mut pichromatic::image::ImageMetadata,
    ) {
        use pichromatic::cst::ColorSpaceTag;
        use pichromatic::film::{FilmFormat, FilmOutput, FilmParams, StockId};

        // Film requires ACEScg; without it both backends no-op (CST must run first).
        if !matches!(meta.color_space, Some(ColorSpaceTag::AcesCg)) {
            return;
        }

        let stock = match self.config.stock.value.as_str() {
            "BwStub" => StockId::BwStub,
            "ColorNeg200" => StockId::ColorNeg200,
            "Portra400" => StockId::Portra400,
            "Ektar100" => StockId::Ektar100,
            "FujiPro400H" => StockId::FujiPro400H,
            "EktachromeE100" => StockId::EktachromeE100,
            "TriX400" => StockId::TriX400,
            other => panic!("Unknown film stock: {other}"),
        };
        let film_format = match self.config.film_format.value.as_str() {
            "Film35mm" => FilmFormat::Film35mm,
            "Film6x6" => FilmFormat::Film6x6,
            "Film4x5" => FilmFormat::Film4x5,
            other => panic!("Unknown film format: {other}"),
        };
        let output = match self.config.output.value.as_str() {
            "NegativeLinear" => FilmOutput::NegativeLinear,
            "PositiveLinear" => FilmOutput::PositiveLinear,
            other => panic!("Unknown film output: {other}"),
        };

        let params = FilmParams {
            stock,
            film_format,
            seed: self.config.seed.value,
            output,
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
            use pichromatic::cst::ColorSpaceTag;
            use pichromatic::film::{FilmFormat, FilmOutput, FilmParams, StockId};

            if !matches!(meta.color_space, Some(ColorSpaceTag::AcesCg)) {
                return;
            }

            let stock = match self.config.stock.value.as_str() {
                "BwStub" => StockId::BwStub,
                "ColorNeg200" => StockId::ColorNeg200,
                "Portra400" => StockId::Portra400,
                "Ektar100" => StockId::Ektar100,
                "FujiPro400H" => StockId::FujiPro400H,
                "EktachromeE100" => StockId::EktachromeE100,
                "TriX400" => StockId::TriX400,
                other => panic!("Unknown film stock: {other}"),
            };
            let film_format = match self.config.film_format.value.as_str() {
                "Film35mm" => FilmFormat::Film35mm,
                "Film6x6" => FilmFormat::Film6x6,
                "Film4x5" => FilmFormat::Film4x5,
                other => panic!("Unknown film format: {other}"),
            };
            let output = match self.config.output.value.as_str() {
                "NegativeLinear" => FilmOutput::NegativeLinear,
                "PositiveLinear" => FilmOutput::PositiveLinear,
                other => panic!("Unknown film output: {other}"),
            };

            let params = FilmParams {
                stock,
                film_format,
                seed: self.config.seed.value,
                output,
            };

            pichromatic::film::process_gpu(ctx, gpu_buf, meta, &params).await
                .expect("GPU film process failed");
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
    fn test_film_deserialization() {
        let json_str = r#"{"stock": "Portra400", "film_format": "Film35mm", "seed": 1, "output": "PositiveLinear"}"#;
        let map: toml::Table = serde_json::from_str(json_str).unwrap();
        let _film: Film = map.try_into().unwrap();
    }
}
