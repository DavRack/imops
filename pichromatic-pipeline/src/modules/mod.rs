use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use pichromatic::cst::ColorSpaceTag;

#[cfg(test)]
pub mod common;
pub mod exp;
pub mod gamma;
pub mod contrast;
pub mod lch;
pub mod sigmoid;
pub mod vignette;
pub mod cst;
pub mod baseline_exp;
pub mod film;
pub mod demosaic;
pub mod bm3d;
pub mod cfa_coeffs;
pub mod chroma_denoise;
pub mod luma_guided_chroma_denoise;
pub mod highlight_reconstruction;

pub use exp::Exp;
pub use gamma::Gamma;
pub use contrast::Contrast;
pub use lch::LCH;
pub use sigmoid::SigmoidToneMap;
pub use vignette::Vignette;
pub use cst::CST;
pub use baseline_exp::BaselineExposureCompensation;
pub use film::Film;
pub use demosaic::Demosaic;
pub use bm3d::BM3D;
pub use cfa_coeffs::CFACoeffs;
pub use chroma_denoise::ChromaDenoise;
pub use luma_guided_chroma_denoise::LumaGuidedChromaDenoise;
pub use highlight_reconstruction::HighlightReconstruction;

// ─── Parameter wrapper type ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter<T> {
    pub value: T,
    pub description: &'static str,
    pub choices: Option<Vec<String>>,
}

impl<T> Parameter<T> {
    pub fn new(value: T, description: &'static str) -> Self {
        Self { value, description, choices: None }
    }
    pub fn new_with_choices(value: T, description: &'static str, choices: Vec<String>) -> Self {
        Self { value, description, choices: Some(choices) }
    }
}

impl<T: Serialize> Serialize for Parameter<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Parameter", 3)?;
        state.serialize_field("value", &self.value)?;
        state.serialize_field("description", &self.description)?;
        state.serialize_field("choices", &self.choices)?;
        state.end()
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Parameter<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let json_val = serde_json::Value::deserialize(deserializer)?;
        match json_val {
            serde_json::Value::Object(mut map) => {
                if let Some(val_json) = map.remove("value") {
                    let value = T::deserialize(val_json).map_err(serde::de::Error::custom)?;
                    let description = match map.remove("description") {
                        Some(serde_json::Value::String(s)) => Box::leak(s.into_boxed_str()),
                        _ => "",
                    };
                    let choices = match map.remove("choices") {
                        Some(val) => serde_json::from_value(val).ok(),
                        None => None,
                    };
                    Ok(Self { value, description, choices })
                } else {
                    let value = T::deserialize(serde_json::Value::Object(map)).map_err(serde::de::Error::custom)?;
                    Ok(Self { value, description: "", choices: None })
                }
            }
            primitive => {
                let value = T::deserialize(primitive).map_err(serde::de::Error::custom)?;
                Ok(Self { value, description: "", choices: None })
            }
        }
    }
}

// ─── Enums & Options ────────────────────────────────────────────────────────

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DemosaicAlgorithmType {
    Amaze,
    Markesteijn,
    Fast,
    SuperFast,
    SuperSuperFast,
}

impl<'de> Deserialize<'de> for DemosaicAlgorithmType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_lowercase().as_str() {
            "amaze" => Ok(Self::Amaze),
            "markesteijn" => Ok(Self::Markesteijn),
            "fast" => Ok(Self::Fast),
            "superfast" => Ok(Self::SuperFast),
            "supersuperfast" => Ok(Self::SuperSuperFast),
            _ => Err(serde::de::Error::custom(format!(
                "unknown demosaic algorithm '{s}', expected one of: amaze, markesteijn, fast, superfast, supersuperfast"
            ))),
        }
    }
}

impl Default for DemosaicAlgorithmType {
    fn default() -> Self {
        Self::Amaze
    }
}

impl DemosaicAlgorithmType {
    pub const VARIANTS: &'static [Self] = &[
        Self::Amaze,
        Self::Markesteijn,
        Self::Fast,
        Self::SuperFast,
        Self::SuperSuperFast,
    ];

    pub fn to_str(self) -> &'static str {
        match self {
            Self::Amaze => "amaze",
            Self::Markesteijn => "markesteijn",
            Self::Fast => "fast",
            Self::SuperFast => "superfast",
            Self::SuperSuperFast => "supersuperfast",
        }
    }
}

pub const SUPPORTED_COLOR_SPACES: &[ColorSpaceTag] = &[
    ColorSpaceTag::Srgb,
    ColorSpaceTag::AcesCg,
    ColorSpaceTag::Oklch,
    ColorSpaceTag::XyzD65,
];

// ─── Reflection Schemas ──────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModuleSchema {
    pub name: String,
    pub description: String,
    pub fields: Vec<FieldSchema>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FieldSchema {
    pub name: String,
    pub field_type: String,
    pub default_value: serde_json::Value,
    pub description: String,
    pub choices: Option<Vec<String>>,
}

pub fn fields_from_config<C: Serialize>(config: &C) -> Vec<FieldSchema> {
    let val = match serde_json::to_value(config) {
        Ok(v) => v,
        Err(e) => panic!("fields_from_config to_value failed: {}", e),
    };
    let mut fields = Vec::new();

    if let serde_json::Value::Object(map) = val {
        for (field_name, field_val) in map {
            if let serde_json::Value::Object(param_map) = field_val {
                let value = param_map.get("value").cloned().unwrap_or(serde_json::Value::Null);
                let description = param_map.get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let choices = param_map.get("choices")
                    .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());

                let field_type = match &value {
                    serde_json::Value::Bool(_) => "boolean",
                    serde_json::Value::Number(n) => {
                        if n.is_f64() {
                            "float"
                        } else {
                            "integer"
                        }
                    }
                    serde_json::Value::String(_) => "string",
                    serde_json::Value::Array(_) => "array",
                    _ => "object",
                }.to_string();

                fields.push(FieldSchema {
                    name: field_name,
                    field_type,
                    default_value: value,
                    description,
                    choices,
                });
            }
        }
    }
    fields
}

// ─── PipelineModule Trait & Wrapper Struct ──────────────────────────────────

pub trait PipelineModule {
    fn process(&self, backend: &crate::backend::Backend, image: &mut crate::backend::PipelineImage) {
        match backend {
            crate::backend::Backend::Cpu => {
                let cpu_img = image.ensure_cpu(None);
                self.process_cpu(cpu_img);
            }
            crate::backend::Backend::Wgpu(ctx) => {
                let (gpu_buf, meta) = image.ensure_gpu(ctx);
                self.process_gpu(ctx, gpu_buf, meta);
            }
        }
    }

    fn process_async<'a>(
        &'a self,
        backend: &'a crate::backend::Backend,
        image: &'a mut crate::backend::PipelineImage,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
        Box::pin(async move {
            match backend {
                crate::backend::Backend::Cpu => {
                    let cpu_img = image.ensure_cpu(None);
                    self.process_cpu(cpu_img);
                }
                crate::backend::Backend::Wgpu(ctx) => {
                    let (gpu_buf, meta) = image.ensure_gpu(ctx);
                    self.process_gpu_async(ctx, gpu_buf, meta).await;
                }
            }
        })
    }

    fn process_cpu(&self, image: &mut Image);

    /// Native GPU implementation. Must not download and run CPU.
    /// Modules without a GPU path leave this unimplemented; select `Backend::Cpu` instead.
    /// `meta` is mutable so modules can update color_space and other fields (same as CPU).
    fn process_gpu(
        &self,
        _ctx: &pichromatic::gpu::GpuContext,
        _gpu_buf: &pichromatic::gpu::GpuImageBuffer,
        _meta: &mut pichromatic::image::ImageMetadata,
    ) {
        unimplemented!(
            "GPU backend selected but {} has no native GPU path — use Backend::Cpu",
            self.schema().name
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

    fn name(&self) -> String {
        self.schema().name
    }

    fn schema(&self) -> ModuleSchema;

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule>;
}

pub struct Module<T: Debug> {
    pub name: String,
    pub cache: Option<Image>,
    pub config: T,
}

impl<T: Default + Debug> Default for Module<T> {
    fn default() -> Self {
        let full_type_name = std::any::type_name::<T>();
        let name = full_type_name.split("::").last().unwrap_or(full_type_name).to_string();
        Self {
            name,
            cache: None,
            config: T::default(),
        }
    }
}

pub fn get_default_modules() -> Vec<Box<dyn PipelineModule>> {
    vec![
        Box::new(Module::<Demosaic>::default()),
        Box::new(Module::<ChromaDenoise>::default()),
        Box::new(Module::<LumaGuidedChromaDenoise>::default()),
        Box::new(Module::<CFACoeffs>::default()),
        Box::new(Module::<Vignette>::default()),
        Box::new(Module::<HighlightReconstruction>::default()),
        Box::new(Module::<BaselineExposureCompensation>::default()),
        Box::new(Module::<Exp>::default()),
        Box::new(Module::<Contrast>::default()),
        Box::new(Module::<CST>::default()),
        Box::new(Module::<Film>::default()),
        Box::new(Module::<LCH>::default()),
        Box::new(Module::<SigmoidToneMap>::default()),
        Box::new(Module::<Gamma>::default()),
        Box::new(Module::<BM3D>::default()),
    ]
}

pub fn get_pipeline_schema() -> Vec<ModuleSchema> {
    get_default_modules()
        .iter()
        .map(|m| m.schema())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demosaic_algorithm_type_case_insensitive() {
        let amaze_lower: DemosaicAlgorithmType = serde_json::from_str("\"amaze\"").unwrap();
        let amaze_cap: DemosaicAlgorithmType = serde_json::from_str("\"Amaze\"").unwrap();
        let amaze_upper: DemosaicAlgorithmType = serde_json::from_str("\"AMAZE\"").unwrap();

        assert_eq!(amaze_lower, DemosaicAlgorithmType::Amaze);
        assert_eq!(amaze_cap, DemosaicAlgorithmType::Amaze);
        assert_eq!(amaze_upper, DemosaicAlgorithmType::Amaze);

        let mark_mixed: DemosaicAlgorithmType = serde_json::from_str("\"MarkEsteijn\"").unwrap();
        assert_eq!(mark_mixed, DemosaicAlgorithmType::Markesteijn);

        let invalid: Result<DemosaicAlgorithmType, _> = serde_json::from_str("\"nonexistent\"");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_parse_config_with_capitalized_demosaic_algorithm() {
        let json_config = r#"{
            "pipeline_modules": [
                { "name": "Demosaic", "algorithm": "Amaze" }
            ]
        }"#;
        let parsed = crate::config::parse_config(json_config.to_string());
        assert_eq!(parsed.pipeline_modules.len(), 1);
        assert_eq!(parsed.pipeline_modules[0].schema().name, "Demosaic");
    }
}
