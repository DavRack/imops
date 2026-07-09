use std::fmt::Debug;

use pichromatic::demosaic::demosaic_algorithms;
use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use pichromatic::cst::ColorSpaceTag;

// ─── Parameter wrapper type (with Serde compatibility) ───────────────────────

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
        let v = serde_json::Value::deserialize(deserializer)?;
        match v {
            serde_json::Value::Object(mut map) => {
                let value_json = map.remove("value")
                    .ok_or_else(|| serde::de::Error::missing_field("value"))?;
                let value = T::deserialize(value_json).map_err(serde::de::Error::custom)?;
                let description = match map.remove("description") {
                    Some(serde_json::Value::String(s)) => Box::leak(s.into_boxed_str()),
                    _ => "",
                };
                let choices = match map.remove("choices") {
                    Some(val) => serde_json::from_value(val).ok(),
                    None => None,
                };
                Ok(Self { value, description, choices })
            }
            primitive => {
                let value = T::deserialize(primitive).map_err(serde::de::Error::custom)?;
                Ok(Self { value, description: "", choices: None })
            }
        }
    }
}

// ─── Enums and Options part of the code itself ────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DemosaicAlgorithmType {
    Amaze,
    Markesteijn,
    Fast,
    SuperFast,
    SuperSuperFast,
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
    pub field_type: String, // "float", "integer", "string", "boolean", "array"
    pub default_value: serde_json::Value,
    pub description: String,
    pub choices: Option<Vec<String>>,
}

pub fn fields_from_config<C: Serialize>(config: &C) -> Vec<FieldSchema> {
    let val = serde_json::to_value(config).unwrap();
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


pub trait PipelineModule {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image;

    fn schema(&self) -> ModuleSchema;

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule>;
}


pub struct Module<T: Debug>{
    pub name: String,
    pub cache: Option<Image>,
    pub config: T,
    // pub mask: Option<Box<dyn Mask>>
}

impl<T: Default + Debug> Default for Module<T> {
    fn default() -> Self {
        Self {
            name: std::any::type_name::<T>().split("::").last().unwrap().to_string(),
            cache: None,
            config: T::default(),
        }
    }
}


// impl<T> Module<T>{
//     pub fn from_toml<'a>(module: Map<String, toml::Value>) -> Box<Self>
// where
//         T: Deserialize<'a> + Default,
//         Self: Sized
//     {
//         let cfg: T = module.clone().try_into::<T>().expect(any::type_name::<Self>());
//         let module = Module{
//             name: module["name"].to_string(),
//             cache: None,
//             config: cfg,
//             mask: None
//         };
//         Box::new(module)
//     }
// }



#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LCH {
    pub lc: Parameter<SubPixel>,
    pub cc: Parameter<SubPixel>,
    pub hc: Parameter<SubPixel>,
}

impl Default for LCH {
    fn default() -> Self {
        Self {
            lc: Parameter::new(1.0, "Lightness coefficient multiplier."),
            cc: Parameter::new(1.0, "Chroma (saturation) coefficient multiplier."),
            hc: Parameter::new(1.0, "Hue coefficient multiplier."),
        }
    }
}

impl PipelineModule for Module<LCH> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
         image.lch([
            self.config.lc.value,
            self.config.cc.value,
            self.config.hc.value,
        ])
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "LCH".to_string(),
            description: "Adjust lightness, chroma, and hue coefficients.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: LCH = module.try_into().expect("Invalid LCH config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct HighlightReconstruction {
}

impl PipelineModule for Module<HighlightReconstruction> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.highlight_reconstruction(
            image.metadata.wb_coeffs.expect(
                "wb_coeffs needs to be set to perform highlight_reconstruction"
            )
        )
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "HighlightReconstruction".to_string(),
            description: "Reconstruct clipped highlight details using white balance coefficients.".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<HighlightReconstruction> {
            name: self.schema().name,
            cache: None,
            config: HighlightReconstruction {},
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Exp {
    pub ev: Parameter<SubPixel>
}

impl Default for Exp {
    fn default() -> Self {
        Self {
            ev: Parameter::new(0.0, "Exposure compensation value in EV."),
        }
    }
}

impl PipelineModule for Module<Exp> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.exp(self.config.ev.value)
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Exp".to_string(),
            description: "Apply manual exposure compensation in EV.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: Exp = module.try_into().expect("Invalid Exp config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BaselineExposureCompensation {
}

impl PipelineModule for Module<BaselineExposureCompensation> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        let ev = image.metadata.baseline_exposure.unwrap_or(0.0);
        if ev.abs() > 1e-6 {
            return image.exp(ev)
        }
        image
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "BaselineExposureCompensation".to_string(),
            description: "Automatically apply baseline exposure compensation from DNG metadata.".to_string(),
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ToneMap {
}

impl PipelineModule for Module<ToneMap> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.tone_map()
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "ToneMap".to_string(),
            description: "Apply legacy sigmoid tone mapping curve (kept for backward compatibility).".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<ToneMap> {
            name: self.schema().name,
            cache: None,
            config: ToneMap {},
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Contrast {
    pub c: Parameter<SubPixel>
}

impl Default for Contrast {
    fn default() -> Self {
        Self {
            c: Parameter::new(1.0, "Contrast adjustment factor."),
        }
    }
}

impl PipelineModule for Module<Contrast> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.contrast(self.config.c.value)
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Contrast".to_string(),
            description: "Adjust image contrast.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: Contrast = module.try_into().expect("Invalid Contrast config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}



// ponytail: BM3D module
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BM3D {
    pub intensity: Parameter<SubPixel>,
}

impl Default for BM3D {
    fn default() -> Self {
        Self {
            intensity: Parameter::new(0.0, "Denoising intensity factor."),
        }
    }
}

impl PipelineModule for Module<BM3D> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.bm3d(self.config.intensity.value)
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "BM3D".to_string(),
            description: "Apply BM3D image denoising.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: BM3D = module.try_into().expect("Invalid BM3D config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CFACoeffs {
}

impl PipelineModule for Module<CFACoeffs> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.cfa_coeffs(
            image.metadata.wb_coeffs.expect(
                "wb_coeffs needs to be set to perform highlight_reconstruction"
            )
        )
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "CFACoeffs".to_string(),
            description: "Apply Color Filter Array (CFA) white balance coefficients.".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<CFACoeffs> {
            name: self.schema().name,
            cache: None,
            config: CFACoeffs {},
        })
    }
}

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
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        let target_color_space: ColorSpaceTag = serde_plain::from_str(
            self.config.target_color_space.value.trim()
        ).expect(
            &format!("Color space not recognized: {}", self.config.target_color_space.value)
        );
        return match image.metadata.color_space {
            Some(_) => image.cst(target_color_space),
            None => image.camera_cst(
                target_color_space,
                &image.metadata.calibration_matrix_d65.clone().expect(
                    "A calibration matrix must be set to perform a camera cst transform"
                )
            )
        }
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        println!("DEBUG DEMOSAIC: starting algorithm = {:?}, width = {}, height = {}", 
                 self.config.algorithm.value, image.metadata.width, image.metadata.height);
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
        image.metadata.width = new_image.metadata.width;
        image.metadata.height = new_image.metadata.height;
        image.metadata.color_space = None;
        return image;
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Demosaic".to_string(),
            description: "Demosaic raw image data.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: Demosaic = module.try_into().expect("Invalid Demosaic config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChromaDenoise {
    pub intensity: Parameter<SubPixel>,
}

impl Default for ChromaDenoise {
    fn default() -> Self {
        Self {
            intensity: Parameter::new(0.0, "Chroma denoising intensity factor."),
        }
    }
}

impl PipelineModule for Module<ChromaDenoise> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.chroma_bm3d(self.config.intensity.value)
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "ChromaDenoise".to_string(),
            description: "Apply chroma-only BM3D denoising.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: ChromaDenoise = module.try_into().expect("Invalid ChromaDenoise config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LumaGuidedChromaDenoise {
    pub radius: Parameter<usize>,
    pub epsilon: Parameter<f32>,
}

impl Default for LumaGuidedChromaDenoise {
    fn default() -> Self {
        Self {
            radius: Parameter::new(4, "Guided filter radius in pixels."),
            epsilon: Parameter::new(0.01, "Guided filter edge-preservation epsilon (linear light)."),
        }
    }
}

impl PipelineModule for Module<LumaGuidedChromaDenoise> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.chroma_denoise(self.config.radius.value, self.config.epsilon.value)
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "LumaGuidedChromaDenoise".to_string(),
            description: "Apply luma-guided chroma denoising.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: LumaGuidedChromaDenoise = module.try_into().expect("Invalid LumaGuidedChromaDenoise config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Vignette {
    pub strength: Parameter<f32>,
}

impl Default for Vignette {
    fn default() -> Self {
        Self {
            strength: Parameter::new(1.0, "Correction strength modifier."),
        }
    }
}

impl PipelineModule for Module<Vignette> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.vignette(self.config.strength.value)
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Vignette".to_string(),
            description: "Apply radial vignette correction.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: Vignette = module.try_into().expect("Invalid Vignette config");
        Box::new(Module {
            name: self.schema().name,
            cache: None,
            config,
        })
    }
}

/// Dynamic list of all default pipeline modules acting as templates.
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
        Box::new(Module::<LCH>::default()),
        Box::new(Module::<ToneMap>::default()),
        Box::new(Module::<BM3D>::default()),
    ]
}

/// Generates a list of all available pipeline module schemas.
pub fn get_pipeline_schema() -> Vec<ModuleSchema> {
    get_default_modules()
        .iter()
        .map(|m| m.schema())
        .collect()
}
