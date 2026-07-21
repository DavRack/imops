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
                // Mark that buffer is now absolute luminance. Mid-gray relative 0.185
                // under sunny-16 metering → L ≈ K·0.185·N²/(t·S) ≈ 148.
                let _ = MIDDLE_GRAY;
            }
            _ => {
                // No capture exposure metadata — leave relative. Film will still run
                // but absolute calibration assumes Baseline ran with EXIF present.
                eprintln!(
                    "BaselineExposureCompensation: missing shutter/f-number/ISO; \
                     skipping absolute luminance conversion"
                );
            }
        }
        image
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct SigmoidToneMap {}

impl PipelineModule for Module<SigmoidToneMap> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        image.sigmoid_tone_map()
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "SigmoidToneMap".to_string(),
            description: "Sigmoid display tone map with ACES reference gamut compression (ACEScg).".to_string(),
            fields: vec![],
        }
    }

    fn create(&self, _module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        Box::new(Module::<SigmoidToneMap> {
            name: self.schema().name,
            cache: None,
            config: SigmoidToneMap {},
        })
    }
}

/// Power-law display encode (sRGB ≈ γ 2.2, BT.1886 = γ 2.4).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Gamma {
    pub gamma: Parameter<SubPixel>,
}

impl Default for Gamma {
    fn default() -> Self {
        Self {
            gamma: Parameter::new(2.2, "Display gamma (2.2 ≈ sRGB, 2.4 = BT.1886)."),
        }
    }
}

impl PipelineModule for Module<Gamma> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        image.gamma(self.config.gamma.value)
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Gamma".to_string(),
            description: "Power-law display encode: out = linear^(1/γ). Use γ=2.2 for sRGB-like, γ=2.4 for BT.1886. Apply after CST to a linear display space.".to_string(),
            fields: fields_from_config(&Gamma::default()),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: Gamma = module.try_into().expect("Invalid Gamma config");
        Box::new(Module::<Gamma> {
            name: self.schema().name,
            cache: None,
            config,
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        use pichromatic::film::{FilmFormat, FilmOutput, FilmParams, StockId};

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
        image
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Film".to_string(),
            description: "Physically-based analog film simulation. NegativeLinear exports the densitometric scan; PositiveLinear inverts with the stock film-base Dmin and mid-gray gain.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: toml::map::Map<String, toml::Value>) -> Box<dyn PipelineModule> {
        let config: Film = module.try_into().expect("Invalid Film config");
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
        Box::new(Module::<Film>::default()),
        Box::new(Module::<LCH>::default()),
        Box::new(Module::<SigmoidToneMap>::default()),
        Box::new(Module::<Gamma>::default()),
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
