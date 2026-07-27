use serde::{Deserialize, Serialize};
use pichromatic::image::ImageOrientation;
use pichromatic::pixel::Image;
use pichromatic::rotation::{rotate_gpu, QuarterTurn};
use crate::backend::{Backend, PipelineImage};
use super::{fields_from_config, Module, ModuleSchema, Parameter, PipelineModule};

/// User-facing rotation angle for the Rotation module.
#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RotationAngle {
    Auto,
    #[serde(rename = "90")]
    Deg90,
    #[serde(rename = "180")]
    Deg180,
    #[serde(rename = "270")]
    Deg270,
}

impl<'de> Deserialize<'de> for RotationAngle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "90" => Ok(Self::Deg90),
            "180" => Ok(Self::Deg180),
            "270" => Ok(Self::Deg270),
            _ => Err(serde::de::Error::custom(format!(
                "unknown rotation '{s}', expected one of: auto, 90, 180, 270"
            ))),
        }
    }
}

impl Default for RotationAngle {
    fn default() -> Self {
        Self::Auto
    }
}

impl RotationAngle {
    pub const VARIANTS: &'static [Self] =
        &[Self::Auto, Self::Deg90, Self::Deg180, Self::Deg270];

    pub fn to_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Deg90 => "90",
            Self::Deg180 => "180",
            Self::Deg270 => "270",
        }
    }

    fn resolve(self, orientation: ImageOrientation) -> Option<QuarterTurn> {
        match self {
            Self::Auto => QuarterTurn::from_orientation(orientation),
            Self::Deg90 => Some(QuarterTurn::Deg90),
            Self::Deg180 => Some(QuarterTurn::Deg180),
            Self::Deg270 => Some(QuarterTurn::Deg270),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct Rotation {
    pub angle: Parameter<RotationAngle>,
}

impl Default for Rotation {
    fn default() -> Self {
        Self {
            angle: Parameter::new_with_choices(
                RotationAngle::Auto,
                "Image rotation. Auto uses EXIF/DNG orientation (90/180/270 only).",
                RotationAngle::VARIANTS
                    .iter()
                    .map(|v| v.to_str().to_string())
                    .collect(),
            ),
        }
    }
}

impl Module<Rotation> {
    fn apply_cpu(turn: QuarterTurn, image: &mut Image) {
        image.rotate(turn);
        // Avoid double-applying EXIF orientation on a second pass.
        image.metadata.orientation = ImageOrientation::Normal;
    }
}

impl PipelineModule for Module<Rotation> {
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
                let Some(turn) = self.config.angle.value.resolve(cpu_img.metadata.orientation) else {
                    return;
                };
                Self::apply_cpu(turn, cpu_img);
            }
            Backend::Wgpu(ctx) => {
                let orientation = match image {
                    PipelineImage::Gpu(_, meta, _) => meta.orientation,
                    PipelineImage::Cpu(img) => img.metadata.orientation,
                };
                let Some(turn) = self.config.angle.value.resolve(orientation) else {
                    return;
                };

                let (old_buf, mut meta, raw) = match std::mem::replace(
                    image,
                    PipelineImage::Cpu(Image::default()),
                ) {
                    PipelineImage::Gpu(buf, meta, raw) => (buf, meta, raw),
                    PipelineImage::Cpu(img) => {
                        let width = img.metadata.width.max(1);
                        let height = img.metadata.height.max(1);
                        let buf = ctx.acquire_rgba_buffer(width, height);
                        if !img.rgb_data.is_empty() {
                            ctx.update_buffer_from_image(&buf, &img);
                        }
                        (buf, img.metadata, img.raw_data)
                    }
                };

                let new_buf = rotate_gpu(ctx, &old_buf, turn);
                let (nw, nh) = turn.output_size(old_buf.width, old_buf.height);
                ctx.recycle_rgba_buffer(old_buf);
                meta.width = nw;
                meta.height = nh;
                meta.orientation = ImageOrientation::Normal;
                *image = PipelineImage::Gpu(new_buf, meta, raw);
            }
        }
    }

    fn process_cpu(&self, image: &mut Image) {
        let Some(turn) = self.config.angle.value.resolve(image.metadata.orientation) else {
            return;
        };
        Self::apply_cpu(turn, image);
    }

    fn schema(&self) -> ModuleSchema {
        ModuleSchema {
            name: "Rotation".to_string(),
            description: "Rotate the image clockwise by 90°, 180°, or 270°. Auto applies EXIF/DNG orientation when it is a pure rotation.".to_string(),
            fields: fields_from_config(&self.config),
        }
    }

    fn create(&self, module: serde_json::Map<String, serde_json::Value>) -> Box<dyn PipelineModule> {
        let config: Rotation =
            serde_json::from_value(serde_json::Value::Object(module)).expect("Invalid Rotation config");
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
    use crate::modules::common::test_pipeline_module_cpu_vs_gpu;

    #[test]
    fn test_rotation_module_cpu_vs_gpu() {
        let module = Module::<Rotation> {
            name: "Rotation".to_string(),
            cache: None,
            config: Rotation {
                angle: Parameter::new_with_choices(
                    RotationAngle::Deg90,
                    "test",
                    vec!["auto".into(), "90".into(), "180".into(), "270".into()],
                ),
            },
        };
        test_pipeline_module_cpu_vs_gpu(&module, 7);
    }

    #[test]
    fn test_rotation_angle_parse() {
        let auto: RotationAngle = serde_json::from_str("\"auto\"").unwrap();
        let d90: RotationAngle = serde_json::from_str("\"90\"").unwrap();
        assert_eq!(auto, RotationAngle::Auto);
        assert_eq!(d90, RotationAngle::Deg90);
    }
}
