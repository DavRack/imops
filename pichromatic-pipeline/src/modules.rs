use serde::{Deserialize, Serialize};
use pichromatic::pixel::Image;
use pichromatic::cst::ColorSpaceTag;

pub trait GenericModule {
    fn set_cache(&mut self, cache: Image);
    fn get_cache(&self) -> Option<Image>;
    fn get_name(&self) -> String;
    fn get_mask(&self) -> Option<Box<dyn Mask>>;
}

pub trait PipelineModule: GenericModule{
    fn process(&self, image: Image, raw_image: &ImageMetadata) -> Image;
}

pub struct ImageMetadata {}

impl<T> GenericModule for Module<T> {
    fn set_cache(&mut self, cache: Image) {
        self.cache = Some(cache)
    }
    fn get_cache(&self) -> Option<Image>{
        match &self.cache {
            Some(cache) => Some(cache.clone()),
            None => None,
        }
    }
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn get_mask(&self) -> Option<Box<dyn Mask>> {
        if let Some(v) = &self.mask{
            let a = dyn_clone::clone_box(&**v);
            return Some(a)
        }else {
            return None
        }
    }
}

#[derive(Clone)]
pub struct FormedImage {
    pub raw_image: RawImage,
    pub data: RgbF32,
}

pub struct Module<T>{
    pub name: String,
    pub cache: Option<Image>,
    pub config: T,
    pub mask: Option<Box<dyn Mask>>
}

impl<T> Module<T>{
    pub fn from_toml<'a>(module: Map<String, toml::Value>) -> Box<Self>
    where
        T: Deserialize<'a> + Default,
        Self: Sized
    {
        let cfg: T = module.clone().try_into::<T>().expect(any::type_name::<Self>());
        let module = Module{
            name: module["name"].to_string(),
            cache: None,
            config: cfg,
            mask: None
        };
        Box::new(module)
    }
}



#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct  LCH{
    pub lc: SubPixel,
    pub cc: SubPixel,
    pub hc: SubPixel,
}

impl PipelineModule for Module<LCH> {
    fn process(&self, mut image: Image, _raw_image: &RawImage) -> Image {
        return image.lch([
            self.lc,
            self.cc,
            self.hc,
        ])
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct HighlightReconstruction {
}

impl PipelineModule for Module<HighlightReconstruction> {
    fn process(&self, mut image: Image, raw_image: &RawImage) -> Image {
        let wb_coeffs = [0.0, 0.0, 0.0];
        return image.highlight_reconstruction(wb_coeffs)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Exp {
    pub ev: SubPixel
}

impl PipelineModule for Module<Exp> {
    fn process(&self, mut image: Image, _raw_image: &RawImage) -> Image {
        return image.exp(self.ev)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ToneMap {
}

impl PipelineModule for Module<ToneMap> {
    fn process(&self, mut image: Image, _raw_image: &RawImage) -> Image {
        return image.tone_map()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Contrast {
    pub c: SubPixel
}

impl PipelineModule for Module<Contrast> {
    fn process(&self, mut image: Image, _raw_image: &RawImage) -> Image {
        return image.contrast(self.c)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CFACoeffs {
}

impl PipelineModule for Module<CFACoeffs> {
    fn process(&self, mut image: Image, raw_image: &RawImage) -> Image {
        let wb_coeffs = [0.0, 0.0, 0.0];
        return image.cfa_coeffs(wb_coeffs)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CST {
    pub target_color_space: ColorSpaceTag,
}

impl PipelineModule for Module<CST> {
    fn process(&self, mut image: Image, raw_image: &RawImage) -> Image {
        return match image.color_space {
            Some(_) => image.cst(self.target_color_space),
            None => image.camera_cst(self.target_color_space, camera_color_matrix)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Demosaic {
    pub algorithm: String,
}

impl PipelineModule for Module<Demosaic> {
    fn process( &self, image: Image, raw_image: &RawImage) -> Image {
        Image::demosaic(raw_image_data, image_dimensions, crop_area, black_level, white_level, cfa, demosaic_algorithm)
    }
}


// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
// pub struct LocalExpousure {
//     pub c: SubPixel,
//     pub m: SubPixel,
//     pub p: SubPixel,
// }

// impl PipelineModule for Module<LocalExpousure> {
//     fn process(&self, mut image: Image, _raw_image: &RawImage) -> Image {
//     }
// }

// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
// pub struct ChromaDenoise {
// 	  pub a: SubPixel,
//     pub b: SubPixel,
//     pub strength: SubPixel,
//     #[serde(default)]
//     pub use_ai: bool,
// }

// impl PipelineModule for Module<ChromaDenoise> {
//     fn process(&self, image: Image, _raw_image: &RawImage) -> Image {
//     }
// }

// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
// pub struct  Crop{
//     pub factor: usize,
// }

// impl PipelineModule for Module<Crop> {
//     fn process(&self, mut image: Image, _raw_image: &RawImage) -> Image {
//     }
// }

