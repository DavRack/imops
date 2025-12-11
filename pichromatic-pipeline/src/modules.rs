use pichromatic::demosaic::demosaic_algorithms;
use serde::{Deserialize, Serialize};
use pichromatic::pixel::{Image, SubPixel};
use pichromatic::cst::ColorSpaceTag;

pub trait GenericModule {
    // fn set_cache(&mut self, cache: Image);
    // fn get_cache(&self) -> Option<Image>;
    fn get_name(&self) -> String;
    // fn get_mask(&self) -> Option<Box<dyn Mask>>;
}

pub trait PipelineModule: GenericModule{
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image;
}

impl<T> GenericModule for Module<T> {
    // fn set_cache(&mut self, cache: Image) {
    //     self.cache = Some(cache)
    // }
    // fn get_cache(&self) -> Option<Image>{
    //     match &self.cache {
    //         Some(cache) => Some(cache.clone()),
    //         None => None,
    //     }
    // }
    fn get_name(&self) -> String {
        self.name.clone()
    }

    // fn get_mask(&self) -> Option<Box<dyn Mask>> {
    //     if let Some(v) = &self.mask{
    //         let a = dyn_clone::clone_box(&**v);
    //         return Some(a)
    //     }else {
    //         return None
    //     }
    // }
}

pub struct Module<T>{
    pub name: String,
    pub cache: Option<Image>,
    pub config: T,
    // pub mask: Option<Box<dyn Mask>>
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



#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct  LCH{
    pub lc: SubPixel,
    pub cc: SubPixel,
    pub hc: SubPixel,
}

impl PipelineModule for Module<LCH> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
         image.lch([
            self.config.lc,
            self.config.cc,
            self.config.hc,
        ])
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
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Exp {
    pub ev: SubPixel
}

impl PipelineModule for Module<Exp> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.exp(self.config.ev)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct ToneMap {
}

impl PipelineModule for Module<ToneMap> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.tone_map()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct Contrast {
    pub c: SubPixel
}

impl PipelineModule for Module<Contrast> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        return image.contrast(self.config.c)
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
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct CST {
    pub target_color_space: String,
}

impl PipelineModule for Module<CST> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        let target_color_space: ColorSpaceTag = serde_plain::from_str(
            &self.config.target_color_space
        ).expect(
            &format!("Color space not recognized: {}", self.config.target_color_space)
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
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct Demosaic {
    pub algorithm: String,
}

impl PipelineModule for Module<Demosaic> {
    fn process<'a>(&self, image: &'a mut Image) -> &'a mut Image {
        let new_image = Image::demosaic(
            image.clone(),
            demosaic_algorithms::Markesteijn{}
        );
        image.rgb_data = new_image.rgb_data;
        image.metadata.width = new_image.metadata.width;
        image.metadata.height = new_image.metadata.height;
        image.metadata.color_space = None;
        return image;
    }
}


// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
// pub struct LocalExpousure {
//     pub c: SubPixel,
//     pub m: SubPixel,
//     pub p: SubPixel,
// }

// impl PipelineModule for Module<LocalExpousure> {
//     fn process(&self, mut image: Image, _image_metadata: &ImageMetadata) -> Image {
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
//     fn process(&self, image: Image, _image_metadata: &ImageMetadata) -> Image {
//     }
// }

// #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
// pub struct  Crop{
//     pub factor: usize,
// }

// impl PipelineModule for Module<Crop> {
//     fn process(&self, mut image: Image, _image_metadata: &ImageMetadata) -> Image {
//     }
// }

