use std::{any, fmt::Debug};
use std::hash::{DefaultHasher, Hash, Hasher};
use serde::{Deserialize, Serialize};
use toml::map::Map;

use crate::modules::{
    CFACoeffs, CST, Contrast, Demosaic, Exp, HighlightReconstruction, LCH, Module, PipelineModule, ToneMap};

#[derive(Serialize, Deserialize, Debug)]
pub struct RawConfig {
    pub pipeline_modules: Vec<toml::Table>
}

pub struct PipelineConfig {
    pub pipeline_modules: Vec<Box<dyn PipelineModule>>
}

fn chained_hash<T>(module: &T, prev_hash: u64) -> u64 where T: Debug {
    let mut hasher = DefaultHasher::new();
    prev_hash.hash(&mut hasher);
    format!("{:?}", module).hash(&mut hasher);
    return hasher.finish();
}

pub fn from_toml<'a, T>(module: Map<String, toml::Value>, prev_hash: u64) -> Box<dyn PipelineModule>
    where
    T: Deserialize<'a> + Default + 'static,
    T: Debug,
    Module<T>: PipelineModule
{
    // let mask: Option<Box<dyn Mask>> = match module.get("mask"){
    //     Some(mask_cfg) => match mask_cfg["name"].as_str().unwrap(){
    //         "LuminanceGradient" => Some(Box::new(mask_cfg.clone().try_into::<LuminanceGradient>().unwrap())),
    //         "Constant" =>          Some(Box::new(mask_cfg.clone().try_into::<Constant>().unwrap())),
    //         v => panic!("wrong mask function name: {:}", v),
    //     },
    //     None => None,
    // };

    let cfg: T = module.clone().try_into::<T>().expect(any::type_name::<T>());
    let module = Module{
        name: module["name"].to_string(),
        cache: None,
        chained_hash: chained_hash(&cfg, prev_hash),
        config: cfg,
        // mask
    };
    Box::new(module)
}


pub fn parse_config(config: String) -> PipelineConfig{
    let data: RawConfig = toml::from_str(config.as_str()).unwrap();


    let mut config = PipelineConfig{
        pipeline_modules: vec![]
    };

    let mut current_hash: u64 = 0;

    for module in data.pipeline_modules {
        let pipeline_module: Box<dyn PipelineModule> = match module["name"].as_str().unwrap() {
            "HighlightReconstruction" =>    from_toml::<HighlightReconstruction>(module, current_hash),
            // "LocalExpousure" =>             from_toml::<LocalExpousure>(module),
            // "ChromaDenoise" =>              from_toml::<ChromaDenoise>(module),
            "CFACoeffs" =>                  from_toml::<CFACoeffs>(module, current_hash),
            "Contrast" =>                   from_toml::<Contrast>(module, current_hash),
            "ToneMap" =>                    from_toml::<ToneMap>(module, current_hash),
            // "Crop" =>                       from_toml::<Crop>(module),
            "CST" =>                        from_toml::<CST>(module, current_hash),
            "Exp" =>                        from_toml::<Exp>(module, current_hash),
            "LCH" =>                        from_toml::<LCH>(module, current_hash),
            // "LS" =>                         from_toml::<LS>(module),
            "Demosaic" =>                   from_toml::<Demosaic>(module, current_hash),
            v => panic!("wrong pipeline module name {:}", v)
        };
        current_hash = pipeline_module.get_chained_hash();

        config.pipeline_modules.push(pipeline_module);
    }
    return config
}

