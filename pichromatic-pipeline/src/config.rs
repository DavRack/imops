// use core::panic;
// use std::any;

// use serde::{Deserialize, Serialize};
// use toml::map::Map;

// use crate::modules::PipelineModule;

// use crate::{imops::{self, *, Demosaic}, mask::{Constant, LuminanceGradient, Mask}};

use crate::modules::PipelineModule;

// #[derive(Serialize, Deserialize, Debug)]
// pub struct RawConfig {
//     pub pipeline_modules: Vec<toml::Table>
// }
pub struct PipelineConfig {
    pub pipeline_modules: Vec<Box<dyn PipelineModule>>
}

// pub fn from_toml<'a, T>(module: Map<String, toml::Value>) -> Box<dyn PipelineModule>
//     where
//     T: Deserialize<'a> + Default + 'static,
//     imops::Module<T>: imops::PipelineModule
// {
//     let mask: Option<Box<dyn Mask>> = match module.get("mask"){
//         Some(mask_cfg) => match mask_cfg["name"].as_str().unwrap(){
//             "LuminanceGradient" => Some(Box::new(mask_cfg.clone().try_into::<LuminanceGradient>().unwrap())),
//             "Constant" =>          Some(Box::new(mask_cfg.clone().try_into::<Constant>().unwrap())),
//             v => panic!("wrong mask function name: {:}", v),
//         },
//         None => None,
//     };

//     let cfg: T = module.clone().try_into::<T>().expect(any::type_name::<T>());
//     let module = Module{
//         name: module["name"].to_string(),
//         cache: None,
//         config: cfg,
//         mask
//     };
//     Box::new(module)
// }


// pub fn parse_config(config_path: String) -> PipelineConfig{
//     let data_string = &std::fs::read_to_string(config_path).unwrap();
//     let data: RawConfig = toml::from_str(data_string).unwrap();


//     let mut config = PipelineConfig{
//         pipeline_modules: vec![]
//     };

//     for module in data.pipeline_modules {
//         let pipeline_module: Box<dyn PipelineModule> = match module["name"].as_str().unwrap() {
//             "HighlightReconstruction" =>    from_toml::<HighlightReconstruction>(module),
//             "LocalExpousure" =>             from_toml::<LocalExpousure>(module),
//             "ChromaDenoise" =>              from_toml::<ChromaDenoise>(module),
//             "CFACoeffs" =>                  from_toml::<CFACoeffs>(module),
//             "Contrast" =>                   from_toml::<Contrast>(module),
//             "Sigmoid" =>                    from_toml::<Sigmoid>(module),
//             "Crop" =>                       from_toml::<Crop>(module),
//             "CST" =>                        from_toml::<CST>(module),
//             "Exp" =>                        from_toml::<Exp>(module),
//             "LCH" =>                        from_toml::<LCH>(module),
//             "LS" =>                         from_toml::<LS>(module),
//             "Demosaic" =>                   from_toml::<Demosaic>(module),
//             v => panic!("wrong pipeline module name {:}", v)
//         };

//         config.pipeline_modules.push(pipeline_module);
//     }
//     return config
// }

