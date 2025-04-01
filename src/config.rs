use core::panic;

use serde::{Deserialize, Serialize};

use crate::imops::*;

#[derive(Serialize, Deserialize, Debug)]
pub struct RawConfig {
    pub pipeline_modules: Vec<toml::Table>
}
pub struct PipelineConfig {
    pub pipeline_modules: Vec<Box<dyn PipelineModule>>
}

pub fn parse_config(config_path: String) -> PipelineConfig{
    let data_string = &std::fs::read_to_string(config_path).unwrap();
    let data: RawConfig = toml::from_str(data_string).unwrap();


    let mut config = PipelineConfig{
        pipeline_modules: vec![]
    };

    for module in data.pipeline_modules {
        let pipeline_module: Box<dyn PipelineModule> = match module["name"].as_str().unwrap() {
            "HighlightReconstruction" =>    Module::<HighlightReconstruction>::from_toml(module),
            "LocalExpousure" =>             Module::<LocalExpousure>::from_toml(module),
            "ChromaDenoise" =>              Module::<ChromaDenoise>::from_toml(module),
            "CFACoeffs" =>                  Module::<CFACoeffs>::from_toml(module),
            "Contrast" =>                   Module::<Contrast>::from_toml(module),
            "Sigmoid" =>                    Module::<Sigmoid>::from_toml(module),
            "Crop" =>                       Module::<Crop>::from_toml(module),
            "CST" =>                        Module::<CST>::from_toml(module),
            "Exp" =>                        Module::<Exp>::from_toml(module),
            "LCH" =>                        Module::<LCH>::from_toml(module),
            "LS" =>                         Module::<LS>::from_toml(module),
            v => panic!("wrong pipeline module name {:}", v)
        };

        config.pipeline_modules.push(pipeline_module);
    }
    return config
}

