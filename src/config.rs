use core::panic;

use serde::{Deserialize, Serialize};

use crate::imops::{self, PipelineModule};

#[derive(Serialize, Deserialize, Debug)]
pub struct RawConfig {
    pub pipeline_modules: Vec<toml::Table>
}

pub struct PipelineConfig {
    pub pipeline_modules: Vec<Box<dyn imops::PipelineModule>>
}

pub fn parse_config(config_path: String) -> PipelineConfig{
    let data_string = &std::fs::read_to_string(config_path).unwrap();
    let data: RawConfig = toml::from_str(data_string).unwrap();


    let mut config = PipelineConfig{
        pipeline_modules: vec![]
    };

    for module in data.pipeline_modules {
        let pipeline_module: Box<dyn imops::PipelineModule> = match module["name"].as_str().unwrap() {
            "CFACoeffs" => Box::new(module.clone().try_into::<imops::CFACoeffs>().unwrap()),
            "CST" => Box::new(module.clone().try_into::<imops::CST>().unwrap()),
            "Exp" => Box::new(module.clone().try_into::<imops::Exp>().unwrap()),
            "Contrast" => Box::new(module.clone().try_into::<imops::Contrast>().unwrap()),
            "Sigmoid" => Box::new(module.clone().try_into::<imops::Sigmoid>().unwrap()),
            "LocalExpousure" => Box::new(module.clone().try_into::<imops::LocalExpousure>().unwrap()),
            "LS" => Box::new(module.clone().try_into::<imops::LS>().unwrap()),
            v => panic!("wrong pipeline module name {:}", v)
        };

        config.pipeline_modules.push(pipeline_module);
    }
    return config
}

