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
            "CFACoeffs" => imops::CFACoeffs::from_toml(module),
            "CST" => imops::CST::from_toml(module),
            "Exp" => imops::Exp::from_toml(module),
            "Contrast" => imops::Contrast::from_toml(module),
            "Sigmoid" => imops::Sigmoid::from_toml(module),
            "LocalExpousure" => imops::LocalExpousure::from_toml(module),
            "LS" => imops::LS::from_toml(module),
            "LCH" => imops::LCH::from_toml(module),
            "ChromaDenoise" => imops::ChromaDenoise::from_toml(module),
            "HighlightReconstruction" => imops::HighlightReconstruction::from_toml(module),
            "Crop" =>imops::Crop::from_toml(module),
            v => panic!("wrong pipeline module name {:}", v)
        };

        config.pipeline_modules.push(pipeline_module);
    }
    return config
}

