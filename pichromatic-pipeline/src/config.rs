use serde::{Deserialize, Serialize};
use crate::modules::{get_default_modules, PipelineModule};

#[derive(Serialize, Deserialize, Debug)]
pub struct RawConfig {
    pub pipeline_modules: Vec<toml::Table>
}

pub struct PipelineConfig {
    pub pipeline_modules: Vec<Box<dyn PipelineModule>>
}

pub fn parse_config(config: String) -> PipelineConfig {
    let data: RawConfig = toml::from_str(config.as_str()).or(
        serde_json::from_str(config.as_str())
    ).expect("cant decode config from string");

    let mut config = PipelineConfig {
        pipeline_modules: vec![]
    };

    let default_modules = get_default_modules();

    for module in data.pipeline_modules {
        let name = module["name"].as_str().expect("Module must have a name field");
        let template = default_modules.iter()
            .find(|m| m.schema().name.to_lowercase() == name.to_lowercase())
            .expect(&format!("wrong pipeline module name {}", name));
        let pipeline_module = template.create(module);

        config.pipeline_modules.push(pipeline_module);
    }
    return config
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_imgconfig_film() {
        let cfg = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../imgconfig-film.toml"
        ))
        .expect("imgconfig-film.toml");
        let parsed = super::parse_config(cfg);
        let names: Vec<_> = parsed
            .pipeline_modules
            .iter()
            .map(|m| m.schema().name)
            .collect();
        assert!(
            names.iter().any(|n| n == "Film"),
            "expected Film in pipeline, got {names:?}"
        );
    }
}
