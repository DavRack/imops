use serde::{Deserialize, Serialize};
use crate::modules::{get_default_modules, PipelineModule};

#[derive(Serialize, Deserialize, Debug)]
pub struct RawConfig {
    pub pipeline_modules: Vec<serde_json::Map<String, serde_json::Value>>
}

pub struct PipelineConfig {
    pub pipeline_modules: Vec<Box<dyn PipelineModule>>
}

pub fn parse_config(config: String) -> PipelineConfig {
    let raw_modules = serde_json::from_str::<RawConfig>(&config)
        .map(|d| d.pipeline_modules)
        .or_else(|_| toml::from_str::<RawConfig>(&config).map(|d| d.pipeline_modules))
        .expect("cant decode config from string");

    let mut config = PipelineConfig {
        pipeline_modules: vec![]
    };

    let default_modules = get_default_modules();

    for (i, mut module) in raw_modules.into_iter().enumerate() {
        let name_val = module.remove("name").unwrap_or_else(|| panic!("Module {} missing name key", i));
        let name = name_val.as_str().unwrap_or_else(|| panic!("Module {} name is not a string", i));
        let template = default_modules
            .iter()
            .find(|m| {
                let schema_name = m.schema().name.to_lowercase();
                let requested_name = name.to_lowercase();
                schema_name == requested_name
                    || (requested_name == "tonemap" && schema_name == "sigmoidtonemap")
                    || (requested_name == "inversehd" && schema_name == "inversehdtonemap")
                    || (requested_name == "inverse_hd" && schema_name == "inversehdtonemap")
                    || (requested_name == "inverse_hd_tone_map" && schema_name == "inversehdtonemap")
            })
            .unwrap_or_else(|| panic!("wrong pipeline module name '{}'", name));
        let pipeline_module = template.create(module);

        config.pipeline_modules.push(pipeline_module);
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_film_config() {
        let json = r#"{"pipeline_modules": [{"name": "Film", "stock": "Portra400", "film_format": "Film35mm", "seed": 1, "output": "PositiveLinear"}]}"#;
        let pipeline = parse_config(json.to_string());
        assert_eq!(pipeline.pipeline_modules.len(), 1);
    }
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

    #[test]
    fn parses_json_config() {
        let json = r#"{
            "pipeline_modules": [
                { "name": "Demosaic", "algorithm": "markesteijn" },
                { "name": "CFACoeffs" },
                { "name": "LumaGuidedChromaDenoise", "radius": 2, "epsilon": 0.01 },
                { "name": "HighlightReconstruction" },
                { "name": "Vignette", "strength": 1.0 },
                { "name": "BaselineExposureCompensation" },
                { "name": "Exp", "ev": 0.0 },
                { "name": "CST", "target_color_space": "AcesCg" },
                { "name": "Film", "stock": "Portra400", "film_format": "Film35mm", "seed": 1, "output": "PositiveLinear" },
                { "name": "Exp", "ev": 1.5 },
                { "name": "Contrast", "c": 1.5 },
                { "name": "SigmoidToneMap" },
                { "name": "CST", "target_color_space": "LinearSrgb" }
            ]
        }"#;
        let _parsed = super::parse_config(json.to_string());
    }

    #[test]
    fn test_individual_modules() {
        let json_film = r#"{"pipeline_modules": [{"name": "Film", "stock": "Portra400", "film_format": "Film35mm", "seed": 1, "output": "PositiveLinear"}]}"#;
        let _ = super::parse_config(json_film.to_string());

        let json_sigmoid = r#"{"pipeline_modules": [{"name": "SigmoidToneMap"}]}"#;
        let _ = super::parse_config(json_sigmoid.to_string());

        let json_gamma = r#"{"pipeline_modules": [{"name": "Gamma", "gamma": 2.2}]}"#;
        let _ = super::parse_config(json_gamma.to_string());

        let json_inv_hd = r#"{"pipeline_modules": [{"name": "InverseHdToneMap", "strength": 1.2}]}"#;
        let parsed_inv_hd = super::parse_config(json_inv_hd.to_string());
        assert_eq!(parsed_inv_hd.pipeline_modules.len(), 1);
        assert_eq!(parsed_inv_hd.pipeline_modules[0].schema().name, "InverseHdToneMap");

        let toml_inv_hd = r#"
            [[pipeline_modules]]
            name = "InverseHdToneMap"
            strength = 1.5
        "#;
        let parsed_toml = super::parse_config(toml_inv_hd.to_string());
        assert_eq!(parsed_toml.pipeline_modules.len(), 1);
        assert_eq!(parsed_toml.pipeline_modules[0].schema().name, "InverseHdToneMap");
    }
}
