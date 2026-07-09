use imops::file_helpers::*;
use std::time::Instant;
use pichromatic_pipeline::config::PipelineConfig;
use pichromatic_pipeline::modules::{CFACoeffs, CST, Contrast, Demosaic, DemosaicAlgorithmType, Exp, HighlightReconstruction, LCH, Module, Parameter, PipelineModule, ToneMap};
use pichromatic_pipeline::pipeline::run_pixel_pipeline;

fn main() {
    let input_path = "test_data/test.dng";
    let output_path = "result.ppm";

    let raw_image = rawler::decode_file(input_path).unwrap();
    let t1 = Instant::now();
    let mut image = pichromatic_pipeline::extern_pipeline::parse_raw_image(raw_image);

    let pipeline1: Vec<Box<dyn PipelineModule>> = vec![
        Box::new(Module {
            name: "Demosaic".to_string(),
            cache: None,
            config: Demosaic{ algorithm: Parameter::new(DemosaicAlgorithmType::Markesteijn, "") },
        }),
        Box::new(Module {
            name: "CFACoeffs".to_string(),
            cache: None,
            config: CFACoeffs {},
        }),
        Box::new(Module {
            name: "HighlightReconstruction".to_string(),
            cache: None,
            config: HighlightReconstruction {},
        }),
        Box::new(Module {
            name: "Exp".to_string(),
            cache: None,
            config: Exp { ev: Parameter::new(1.75, "")},
        }),
        Box::new(Module {
            name: "Contrast".to_string(),
            cache: None,
            config: Contrast { c: Parameter::new(1.75, "")},
        }),
        Box::new(Module {
            name: "CST".to_string(),
            cache: None,
            config: CST { target_color_space: Parameter::new("XyzD65".to_string(), "")},
        }),
        Box::new(Module {
            name: "LHC".to_string(),
            cache: None,
            config: LCH{
                lc: Parameter::new(1.0, ""),
                cc: Parameter::new(1.3, ""),
                hc: Parameter::new(1.0, ""),
            },
        }),
        Box::new(Module {
            name: "Sigmoid (Soft)".to_string(),
            cache: None,
            config: ToneMap {},
        }),
        Box::new(Module {
            name: "CST".to_string(),
            cache: None,
            config: CST { target_color_space: Parameter::new("Srgb".to_string(), "")},
        })
    ];
    let mut pipeline1 = PipelineConfig{
        pipeline_modules: pipeline1,
    };
    run_pixel_pipeline(&mut image, &mut pipeline1);
    println!("total pipeline time: {}ms", t1.elapsed().as_millis());
    println!("total pipeline fps: {}fps", 1000/t1.elapsed().as_millis());

    let (pixels, width, height) = to_u8(&mut image);
    save_bmp(output_path, width, height, pixels).unwrap();
}
