use imops::config::PipelineConfig;
use imops::imops::{CFACoeffs, CST, ChromaDenoise, Contrast, Demosaic, Exp, HighlightReconstruction, LCH, Module, PipelineImage, PipelineModule, Sigmoid};

use imops::pipeline::run_pixel_pipeline;
use imops::{visual_viewwer};

use rawler::{decode_file, RawImage};
use std::time::Instant;
use std::vec::Vec;

fn main() {
    let raw_image_path = "test_data/maya.dng";
    // let raw_image_path = "test_data/raw_sample.NEF";
    // let raw_image_path = "test_data/test.dng";

    // 1. Load the raw image.
    let raw_image = decode_file(raw_image_path).expect("Failed to load raw image");

    

    visual_viewwer::run_viewer("Pipeline Comparison", move || {
        // --- Define Pipeline 1: Softer Sigmoid ---
        let pipeline1: Vec<Box<dyn PipelineModule>> = vec![
            Box::new(Module {
                name: "Demosaic".to_string(),
                cache: None,
                config: Demosaic { algorithm: "markesteijn".to_string() },
                mask: None,
            }),
            Box::new(Module {
                name: "CFACoeffs".to_string(),
                cache: None,
                config: CFACoeffs { },
                mask: None,
            }),
            Box::new(Module {
                name: "HighlightReconstruction".to_string(),
                cache: None,
                config: HighlightReconstruction {},
                mask: None,
            }),
            Box::new(Module {
                name: "Exp".to_string(),
                cache: None,
                config: Exp { ev: 1.5},
                mask: None,
            }),
            Box::new(Module {
                name: "Contrast".to_string(),
                cache: None,
                config: Contrast { c: 1.75},
                mask: None,
            }),
            Box::new(Module {
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::CameraToXYZ},
                mask: None,
            }),
            Box::new(Module {
                name: "NR".to_string(),
                cache: None,
                config: ChromaDenoise{
                    a: 0.0,
                    b: 0.0,
                    strength: 0.1,
                    use_ai: false,
                },
                mask: None,
            }),
            Box::new(Module {
                name: "LHC".to_string(),
                cache: None,
                config: LCH{
                    lc: 1.0,
                    cc: 1.15,
                    hc: 1.0,
                },
                mask: None,
            }),
            Box::new(Module {
                name: "Sigmoid (Soft)".to_string(),
                cache: None,
                config: Sigmoid { c: 6.0 },
                mask: None,
            }),
            Box::new(Module {
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::XYZTOsRGB},
                mask: None,
            })
        ];

        // --- Define Pipeline 2: Harder Sigmoid ---
        let pipeline2: Vec<Box<dyn PipelineModule>> = vec![
            Box::new(Module {
                name: "Demosaic".to_string(),
                cache: None,
                config: Demosaic { algorithm: "markesteijn".to_string() },
                mask: None,
            }),
            Box::new(Module {
                name: "CFACoeffs".to_string(),
                cache: None,
                config: CFACoeffs { },
                mask: None,
            }),
            Box::new(Module {
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::CameraToXYZ},
                mask: None,
            }),
            Box::new(Module {
                name: "Sigmoid (Soft)".to_string(),
                cache: None,
                config: Sigmoid { c: 6.0 },
                mask: None,
            }),
            Box::new(Module {
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::XYZTOsRGB},
                mask: None,
            })
        ];

        // --- Process both pipelines starting from a default PipelineImage ---
        println!("Processing pipeline 1...");
        let now = Instant::now();
        // let result1 = process_pipeline(img, &pipeline1, &raw_image);
        let pipeline1 = PipelineConfig{
            pipeline_modules: pipeline1,
        };
        let result1 = run_pixel_pipeline(raw_image.clone(), pipeline1);
        println!("Pipeline 1 execution time: {}", now.elapsed().as_millis());
        println!("Processing pipeline 2...");
        let now = Instant::now();
        let pipeline2 = PipelineConfig{
            pipeline_modules: pipeline2,
        };
        let result2 = run_pixel_pipeline(raw_image.clone(), pipeline2);
        println!("Pipeline 2 execution time: {}", now.elapsed().as_millis());

        // Compare the results of the two pipelines
        (result1, result2)
    });

    println!("Visual tests finished.");
}
