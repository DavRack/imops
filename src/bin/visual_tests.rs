use imops::imops::{CFACoeffs, CST, Contrast, Demosaic, Exp, FormedImage, HighlightReconstruction, LCH, Module, PipelineImage, PipelineModule, Sigmoid};

use imops::{demosaic, visual_viewwer};
use imops::conditional_paralell::prelude::*;

use rawler::{decode_file, RawImage};
use std::time::Instant;
use std::vec::Vec;

// This helper function processes an image through a series of modules.
fn process_pipeline(
    mut image: PipelineImage,
    pipeline: &Vec<Box<dyn PipelineModule>>,
    raw_image: &RawImage,
) -> PipelineImage {
    for module in pipeline {
        image = module.process(image, raw_image);
    }
    image
}

fn main() {
    let raw_image_path = "test_data/raw_sample.NEF";
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
                config: Exp { ev: 2.0},
                mask: None,
            }),
            Box::new(Module {
                name: "Contrast".to_string(),
                cache: None,
                config: Contrast { c: 1.25},
                mask: None,
            }),
            Box::new(Module {
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::CameraToXYZ},
                mask: None,
            }),
            Box::new(Module {
                name: "LHC".to_string(),
                cache: None,
                config: LCH{
                    lc: 1.0,
                    cc: 1.0,
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
                config: CST { color_space: imops::imops::ColorSpaceMatrix::XYZTORGB},
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
                name: "Exp".to_string(),
                cache: None,
                config: Exp { ev: 2.0},
                mask: None,
            }),
            Box::new(Module {
                name: "Contrast".to_string(),
                cache: None,
                config: Contrast { c: 1.25},
                mask: None,
            }),
            Box::new(Module {
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::CameraToXYZ},
                mask: None,
            }),
            Box::new(Module {
                name: "LCH".to_string(),
                cache: None,
                config: LCH{
                    lc: 1.0,
                    cc: 2.0,
                    hc: 1.0,
                },
                mask: None,
            }),
            Box::new(Module {
                name: "Sigmoid (Soft)".to_string(),
                cache: None,
                config: Sigmoid { c: 8.0 },
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
        let initial_image = PipelineImage::default();
        println!("Processing pipeline 1...");
        let img = initial_image.clone();
        let now = Instant::now();
        let result1 = process_pipeline(img, &pipeline1, &raw_image);
        println!("Pipeline 1 execution time: {}", now.elapsed().as_millis());
        println!("Processing pipeline 2...");
        let img = initial_image.clone();
        let now = Instant::now();
        let result2 = process_pipeline(img, &pipeline2, &raw_image);
        println!("Pipeline 2 execution time: {}", now.elapsed().as_millis());

        // Compare the results of the two pipelines
        (result1, result2)
    });

    println!("Visual tests finished.");
}
