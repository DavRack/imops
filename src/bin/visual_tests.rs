use imops::imops::{CFACoeffs, Contrast, Demosaic, Exp, Module, PipelineImage, PipelineModule, Sigmoid, CST};

use imops::visual_viewwer;

use rawler::{decode_file, RawImage};
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
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::CameraToXYZ},
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
                config: Sigmoid { c: 8.0 },
                mask: None,
            }),
            Box::new(Module {
                name: "CST".to_string(),
                cache: None,
                config: CST { color_space: imops::imops::ColorSpaceMatrix::XYZTOsRGB},
                mask: None,
            }),
        ];

        // --- Process both pipelines starting from a default PipelineImage ---
        let initial_image = PipelineImage::default();
        println!("Processing pipeline 1...");
        let result1 = process_pipeline(initial_image.clone(), &pipeline1, &raw_image);
        println!("Processing pipeline 2...");
        let result2 = process_pipeline(initial_image.clone(), &pipeline2, &raw_image);

        // Compare the results of the two pipelines
        (result1, result2)
    });

    println!("Visual tests finished.");
}
