use filsimrs::file_helpers::*;
use std::time::Instant;
use pichromatic::cfa::CFA;
use pichromatic::demosaic::{Dim2, Point, Rect};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::Image;
use pichromatic_pipeline::config::PipelineConfig;
use pichromatic_pipeline::modules::{CFACoeffs, CST, Contrast, Demosaic, Exp, HighlightReconstruction, LCH, Module, PipelineModule, ToneMap};
use pichromatic_pipeline::pipeline::run_pixel_pipeline;
use rawler::RawImageData;
use rawler::imgop::xyz::Illuminant;
use pichromatic_pipeline;


fn main() {

    let input_path = "test_data/test.dng";
    let output_path = "result.ppm";

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let raw_image = rawler::decode_file(input_path).unwrap();
    let t1 = Instant::now();
    let calibration_matrix_d65 = raw_image.camera.color_matrix[&Illuminant::D65].clone();
    let wb_coeffs = raw_image.wb_coeffs;
    let raw_image_dimentions = raw_image.dim();
    let raw_image_crop_area = raw_image.crop_area.unwrap();
    let raw_image_white_level = raw_image.whitelevel.as_bayer_array()[0];
    let raw_image_black_level = raw_image.blacklevel.as_bayer_array()[0];
    let raw_image_cfa = raw_image.camera.cfa.to_string();
    let raw_image_data = match raw_image.data {
        RawImageData::Integer(data) => data,
        _ => panic!(""),
    };
    let image_metadata = ImageMetadata{
        width: raw_image_dimentions.w,
        height: raw_image_dimentions.h,
        crop_area: Some(Rect{
            p: Point {
                x: raw_image_crop_area.p.x,
                y: raw_image_crop_area.p.y
            },
            d: Dim2 {
                w: raw_image_crop_area.d.w,
                h: raw_image_crop_area.d.h
            },
        }),
        black_level: Some(raw_image_black_level),
        white_level: Some(raw_image_white_level),
        wb_coeffs: Some(wb_coeffs),
        cfa: Some(CFA::new(&raw_image_cfa)),
        calibration_matrix_d65: Some(calibration_matrix_d65),
        color_space: None,
    };

    let image = Image {
        raw_data: raw_image_data,
        rgb_data: vec![],
        metadata: image_metadata,
    };

    let pipeline1: Vec<Box<dyn PipelineModule>> = vec![
        Box::new(Module {
            name: "Demosaic".to_string(),
            cache: None,
            config: Demosaic{ algorithm: "markesteijn".to_string() },
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "CFACoeffs".to_string(),
            cache: None,
            config: CFACoeffs {},
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "HighlightReconstruction".to_string(),
            cache: None,
            config: HighlightReconstruction {},
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "Exp".to_string(),
            cache: None,
            config: Exp { ev: 1.75},
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "Contrast".to_string(),
            cache: None,
            config: Contrast { c: 1.75},
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "CST".to_string(),
            cache: None,
            config: CST { target_color_space: "XyzD65".to_string()},
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "LHC".to_string(),
            cache: None,
            config: LCH{
                lc: 1.0,
                cc: 1.3,
                hc: 1.0,
            },
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "Sigmoid (Soft)".to_string(),
            cache: None,
            config: ToneMap {},
            chained_hash: 0,
            // mask: None,
        }),
        Box::new(Module {
            name: "CST".to_string(),
            cache: None,
            config: CST { target_color_space: "Srgb".to_string()},
            chained_hash: 0,
            // mask: None,
        })
    ];
    let pipeline1 = PipelineConfig{
        pipeline_modules: pipeline1,
    };
    let mut result1 = run_pixel_pipeline(image, pipeline1);
    println!("total pipeline time: {}ms", t1.elapsed().as_millis());
    println!("total pipeline fps: {}fps", 1000/t1.elapsed().as_millis());

    let (pixels, width, height) = to_u8(&mut result1);
    save_bmp(output_path, width, height, pixels).unwrap();
}
