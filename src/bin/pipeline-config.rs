use std::time::Instant;

use filsimrs::file_helpers::*;
use pichromatic::{cfa::CFA, demosaic::{Dim2, Point, Rect, crop_and_normalize}, image::ImageMetadata, pixel::Image};
use pichromatic_pipeline::{config, pipeline::run_pixel_pipeline};
use rawler::{RawImageData, imgop::xyz::Illuminant};

fn main() {

    let input_path = "test_data/raw_sample.NEF";
    let config_path = "imgconfig.json";
    let output_path = "result.ppm";

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let mut raw_image = rawler::decode_file(input_path).unwrap();
    let t1 = Instant::now();
    let calibration_matrix_d65 = raw_image.camera.color_matrix[&Illuminant::D65].clone();
    let wb_coeffs = raw_image.wb_coeffs;
    let raw_image_dimentions = raw_image.dim();
    let raw_image_crop_area = raw_image.crop_area.unwrap();
    let raw_image_white_level = raw_image.whitelevel.as_bayer_array()[0];
    let raw_image_black_level = raw_image.blacklevel.as_bayer_array()[0];
    let raw_image_cfa = raw_image.camera.cfa.to_string();
    let _ = raw_image.apply_scaling();
    let raw_image_data = match raw_image.data {
        RawImageData::Float(data) => data,
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

    let mut image = Image{
        raw_data: raw_image_data,
        rgb_data: vec![],
        metadata: image_metadata,
    };
    let normalized_raw_data = crop_and_normalize(
        &image,
    );
    image.raw_data = normalized_raw_data;
    image.metadata.width = image.metadata.crop_area.unwrap().d.w;
    image.metadata.height = image.metadata.crop_area.unwrap().d.h;

    let config_data = &std::fs::read_to_string(config_path).unwrap();
    let mut pipeline = config::parse_config(config_data.to_string());

    let mut result1 = run_pixel_pipeline(image, &mut pipeline);
    println!("total pipeline time: {}ms", t1.elapsed().as_millis());
    println!("total pipeline fps: {}fps", 1000/t1.elapsed().as_millis());

    let (pixels, width, height) = to_u8(&mut result1);
    save_bmp(output_path, width, height, pixels).unwrap();
}



