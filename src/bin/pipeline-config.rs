use std::time::Instant;

use imops::file_helpers::*;
use pichromatic::{cfa::CFA, demosaic::{Dim2, Point, Rect, crop_and_normalize}, image::ImageMetadata, pixel::Image};
use pichromatic_pipeline::{config, pipeline::run_pixel_pipeline};
use rawler::{RawImageData, imgop::xyz::Illuminant};

fn main() {

    let input_path = "test_data/test.dng";
    let config_path = "imgconfig.toml";
    let output_path = "result.ppm";

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let raw_image = rawler::decode_file(input_path).unwrap();
    let t1 = Instant::now();
    let mut image = pichromatic_pipeline::extern_pipeline::parse_raw_image(raw_image);

    let config_data = &std::fs::read_to_string(config_path).unwrap();
    let mut pipeline = config::parse_config(config_data.to_string());

    run_pixel_pipeline(&mut image, &mut pipeline);
    println!("total pipeline time: {}ms", t1.elapsed().as_millis());
    println!("total pipeline fps: {}fps", 1000/t1.elapsed().as_millis());

    let (pixels, width, height) = to_u8(&mut image);
    save_bmp(output_path, width, height, pixels).unwrap();
}



