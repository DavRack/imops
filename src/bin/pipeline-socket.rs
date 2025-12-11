use std::io::{self, Read};
use std::os::unix::net::UnixListener;
use std::fs;
use std::path::Path;

use filsimrs::file_helpers::{save_bmp, to_u8};
use pichromatic::cfa::CFA;
use pichromatic::demosaic::{Dim2, Point, Rect};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::Image;
use pichromatic_pipeline::config;
use pichromatic_pipeline::pipeline::run_pixel_pipeline;
use rawler::RawImageData;
use rawler::imgop::xyz::Illuminant;

fn main() -> io::Result<()> {

    let output_path = "result.ppm";
    let input_path = "test_data/test.dng";
    let socket_path = "/tmp/rust_listener.sock";

    let mut raw_image = get_raw_image(input_path.to_string());

    // 1. Cleanup: Remove the socket file if it already exists
    if Path::new(socket_path).exists() {
        fs::remove_file(socket_path)?;
    }

    // 2. Bind to the socket path (creates the file)
    let listener = UnixListener::bind(socket_path)?;
    println!("Listening on {}", socket_path);

    // 3. Accept incoming connections in a loop
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                // Handle the connection (Read data)
                let mut buffer = [0; 1024]; // 1KB buffer
                match stream.read(&mut buffer) {
                    Ok(size) => {
                        // Convert bytes to string for display (lossy in case of binary data)
                        let config = String::from_utf8_lossy(&buffer[..size]);
                        println!("Received:");
                        println!("{}", config);
                        process_image(&mut raw_image, config.to_string(), output_path.to_string());
                        println!("finish");
                    }
                    Err(e) => eprintln!("Failed to read from client: {}", e),
                }
            }
            Err(e) => {
                eprintln!("Connection failed: {}", e);
            }
        }
    }

    Ok(())
}
fn process_image(image: &mut Image, config: String, output_path: String){
    let pipeline = config::parse_config(config);
    let mut result1 = run_pixel_pipeline(image.clone(), pipeline);
    let (pixels, width, height) = to_u8(&mut result1);
    save_bmp(&output_path, width, height, pixels).unwrap();
}


fn get_raw_image(input_path: String) -> Image{

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let raw_image = rawler::decode_file(input_path).unwrap();
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

    let image = Image{
        raw_data: raw_image_data,
        rgb_data: vec![],
        metadata: image_metadata,
    };
    return image
}
