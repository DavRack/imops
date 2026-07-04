use std::io::{self, Read};
use std::os::unix::net::UnixListener;
use std::fs;
use std::path::Path;
use std::time::Instant;

use imops::file_helpers::{save_bmp, to_u8};
use pichromatic::cfa::CFA;
use pichromatic::demosaic::{Dim2, Point, Rect, crop_and_normalize};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::Image;
use pichromatic_pipeline::config::{self, PipelineConfig};
use pichromatic_pipeline::pipeline::run_pixel_pipeline;
use rawler::RawImageData;
use rawler::imgop::xyz::Illuminant;

fn main() -> io::Result<()> {

    let output_path = "result.ppm";
    let input_path = "test_data/test.dng";
    let socket_path = "/tmp/rust_listener.sock";

    let mut raw_image = get_raw_image(input_path.to_string());
    let metadata = raw_image.metadata.clone();

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
                        let mut pipeline = config::parse_config(config.to_string());
                        raw_image = process_image(raw_image, &mut pipeline, output_path.to_string());
                        raw_image.metadata = metadata.clone();
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
fn process_image(mut image: Image, pipeline: &mut PipelineConfig, output_path: String) -> Image{
    let t1 = Instant::now();
    run_pixel_pipeline(&mut image, pipeline);
    println!("finish pipeline with total time: {:.2?}", t1.elapsed());
    let (pixels, width, height) = to_u8(&mut image);
    save_bmp(&output_path, width, height, pixels).unwrap();
    return image
}


fn get_raw_image(input_path: String) -> Image{
    let raw_image = rawler::decode_file(input_path).unwrap();
    pichromatic_pipeline::extern_pipeline::parse_raw_image(raw_image)
}
