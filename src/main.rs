#![warn(unused_extern_crates)]
mod demosaic;
mod imops;
mod config;

use clap::Parser as Clap_parser;
use imops::FormedImage;
use ndarray::{s, Array2, Array3};
use rawler::pixarray::RgbF32;
use rawler::{self};
use core::panic;
use std::io::Cursor;
use std::time::Instant;

#[derive(Clap_parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// code query
    #[arg(name="input path", value_name = "input_path")]
    input_path: String,

    #[arg(short, name="output path", default_value="result.jpg", value_name = "output_path")]
    output_path: String,

    #[arg(short, name="config path", default_value="imgconfig.toml", value_name = "config_path")]
    config_path: String,
}

fn small(v: Array3<f32>) -> Array3<f32> {
    let f = 1;
    let s = v.shape();
    let mut nv = Array3::zeros(((s[0] / f) + 1, (s[1] / f) + 1, s[2]));
    for i in (0..s[0]).step_by(f) {
        for j in (0..s[1]).step_by(f) {
            for x in 0..3 {
                nv[[i / f, j / f, x]] = v[[i, j, x]];
            }
        }
    }
    return nv;
}

fn debayer(image: &mut rawler::RawImage) -> imops::FormedImage {
    let debayer_image: imops::FormedImage;
    let _ = image.apply_scaling();
    if let rawler::RawImageData::Float(ref im) = image.data {
        let cfa = image.camera.cfa.clone();
        let cfa = demosaic::get_cfa(cfa, image.crop_area.unwrap());
        let (nim, width, height) = demosaic::crop(image.dim(), image.crop_area.unwrap(), im.to_vec());


        debayer_image = imops::FormedImage{
            raw_image: image.clone(),
            data: demosaic::demosaic_algorithms::linear_interpolation(width, height, cfa, nim),
        };
        return debayer_image;

    } else {
        panic!("Don't know how to process non-integer raw files");
    }
}

fn to_vec(data: RgbF32) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let height = data.height;
    let width = data.width;
    let data2 = data.flatten().iter().map(|x| (x * 255.0) as u8).collect();
    let img = image::RgbImage::from_vec(
        width as u32,
        height as u32,
        data2,
    )
    .unwrap();
    return img;
}

fn run_pixel_pipeline(image: &mut imops::FormedImage, pixel_pipeline: config::PipelineConfig) -> FormedImage {
    let modules = pixel_pipeline.pipeline_modules;

    let mut image = image.clone();
    for module in modules {
        image = module.process(&mut image);
    }
    return image;
}

fn main() {
    let args = Box::leak(Box::new(Args::parse()));

    let path = args.input_path.clone();

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let decode = Instant::now();
    let mut raw_image = rawler::decode_file(path).unwrap();
    let rotation = raw_image.orientation;

    println!("rotation: {:.2?}", rotation);
    println!("decode file: {:.2?}", decode.elapsed());

    let now = Instant::now();
    let mut debayer_image = debayer(&mut raw_image);
    println!("debayer: {:.2?}", now.elapsed());

    // debayer_image.data = small(debayer_image.data);
    let now = Instant::now();

    let config = config::parse_config(args.config_path.clone());

    debayer_image = run_pixel_pipeline(&mut debayer_image, config);
    // println!("img size: {:?}", debayer_image.data.shape());
    let mut img = image::DynamicImage::ImageRgb8(to_vec(debayer_image.data));
    img = match rotation {
        rawler::Orientation::Rotate90 => img.rotate90(),
        rawler::Orientation::Rotate180 => img.rotate180(),
        rawler::Orientation::Rotate270 => img.rotate270(),
        _ => img,
    };
    println!("pixel pipeline time: {:.2?}", now.elapsed());

    let now = Instant::now();
    img.write_to(
        &mut Cursor::new(img.as_bytes().to_owned()),
        image::ImageFormat::Jpeg,
    )
    .unwrap();
    img.save(args.output_path.clone()).unwrap();
    println!("jpeg save: {:.2?}", now.elapsed());
    println!("total time: {:.2?}", decode.elapsed());
}
