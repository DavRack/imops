#![warn(unused_extern_crates)]
use imops::{config, pipeline::run_pixel_pipeline};
use clap::Parser as Clap_parser;
use rawler::{self};
use std::io::Cursor;
use std::time::Instant;

#[derive(Clap_parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// code query
    #[arg(name = "input path", value_name = "input_path")]
    input_path: String,

    #[arg(
        short,
        name = "output path",
        default_value = "result.jpg",
        value_name = "output_path"
    )]
    output_path: String,

    #[arg(
        short,
        name = "config path",
        default_value = "imgconfig.toml",
        value_name = "config_path"
    )]
    config_path: String,
}

fn to_vec(data: imops::imops::PipelineImage) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let height = data.height;
    let width = data.width;
    let data2 = data.data.iter().flatten().map(|x| (x * 255.0) as u8).collect();
    let img = image::RgbImage::from_vec(width as u32, height as u32, data2).unwrap();
    return img;
}

fn main() {
    let args = Box::leak(Box::new(Args::parse()));

    let path = args.input_path.clone();

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let decode = Instant::now();
    let raw_image = rawler::decode_file(path).unwrap();
    let rotation = raw_image.orientation;

    println!("decode file: {:.2?}", decode.elapsed());

    let now = Instant::now();

    let config = config::parse_config(args.config_path.clone());

    let debayer_image = run_pixel_pipeline(raw_image, config);
    println!("pixel pipeline time: {:.2?}", now.elapsed());
    println!("pixel pipeline fps: {:.2?}", 1.0/now.elapsed().as_secs_f32());
    // println!("img size: {:?}", debayer_image.data.shape());
    let mut img = image::DynamicImage::ImageRgb8(to_vec(debayer_image));
    img = match rotation {
        rawler::Orientation::Rotate90 => img.rotate90(),
        rawler::Orientation::Rotate180 => img.rotate180(),
        rawler::Orientation::Rotate270 => img.rotate270(),
        _ => img,
    };

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