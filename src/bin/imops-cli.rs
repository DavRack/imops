#![warn(unused_extern_crates)]
use pichromatic::pixel::{Image};
use clap::Parser as Clap_parser;
use pichromatic_pipeline::pipeline::run_pixel_pipeline;
use pichromatic_pipeline::config;
use pichromatic::cfa::CFA;
use pichromatic::demosaic::{Dim2, Point, Rect, crop_and_normalize};
use pichromatic::image::ImageMetadata;
use rawler::{self, RawImageData};
use rawler::imgop::xyz::Illuminant;
use std::time::Instant;
use std::fs::File;
use std::io::{BufWriter, Write};

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

fn to_vec(data: &Image) -> image::RgbImage {
    let height = data.metadata.height;
    let width = data.metadata.width;
    let data2 = data
        .rgb_data
        .iter()
        .flatten()
        .map(|x| (x.max(0.0).min(1.0) * 255.0) as u8)
        .collect();
    let img = image::RgbImage::from_vec(width as u32, height as u32, data2).unwrap();
    return img;
}

fn main() {
    let args = Box::leak(Box::new(Args::parse()));

    let path = args.input_path.clone();

    let decode = Instant::now();
    let mut raw_image = rawler::decode_file(path).unwrap();
    let rotation = raw_image.orientation;

    println!("decode file: {:.2?}", decode.elapsed());

    let now = Instant::now();

    let config_data = std::fs::read_to_string(args.config_path.clone()).expect("Cannot read config file");
    let mut config = config::parse_config(config_data);

    let calibration_matrix_d65 = raw_image.camera.color_matrix[&Illuminant::D65].clone();
    let wb_coeffs = raw_image.wb_coeffs;
    let raw_image_dimensions = raw_image.dim();
    let raw_image_crop_area = raw_image.crop_area.unwrap();
    let raw_image_white_level = raw_image.whitelevel.as_bayer_array()[0];
    let raw_image_black_level = raw_image.blacklevel.as_bayer_array()[0];
    let raw_image_cfa = raw_image.camera.cfa.to_string();
    let _ = raw_image.apply_scaling();
    let raw_image_data = match raw_image.data {
        RawImageData::Float(data) => data,
        _ => panic!("non float data")
    };
    let image_metadata = ImageMetadata {
        width: raw_image_dimensions.w,
        height: raw_image_dimensions.h,
        crop_area: Some(Rect {
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

    let mut image = Image {
        raw_data: raw_image_data,
        rgb_data: vec![],
        metadata: image_metadata,
    };
    let normalized_raw_data = crop_and_normalize(&image);
    image.raw_data = normalized_raw_data;
    image.metadata.width = image.metadata.crop_area.unwrap().d.w;
    image.metadata.height = image.metadata.crop_area.unwrap().d.h;

    run_pixel_pipeline(&mut image, &mut config);
    println!("pixel pipeline time: {:.2?}", now.elapsed());
    println!("pixel pipeline fps: {:.2?}", 1.0/now.elapsed().as_secs_f32());

    let mut img = image::DynamicImage::ImageRgb8(to_vec(&image));
    img = match rotation {
        rawler::Orientation::Rotate90 => img.rotate90(),
        rawler::Orientation::Rotate180 => img.rotate180(),
        rawler::Orientation::Rotate270 => img.rotate270(),
        _ => img,
    };

    let now = Instant::now();
    img.save(args.output_path.clone()).unwrap();
    println!("jpeg save: {:.2?}", now.elapsed());
    println!("total time: {:.2?}", decode.elapsed());
}

#[allow(dead_code)]
fn to_u8(image: &mut Image) -> (Vec<[u8;3]>, usize, usize){
    let new_image = image.rgb_data.iter().map(|pixel|{
        pixel.map(|sub_pixel| (sub_pixel.clamp(0.0, 1.0) * 255.0) as u8)
    }).collect();
    return (new_image, image.metadata.width, image.metadata.height)
}

#[allow(dead_code)]
pub fn save_bmp(
    path: &str,
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>,
) -> std::io::Result<()> {
    assert_eq!(
        width * height,
        pixels.len(),
        "Pixel buffer size does not match image dimensions"
    );

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let padding_size = (4 - (width * 3) % 4) % 4;
    let row_size = (width * 3) + padding_size;
    let file_size = 14 + 40 + (row_size * height);

    writer.write_all(&[0x42, 0x4D])?;
    writer.write_all(&(file_size as u32).to_le_bytes())?;
    writer.write_all(&[0; 4])?;
    writer.write_all(&(54u32).to_le_bytes())?;

    writer.write_all(&(40u32).to_le_bytes())?;
    writer.write_all(&(width as i32).to_le_bytes())?;
    writer.write_all(&(height as i32).to_le_bytes())?;
    writer.write_all(&(1u16).to_le_bytes())?;
    writer.write_all(&(24u16).to_le_bytes())?;
    writer.write_all(&(0u32).to_le_bytes())?;
    writer.write_all(&(0u32).to_le_bytes())?;
    writer.write_all(&(0i32).to_le_bytes())?;
    writer.write_all(&(0i32).to_le_bytes())?;
    writer.write_all(&(0u32).to_le_bytes())?;
    writer.write_all(&(0u32).to_le_bytes())?;

    let padding = vec![0u8; padding_size]; 

    for y in (0..height).rev() {
        let start_index = y * width;
        let end_index = start_index + width;
        let row_pixels = &pixels[start_index..end_index];

        for pixel in row_pixels {
            writer.write_all(&[pixel[2], pixel[1], pixel[0]])?;
        }
        writer.write_all(&padding)?;
    }

    writer.flush()?;
    Ok(())
}
