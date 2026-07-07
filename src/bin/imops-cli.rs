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
    let file_bytes = std::fs::read(path).expect("failed to read input file");
    let decode_params = rawler::decoders::RawDecodeParams::default();
    let mut raw_file = rawler::rawsource::RawSource::new_from_slice(&file_bytes);
    let raw_image = rawler::decode(&mut raw_file, &decode_params).expect("failed to decode raw image");
    let rotation = raw_image.orientation;
    let mut image = pichromatic_pipeline::extern_pipeline::parse_raw_image(raw_image);

    // Extract DNG metadata directly from the bytes and consolidate
    if let Some(parser) = pichromatic_pipeline::dng_metadata::DngMetadataParser::new(&file_bytes) {
        let dng_meta = parser.parse();
        pichromatic_pipeline::extern_pipeline::consolidate_dng_metadata(&mut image, &dng_meta);
    }
    println!("decode file: {:.2?}", decode.elapsed());

    let now = Instant::now();

    let config_data = std::fs::read_to_string(args.config_path.clone()).expect("Cannot read config file");
    let mut config = config::parse_config(config_data);

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
