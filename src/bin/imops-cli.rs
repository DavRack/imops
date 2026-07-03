// use pichromatic::image::metadata::{Cicp, CicpColorPrimaries};
// use imops::{config, pipeline::run_pixel_pipeline};
#![warn(unused_extern_crates)]
use pichromatic::pixel::{Image};
use clap::Parser as Clap_parser;
use pichromatic_pipeline::pipeline::run_pixel_pipeline;
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

// fn to_vec(data: imops::imops::PipelineImage) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
//     let height = data.height;
//     let width = data.width;
//     let data2 = data
//         .data
//         .iter()
//         .flatten()
//         .map(|x| (x.max(0.0).min(1.0) * 255.0) as u8)
//         .collect();
//     let img = image::RgbImage::from_vec(width as u32, height as u32, data2).unwrap();
//     return img;
// }

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


    img.set_color_space(Cicp::DISPLAY_P3).unwrap();
    println!("{:?}",img.color_space());
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

fn to_u8(image: &mut Image) -> (Vec<[u8;3]>, usize, usize){
    let new_image = image.rgb_data.iter().map(|pixel|{
        pixel.map(|sub_pixel| (sub_pixel.clamp(0.0, 1.0) * 255.0) as u8)
    }).collect();
    return (new_image, image.metadata.width, image.metadata.height)
}


pub fn save_bmp(
    path: &str,
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>, // Kept as requested
) -> std::io::Result<()> {
    assert_eq!(
        width * height,
        pixels.len(),
        "Pixel buffer size does not match image dimensions"
    );

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // 1. Calculate Padding
    let padding_size = (4 - (width * 3) % 4) % 4;
    let row_size = (width * 3) + padding_size;

    // 2. File Header (14 bytes)
    // Calculated as usize initially
    let file_size = 14 + 40 + (row_size * height);

    // Signature "BM"
    writer.write_all(&[0x42, 0x4D])?;
    
    // File size (u32, little endian)
    // CRITICAL FIX: Cast to u32. 
    // Your previous code wrote 8 bytes here (usize) which broke the header.
    writer.write_all(&(file_size as u32).to_le_bytes())?;
    
    // Reserved 1 & 2 (u16, u16)
    writer.write_all(&[0; 4])?;
    
    // Pixel data offset (u32). 14 (File Header) + 40 (Info Header) = 54
    writer.write_all(&(54u32).to_le_bytes())?;

    // 3. DIB Header / Info Header (40 bytes - BITMAPINFOHEADER)
    writer.write_all(&(40u32).to_le_bytes())?;
    
    // Width (i32)
    writer.write_all(&(width as i32).to_le_bytes())?;
    
    // Height (i32) - Positive means bottom-up
    writer.write_all(&(height as i32).to_le_bytes())?;
    
    // Planes (always 1)
    writer.write_all(&(1u16).to_le_bytes())?;
    
    // Bits per pixel (24 for RGB)
    writer.write_all(&(24u16).to_le_bytes())?;
    
    // Compression (0 = BI_RGB)
    writer.write_all(&(0u32).to_le_bytes())?;
    
    // Image size (can be 0 for BI_RGB)
    writer.write_all(&(0u32).to_le_bytes())?;
    
    // X & Y pixels per meter (0 is fine)
    writer.write_all(&(0i32).to_le_bytes())?;
    writer.write_all(&(0i32).to_le_bytes())?;
    
    // Colors used & Important colors
    writer.write_all(&(0u32).to_le_bytes())?;
    writer.write_all(&(0u32).to_le_bytes())?;

    // 4. Pixel Data
    let padding = vec![0u8; padding_size]; 

    for y in (0..height).rev() {
        let start_index = y * width;
        let end_index = start_index + width;
        let row_pixels = &pixels[start_index..end_index];

        for pixel in row_pixels {
            // Write BGR
            writer.write_all(&[pixel[2], pixel[1], pixel[0]])?;
        }

        // Write padding
        writer.write_all(&padding)?;
    }

    writer.flush()?;
    Ok(())
}
