use pichromatic::cfa::CFA;
use pichromatic::demosaic::{Dim2, Point, Rect, demosaic_algorithms};
use pichromatic::pixel::{Image};
use pichromatic::cst::ColorSpaceTag;
use rawler::RawImageData;
use rawler::imgop::xyz::Illuminant;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use rayon::prelude::*;

fn main() {

    let input_path = "test_data/test.dng";
    let output_path = "result.ppm";

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let raw_image = rawler::decode_file(input_path).unwrap();
    let t1 = Instant::now();
    let camera_color_matrix = raw_image.camera.color_matrix[&Illuminant::D65].clone();
    let wb_coeffs = raw_image.wb_coeffs;
    let raw_image_dimentions = raw_image.dim();
    let raw_image_crop_area = raw_image.crop_area.unwrap();
    let raw_image_white_level = raw_image.whitelevel.as_bayer_array()[0];
    let raw_image_black_level = raw_image.blacklevel.as_bayer_array()[0];
    let raw_image_cfa = raw_image.camera.cfa.to_string();
    let raw_image_data = match raw_image.data {
        RawImageData::Integer(data) => data,
        _ => panic!("non integer data")
    };

    let wcs = ColorSpaceTag::AcesCg;


    // image processing pipeline example, this insnt by any means a complete pipeline
    // but the minimal steps to get a "correct" sRGB image from a raw file
    let mut image = Image::demosaic(
        &raw_image_data,
        Dim2{
            w: raw_image_dimentions.w,
            h: raw_image_dimentions.h
        },
        Rect{
            p: Point {
                x: raw_image_crop_area.p.x,
                y: raw_image_crop_area.p.y
            },
            d: Dim2 {
                w: raw_image_crop_area.d.w,
                h: raw_image_crop_area.d.h
            },
        },
        raw_image_black_level,
        raw_image_white_level,
        CFA::new(&raw_image_cfa),
        demosaic_algorithms::Markesteijn{}
    );
    let image = image.cfa_coeffs(wb_coeffs)
        .highlight_reconstruction(wb_coeffs)
        .camera_cst(wcs, camera_color_matrix)
        .tone_map()
        .cst(ColorSpaceTag::Srgb);

    println!("pipeline time: {}ms", t1.elapsed().as_millis());
    println!("pipeline fps: {}fps", 1000/t1.elapsed().as_millis());

    let (img, w, h) = to_u8(image);
    save_bmp(output_path, w, h, img).unwrap();
    
}

fn to_u8(image: &mut Image) -> (Vec<[u8;3]>, usize, usize){
    let new_image = image.data.par_iter().map(|pixel|{
        pixel.map(|sub_pixel| (sub_pixel.clamp(0.0, 1.0) * 255.0) as u8)
    }).collect();
    return (new_image, image.width, image.height)
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
