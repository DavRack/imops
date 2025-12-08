use pichromatic::demosaic::demosaic_algorithms;
use pichromatic::pixel::{Image};
use pichromatic::cst::ColorSpaceTag;
use rawler::imgop::xyz::Illuminant;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn main() {
    let input_path = "test_data/test.dng";
    let output_path = "result.ppm";

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let raw_image = rawler::decode_file(input_path).unwrap();
    let camera_color_matrix = raw_image.camera.color_matrix[&Illuminant::D65].clone();
    let wb_coeffs = raw_image.wb_coeffs;
    let wcs = ColorSpaceTag::AcesCg;

    let t1 = Instant::now();
    let t = Instant::now();
    let mut image = Image::demosaic(raw_image, demosaic_algorithms::Fast{});
    println!("demosaic time: {}ms", t.elapsed().as_millis());
    let t = Instant::now();
    let image = image.cfa_coeffs(wb_coeffs);
    let image = image.highlight_reconstruction(wb_coeffs);
    println!("wb_coeffs time: {}ms", t.elapsed().as_millis());
    let t = Instant::now();
    let image = image.exp(1.0);
    let image = image.contrast(1.25);
    println!("exp time: {}ms", t.elapsed().as_millis());
    let t = Instant::now();
    let image = image.camera_cst(wcs, camera_color_matrix);
    println!("cst time: {}ms", t.elapsed().as_millis());
    let t = Instant::now();
    let image = image.tone_map();
    println!("tone_map time: {}ms", t.elapsed().as_millis());
    let t = Instant::now();
    let image = image.cst(ColorSpaceTag::Srgb);
    println!("cst time: {}ms\n", t.elapsed().as_millis());
    println!("pipeline time: {}ms", t1.elapsed().as_millis());
    println!("pipeline fps: {}fps", 1000/t1.elapsed().as_millis());

    let (img, w, h) = to_u8(image);
    save_bmp(output_path, w, h, img).unwrap();
    
}

fn to_u8(image: &mut Image) -> (Vec<[u8;3]>, usize, usize){
    let new_image = image.data.iter().map(|pixel|{
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
