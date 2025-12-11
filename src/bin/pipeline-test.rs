use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use color::ColorSpaceTag;
use pichromatic::cfa::CFA;
use pichromatic::demosaic::{Dim2, Point, Rect, demosaic_algorithms};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::Image;
use pichromatic_pipeline::config::PipelineConfig;
use pichromatic_pipeline::modules::{CFACoeffs, CST, Contrast, Demosaic, Exp, HighlightReconstruction, LCH, Module, PipelineModule, ToneMap};
use pichromatic_pipeline::pipeline::run_pixel_pipeline;
use rawler::RawImageData;
use rawler::imgop::xyz::Illuminant;
use pichromatic_pipeline;


fn main() {

    let input_path = "test_data/test.dng";
    let output_path = "result.ppm";

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let raw_image = rawler::decode_file(input_path).unwrap();
    let t1 = Instant::now();
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

    let image = Image {
        raw_data: raw_image_data,
        rgb_data: vec![],
        metadata: image_metadata,
    };

    let pipeline1: Vec<Box<dyn PipelineModule>> = vec![
        Box::new(Module {
            name: "Demosaic".to_string(),
            cache: None,
            config: Demosaic{ algorithm: "markesteijn".to_string() },
            // mask: None,
        }),
        Box::new(Module {
            name: "CFACoeffs".to_string(),
            cache: None,
            config: CFACoeffs {},
            // mask: None,
        }),
        Box::new(Module {
            name: "HighlightReconstruction".to_string(),
            cache: None,
            config: HighlightReconstruction {},
            // mask: None,
        }),
        Box::new(Module {
            name: "Exp".to_string(),
            cache: None,
            config: Exp { ev: 1.75},
            // mask: None,
        }),
        Box::new(Module {
            name: "Contrast".to_string(),
            cache: None,
            config: Contrast { c: 1.75},
            // mask: None,
        }),
        Box::new(Module {
            name: "CST".to_string(),
            cache: None,
            config: CST { target_color_space: ColorSpaceTag::XyzD65},
            // mask: None,
        }),
        Box::new(Module {
            name: "LHC".to_string(),
            cache: None,
            config: LCH{
                lc: 1.0,
                cc: 1.3,
                hc: 1.0,
            },
            // mask: None,
        }),
        Box::new(Module {
            name: "Sigmoid (Soft)".to_string(),
            cache: None,
            config: ToneMap {},
            // mask: None,
        }),
        Box::new(Module {
            name: "CST".to_string(),
            cache: None,
            config: CST { target_color_space: ColorSpaceTag::Srgb},
            // mask: None,
        })
    ];
    let pipeline1 = PipelineConfig{
        pipeline_modules: pipeline1,
    };
    let mut result1 = run_pixel_pipeline(image, pipeline1);
    println!("total pipeline time: {}ms", t1.elapsed().as_millis());
    println!("total pipeline fps: {}fps", 1000/t1.elapsed().as_millis());

    let (pixels, width, height) = to_u8(&mut result1);
    save_bmp(output_path, width, height, pixels).unwrap();
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
