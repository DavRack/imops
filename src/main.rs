mod demosaic;
mod imops;

use clap::Parser as Clap_parser;
use imops::FormedImage;
use ndarray::{s, Array2, Array3};
use rawler::{self, imgop::xyz::Illuminant};
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

    // /// code query
    // #[arg(name="query", value_name = "regex")]
    // query: String,

    // /// Select the tree sitter node kind
    // #[arg(long, short, name = "kind", value_name = "regex")]
    // kind: Option<String>,

    // /// Select the files to search
    // #[arg(trailing_var_arg = true, name = "files", value_name = "file names")]
    // files: Option<Vec<String>>,

    // /// Maximun recursive folder find depth
    // #[arg(long, short, value_name = "Int")]
    // max_recursive_depth: Option<u8>,

    // /// Config file path
    // #[arg(long, short, value_name = "String")]
    // config_file_path: Option<String>,

    // /// Before match context lines
    // #[arg(long, short, default_value = "5", value_name = "Int")]
    // before_context: usize,

    // /// After match context lines
    // #[arg(long, short, default_value = "5", value_name = "Int")]
    // after_context: usize,

    // /// Show all matches (if false will show the top most node match only)
    // #[arg(long, short, default_value = "false", value_name = "Int")]
    // show_all_matches: bool,
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

pub fn apply_color_correction(image: &Array3<f32>, color_matrix: &Array2<f32>) -> Array3<f32> {
    // 1. Ensure the color matrix is 3x3
    if color_matrix.shape() != &[3, 3] {
        panic!(
            "Color matrix must be 3x3, but got {:?}",
            color_matrix.shape()
        );
    }
    // 2. Get the dimensions of the input image
    let (rows, cols, _) = image.dim();

    // 3. Create a new array to store the corrected image.
    let mut corrected_image = Array3::<f32>::zeros((rows, cols, 3));

    // 4. Iterate over each pixel in the image
    for r in 0..rows {
        for c in 0..cols {
            // 5. Extract the RGB values for the current pixel as a slice
            let pixel = image.slice(s![r, c, ..]);

            // 6. No need to convert to f32, already is f32

            // 7. Perform the matrix multiplication: corrected_pixel = color_matrix * pixel
            let corrected_pixel_f32 = color_matrix.dot(&pixel);

            // 8. No need to convert back to original type, already f32.
            //    Still clamping to 0 to handle potential negative values after correction.
            let corrected_pixel_typed: ndarray::Array1<f32> = corrected_pixel_f32.mapv(|val| {
                f32::max(0.0, val) // Clamp negative values to 0.
                                   // You might need to add upper clamping if your f32 image has a max range like 0-1 or 0-255 (represented as f32).
            });

            // 9. Assign the corrected pixel values to the corrected image
            corrected_image
                .slice_mut(s![r, c, ..])
                .assign(&corrected_pixel_typed);
        }
    }

    corrected_image
}

fn debayer(image: &mut rawler::RawImage) -> imops::FormedImage {
    let mut debayer_image = imops::FormedImage {
        height: image.height,
        width: image.width,
        white: 16383.0,
        black: 1023.0,
        data: Array3::zeros((1, 1, 1)),
    };
    let _ = image.apply_scaling();
    if let rawler::RawImageData::Float(ref im) = image.data {
        let cfa = image.camera.cfa.clone();
        let cfa = demosaic::get_cfa(cfa, image.crop_area.unwrap());
        let (nim, width, height) =
            demosaic::crop(image.dim(), image.crop_area.unwrap(), im.to_vec());
        debayer_image.data =
            demosaic::demosaic_algorithms::linear_interpolation(width, height, cfa, nim);
        // let matrix = image.camera.color_matrix[&Illuminant::A].clone();
        // let matrix2 = image.camera.color_matrix.clone();
        // println!("{:?}", matrix2);
        // let m = Array2::from_shape_vec((3, 3), matrix).unwrap();
        // // let m2 = debayer_image.data;
        // // println!("{:?}", m2.shape());
        // // debayer_image.data = m2.clone();
        // let m = apply_color_correction(&debayer_image.data, &m);
        // debayer_image.data = m
    } else {
        eprintln!("Don't know how to process non-integer raw files");
    }
    return debayer_image;
}

fn to_vec(data: Array3<f32>) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let shape = data.shape();
    let height = shape[0];
    let width = shape[1];
    let data2 = data.map(|x| (x * 255.0) as u8);
    let img = image::RgbImage::from_vec(
        width as u32,
        height as u32,
        data2.as_slice().unwrap().to_vec(),
    )
    .unwrap();
    return img;
}

fn run_pixel_pipeline(image: &mut imops::FormedImage) -> FormedImage {
    let modules: Vec<Box<dyn imops::PipelineModule>> = vec![
        Box::new(imops::Exp { ev: 6.0 }),
        Box::new(imops::Wb {
            r: 1.0,
            g: 0.5,
            b: 1.0,
        }),
        Box::new(imops::Contrast { c: 1.5 }),
        Box::new(imops::Sigmoid { c: 6.0 }),
    ];

    let mut image = image.clone();
    for module in modules {
        image = module.process(&mut image);
    }

    return image;
}

fn main() {
    let args = Box::leak(Box::new(Args::parse()));
    println!("{:?}", args);

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

    debayer_image.data = small(debayer_image.data);
    let now = Instant::now();
    debayer_image = run_pixel_pipeline(&mut debayer_image);
    println!("img size: {:?}", debayer_image.data.shape());
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
