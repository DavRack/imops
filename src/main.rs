use ndarray::Array3;
use rayon::iter::IntoParallelRefIterator;
use serde_derive::Deserialize;
use std::{time::Instant, usize};
use rawler;
mod demosaic;
use std::io::Cursor;
mod imops;

#[derive(Default, Debug, Deserialize)]
struct Channels {
    r: bool,
    g: bool,
    b: bool,
}
struct DebayerImage {
    height: usize,
    width: usize,
    black: f32,
    white: f32,
    data: Array3<f32>,
}

fn debayer(image: &mut rawler::RawImage) -> DebayerImage {
    let mut debayer_image = DebayerImage {
        height: image.height,
        width: image.width,
        white: 16383.0,
        black: 1023.0,
        data: Array3::zeros((1,1,1))
    };
    image.apply_scaling();
    if let rawler::RawImageData::Float(ref im) = image.data {
        // debayer_image.white = im.iter().max().unwrap().clone() as f32;
        // debayer_image. black = im.iter().min().unwrap().clone() as f32;

        // let nim = demosaic::debug_data();
        // let width = 6;
        // let height = 12;
        // debayer_image.data = demosaic::photosite(width, height, 0.0, 2.0, nim);

        let cfa = image.camera.cfa.clone();

        let (nim, width, height) = demosaic::crop(image.dim(), image.crop_area.unwrap(), im.to_vec());
        debayer_image.data = demosaic::linear_interpolation(
            width,
            height,
            cfa,
            nim,
        );
    } else {
        eprintln!("Don't know how to process non-integer raw files");
    }
    return debayer_image;
}

fn to_vec(data: Array3<f32>) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>>{
    let shape = data.shape();
    let height = shape[0];
    let width = shape[1];
    println!("h{:}, {:}", height, width);
    let data2 = data.map(|x| (x*255.0) as u8);
    let img = image::RgbImage::from_vec(width as u32, height as u32, data2.as_slice().unwrap().to_vec()).unwrap();
    return img;
}

fn main() {
    let path = "test2.RAF";
    // let mut file = BufReader::new(File::open(path).unwrap());
    // demosaic::test_mosaic();
    // return;

    // // Decode the file to extract the raw pixels and its associated metadata
    // let raw_image = RawImage::decode(&mut file).unwrap();
    let mut raw_image = rawler::decode_file(path).unwrap();
    let now = Instant::now();
    // let image = rawloader::decode_file(path).unwrap();
    let mut debayer_image = debayer(&mut raw_image);

    debayer_image.data = imops::exp(6.0, &mut debayer_image.data);
    debayer_image.data = imops::wb(3.0, &mut debayer_image.data);
    debayer_image.data = imops::sigmoid(6.0, &mut debayer_image.data);
    // debayer_image.data = imops::get_channel(2, &mut debayer_image.data);
    // debayer_image.data = imops::film_curve(0.9, 0.8, 0.4, 0.5, 32.0, &mut debayer_image.data);

    let img = image::DynamicImage::ImageRgb8(to_vec(debayer_image.data));
    println!("{:?}", img.height());
    let elapsed = now.elapsed();
    println!("pixel pipeline time: {:.2?}",elapsed);
    img.write_to(&mut Cursor::new(img.as_bytes().to_owned()), image::ImageFormat::Jpeg).unwrap();
    img.save("test.jpg").unwrap();
}
