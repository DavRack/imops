#![warn(unused_extern_crates)]
use clap::Parser as Clap_parser;
use pichromatic_pipeline::config;
use pichromatic_pipeline::pipeline::run_pixel_pipeline;
use std::time::Instant;

#[derive(Clap_parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input RAW / DNG path
    #[arg(name = "input path", value_name = "input_path")]
    input_path: String,

    /// Output path (`.exr` = 16-bit lossless OpenEXR, `.jpg`/`.jpeg` = 8-bit JPEG)
    #[arg(
        short,
        name = "output path",
        default_value = "result.exr",
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

fn main() {
    let args = Box::leak(Box::new(Args::parse()));

    let path = args.input_path.clone();

    let decode = Instant::now();
    let file_bytes = std::fs::read(path).expect("failed to read input file");
    let decode_params = rawler::decoders::RawDecodeParams::default();
    let mut raw_file = rawler::rawsource::RawSource::new_from_slice(&file_bytes);
    let raw_image =
        rawler::decode(&mut raw_file, &decode_params).expect("failed to decode raw image");
    let rotation = raw_image.orientation;
    let mut image = pichromatic_pipeline::extern_pipeline::parse_raw_image(raw_image);

    if let Some(parser) = pichromatic_pipeline::dng_metadata::DngMetadataParser::new(&file_bytes) {
        let dng_meta = parser.parse();
        pichromatic_pipeline::extern_pipeline::consolidate_dng_metadata(&mut image, &dng_meta);
    }
    println!("decode file: {:.2?}", decode.elapsed());

    let now = Instant::now();

    let config_data =
        std::fs::read_to_string(args.config_path.clone()).expect("Cannot read config file");
    let mut config = config::parse_config(config_data);

    run_pixel_pipeline(&mut image, &mut config);
    println!("pixel pipeline time: {:.2?}", now.elapsed());
    println!(
        "pixel pipeline fps: {:.2?}",
        1.0 / now.elapsed().as_secs_f32()
    );

    let now = Instant::now();
    let format = imops::output::save_image(&args.output_path, &image, rotation)
        .unwrap_or_else(|e| panic!("{e}"));
    match format {
        imops::output::OutputFormat::Exr => {
            println!("exr save (f16 ZIP16 lossless): {:.2?}", now.elapsed());
        }
        imops::output::OutputFormat::Jpeg => {
            println!("jpeg save: {:.2?}", now.elapsed());
        }
    }
    println!("wrote {}", args.output_path);
    println!("total time: {:.2?}", decode.elapsed());
}
