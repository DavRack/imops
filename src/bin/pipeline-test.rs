use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::extern_pipeline::get_raw_img_internal;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let input_path = if args.len() > 1 {
        &args[1]
    } else {
        "test_data/color_check.ARW"
    };
    let config_path = if args.len() > 2 {
        &args[2]
    } else {
        "imgconfig-film.toml"
    };

    println!("Reading RAW file: {input_path}");
    let file_bytes = fs::read(input_path).expect("Failed to read raw file");
    let image = get_raw_img_internal(&file_bytes);

    println!("=== PARSED METADATA ===");
    println!("Dimensions: {}x{}", image.metadata.width, image.metadata.height);
    println!("CFA: {:?}", image.metadata.cfa);
    println!("WB Coeffs: {:?}", image.metadata.wb_coeffs);
    println!("Calibration Matrix D65: {:?}", image.metadata.calibration_matrix_d65);
    println!("Baseline Exposure: {:?}", image.metadata.baseline_exposure);
    println!("Shutter Seconds: {:?}", image.metadata.shutter_seconds);
    println!("F-Number: {:?}", image.metadata.f_number);
    println!("ISO: {:?}", image.metadata.iso);
    println!("Color Space: {:?}", image.metadata.color_space);

    let config_str = fs::read_to_string(config_path).expect("Failed to read config file");
    let pipeline = parse_config(config_str);

    let mut img = image;
    for (i, module) in pipeline.pipeline_modules.iter().enumerate() {
        module.process(&mut img);
        let rgb_len = img.rgb_data.len();
        let (r_mean, g_mean, b_mean) = if rgb_len > 0 {
            let (mut r, mut g, mut b) = (0.0f64, 0.0f64, 0.0f64);
            for px in &img.rgb_data {
                r += px[0] as f64;
                g += px[1] as f64;
                b += px[2] as f64;
            }
            (r / rgb_len as f64, g / rgb_len as f64, b / rgb_len as f64)
        } else {
            (0.0, 0.0, 0.0)
        };
        println!(
            "Module {:2}: {:<30} | rgb_len={:<8} | mean=[{:.4}, {:.4}, {:.4}] | color_space={:?}",
            i,
            module.schema().name,
            rgb_len,
            r_mean,
            g_mean,
            b_mean,
            img.metadata.color_space
        );
    }
}
