use color::ColorSpaceTag;
use pichromatic::film::exposure::radiance::{absolute_luminance_gain, sunny16_exposure};
use pichromatic::film::scan::mean_rgb;
use pichromatic::film::stock::StockId;
use pichromatic::film::units::IsoSpeed;
use pichromatic::film::{FilmFormat, FilmOutput, FilmParams};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::{Image, PixelOps};

fn main() {
    println!("neutrals sceneY → PositiveLinear");
    for y in [0.0f32, 0.005, 0.01, 0.02, 0.05, 0.1, 0.185] {
        let e = sunny16_exposure(IsoSpeed(200.0));
        let gain = absolute_luminance_gain(e.shutter_seconds as f64, e.f_number as f64, e.iso as f64)
            as f32;
        let fill = [y * gain; 3];
        let mut img = Image {
            rgb_data: vec![fill; 64 * 64],
            raw_data: vec![],
            metadata: ImageMetadata {
                width: 64,
                height: 64,
                color_space: Some(ColorSpaceTag::AcesCg),
                ..Default::default()
            },
        };
        img.film(&FilmParams {
            stock: StockId::ColorNeg200,
            film_format: FilmFormat::Film35mm,
            seed: 1,
            output: FilmOutput::PositiveLinear,
        })
        .unwrap();
        let out = mean_rgb(&img.rgb_data);
        println!(
            "  {y:.3} → Y={:.4} rgb=[{:.4},{:.4},{:.4}]",
            out.luminance(),
            out[0],
            out[1],
            out[2]
        );
    }
    for (label, fill_rel) in [
        ("red", [0.4f32, 0.05, 0.05]),
        ("green", [0.05, 0.4, 0.05]),
        ("blue", [0.05, 0.05, 0.4]),
    ] {
        let e = sunny16_exposure(IsoSpeed(200.0));
        let gain = absolute_luminance_gain(e.shutter_seconds as f64, e.f_number as f64, e.iso as f64)
            as f32;
        let fill = [fill_rel[0] * gain, fill_rel[1] * gain, fill_rel[2] * gain];
        let mut img = Image {
            rgb_data: vec![fill; 64 * 64],
            raw_data: vec![],
            metadata: ImageMetadata {
                width: 64,
                height: 64,
                color_space: Some(ColorSpaceTag::AcesCg),
                ..Default::default()
            },
        };
        img.film(&FilmParams {
            stock: StockId::ColorNeg200,
            film_format: FilmFormat::Film35mm,
            seed: 1,
            output: FilmOutput::PositiveLinear,
        })
        .unwrap();
        let out = mean_rgb(&img.rgb_data);
        println!(
            "  {label}: out=[{:.3},{:.3},{:.3}]",
            out[0], out[1], out[2]
        );
    }
}
