use pichromatic::color::ColorSpaceTag;
use pichromatic::film::exposure::radiance::{absolute_luminance_gain, sunny16_exposure};
use pichromatic::film::scan::mean_rgb;
use pichromatic::film::stock::StockId;
use pichromatic::film::units::IsoSpeed;
use pichromatic::film::{FilmFormat, FilmOutput, FilmParams};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::{Image, PixelOps};

fn main() {
    let raw_image_path = "test_data/plaza.dng";
    println!("Loading {raw_image_path}...");
    let file_bytes = std::fs::read(raw_image_path).expect("Failed to read raw image");
    let decode_params = rawler::decoders::RawDecodeParams::default();
    let mut raw_file = rawler::rawsource::RawSource::new_from_slice(&file_bytes);
    let raw_image = rawler::decode(&mut raw_file, &decode_params).expect("Failed to decode");

    for ev_boost in [0.0f32, 3.0] {
        println!("\n=== Testing CineStill 50D with EV boost = {ev_boost} ===");

        let make_pipeline =
            |enable_hal: bool| -> Vec<Box<dyn pichromatic_pipeline::modules::PipelineModule>> {
                use pichromatic_pipeline::modules::*;
                vec![
                    Box::new(Module {
                        name: "Demosaic".to_string(),
                        cache: None,
                        config: Demosaic {
                            algorithm: Parameter::new(DemosaicAlgorithmType::Markesteijn, ""),
                        },
                    }),
                    Box::new(Module {
                        name: "HighlightReconstruction".to_string(),
                        cache: None,
                        config: HighlightReconstruction {},
                    }),
                    Box::new(Module {
                        name: "CFACoeffs".to_string(),
                        cache: None,
                        config: CFACoeffs {},
                    }),
                    Box::new(Module {
                        name: "Baselineexp".to_string(),
                        cache: None,
                        config: BaselineExposureCompensation {},
                    }),
                    Box::new(Module {
                        name: "Exp".to_string(),
                        cache: None,
                        config: Exp {
                            ev: Parameter::new(ev_boost, ""),
                        },
                    }),
                    Box::new(Module {
                        name: "CST".to_string(),
                        cache: None,
                        config: CST {
                            target_color_space: Parameter::new("AcesCg".to_string(), ""),
                        },
                    }),
                    Box::new(Module {
                        name: "Film".to_string(),
                        cache: None,
                        config: Film {
                            stock: Parameter::new("CineStill50D".to_string(), ""),
                            film_format: Parameter::new("Film35mm".to_string(), ""),
                            render_width_mm: Parameter::new(Some(35.0), ""),
                            seed: Parameter::new(1, ""),
                            enable_halation: Parameter::new(enable_hal, ""),
                            output: Parameter::new("PositiveLinear".to_string(), ""),
                            compensate_box_speed: Parameter::new(true, ""),
                            scanner_s_curve: Parameter::new(0.0, ""),
                        },
                    }),
                ]
            };

        let mut img_on = pichromatic_pipeline::extern_pipeline::parse_raw_image(raw_image.clone());
        if let Some(parser) =
            pichromatic_pipeline::dng_metadata::DngMetadataParser::new(&file_bytes)
        {
            pichromatic_pipeline::extern_pipeline::consolidate_dng_metadata(
                &mut img_on,
                &parser.parse(),
            );
        }
        let mut img_off = img_on.clone();

        let mut config_on = pichromatic_pipeline::config::PipelineConfig {
            pipeline_modules: make_pipeline(true),
        };
        let mut config_off = pichromatic_pipeline::config::PipelineConfig {
            pipeline_modules: make_pipeline(false),
        };

        pichromatic_pipeline::pipeline::run_pixel_pipeline_with_backend(
            &mut img_on,
            &mut config_on,
            &pichromatic_pipeline::backend::Backend::Cpu,
        );
        pichromatic_pipeline::pipeline::run_pixel_pipeline_with_backend(
            &mut img_off,
            &mut config_off,
            &pichromatic_pipeline::backend::Backend::Cpu,
        );

        let mut max_diff = [0.0f32; 3];
        let mut mean_diff = [0.0f64; 3];
        let mut count_changed = 0usize;
        for (a, b) in img_on.rgb_data.iter().zip(img_off.rgb_data.iter()) {
            let d = [
                (a[0] - b[0]).abs(),
                (a[1] - b[1]).abs(),
                (a[2] - b[2]).abs(),
            ];
            for c in 0..3 {
                max_diff[c] = max_diff[c].max(d[c]);
                mean_diff[c] += d[c] as f64;
            }
            if d[0] > 1e-4 || d[1] > 1e-4 || d[2] > 1e-4 {
                count_changed += 1;
            }
        }
        let total = img_on.rgb_data.len();
        println!(
            "  Pixels changed (>1e-4): {} / {} ({:.2}%)",
            count_changed,
            total,
            (count_changed as f64 / total as f64) * 100.0
        );
        println!(
            "  Max channel diffs: R={:.6}, G={:.6}, B={:.6}",
            max_diff[0], max_diff[1], max_diff[2]
        );
        println!(
            "  Mean channel diffs: R={:.8}, G={:.8}, B={:.8}",
            mean_diff[0] / total as f64,
            mean_diff[1] / total as f64,
            mean_diff[2] / total as f64
        );
    }
}
