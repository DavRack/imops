use clap::Parser;
use pichromatic::film::fixtures::colorchecker_image;
use pichromatic_pipeline::backend::Backend;
use pichromatic_pipeline::config::parse_config;
use pichromatic_pipeline::pipeline::run_pixel_pipeline_with_backend;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::time::Instant;

const PATCH_GRID_NAMES: [[&str; 6]; 4] = [
    [
        "Bluish Green",
        "Blue Flower",
        "Foliage",
        "Blue Sky",
        "Light Skin",
        "Dark Skin",
    ],
    [
        "Orange Yellow",
        "Purplish Blue",
        "Moderate Red",
        "Purple",
        "Yellow Green",
        "Orange",
    ],
    ["Cyan", "Magenta", "Yellow", "Red", "Green", "Blue"],
    [
        "White 9.5",
        "Neutral 8",
        "Neutral 6.5",
        "Neutral 5",
        "Neutral 3.5",
        "Black 2",
    ],
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchData {
    pub index: usize,
    pub name: String,
    pub row: usize,
    pub col: usize,
    pub rgb_u8: [u8; 3],
    pub rgb_norm: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StockData {
    pub source_image: String,
    pub image_width: u32,
    pub image_height: u32,
    pub patches: Vec<PatchData>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Database {
    #[serde(default)]
    pub stocks: BTreeMap<String, StockData>,
}

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Generate ColorChecker test pattern and process with pipeline",
    long_about = None
)]
struct Args {
    /// Output path (.png, .exr, .jpg)
    #[arg(short, long, default_value = "color_check.png")]
    output: String,

    /// Patch size in pixels (6x4 grid)
    #[arg(short, long, default_value_t = 200)]
    patch_size: usize,

    /// Optional config path (JSON or TOML). If not specified, default Film pipeline is used.
    #[arg(short, long)]
    config: Option<String>,
}

fn sample_colorchecker_patches(
    image: &pichromatic::pixel::Image,
    patch_size: usize,
) -> Vec<PatchData> {
    let width = image.metadata.width;
    let height = image.metadata.height;
    let half_box = (patch_size / 4).max(1);

    let mut patches = Vec::with_capacity(24);
    let mut index = 1;

    for row in 0..4 {
        for col in 0..6 {
            let cx = col * patch_size + patch_size / 2;
            let cy = row * patch_size + patch_size / 2;

            let sample_x0 = cx.saturating_sub(half_box);
            let sample_x1 = (cx + half_box).min(width);
            let sample_y0 = cy.saturating_sub(half_box);
            let sample_y1 = (cy + half_box).min(height);

            let mut sum_r = 0.0f64;
            let mut sum_g = 0.0f64;
            let mut sum_b = 0.0f64;
            let mut count = 0usize;

            for y in sample_y0..sample_y1 {
                for x in sample_x0..sample_x1 {
                    let px = image.rgb_data[y * width + x];
                    sum_r += px[0] as f64;
                    sum_g += px[1] as f64;
                    sum_b += px[2] as f64;
                    count += 1;
                }
            }

            let n = count.max(1) as f64;
            let mean_r = sum_r / n;
            let mean_g = sum_g / n;
            let mean_b = sum_b / n;

            let rgb_norm = [
                mean_r.clamp(0.0, 1.0) as f32,
                mean_g.clamp(0.0, 1.0) as f32,
                mean_b.clamp(0.0, 1.0) as f32,
            ];

            let rgb_u8 = [
                (rgb_norm[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                (rgb_norm[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                (rgb_norm[2] * 255.0).round().clamp(0.0, 255.0) as u8,
            ];

            patches.push(PatchData {
                index,
                name: PATCH_GRID_NAMES[row][col].to_string(),
                row: row + 1,
                col: col + 1,
                rgb_u8,
                rgb_norm,
            });

            index += 1;
        }
    }

    patches
}

fn center_text(text: &str, width: usize) -> String {
    if text.len() >= width {
        text[..width].to_string()
    } else {
        let pad = width - text.len();
        let left = pad / 2;
        let right = pad - left;
        format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
    }
}

fn print_ascii_table(patches: &[PatchData]) {
    assert_eq!(patches.len(), 24);
    let cell_width = 17;
    let divider: String = (0..6)
        .map(|_| format!("+{:-<width$}", "", width = cell_width))
        .collect::<Vec<_>>()
        .join("")
        + "+";

    println!("\nColorChecker Patches (RGB):");
    for row in 0..4 {
        println!("{}", divider);

        let mut name_line = String::from("|");
        for col in 0..6 {
            let idx = row * 6 + col;
            let cell = center_text(&patches[idx].name, cell_width);
            name_line.push_str(&cell);
            name_line.push('|');
        }
        println!("{}", name_line);

        let mut rgb_line = String::from("|");
        for col in 0..6 {
            let idx = row * 6 + col;
            let p = &patches[idx];
            let rgb_str = format!("({}, {}, {})", p.rgb_u8[0], p.rgb_u8[1], p.rgb_u8[2]);
            let cell = center_text(&rgb_str, cell_width);
            rgb_line.push_str(&cell);
            rgb_line.push('|');
        }
        println!("{}", rgb_line);
    }
    println!("{}\n", divider);
}

fn main() {
    let args = Args::parse();

    println!(
        "Generating ColorChecker (24 patches, {}x{} px per patch, total {}x{})...",
        args.patch_size,
        args.patch_size,
        args.patch_size * 6,
        args.patch_size * 4
    );

    let (mut image, _) = colorchecker_image(args.patch_size);

    let e = pichromatic::film::exposure::radiance::sunny16_exposure(
        pichromatic::film::units::IsoSpeed(400.0),
    );
    image.metadata.shutter_seconds = Some(e.shutter_seconds);
    image.metadata.f_number = Some(e.f_number);
    image.metadata.iso = Some(e.iso);

    let mut config = if let Some(ref config_path) = args.config {
        println!("Loading pipeline config from: {config_path}");
        let config_str = fs::read_to_string(config_path)
            .unwrap_or_else(|e| panic!("Failed to read config file '{config_path}': {e}"));
        parse_config(config_str)
    } else {
        println!(
            "Using default pipeline: BaselineExposureCompensation -> Film (Portra400, Film35mm, halation=true, seed=1, mode=PositiveInverseHd) -> CST (sRGB)"
        );
        let default_config_json = serde_json::json!({
            "pipeline_modules": [
                {
                    "name": "BaselineExposureCompensation"
                },
                {
                    "name": "Film",
                    "stock": "Portra400",
                    "film_format": "Film35mm",
                    "enable_halation": true,
                    "seed": 1,
                    "output": "PositiveInverseHd"
                },
                {
                    "name": "CST",
                    "target_color_space": "Srgb"
                }
            ]
        })
        .to_string();
        parse_config(default_config_json)
    };

    let start = Instant::now();
    run_pixel_pipeline_with_backend(&mut image, &mut config, &Backend::Cpu);
    println!("Pipeline execution completed in {:.2?}", start.elapsed());

    let save_start = Instant::now();
    let format = imops::output::save_image(&args.output, &image, rawler::Orientation::Normal)
        .unwrap_or_else(|e| panic!("Failed to save image: {e}"));

    println!(
        "Saved {:?} output to '{}' ({:.2?})",
        format,
        args.output,
        save_start.elapsed()
    );

    let patches = sample_colorchecker_patches(&image, args.patch_size);

    let mut database = Database::default();
    let stock_key = "synthetic_colorchecker".to_string();
    database.stocks.insert(
        stock_key,
        StockData {
            source_image: args.output.clone(),
            image_width: image.metadata.width as u32,
            image_height: image.metadata.height as u32,
            patches: patches.clone(),
        },
    );

    let toml_str = toml::to_string_pretty(&database)
        .unwrap_or_else(|e| panic!("Failed to serialize database to TOML: {e}"));
    fs::write("color_check_bin.toml", &toml_str)
        .unwrap_or_else(|e| panic!("Failed to write color_check_bin.toml: {e}"));
    println!("Saved patch measurements to 'color_check_bin.toml'");

    print_ascii_table(&patches);
}
