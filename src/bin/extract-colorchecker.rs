use clap::Parser;
use image::{GenericImageView, Rgba};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// 24 standard ColorChecker patch names in row-major order (Row 1..4, Col 1..6).
pub const PATCH_NAMES: [&str; 24] = [
    // Row 1
    "Bluish Green",
    "Blue Flower",
    "Foliage",
    "Blue Sky",
    "Light Skin",
    "Dark Skin",
    // Row 2
    "Orange Yellow",
    "Purplish Blue",
    "Moderate Red",
    "Purple",
    "Yellow Green",
    "Orange",
    // Row 3
    "Cyan",
    "Magenta",
    "Yellow",
    "Red",
    "Green",
    "Blue",
    // Row 4
    "White 9.5",
    "Neutral 8",
    "Neutral 6.5",
    "Neutral 5",
    "Neutral 3.5",
    "Black 2",
];

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Extract 24 ColorChecker patch RGB values from images and save to TOML",
    long_about = None
)]
struct Args {
    /// Input image file path or directory (defaults to Colorplus-0EV-Frontier.png)
    #[arg(
        short,
        long,
        default_value = "film-data/color-check/data/Colorplus-0EV-Frontier.png"
    )]
    input: PathBuf,

    /// Output TOML file path
    #[arg(
        short,
        long,
        default_value = "film-data/color-check/colorchecker_patches.toml"
    )]
    output: PathBuf,

    /// Process all images in the input directory
    #[arg(short, long)]
    all: bool,

    /// Optional comma-separated filename filters (e.g. "0ev,frontier" matches files containing both)
    #[arg(short, long)]
    filter: Option<String>,

    /// Sampling box size in pixels centered on each patch (e.g. 32 for 32x32)
    #[arg(long, default_value_t = 32)]
    sample_size: u32,

    /// Print detailed patch values to stdout
    #[arg(short, long, default_value_t = true)]
    verbose: bool,
}

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

/// Extract patch RGB data from an image.
pub fn extract_patches(img_path: &Path, sample_size: u32) -> Result<StockData, String> {
    let img = image::io::Reader::open(img_path)
        .map_err(|e| format!("Failed to open image file '{}': {e}", img_path.display()))?
        .with_guessed_format()
        .map_err(|e| {
            format!(
                "Failed to determine image format for '{}': {e}",
                img_path.display()
            )
        })?
        .decode()
        .map_err(|e| format!("Failed to decode image '{}': {e}", img_path.display()))?;

    let (width, height) = img.dimensions();

    let patch_grid_names: [[&str; 4]; 6] = [
        ["White 9.5", "Cyan", "Orange Yellow", "Bluish Green"],
        ["Neutral 8", "Magenta", "Purplish Blue", "Blue Flower"],
        ["Neutral 6.5", "Yellow", "Moderate Red", "Foliage"],
        ["Neutral 5", "Red", "Purple", "Blue Sky"],
        ["Neutral 3.5", "Green", "Yellow Green", "Light Skin"],
        ["Black 2", "Blue", "Orange", "Dark Skin"],
    ];

    // Base reference coordinates measured at 3128 x 2082
    let base_width = 3128.0f64;
    let base_height = 2082.0f64;
    let scale_x = width as f64 / base_width;
    let scale_y = height as f64 / base_height;

    let base_patch_coords: [[(f64, f64); 4]; 6] = [
        [
            (2420.0, 460.0),
            (2550.0, 460.0),
            (2680.0, 460.0),
            (2805.0, 460.0),
        ],
        [
            (2420.0, 595.0),
            (2550.0, 595.0),
            (2680.0, 595.0),
            (2805.0, 595.0),
        ],
        [
            (2420.0, 725.0),
            (2550.0, 725.0),
            (2680.0, 725.0),
            (2805.0, 725.0),
        ],
        [
            (2420.0, 850.0),
            (2550.0, 850.0),
            (2680.0, 850.0),
            (2805.0, 850.0),
        ],
        [
            (2420.0, 975.0),
            (2550.0, 975.0),
            (2680.0, 975.0),
            (2805.0, 975.0),
        ],
        [
            (2335.0, 1130.0),
            (2550.0, 1100.0),
            (2680.0, 1100.0),
            (2805.0, 1100.0),
        ],
    ];

    let half_size = (sample_size as f64 * (scale_x + scale_y) / 4.0)
        .round()
        .max(2.0) as u32;

    let cols = 4usize;
    let rows = 6usize;
    let mut patches = Vec::with_capacity(24);

    let mut idx = 1usize;
    for r in 0..rows {
        for c in 0..cols {
            let (bx, by) = base_patch_coords[r][c];
            let cx = (bx * scale_x).round() as u32;
            let cy = (by * scale_y).round() as u32;

            let sample_x0 = cx.saturating_sub(half_size);
            let sample_x1 = (cx + half_size).min(width);
            let sample_y0 = cy.saturating_sub(half_size);
            let sample_y1 = (cy + half_size).min(height);

            let mut sum_r = 0.0f64;
            let mut sum_g = 0.0f64;
            let mut sum_b = 0.0f64;
            let mut count = 0usize;

            for y in sample_y0..sample_y1 {
                for x in sample_x0..sample_x1 {
                    let pixel = img.get_pixel(x, y);
                    let Rgba([pr, pg, pb, _]) = pixel;
                    sum_r += pr as f64;
                    sum_g += pg as f64;
                    sum_b += pb as f64;
                    count += 1;
                }
            }

            let n = count.max(1) as f64;
            let mean_r = sum_r / n;
            let mean_g = sum_g / n;
            let mean_b = sum_b / n;

            let rgb_u8 = [
                mean_r.round().clamp(0.0, 255.0) as u8,
                mean_g.round().clamp(0.0, 255.0) as u8,
                mean_b.round().clamp(0.0, 255.0) as u8,
            ];

            let rgb_norm = [
                ((mean_r / 255.0 * 10000.0).round() / 10000.0) as f32,
                ((mean_g / 255.0 * 10000.0).round() / 10000.0) as f32,
                ((mean_b / 255.0 * 10000.0).round() / 10000.0) as f32,
            ];

            patches.push(PatchData {
                index: idx,
                name: patch_grid_names[r][c].to_string(),
                row: r + 1,
                col: c + 1,
                rgb_u8,
                rgb_norm,
            });
            idx += 1;
        }
    }

    Ok(StockData {
        source_image: img_path.to_string_lossy().to_string(),
        image_width: width,
        image_height: height,
        patches,
    })
}

fn stock_key_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown_stock".to_string())
}

fn main() {
    let args = Args::parse();

    let mut images_to_process: Vec<PathBuf> = Vec::new();

    let scan_directory = args.all || args.filter.is_some() || args.input.is_dir();

    if scan_directory {
        let dir = if args.input.is_dir() {
            &args.input
        } else {
            Path::new("film-data/color-check/data")
        };

        let filter_terms: Vec<String> = args
            .filter
            .as_ref()
            .map(|f| {
                f.split(',')
                    .map(|s| s.trim().to_lowercase())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    let ext = ext.to_lowercase();
                    if ext == "png"
                        || ext == "jpg"
                        || ext == "jpeg"
                        || ext == "tif"
                        || ext == "tiff"
                    {
                        let fname = p
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_lowercase();
                        let matches = filter_terms.iter().all(|term| fname.contains(term));
                        if matches {
                            images_to_process.push(p);
                        }
                    }
                }
            }
        }
        images_to_process.sort();
    } else {
        images_to_process.push(args.input.clone());
    }

    if images_to_process.is_empty() {
        eprintln!("No images found to process at '{}'", args.input.display());
        std::process::exit(1);
    }

    // Load existing database if available, to merge/update
    let mut database: Database = if args.output.exists() {
        match fs::read_to_string(&args.output) {
            Ok(content) => toml::from_str(&content).unwrap_or_default(),
            Err(_) => Database::default(),
        }
    } else {
        Database::default()
    };

    println!(
        "Processing {} image(s) with sample_size={}x{}px...",
        images_to_process.len(),
        args.sample_size,
        args.sample_size
    );

    for img_path in &images_to_process {
        let stock_key = stock_key_from_path(img_path);
        match extract_patches(img_path, args.sample_size) {
            Ok(stock_data) => {
                if args.verbose {
                    println!(
                        "\n=== Stock: {} ({}x{}) ===",
                        stock_key, stock_data.image_width, stock_data.image_height
                    );
                    println!(
                        "{:<3} | {:<15} | {:<5} | {:<14} | {:<20}",
                        "#", "Name", "Grid", "RGB (Mean)", "RGB Norm (0..1)"
                    );
                    println!(
                        "{:-<3}-+-{:-<15}-+-{:-<5}-+-{:-<14}-+-{:-<20}",
                        "", "", "", "", ""
                    );
                    for p in &stock_data.patches {
                        println!(
                            "{:<3} | {:<15} | ({},{}) | [{:3}, {:3}, {:3}] | [{:.4}, {:.4}, {:.4}]",
                            p.index,
                            p.name,
                            p.row,
                            p.col,
                            p.rgb_u8[0],
                            p.rgb_u8[1],
                            p.rgb_u8[2],
                            p.rgb_norm[0],
                            p.rgb_norm[1],
                            p.rgb_norm[2],
                        );
                    }
                }
                database.stocks.insert(stock_key, stock_data);
            }
            Err(e) => {
                eprintln!("Error processing '{}': {}", img_path.display(), e);
            }
        }
    }

    // Ensure output directory exists
    if let Some(parent) = args.output.parent() {
        if !parent.exists() {
            let _ = fs::create_dir_all(parent);
        }
    }

    // Serialize to TOML
    match toml::to_string_pretty(&database) {
        Ok(toml_str) => {
            if let Err(e) = fs::write(&args.output, toml_str) {
                eprintln!("Failed to write TOML to '{}': {}", args.output.display(), e);
                std::process::exit(1);
            }
            println!(
                "\nSuccessfully saved database with {} stock(s) to '{}'",
                database.stocks.len(),
                args.output.display()
            );
        }
        Err(e) => {
            eprintln!("Failed to serialize database to TOML: {e}");
            std::process::exit(1);
        }
    }
}
