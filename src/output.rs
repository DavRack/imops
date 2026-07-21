//! Image export helpers (OpenEXR archival + JPEG preview).

use color::ColorSpaceTag;
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::Image;
use std::path::Path;

/// Output format selected by path extension (or explicit override).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// 16-bit half-float RGB, ZIP16 lossless OpenEXR.
    Exr,
    /// 8-bit display JPEG (kept for quick previews / switching).
    Jpeg,
}

impl OutputFormat {
    pub fn from_path(path: &str) -> Self {
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => Self::Jpeg,
            _ => Self::Exr,
        }
    }
}

/// Apply EXIF/DNG orientation to an interleaved RGB buffer (top-left origin).
pub fn apply_orientation(
    rgb: &[[f32; 3]],
    width: usize,
    height: usize,
    orientation: rawler::Orientation,
) -> (usize, usize, Vec<[f32; 3]>) {
    match orientation {
        rawler::Orientation::Rotate90 => {
            let (nw, nh) = (height, width);
            let mut out = vec![[0.0; 3]; nw * nh];
            for y in 0..height {
                for x in 0..width {
                    // (x,y) → (h-1-y, x)
                    out[x * nw + (height - 1 - y)] = rgb[y * width + x];
                }
            }
            (nw, nh, out)
        }
        rawler::Orientation::Rotate180 => {
            let mut out = vec![[0.0; 3]; width * height];
            for y in 0..height {
                for x in 0..width {
                    out[(height - 1 - y) * width + (width - 1 - x)] = rgb[y * width + x];
                }
            }
            (width, height, out)
        }
        rawler::Orientation::Rotate270 => {
            let (nw, nh) = (height, width);
            let mut out = vec![[0.0; 3]; nw * nh];
            for y in 0..height {
                for x in 0..width {
                    // (x,y) → (y, w-1-x)
                    out[(width - 1 - x) * nw + y] = rgb[y * width + x];
                }
            }
            (nw, nh, out)
        }
        _ => (width, height, rgb.to_vec()),
    }
}

fn color_space_name(cs: Option<ColorSpaceTag>) -> &'static str {
    match cs {
        Some(ColorSpaceTag::Srgb) => "sRGB",
        Some(ColorSpaceTag::LinearSrgb) => "Linear sRGB",
        Some(ColorSpaceTag::DisplayP3) => "Display P3",
        Some(ColorSpaceTag::AcesCg) => "ACEScg",
        Some(ColorSpaceTag::Aces2065_1) => "ACES2065-1",
        Some(ColorSpaceTag::Rec2020) => "Rec.2020",
        Some(ColorSpaceTag::XyzD65) => "XYZ D65",
        Some(ColorSpaceTag::XyzD50) => "XYZ D50",
        Some(ColorSpaceTag::Oklab) => "Oklab",
        Some(ColorSpaceTag::Oklch) => "Oklch",
        Some(_) => "Unknown",
        None => "Unspecified",
    }
}

/// CIE xy chromaticities for common RGB spaces (OpenEXR `chromaticities` attr).
fn chromaticities_for(cs: Option<ColorSpaceTag>) -> exr::meta::attribute::Chromaticities {
    use exr::prelude::Vec2;
    match cs {
        // ACES AP1 (ACEScg) + ACES white (~D60)
        Some(ColorSpaceTag::AcesCg) | Some(ColorSpaceTag::Aces2065_1) => {
            exr::meta::attribute::Chromaticities {
                red: Vec2(0.713, 0.293),
                green: Vec2(0.165, 0.830),
                blue: Vec2(0.128, 0.044),
                white: Vec2(0.32168, 0.33767),
            }
        }
        // Display P3
        Some(ColorSpaceTag::DisplayP3) => exr::meta::attribute::Chromaticities {
            red: Vec2(0.680, 0.320),
            green: Vec2(0.265, 0.690),
            blue: Vec2(0.150, 0.060),
            white: Vec2(0.3127, 0.3290),
        },
        // Rec.2020
        Some(ColorSpaceTag::Rec2020) => exr::meta::attribute::Chromaticities {
            red: Vec2(0.708, 0.292),
            green: Vec2(0.170, 0.797),
            blue: Vec2(0.131, 0.046),
            white: Vec2(0.3127, 0.3290),
        },
        // sRGB / linear sRGB / default Rec.709 primaries, D65
        _ => exr::meta::attribute::Chromaticities {
            red: Vec2(0.64, 0.33),
            green: Vec2(0.30, 0.60),
            blue: Vec2(0.15, 0.06),
            white: Vec2(0.3127, 0.3290),
        },
    }
}

/// Write 16-bit half-float RGB OpenEXR with ZIP16 lossless compression and metadata tags.
pub fn save_exr(
    path: &str,
    width: usize,
    height: usize,
    pixels: &[[f32; 3]],
    meta: &ImageMetadata,
) -> Result<(), String> {
    use exr::prelude::*;

    assert_eq!(width * height, pixels.len());

    let mut layer_attributes = LayerAttributes::named("rgb");
    layer_attributes.software_name = Some(Text::from("imops / pichromatic"));
    layer_attributes.owner = Some(Text::from("imops"));
    layer_attributes.comments = Some(Text::from(
        format!(
            "colorSpace={}; linear scene/display buffer as produced by the pixel pipeline",
            color_space_name(meta.color_space)
        )
        .as_str(),
    ));
    layer_attributes.exposure = meta.shutter_seconds;
    layer_attributes.aperture = meta.f_number;
    layer_attributes.iso_speed = meta.iso;

    if let Some(ref model) = meta.unique_camera_model {
        layer_attributes.other.insert(
            Text::from("cameraModel"),
            AttributeValue::Text(Text::from(model.as_str())),
        );
    }
    if let Some(ref serial) = meta.camera_serial_number {
        layer_attributes.other.insert(
            Text::from("cameraSerialNumber"),
            AttributeValue::Text(Text::from(serial.as_str())),
        );
    }
    if let Some(ev) = meta.baseline_exposure {
        layer_attributes.other.insert(
            Text::from("baselineExposure"),
            AttributeValue::F32(ev),
        );
    }
    layer_attributes.other.insert(
        Text::from("colorSpace"),
        AttributeValue::Text(Text::from(color_space_name(meta.color_space))),
    );
    // Explicit transfer / encoding hint for viewers (tev, Nuke, etc.).
    // `ColorSpaceTag::Srgb` from the `color` crate is *encoded* sRGB (OETF
    // applied by CST). `LinearSrgb` is scene/display-linear — do not claim OETF.
    let transfer = match meta.color_space {
        Some(ColorSpaceTag::Srgb) | Some(ColorSpaceTag::DisplayP3) => "sRGB-OETF / display-referred",
        Some(ColorSpaceTag::AcesCg)
        | Some(ColorSpaceTag::Aces2065_1)
        | Some(ColorSpaceTag::LinearSrgb)
        | Some(ColorSpaceTag::Rec2020)
        | Some(ColorSpaceTag::XyzD65)
        | Some(ColorSpaceTag::XyzD50) => "linear",
        _ => "unspecified",
    };
    layer_attributes.other.insert(
        Text::from("transferFunction"),
        AttributeValue::Text(Text::from(transfer)),
    );

    let pixels = pixels.to_vec();
    let layer = Layer::new(
        (width, height),
        layer_attributes,
        Encoding::SMALL_LOSSLESS, // ZIP16 — lossless
        SpecificChannels::rgb(move |Vec2(x, y)| {
            let p: [f32; 3] = pixels[y * width + x];
            (
                f16::from_f32(p[0]),
                f16::from_f32(p[1]),
                f16::from_f32(p[2]),
            )
        }),
    );

    let mut exr_image = Image::from_layer(layer);
    exr_image.attributes.pixel_aspect = 1.0;
    exr_image.attributes.chromaticities = Some(chromaticities_for(meta.color_space));
    exr_image.attributes.display_window = IntegerBounds::from_dimensions((width, height));

    exr_image
        .write()
        .to_file(path)
        .map_err(|e| format!("EXR write failed: {e}"))
}

/// Write 8-bit JPEG (display preview). Clamps to [0, 1] then quantizes.
pub fn save_jpeg(
    path: &str,
    width: usize,
    height: usize,
    pixels: &[[f32; 3]],
) -> Result<(), String> {
    assert_eq!(width * height, pixels.len());
    let data: Vec<u8> = pixels
        .iter()
        .flat_map(|p| {
            p.iter()
                .map(|&c| (c.max(0.0).min(1.0) * 255.0) as u8)
                .collect::<Vec<_>>()
        })
        .collect();
    let img = image::RgbImage::from_vec(width as u32, height as u32, data)
        .ok_or_else(|| "failed to build JPEG buffer".to_string())?;
    img.save(path)
        .map_err(|e| format!("JPEG write failed: {e}"))
}

/// Save pipeline output; format from path extension (`.exr` default, `.jpg`/`.jpeg` → JPEG).
pub fn save_image(
    path: &str,
    image: &Image,
    orientation: rawler::Orientation,
) -> Result<OutputFormat, String> {
    let format = OutputFormat::from_path(path);
    let (w, h, pixels) = apply_orientation(
        &image.rgb_data,
        image.metadata.width,
        image.metadata.height,
        orientation,
    );
    match format {
        OutputFormat::Exr => {
            save_exr(path, w, h, &pixels, &image.metadata)?;
        }
        OutputFormat::Jpeg => {
            save_jpeg(path, w, h, &pixels)?;
        }
    }
    Ok(format)
}
