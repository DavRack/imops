use pichromatic::exp;
use pichromatic::gpu::GpuContext;
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::Image;
use pichromatic_pipeline::backend::Backend;
use pichromatic_pipeline::modules::{
    BaselineExposureCompensation, CFACoeffs, Contrast, Demosaic, DemosaicAlgorithmType, Exp, Film,
    HighlightReconstruction, LumaGuidedChromaDenoise, Module, Parameter, PipelineModule,
    SigmoidToneMap, Vignette, CST,
};
use pichromatic_pipeline::pipeline::run_pixel_pipeline_with_backend;

use std::time::Instant;
use std::vec::Vec;

use eframe::{egui, run_native, App, Frame, NativeOptions};
use egui::{pos2, CentralPanel, ColorImage, Context, Rect, Sense, TextureHandle, Vec2};
use image::{DynamicImage, GenericImageView, RgbImage};

fn main() {
    let raw_image_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test_data/vale.dng".to_string());

    // 1. Load the raw image and its bytes
    let file_bytes = std::fs::read(&raw_image_path).expect("Failed to read raw image file");
    let decode_params = rawler::decoders::RawDecodeParams::default();
    let mut raw_file = rawler::rawsource::RawSource::new_from_slice(&file_bytes);
    let raw_image =
        rawler::decode(&mut raw_file, &decode_params).expect("Failed to decode raw image");

    let file_bytes_clone = file_bytes.clone();

    let pipeline1_label = "no halation";
    let pipeline2_label = "halation";

    run_viewer(
        "Pipeline comparison (CPU vs GPU)",
        pipeline1_label,
        pipeline2_label,
        move || {
            let gpu_context = GpuContext::try_new_sync().expect("Failed to initialize GPU context");

            // --- Pipeline 1: CPU ---
            let pipeline1: Vec<Box<dyn PipelineModule>> = vec![
                Box::new(Module {
                    name: "Demosaic".to_string(),
                    cache: None,
                    config: Demosaic {
                        algorithm: Parameter::new(DemosaicAlgorithmType::Markesteijn, ""),
                    },
                }),
                Box::new(Module {
                    name: "Vignette".to_string(),
                    cache: None,
                    config: Vignette {
                        strength: Parameter::new(1.0, ""),
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
                    name: "LumaGuidedChromaDenoise".to_string(),
                    cache: None,
                    config: LumaGuidedChromaDenoise {
                        radius: Parameter::new(2, ""),
                        epsilon: Parameter::new(0.01, ""),
                    },
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
                        ev: Parameter::new(0.0, ""),
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
                    name: "film".to_string(),
                    cache: None,
                    config: Film {
                        stock: Parameter::new("CineStill50D".to_string(), ""),
                        film_format: Parameter::new("FilmSuper8".to_string(), ""),
                        render_width_mm: Parameter::new(None, ""),
                        seed: Parameter::new(1, ""),
                        enable_halation: Parameter::new(false, ""),
                        output: Parameter::new("PositiveLinear".to_string(), ""),
                        compensate_box_speed: Parameter::new(true, ""),
                        scanner_s_curve: Parameter::new(0.0, ""),
                    },
                }),
                Box::new(Module {
                    name: "Exp".to_string(),
                    cache: None,
                    config: Exp {
                        ev: Parameter::new(0.0, ""),
                    },
                }),
                // Box::new(Module {
                //     name: "contrast".to_string(),
                //     cache: None,
                //     config: Contrast {c: Parameter::new(1.0, "") },
                // }),
                Box::new(Module {
                    name: "SigmoidToneMap".to_string(),
                    cache: None,
                    config: SigmoidToneMap {},
                }),
                Box::new(Module {
                    name: "CST".to_string(),
                    cache: None,
                    config: CST {
                        target_color_space: Parameter::new("Srgb".to_string(), ""),
                    },
                }),
            ];

            // --- Pipeline 2: GPU ---
            let pipeline2: Vec<Box<dyn PipelineModule>> = vec![
                Box::new(Module {
                    name: "Demosaic".to_string(),
                    cache: None,
                    config: Demosaic {
                        algorithm: Parameter::new(DemosaicAlgorithmType::Markesteijn, ""),
                    },
                }),
                Box::new(Module {
                    name: "Vignette".to_string(),
                    cache: None,
                    config: Vignette {
                        strength: Parameter::new(1.0, ""),
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
                    name: "LumaGuidedChromaDenoise".to_string(),
                    cache: None,
                    config: LumaGuidedChromaDenoise {
                        radius: Parameter::new(2, ""),
                        epsilon: Parameter::new(0.01, ""),
                    },
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
                        ev: Parameter::new(0.0, ""),
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
                    name: "film".to_string(),
                    cache: None,
                    config: Film {
                        stock: Parameter::new("CineStill50D".to_string(), ""),
                        film_format: Parameter::new("Film35mm".to_string(), ""),
                        render_width_mm: Parameter::new(None, ""),
                        seed: Parameter::new(1, ""),
                        enable_halation: Parameter::new(true, ""),
                        output: Parameter::new("PositiveLinear".to_string(), ""),
                        compensate_box_speed: Parameter::new(true, ""),
                        scanner_s_curve: Parameter::new(0.0, ""),
                    },
                }),
                Box::new(Module {
                    name: "Exp".to_string(),
                    cache: None,
                    config: Exp {
                        ev: Parameter::new(0.0, ""),
                    },
                }),
                // Box::new(Module {
                //     name: "contrast".to_string(),
                //     cache: None,
                //     config: Contrast {c: Parameter::new(1.0, "") },
                // }),
                Box::new(Module {
                    name: "SigmoidToneMap".to_string(),
                    cache: None,
                    config: SigmoidToneMap {},
                }),
                Box::new(Module {
                    name: "CST".to_string(),
                    cache: None,
                    config: CST {
                        target_color_space: Parameter::new("Srgb".to_string(), ""),
                    },
                }),
            ];

            println!("Processing pipeline 1 (CPU)...");
            let now = Instant::now();
            let mut image1 = get_image_from_raw(raw_image.clone(), &file_bytes_clone);
            let mut config1 = pichromatic_pipeline::config::PipelineConfig {
                pipeline_modules: pipeline1,
            };
            run_pixel_pipeline_with_backend(&mut image1, &mut config1, &Backend::Cpu);
            println!("Pipeline 1 execution time: {}ms", now.elapsed().as_millis());

            println!("Processing pipeline 2 (GPU)...");
            let now = Instant::now();
            let mut image2 = get_image_from_raw(raw_image.clone(), &file_bytes_clone);
            let mut config2 = pichromatic_pipeline::config::PipelineConfig {
                pipeline_modules: pipeline2,
            };
            run_pixel_pipeline_with_backend(&mut image2, &mut config2, &Backend::Cpu);
            println!("Pipeline 2 execution time: {}ms", now.elapsed().as_millis());

            (image1, image2)
        },
    );

    println!("Visual tests finished.");
}

fn get_image_from_raw(raw_image: rawler::RawImage, file_bytes: &[u8]) -> Image {
    let mut image = pichromatic_pipeline::extern_pipeline::parse_raw_image(raw_image);
    if let Some(parser) = pichromatic_pipeline::dng_metadata::DngMetadataParser::new(file_bytes) {
        let dng_meta = parser.parse();
        pichromatic_pipeline::extern_pipeline::consolidate_dng_metadata(&mut image, &dng_meta);
    }
    image
}

pub struct ViewerApp {
    before_image: RgbImage,
    after_image: RgbImage,
    before_texture: Option<TextureHandle>,
    after_texture: Option<TextureHandle>,
    zoom_level: f32,
    offset_pixels: Vec2,
    original_image_dims: Vec2,
    slider_position: f32, // 0.0 to 1.0
    label1: String,
    label2: String,
}

impl ViewerApp {
    pub fn new(before_image: RgbImage, after_image: RgbImage, label1: &str, label2: &str) -> Self {
        let (width, height) = before_image.dimensions();
        Self {
            before_image,
            after_image,
            before_texture: None,
            after_texture: None,
            zoom_level: 1.0,
            offset_pixels: Vec2::ZERO,
            original_image_dims: Vec2::new(width as f32, height as f32),
            slider_position: 0.5,
            label1: label1.to_string(),
            label2: label2.to_string(),
        }
    }

    fn load_texture(&self, ctx: &Context, image: &RgbImage, name: &str) -> TextureHandle {
        let (width, height) = image.dimensions();
        let pixels = image.as_flat_samples();
        let color_image =
            ColorImage::from_rgb([width as usize, height as usize], pixels.as_slice());
        ctx.load_texture(name, color_image, Default::default())
    }
}

impl App for ViewerApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        if self.before_texture.is_none() {
            self.before_texture = Some(self.load_texture(ctx, &self.before_image, "before_image"));
        }
        if self.after_texture.is_none() {
            self.after_texture = Some(self.load_texture(ctx, &self.after_image, "after_image"));
        }

        CentralPanel::default().show(ctx, |ui| {
            // --- Sizing ---
            let available_size = ui.available_size();
            let image_aspect_ratio = self.original_image_dims.x / self.original_image_dims.y;
            let container_aspect_ratio = available_size.x / available_size.y;
            let widget_size = if image_aspect_ratio > container_aspect_ratio {
                Vec2::new(available_size.x, available_size.x / image_aspect_ratio)
            } else {
                Vec2::new(available_size.y * image_aspect_ratio, available_size.y)
            };

            let (rect, response) = ui.allocate_at_least(widget_size, Sense::drag());
            let image_rect = Rect::from_center_size(rect.center(), widget_size);

            // --- Zoom and Pan calculations ---
            let fit_zoom = if self.original_image_dims.x > 0.0 {
                widget_size.x / self.original_image_dims.x
            } else {
                1.0
            };

            // Handle pan and zoom on the whole image area
            if response.hovered() {
                // Zoom
                if ui.input(|i| i.raw_scroll_delta.y != 0.0) {
                    let old_zoom_level = self.zoom_level;
                    let zoom_factor = (ui.input(|i| i.raw_scroll_delta.y) / 100.0).exp();
                    self.zoom_level *= zoom_factor;
                    self.zoom_level = self.zoom_level.clamp(1.0, 100.0); // Min zoom is 1.0 (fit-to-screen)

                    if let Some(pointer_pos) = response.hover_pos() {
                        let old_effective_zoom = old_zoom_level * fit_zoom;
                        let new_effective_zoom = self.zoom_level * fit_zoom;

                        if old_effective_zoom > 0.0 && new_effective_zoom > 0.0 {
                            let image_local_pos = pointer_pos - image_rect.min;
                            let point_in_texture =
                                self.offset_pixels + image_local_pos / old_effective_zoom;
                            self.offset_pixels =
                                point_in_texture - image_local_pos / new_effective_zoom;
                        }
                    }
                }
                // Pan
                if response.dragged() {
                    let effective_zoom = self.zoom_level * fit_zoom;
                    if effective_zoom > 0.0 {
                        self.offset_pixels -= response.drag_delta() / effective_zoom;
                    }
                }
            }

            // --- Clamp offset ---
            let visible_texture_dims = self.original_image_dims / self.zoom_level;
            self.offset_pixels.x = self.offset_pixels.x.clamp(
                0.0,
                (self.original_image_dims.x - visible_texture_dims.x).max(0.0),
            );
            self.offset_pixels.y = self.offset_pixels.y.clamp(
                0.0,
                (self.original_image_dims.y - visible_texture_dims.y).max(0.0),
            );

            // If we are not zoomed in, offset should be zero.
            if self.zoom_level <= 1.0 {
                self.offset_pixels = Vec2::ZERO;
            }

            // --- UV calculations ---
            let uv_min = self.offset_pixels / self.original_image_dims;
            let uv_max = (self.offset_pixels + visible_texture_dims) / self.original_image_dims;
            let uv_rect = Rect::from_min_max(pos2(uv_min.x, uv_min.y), pos2(uv_max.x, uv_max.y));

            // --- Drawing ---
            if let (Some(before_texture), Some(after_texture)) =
                (&self.before_texture, &self.after_texture)
            {
                let painter = ui.painter_at(image_rect);

                // Draw before image
                painter.image(
                    before_texture.id(),
                    image_rect,
                    uv_rect,
                    egui::Color32::WHITE,
                );

                // Draw after image (clipped)
                let slider_x = image_rect.min.x + image_rect.width() * self.slider_position;
                let clip_rect =
                    Rect::from_min_max(pos2(slider_x, image_rect.min.y), image_rect.max);

                let painter_clipped = painter.with_clip_rect(clip_rect);
                painter_clipped.image(
                    after_texture.id(),
                    image_rect,
                    uv_rect,
                    egui::Color32::WHITE,
                );

                // --- Labels with white outline ---
                let text_y = image_rect.min.y + 8.0;
                let font_id = egui::FontId::proportional(16.0);
                let outline = egui::Color32::WHITE;
                let fill = egui::Color32::from_black_alpha(200);
                for (label, align, bx) in [
                    (&self.label1, egui::Align2::LEFT_TOP, image_rect.min.x + 8.0),
                    (
                        &self.label2,
                        egui::Align2::RIGHT_TOP,
                        image_rect.max.x - 8.0,
                    ),
                ] {
                    for &(dx, dy) in &[(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                        painter.text(
                            pos2(bx + dx, text_y + dy),
                            align,
                            label,
                            font_id.clone(),
                            outline,
                        );
                    }
                    painter.text(pos2(bx, text_y), align, label, font_id.clone(), fill);
                }

                // Draw slider line
                painter.line_segment(
                    [
                        pos2(slider_x, image_rect.min.y),
                        pos2(slider_x, image_rect.max.y),
                    ],
                    egui::Stroke::new(
                        2.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180),
                    ),
                );

                // --- Slider Interaction ---
                let slider_handle_rect = Rect::from_center_size(
                    pos2(slider_x, image_rect.center().y),
                    Vec2::new(12.0, image_rect.height()),
                );
                let slider_response = ui.interact(
                    slider_handle_rect,
                    response.id.with("slider"),
                    Sense::drag(),
                );

                if slider_response.dragged() {
                    if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
                        let new_slider_x = pointer_pos.x;
                        self.slider_position = ((new_slider_x - image_rect.min.x)
                            / image_rect.width())
                        .clamp(0.0, 1.0);
                    }
                }

                if slider_response.hovered() {
                    ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
            }
        });
    }
}

pub fn to_pipeline_image(img: &DynamicImage) -> Image {
    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb32f();
    let mut rgb_data = Vec::with_capacity((width * height) as usize);

    for pixel in rgb_img.pixels() {
        rgb_data.push([pixel[0], pixel[1], pixel[2]]);
    }

    Image {
        raw_data: std::sync::Arc::from([]),
        rgb_data,
        metadata: ImageMetadata {
            width: width as usize,
            height: height as usize,
            ..Default::default()
        },
    }
}

pub fn to_rgb_image(image: &Image) -> RgbImage {
    let mut rgb_image = RgbImage::new(image.metadata.width as u32, image.metadata.height as u32);
    for (i, pixel_data) in image.rgb_data.iter().enumerate() {
        let x = (i % image.metadata.width) as u32;
        let y = (i / image.metadata.width) as u32;
        // Clamp values to [0, 1] and convert to u8
        let r = (pixel_data[0].max(0.0).min(1.0) * 255.0) as u8;
        let g = (pixel_data[1].max(0.0).min(1.0) * 255.0) as u8;
        let b = (pixel_data[2].max(0.0).min(1.0) * 255.0) as u8;
        rgb_image.put_pixel(x, y, image::Rgb([r, g, b]));
    }
    rgb_image
}

pub fn run_viewer<F>(window_title: &'static str, label1: &str, label2: &str, setup_fn: F)
where
    F: FnOnce() -> (Image, Image) + 'static,
{
    let (pipeline_image_before, pipeline_image_after) = setup_fn();

    let before_rgb = to_rgb_image(&pipeline_image_before);
    let after_rgb = to_rgb_image(&pipeline_image_after);

    let l1 = label1.to_string();
    let l2 = label2.to_string();
    let native_options = NativeOptions::default();
    let _ = run_native(
        window_title,
        native_options,
        Box::new(move |_cc| Box::new(ViewerApp::new(before_rgb, after_rgb, &l1, &l2))),
    );
}

pub fn run_module_viewer(
    window_title: &'static str,
    label1: &str,
    module: Box<dyn PipelineModule>,
    input_path: Option<&'static str>,
    raw_path: Option<&'static str>,
) {
    let input_image_path = input_path.unwrap_or("test_data/test1.tif");
    let _raw_image_path = raw_path.unwrap_or("test_data/raw_sample.NEF");

    run_viewer(window_title, label1, label1, move || {
        let img = image::open(input_image_path).expect("Failed to open input image");
        let pipeline_image_before = to_pipeline_image(&img);

        let pipeline_image_after = pipeline_image_before.clone();
        let mut p_image = pichromatic_pipeline::backend::PipelineImage::Cpu(pipeline_image_after);
        module.process(&pichromatic_pipeline::backend::Backend::Cpu, &mut p_image);
        let pipeline_image_after = p_image.to_cpu(None);

        (pipeline_image_before, pipeline_image_after)
    });
}
