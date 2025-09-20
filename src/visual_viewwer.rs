use eframe::{egui, App, Frame, NativeOptions, run_native};
use egui::{CentralPanel, ColorImage, Context, Rect, TextureHandle, Vec2, pos2, Sense};
use image::{RgbImage, GenericImageView, DynamicImage};
use crate::imops::PipelineImage;
use crate::pixels::{ImageBuffer};
use crate::imops::PipelineModule;
use rawler::decode_file;

pub struct ViewerApp {
    before_image: RgbImage,
    after_image: RgbImage,
    before_texture: Option<TextureHandle>,
    after_texture: Option<TextureHandle>,
    zoom_level: f32,
    offset_pixels: Vec2,
    original_image_dims: Vec2,
    slider_position: f32, // 0.0 to 1.0
}

impl ViewerApp {
    pub fn new(before_image: RgbImage, after_image: RgbImage) -> Self {
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
        }
    }

    fn load_texture(&self, ctx: &Context, image: &RgbImage, name: &str) -> TextureHandle {
        let (width, height) = image.dimensions();
        let pixels = image.as_flat_samples();
        let color_image = ColorImage::from_rgb(
            [width as usize, height as usize],
            pixels.as_slice(),
        );
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
            let fit_zoom = if self.original_image_dims.x > 0.0 { widget_size.x / self.original_image_dims.x } else { 1.0 };

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
                            let point_in_texture = self.offset_pixels + image_local_pos / old_effective_zoom;
                            self.offset_pixels = point_in_texture - image_local_pos / new_effective_zoom;
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
            self.offset_pixels.x = self.offset_pixels.x.clamp(0.0, (self.original_image_dims.x - visible_texture_dims.x).max(0.0));
            self.offset_pixels.y = self.offset_pixels.y.clamp(0.0, (self.original_image_dims.y - visible_texture_dims.y).max(0.0));

            // If we are not zoomed in, offset should be zero.
            if self.zoom_level <= 1.0 {
                self.offset_pixels = Vec2::ZERO;
            }

            // --- UV calculations ---
            let uv_min = self.offset_pixels / self.original_image_dims;
            let uv_max = (self.offset_pixels + visible_texture_dims) / self.original_image_dims;
            let uv_rect = Rect::from_min_max(pos2(uv_min.x, uv_min.y), pos2(uv_max.x, uv_max.y));

            // --- Drawing ---
            if let (Some(before_texture), Some(after_texture)) = (&self.before_texture, &self.after_texture) {
                let painter = ui.painter_at(image_rect);

                // Draw before image
                painter.image(before_texture.id(), image_rect, uv_rect, egui::Color32::WHITE);

                // Draw after image (clipped)
                let slider_x = image_rect.min.x + image_rect.width() * self.slider_position;
                let clip_rect = Rect::from_min_max(
                    pos2(slider_x, image_rect.min.y),
                    image_rect.max,
                );
                
                let painter_clipped = painter.with_clip_rect(clip_rect);
                painter_clipped.image(after_texture.id(), image_rect, uv_rect, egui::Color32::WHITE);

                // Draw slider line
                painter.line_segment(
                    [pos2(slider_x, image_rect.min.y), pos2(slider_x, image_rect.max.y)],
                    egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 180)),
                );
                
                // --- Slider Interaction ---
                let slider_handle_rect = Rect::from_center_size(
                    pos2(slider_x, image_rect.center().y),
                    Vec2::new(12.0, image_rect.height()),
                );
                let slider_response = ui.interact(slider_handle_rect, response.id.with("slider"), Sense::drag());

                if slider_response.dragged() {
                    if let Some(pointer_pos) = ui.input(|i| i.pointer.hover_pos()) {
                        let new_slider_x = pointer_pos.x;
                        self.slider_position = ((new_slider_x - image_rect.min.x) / image_rect.width()).clamp(0.0, 1.0);
                    }
                }
                if slider_response.hovered() {
                    ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
                }
            }
        });
    }
}

pub fn to_pipeline_image(img: &DynamicImage) -> PipelineImage {
    let (width, height) = img.dimensions();
    let rgb_img = img.to_rgb32f();
    let mut data: ImageBuffer = Vec::with_capacity((width * height) as usize);

    for pixel in rgb_img.pixels() {
        data.push([pixel[0], pixel[1], pixel[2]]);
    }

    PipelineImage {
        data,
        width: width as usize,
        height: height as usize,
        max_raw_value: 1.0, // Assuming image data is normalized to [0, 1] for PNG/JPEG
    }
}

pub fn to_rgb_image(pipeline_image: &PipelineImage) -> RgbImage {
    let mut rgb_image = RgbImage::new(pipeline_image.width as u32, pipeline_image.height as u32);
    for (i, pixel_data) in pipeline_image.data.iter().enumerate() {
        let x = (i % pipeline_image.width) as u32;
        let y = (i / pipeline_image.width) as u32;
        // Clamp values to [0, 1] and convert to u8
        let r = (pixel_data[0].max(0.0).min(1.0) * 255.0) as u8;
        let g = (pixel_data[1].max(0.0).min(1.0) * 255.0) as u8;
        let b = (pixel_data[2].max(0.0).min(1.0) * 255.0) as u8;
        rgb_image.put_pixel(x, y,image::Rgb([r, g, b]));
    }
    rgb_image
}

pub fn run_viewer<F>(
    window_title: &'static str,
    setup_fn: F,
)
where
    F: FnOnce() -> (PipelineImage, PipelineImage) + 'static,
{
    let (pipeline_image_before, pipeline_image_after) = setup_fn();

    let before_rgb = to_rgb_image(&pipeline_image_before);
    let after_rgb = to_rgb_image(&pipeline_image_after);

    let native_options = NativeOptions::default();
    let _ = run_native(
        window_title,
        native_options,
        Box::new(|_cc| Box::new(ViewerApp::new(before_rgb, after_rgb))),
    );
}

pub fn run_module_viewer(
    window_title: &'static str,
    module: Box<dyn PipelineModule>,
    input_path: Option<&'static str>,
    raw_path: Option<&'static str>,
) {
    let input_image_path = input_path.unwrap_or("test_data/test1.tif");
    let raw_image_path = raw_path.unwrap_or("test_data/raw_sample.NEF");

    run_viewer(window_title, move || {
        let img = image::open(input_image_path).expect("Failed to open input image");
        let pipeline_image_before = to_pipeline_image(&img);

        let raw_image = decode_file(raw_image_path).expect("Failed to load raw sample image");

        let pipeline_image_after = module.process(pipeline_image_before.clone(), &raw_image);

        (pipeline_image_before, pipeline_image_after)
    });
}
