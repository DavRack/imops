use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use crate::pixel::{Image, Pixel};

#[cfg(target_arch = "wasm32")]
const PRESENT_BLIT_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

// img_w/img_h = source buffer; surf_w/surf_h = canvas (fixed full-res extent)
struct Dims {
    img_w: u32,
    img_h: u32,
    surf_w: u32,
    surf_h: u32,
}

@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> dims: Dims;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Scale-to-fit (contain) into the locked surface so preview/full share the same framing.
    let scale = min(
        f32(dims.surf_w) / f32(dims.img_w),
        f32(dims.surf_h) / f32(dims.img_h)
    );
    let draw_w = f32(dims.img_w) * scale;
    let draw_h = f32(dims.img_h) * scale;
    let off_x = (f32(dims.surf_w) - draw_w) * 0.5;
    let off_y = (f32(dims.surf_h) - draw_h) * 0.5;
    let fx = (in.position.x - off_x) / scale;
    let fy = (in.position.y - off_y) / scale;
    if (fx < 0.0 || fy < 0.0 || fx >= f32(dims.img_w) || fy >= f32(dims.img_h)) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let x = u32(fx);
    let y = u32(fy);
    let i = y * dims.img_w + x;
    let c = clamp(pixels[i].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(c, 1.0);
}
"#;

/// Cached compute pipeline keyed by WGSL identity + bind layout shape.
struct CachedPipeline {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
}

#[derive(Clone, Eq, PartialEq)]
struct PipelineKey {
    label: String,
    source_hash: u64,
    storage_count: u32,
    has_uniform: bool,
}

impl Hash for PipelineKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.source_hash.hash(state);
        self.storage_count.hash(state);
        self.has_uniform.hash(state);
    }
}

fn hash_source(src: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    h.finish()
}

#[cfg(target_arch = "wasm32")]
struct PresentState {
    canvas: web_sys::OffscreenCanvas,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    /// When true, canvas extent stays fixed (full-res) across preview/final presents.
    extent_locked: bool,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    #[allow(dead_code)] // used by wasm canvas present
    instance: wgpu::Instance,
    #[allow(dead_code)] // used by wasm canvas present
    adapter: wgpu::Adapter,
    pipeline_cache: Mutex<HashMap<PipelineKey, CachedPipeline>>,
    /// Reused RGBA working buffers (preview + final sizes). Cap 2 to avoid thrash.
    rgba_pool: Mutex<Vec<GpuImageBuffer>>,
    /// Reused demosaic mosaic upload buffer (avoid create_buffer_init every run).
    demosaic_raw_pool: Mutex<Option<(usize, wgpu::Buffer)>>,
    #[cfg(target_arch = "wasm32")]
    present: Mutex<Option<PresentState>>,
}

unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

pub struct GpuImageBuffer {
    pub buffer: wgpu::Buffer,
    pub width: usize,
    pub height: usize,
}

/// One compute dispatch to encode into a shared command encoder (no submit/wait).
pub struct ComputePassDesc<'a> {
    pub label: &'a str,
    pub wgsl_source: &'a str,
    pub storage_buffers: &'a [&'a wgpu::Buffer],
    pub uniform_bytes: &'a [u8],
    pub workgroups_x: u32,
    pub workgroups_y: u32,
}

static GLOBAL_GPU_CONTEXT: std::sync::OnceLock<Arc<GpuContext>> = std::sync::OnceLock::new();

impl GpuContext {
    pub async fn global() -> Option<Arc<Self>> {
        if let Some(ctx) = GLOBAL_GPU_CONTEXT.get() {
            return Some(ctx.clone());
        }
        match Self::try_new().await {
            Ok(ctx) => {
                let _ = GLOBAL_GPU_CONTEXT.set(ctx.clone());
                Some(ctx)
            }
            Err(_) => None,
        }
    }

    pub fn global_cached() -> Option<Arc<Self>> {
        GLOBAL_GPU_CONTEXT.get().cloned()
    }

    pub fn new_sync() -> Arc<Self> {
        if let Some(ctx) = GLOBAL_GPU_CONTEXT.get() {
            return ctx.clone();
        }
        let ctx = pollster::block_on(Self::new());
        let _ = GLOBAL_GPU_CONTEXT.set(ctx.clone());
        ctx
    }

    pub async fn try_new() -> Result<Arc<Self>, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| "Failed to find a suitable GPU adapter".to_string())?;

        let limits = adapter.limits();
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("[Pichromatic GPU] WebGPU Adapter limits: max_buffer_size={}, max_storage_binding={}", limits.max_buffer_size, limits.max_storage_buffer_binding_size).into());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Pichromatic GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                        max_buffer_size: limits.max_buffer_size,
                        max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage.max(8),
                        ..wgpu::Limits::downlevel_defaults()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create wgpu device: {e}"))?;

        Ok(Arc::new(Self {
            device,
            queue,
            instance,
            adapter,
            pipeline_cache: Mutex::new(HashMap::new()),
            rgba_pool: Mutex::new(Vec::new()),
            demosaic_raw_pool: Mutex::new(None),
            #[cfg(target_arch = "wasm32")]
            present: Mutex::new(None),
        }))
    }

    pub async fn new() -> Arc<Self> {
        Self::try_new().await.expect("Failed to initialize GPU context")
    }

    /// Block until queued GPU work completes. Use at readback / true sync points only.
    pub fn poll_wait(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Attach an OffscreenCanvas surface for GPU present (no CPU readback).
    #[cfg(target_arch = "wasm32")]
    pub fn configure_canvas(&self, canvas: web_sys::OffscreenCanvas) -> Result<(), String> {
        let surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas.clone()))
            .map_err(|e| format!("Failed to create canvas surface: {e}"))?;

        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Rgba8Unorm
                        | wgpu::TextureFormat::Bgra8Unorm
                        | wgpu::TextureFormat::Rgba8UnormSrgb
                        | wgpu::TextureFormat::Bgra8UnormSrgb
                )
            })
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| "No surface formats available".to_string())?;

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Opaque),
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.device, &config);

        let blit_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Present Blit Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Present Blit Shader"),
                source: wgpu::ShaderSource::Wgsl(PRESENT_BLIT_WGSL.into()),
            });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Present Blit Pipeline Layout"),
                bind_group_layouts: &[&blit_bind_group_layout],
                push_constant_ranges: &[],
            });

        let blit_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Present Blit Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        *self.present.lock().unwrap() = Some(PresentState {
            canvas,
            surface,
            config,
            blit_pipeline,
            blit_bind_group_layout,
            extent_locked: false,
        });

        web_sys::console::log_1(
            &format!("[Pichromatic GPU] Canvas surface configured ({format:?})").into(),
        );
        Ok(())
    }

    /// Lock the present canvas to a fixed pixel extent (full-res). Preview/final then
    /// share the same surface framing via scale-to-fit, avoiding 1–2px jumps.
    #[cfg(target_arch = "wasm32")]
    pub fn set_present_extent(&self, width: u32, height: u32) -> Result<(), String> {
        let mut present_guard = self.present.lock().unwrap();
        let present = present_guard
            .as_mut()
            .ok_or_else(|| "Canvas surface not configured".to_string())?;

        let width = width.max(1);
        let height = height.max(1);
        present.canvas.set_width(width);
        present.canvas.set_height(height);
        present.config.width = present.canvas.width().max(1);
        present.config.height = present.canvas.height().max(1);
        present
            .surface
            .configure(&self.device, &present.config);
        present.extent_locked = true;

        web_sys::console::log_1(
            &format!(
                "[Pichromatic GPU] Present extent locked to {}x{}",
                present.config.width, present.config.height
            )
            .into(),
        );
        Ok(())
    }

    /// Present RGBA f32 storage buffer to the configured canvas. Clamps RGB to [0, 1].
    #[cfg(target_arch = "wasm32")]
    pub fn present_image(&self, gpu_buf: &GpuImageBuffer) -> Result<(), String> {
        let mut present_guard = self.present.lock().unwrap();
        let present = present_guard
            .as_mut()
            .ok_or_else(|| "Canvas surface not configured".to_string())?;

        let img_w = (gpu_buf.width as u32).max(1);
        let img_h = (gpu_buf.height as u32).max(1);

        if !present.extent_locked {
            // Fallback before decode locks full-res extent: size to this frame once.
            present.canvas.set_width(img_w);
            present.canvas.set_height(img_h);
            present.config.width = present.canvas.width().max(1);
            present.config.height = present.canvas.height().max(1);
            present
                .surface
                .configure(&self.device, &present.config);
        }

        let surf_w = present.config.width;
        let surf_h = present.config.height;

        let frame = present
            .surface
            .get_current_texture()
            .map_err(|e| format!("Failed to acquire surface texture: {e}"))?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let dims = [img_w, img_h, surf_w, surf_h];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Present Dims Uniform"),
            contents: bytemuck::cast_slice(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Present Blit Bind Group"),
            layout: &present.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Present Encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Present Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&present.blit_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    /// Upload CPU `Image` to a GPU Storage Buffer (layout: RGBA f32 per pixel)
    pub fn upload_image(&self, image: &Image) -> GpuImageBuffer {
        let width = image.metadata.width.max(1);
        let height = image.metadata.height.max(1);

        let mut rgba_data: Vec<f32> = Vec::with_capacity(width * height * 4);
        for pixel in &image.rgb_data {
            rgba_data.push(pixel[0]);
            rgba_data.push(pixel[1]);
            rgba_data.push(pixel[2]);
            rgba_data.push(1.0); // Alpha pad for vec4<f32> alignment
        }

        let target_len = width * height * 4;
        if rgba_data.len() < target_len {
            rgba_data.resize(target_len, 0.0);
        }

        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pichromatic Image Input Buffer"),
            contents: bytemuck::cast_slice(&rgba_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        GpuImageBuffer {
            buffer,
            width,
            height,
        }
    }

    /// Update an existing GPU Storage Buffer with contents of a CPU `Image`
    pub fn update_buffer_from_image(&self, gpu_buf: &GpuImageBuffer, image: &Image) {
        let width = image.metadata.width.max(1);
        let height = image.metadata.height.max(1);

        let mut rgba_data: Vec<f32> = Vec::with_capacity(width * height * 4);
        for pixel in &image.rgb_data {
            rgba_data.push(pixel[0]);
            rgba_data.push(pixel[1]);
            rgba_data.push(pixel[2]);
            rgba_data.push(1.0);
        }

        let target_len = width * height * 4;
        if rgba_data.len() < target_len {
            rgba_data.resize(target_len, 0.0);
        }

        self.queue.write_buffer(&gpu_buf.buffer, 0, bytemuck::cast_slice(&rgba_data));
    }

    /// Download a GPU Storage Buffer (layout: RGBA f32 per pixel) back to a CPU `Image`
    pub fn download_image(&self, gpu_buf: &GpuImageBuffer, original_metadata: &crate::image::ImageMetadata) -> Image {
        let num_pixels = gpu_buf.width * gpu_buf.height;
        let buffer_size = (num_pixels * 4 * std::mem::size_of::<f32>()) as u64;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pichromatic Staging Download Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, &staging_buffer, 0, buffer_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("Failed to map staging buffer for read");

        let data = buffer_slice.get_mapped_range();
        let float_slice: &[f32] = bytemuck::cast_slice(&data);

        let mut rgb_data: Vec<Pixel> = Vec::with_capacity(num_pixels);
        for i in 0..num_pixels {
            let r = float_slice[i * 4];
            let g = float_slice[i * 4 + 1];
            let b = float_slice[i * 4 + 2];
            rgb_data.push([r, g, b]);
        }

        drop(data);
        staging_buffer.unmap();

        let mut meta = original_metadata.clone();
        meta.width = gpu_buf.width;
        meta.height = gpu_buf.height;

        Image {
            rgb_data,
            raw_data: std::sync::Arc::from([]),
            metadata: meta,
        }
    }

    pub async fn download_image_async(&self, gpu_buf: &GpuImageBuffer, original_metadata: &crate::image::ImageMetadata) -> Image {
        let num_pixels = gpu_buf.width * gpu_buf.height;
        let buffer_size = (num_pixels * 4 * std::mem::size_of::<f32>()) as u64;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pichromatic Staging Download Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Encoder"),
        });

        encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, &staging_buffer, 0, buffer_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        #[cfg(target_arch = "wasm32")]
        self.device.poll(wgpu::Maintain::Poll);

        receiver.await.unwrap().expect("Failed to map staging buffer for read");

        let data = buffer_slice.get_mapped_range();
        let float_slice: &[f32] = bytemuck::cast_slice(&data);

        let mut rgb_data: Vec<Pixel> = Vec::with_capacity(num_pixels);
        for i in 0..num_pixels {
            let r = float_slice[i * 4];
            let g = float_slice[i * 4 + 1];
            let b = float_slice[i * 4 + 2];
            rgb_data.push([r, g, b]);
        }

        drop(data);
        staging_buffer.unmap();

        let mut meta = original_metadata.clone();
        meta.width = gpu_buf.width;
        meta.height = gpu_buf.height;

        Image {
            metadata: meta,
            raw_data: std::sync::Arc::from([]),
            rgb_data,
        }
    }

    /// Allocate an uninitialized GPU Storage Buffer for output
    pub fn create_output_buffer(&self, width: usize, height: usize) -> GpuImageBuffer {
        let num_pixels = width * height;
        let buffer_size = (num_pixels * 4 * std::mem::size_of::<f32>()) as u64;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pichromatic Image Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GpuImageBuffer {
            buffer,
            width,
            height,
        }
    }

    /// Acquire a reusable RGBA working buffer for `(width, height)`.
    /// Keeps up to two sizes (preview + final) to avoid multi‑GB realloc thrash.
    pub fn acquire_rgba_buffer(&self, width: usize, height: usize) -> GpuImageBuffer {
        let width = width.max(1);
        let height = height.max(1);
        let mut pool = self.rgba_pool.lock().unwrap();
        if let Some(i) = pool
            .iter()
            .position(|b| b.width == width && b.height == height)
        {
            return pool.remove(i);
        }
        self.create_output_buffer(width, height)
    }

    /// Return an RGBA buffer to the pool for the next pipeline run.
    pub fn recycle_rgba_buffer(&self, buf: GpuImageBuffer) {
        let mut pool = self.rgba_pool.lock().unwrap();
        pool.retain(|b| !(b.width == buf.width && b.height == buf.height));
        pool.push(buf);
        while pool.len() > 2 {
            pool.remove(0);
        }
    }

    /// Acquire/reuse an f32 storage buffer and upload `data` via `write_buffer`
    /// (avoids `create_buffer_init` + WASM heap growth every demosaic).
    pub fn acquire_f32_storage_write(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        let count = data.len().max(1);
        let bytes = bytemuck::cast_slice(data);
        let mut pool = self.demosaic_raw_pool.lock().unwrap();
        if let Some((cap, buf)) = pool.take() {
            if cap >= count {
                self.queue.write_buffer(&buf, 0, bytes);
                return buf;
            }
            // Too small — drop and recreate.
        }
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (count * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    pub fn recycle_f32_storage(&self, buf: wgpu::Buffer, count: usize) {
        *self.demosaic_raw_pool.lock().unwrap() = Some((count.max(1), buf));
    }

    /// Allocate an f32 storage buffer (for intermediate plane / scratch data).
    pub fn create_f32_buffer(&self, count: usize, label: &str) -> wgpu::Buffer {
        let size = (count.max(1) * std::mem::size_of::<f32>()) as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Allocate an f32 storage buffer initialized from CPU data (baked constants).
    pub fn create_f32_buffer_init(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        let src: &[f32] = if data.is_empty() { &[0.0f32] } else { data };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(src),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Download a plain `f32` storage buffer (first `count` elements) back to CPU.
    ///
    /// True sync point: submits + `Maintain::Wait` for map.
    pub fn download_f32(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let buffer_size = (count.max(1) * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pichromatic f32 Download Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("f32 Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("Failed to map f32 staging buffer for read");

        let data = buffer_slice.get_mapped_range();
        let float_slice: &[f32] = bytemuck::cast_slice(&data);
        let out = float_slice[..count.min(float_slice.len())].to_vec();
        drop(data);
        staging_buffer.unmap();
        out
    }

    pub async fn download_f32_async(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let buffer_size = (count.max(1) * std::mem::size_of::<f32>()) as u64;
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pichromatic f32 Download Staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("f32 Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);

        #[cfg(target_arch = "wasm32")]
        self.device.poll(wgpu::Maintain::Poll);

        receiver.await.unwrap().expect("Failed to map f32 staging buffer for read");

        let data = buffer_slice.get_mapped_range();
        let float_slice: &[f32] = bytemuck::cast_slice(&data);
        let out = float_slice[..count.min(float_slice.len())].to_vec();
        drop(data);
        staging_buffer.unmap();
        out
    }

    fn get_or_create_pipeline(
        &self,
        label: &str,
        wgsl_source: &str,
        storage_count: usize,
        has_uniform: bool,
    ) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
        let key = PipelineKey {
            label: label.to_string(),
            source_hash: hash_source(wgsl_source),
            storage_count: storage_count as u32,
            has_uniform,
        };

        {
            let cache = self.pipeline_cache.lock().unwrap();
            if let Some(cached) = cache.get(&key) {
                return (
                    Arc::clone(&cached.pipeline),
                    Arc::clone(&cached.bind_group_layout),
                );
            }
        }

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label} Compute Pipeline")),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let pipeline = Arc::new(compute_pipeline);
        let bind_group_layout = Arc::new(bind_group_layout);

        let mut cache = self.pipeline_cache.lock().unwrap();
        // Another thread may have inserted meanwhile; prefer existing.
        if let Some(cached) = cache.get(&key) {
            return (
                Arc::clone(&cached.pipeline),
                Arc::clone(&cached.bind_group_layout),
            );
        }
        cache.insert(
            key,
            CachedPipeline {
                pipeline: Arc::clone(&pipeline),
                bind_group_layout: Arc::clone(&bind_group_layout),
            },
        );
        (pipeline, bind_group_layout)
    }

    fn make_bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        storage_buffers: &[&wgpu::Buffer],
        uniform_bytes: &[u8],
    ) -> (wgpu::BindGroup, Option<wgpu::Buffer>) {
        let uniform_buffer = if !uniform_bytes.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label} Uniform Buffer")),
                contents: uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            }))
        } else {
            None
        };

        let mut bind_group_entries = Vec::with_capacity(storage_buffers.len() + 1);
        for (i, buf) in storage_buffers.iter().enumerate() {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            });
        }
        if let Some(buf) = &uniform_buffer {
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: storage_buffers.len() as u32,
                resource: buf.as_entire_binding(),
            });
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} Bind Group")),
            layout,
            entries: &bind_group_entries,
        });

        (bind_group, uniform_buffer)
    }

    /// Encode one compute pass. Returns resources that must stay alive until `queue.submit`.
    fn encode_1d_or_2d_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        label: &str,
        wgsl_source: &str,
        storage_buffers: &[&wgpu::Buffer],
        uniform_bytes: &[u8],
        gx: u32,
        gy: u32,
    ) -> (wgpu::BindGroup, Option<wgpu::Buffer>) {
        let has_uniform = !uniform_bytes.is_empty();
        let (pipeline, layout) =
            self.get_or_create_pipeline(label, wgsl_source, storage_buffers.len(), has_uniform);
        let (bind_group, uniform_keep_alive) =
            self.make_bind_group(label, &layout, storage_buffers, uniform_bytes);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label} Compute Pass")),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(gx.max(1), gy.max(1), 1);
        }
        (bind_group, uniform_keep_alive)
    }

    fn fold_workgroups_1d(workgroups_x: u32) -> (u32, u32) {
        let max_x = 65535u32;
        let gx = workgroups_x.min(max_x).max(1);
        let gy = (workgroups_x + max_x - 1) / max_x;
        (gx, gy.max(1))
    }

    /// Encode + submit multiple compute passes in one command buffer (no Wait).
    /// Separate compute passes provide Metal-safe storage visibility between dispatches.
    pub fn dispatch_compute_passes(&self, label: &str, passes: &[ComputePassDesc<'_>]) {
        if passes.is_empty() {
            return;
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{label} Batched Encoder")),
        });
        let mut keep_alive = Vec::with_capacity(passes.len());
        for p in passes {
            let (gx, gy) = if p.workgroups_y > 0 {
                (p.workgroups_x.max(1), p.workgroups_y.max(1))
            } else {
                Self::fold_workgroups_1d(p.workgroups_x)
            };
            keep_alive.push(self.encode_1d_or_2d_pass(
                &mut encoder,
                p.label,
                p.wgsl_source,
                p.storage_buffers,
                p.uniform_bytes,
                gx,
                gy,
            ));
        }
        self.queue.submit(Some(encoder.finish()));
        drop(keep_alive);
        // Intentionally no Maintain::Wait — drain only at readback / poll_wait.
    }

    /// Helper to compile and dispatch a native 2D compute shader (@workgroup_size(16, 16)).
    pub fn dispatch_compute_shader_2d(
        &self,
        label: &str,
        wgsl_source: &str,
        storage_buffer: &GpuImageBuffer,
        uniform_bytes: &[u8],
    ) {
        let gx = (storage_buffer.width as u32 + 15) / 16;
        let gy = (storage_buffer.height as u32 + 15) / 16;
        self.dispatch_compute_shader_2d_multi(
            label,
            wgsl_source,
            &[&storage_buffer.buffer],
            uniform_bytes,
            gx,
            gy,
        );
    }

    /// Compile and dispatch a 2D compute shader with multiple storage buffers.
    /// Submits without waiting — call [`Self::poll_wait`] or a download to sync.
    pub fn dispatch_compute_shader_2d_multi(
        &self,
        label: &str,
        wgsl_source: &str,
        storage_buffers: &[&wgpu::Buffer],
        uniform_bytes: &[u8],
        gx: u32,
        gy: u32,
    ) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{label} Encoder")),
        });
        let keep = self.encode_1d_or_2d_pass(
            &mut encoder,
            label,
            wgsl_source,
            storage_buffers,
            uniform_bytes,
            gx,
            gy,
        );
        self.queue.submit(Some(encoder.finish()));
        drop(keep);
    }

    /// Helper to compile and dispatch a compute shader on a GPU image storage buffer.
    pub fn dispatch_compute_shader(
        &self,
        label: &str,
        wgsl_source: &str,
        storage_buffer: &GpuImageBuffer,
        uniform_bytes: &[u8],
    ) {
        self.dispatch_compute_shader_2d(label, wgsl_source, storage_buffer, uniform_bytes);
    }

    /// Compile and dispatch a compute shader with multiple storage buffers.
    ///
    /// Storage buffers are bound at consecutive bindings starting at 0.
    /// If `uniform_bytes` is non-empty, the uniform buffer is bound at the next binding.
    /// Submits without waiting — call [`Self::poll_wait`] or a download to sync.
    pub fn dispatch_compute_shader_multi(
        &self,
        label: &str,
        wgsl_source: &str,
        storage_buffers: &[&wgpu::Buffer],
        uniform_bytes: &[u8],
        workgroups_x: u32,
    ) {
        let (gx, gy) = Self::fold_workgroups_1d(workgroups_x);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{label} Encoder")),
        });
        let keep = self.encode_1d_or_2d_pass(
            &mut encoder,
            label,
            wgsl_source,
            storage_buffers,
            uniform_bytes,
            gx,
            gy,
        );
        self.queue.submit(Some(encoder.finish()));
        drop(keep);
    }
}
