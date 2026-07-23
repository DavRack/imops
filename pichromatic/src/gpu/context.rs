use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use crate::pixel::{Image, Pixel};

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

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipeline_cache: Mutex<HashMap<PipelineKey, CachedPipeline>>,
}

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

impl GpuContext {
    pub fn new_sync() -> Arc<Self> {
        pollster::block_on(Self::new())
    }

    pub async fn new() -> Arc<Self> {
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
            .expect("Failed to find a suitable GPU adapter");

        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Pichromatic GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                        max_buffer_size: limits.max_buffer_size,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create wgpu device");

        Arc::new(Self {
            device,
            queue,
            pipeline_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Block until queued GPU work completes. Use at readback / true sync points only.
    pub fn poll_wait(&self) {
        self.device.poll(wgpu::Maintain::Wait);
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
            raw_data: vec![],
            metadata: meta,
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

        let mut layout_entries = Vec::new();
        for i in 0..storage_count {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        if has_uniform {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: storage_count as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label} Bind Group Layout")),
            entries: &layout_entries,
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label} Pipeline Layout")),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label} Compute Pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

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
