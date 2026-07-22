use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::pixel::{Image, Pixel};

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub struct GpuImageBuffer {
    pub buffer: wgpu::Buffer,
    pub width: usize,
    pub height: usize,
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

        Arc::new(Self { device, queue })
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
    /// Used only for exact scalar reductions that require CPU `f64` precision
    /// (e.g. grain noise global variance); spatial work stays on GPU.
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

    /// Helper to compile and dispatch a compute shader on a GPU image storage buffer.
    pub fn dispatch_compute_shader(
        &self,
        label: &str,
        wgsl_source: &str,
        storage_buffer: &GpuImageBuffer,
        uniform_bytes: &[u8],
    ) {
        let num_pixels = (storage_buffer.width * storage_buffer.height) as u32;
        let workgroups = (num_pixels + 255) / 256;
        self.dispatch_compute_shader_multi(
            label,
            wgsl_source,
            &[&storage_buffer.buffer],
            uniform_bytes,
            workgroups,
        );
    }

    /// Compile and dispatch a compute shader with multiple storage buffers.
    ///
    /// Storage buffers are bound at consecutive bindings starting at 0.
    /// If `uniform_bytes` is non-empty, the uniform buffer is bound at the next binding.
    pub fn dispatch_compute_shader_multi(
        &self,
        label: &str,
        wgsl_source: &str,
        storage_buffers: &[&wgpu::Buffer],
        uniform_bytes: &[u8],
        workgroups_x: u32,
    ) {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let uniform_buffer = if !uniform_bytes.is_empty() {
            Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label} Uniform Buffer")),
                contents: uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            }))
        } else {
            None
        };

        let mut bind_group_entries = Vec::new();
        let mut layout_entries = Vec::new();

        for (i, buf) in storage_buffers.iter().enumerate() {
            let binding = i as u32;
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            });
        }

        if let Some(buf) = &uniform_buffer {
            let binding = storage_buffers.len() as u32;
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding,
                resource: buf.as_entire_binding(),
            });
        }

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label} Bind Group Layout")),
            entries: &layout_entries,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} Bind Group")),
            layout: &bind_group_layout,
            entries: &bind_group_entries,
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

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{label} Encoder")),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label} Compute Pass")),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups_x.max(1), 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }
}
