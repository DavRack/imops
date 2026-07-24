use std::sync::Arc;
use pichromatic::gpu::{GpuContext, GpuImageBuffer};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::{Image, SubPixel};

#[derive(Clone)]
pub enum Backend {
    Cpu,
    Wgpu(Arc<GpuContext>),
}

pub enum PipelineImage {
    Cpu(Image),
    /// GPU working image + shared Bayer raw (Arc — no per-run clone).
    Gpu(GpuImageBuffer, ImageMetadata, Arc<[SubPixel]>),
}

impl PipelineImage {
    pub fn new_cpu(image: Image) -> Self {
        Self::Cpu(image)
    }

    pub fn new_gpu(ctx: &GpuContext, image: &Image) -> Self {
        let width = image.metadata.width.max(1);
        let height = image.metadata.height.max(1);
        let buffer = ctx.acquire_rgba_buffer(width, height);
        if !image.rgb_data.is_empty() {
            ctx.update_buffer_from_image(&buffer, image);
        }
        Self::Gpu(buffer, image.metadata.clone(), Arc::clone(&image.raw_data))
    }

    pub fn to_cpu(&self, ctx: Option<&GpuContext>) -> Image {
        match self {
            Self::Cpu(img) => img.clone(),
            Self::Gpu(buf, meta, raw_data) => {
                let context = ctx.expect("GpuContext required to download GpuImage");
                let mut img = context.download_image(buf, meta);
                img.raw_data = Arc::clone(raw_data);
                img
            }
        }
    }

    pub async fn to_cpu_async(&self, ctx: Option<&GpuContext>) -> Image {
        match self {
            Self::Cpu(img) => img.clone(),
            Self::Gpu(buf, meta, raw_data) => {
                let context = ctx.expect("GpuContext required to download GpuImage");
                let mut img = context.download_image_async(buf, meta).await;
                img.raw_data = Arc::clone(raw_data);
                img
            }
        }
    }

    pub fn ensure_gpu(&mut self, ctx: &GpuContext) -> (&GpuImageBuffer, &mut ImageMetadata) {
        match self {
            Self::Gpu(buf, meta, _raw_data) => (buf, meta),
            Self::Cpu(img) => {
                let width = img.metadata.width.max(1);
                let height = img.metadata.height.max(1);
                let buf = ctx.acquire_rgba_buffer(width, height);
                if !img.rgb_data.is_empty() {
                    ctx.update_buffer_from_image(&buf, img);
                }
                let meta = img.metadata.clone();
                let raw_data = Arc::clone(&img.raw_data);
                *self = Self::Gpu(buf, meta, raw_data);
                match self {
                    Self::Gpu(buf, meta, _raw_data) => (buf, meta),
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn ensure_cpu(&mut self, ctx: Option<&GpuContext>) -> &mut Image {
        match self {
            Self::Cpu(img) => img,
            Self::Gpu(buf, meta, raw_data) => {
                let context = ctx.expect("GpuContext required to download GpuImage");
                let mut img = context.download_image(buf, meta);
                img.raw_data = Arc::clone(raw_data);
                // Recycle the GPU buffer before dropping it via replace.
                let old = std::mem::replace(self, Self::Cpu(img));
                if let Self::Gpu(buf, _, _) = old {
                    context.recycle_rgba_buffer(buf);
                }
                match self {
                    Self::Cpu(img) => img,
                    _ => unreachable!(),
                }
            }
        }
    }

    /// Recycle the GPU RGBA buffer back into the context pool (if any).
    pub fn recycle_gpu_buffer(self, ctx: &GpuContext) {
        if let Self::Gpu(buf, _, _) = self {
            ctx.recycle_rgba_buffer(buf);
        }
    }
}
