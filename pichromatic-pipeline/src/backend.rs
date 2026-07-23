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
    Gpu(GpuImageBuffer, ImageMetadata, Vec<SubPixel>),
}

impl PipelineImage {
    pub fn new_cpu(image: Image) -> Self {
        Self::Cpu(image)
    }

    pub fn new_gpu(ctx: &GpuContext, image: &Image) -> Self {
        let buffer = ctx.upload_image(image);
        Self::Gpu(buffer, image.metadata.clone(), image.raw_data.clone())
    }

    pub fn to_cpu(&self, ctx: Option<&GpuContext>) -> Image {
        match self {
            Self::Cpu(img) => img.clone(),
            Self::Gpu(buf, meta, raw_data) => {
                let context = ctx.expect("GpuContext required to download GpuImage");
                let mut img = context.download_image(buf, meta);
                img.raw_data = raw_data.clone();
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
                img.raw_data = raw_data.clone();
                img
            }
        }
    }

    pub fn ensure_gpu(&mut self, ctx: &GpuContext) -> (&GpuImageBuffer, &mut ImageMetadata) {
        match self {
            Self::Gpu(buf, meta, _raw_data) => (buf, meta),
            Self::Cpu(img) => {
                let buf = ctx.upload_image(img);
                let meta = img.metadata.clone();
                let raw_data = img.raw_data.clone();
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
                img.raw_data = raw_data.clone();
                *self = Self::Cpu(img);
                match self {
                    Self::Cpu(img) => img,
                    _ => unreachable!(),
                }
            }
        }
    }
}
