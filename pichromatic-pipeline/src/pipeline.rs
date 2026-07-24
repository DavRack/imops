use pichromatic::pixel::Image;
use crate::config;
use crate::backend::{Backend, PipelineImage};

pub extern "C" fn run_pixel_pipeline(
    image: &mut Image,
    pixel_pipeline: &mut config::PipelineConfig,
) {
    run_pixel_pipeline_with_backend(image, pixel_pipeline, &Backend::Cpu);
}

pub fn run_pixel_pipeline_with_backend(
    image: &mut Image,
    pixel_pipeline: &mut config::PipelineConfig,
    backend: &Backend,
) {
    let mut pipeline_image = match backend {
        Backend::Cpu => PipelineImage::Cpu(image.clone()),
        Backend::Wgpu(ctx) => PipelineImage::new_gpu(ctx, image),
    };

    let modules = &mut pixel_pipeline.pipeline_modules;
    for module in modules.iter_mut() {
        module.process(backend, &mut pipeline_image);
    }

    *image = match backend {
        Backend::Cpu => match pipeline_image {
            PipelineImage::Cpu(img) => img,
            PipelineImage::Gpu(_, _, _) => unreachable!("CPU backend produced GPU image"),
        },
        Backend::Wgpu(ctx) => match pipeline_image {
            PipelineImage::Cpu(img) => img,
            PipelineImage::Gpu(buf, meta, raw) => {
                let mut img = ctx.download_image(&buf, &meta);
                img.raw_data = raw;
                ctx.recycle_rgba_buffer(buf);
                img
            }
        },
    };
}

pub async fn run_pixel_pipeline_with_backend_async(
    image: &mut Image,
    pixel_pipeline: &mut config::PipelineConfig,
    backend: &Backend,
) {
    let mut pipeline_image = match backend {
        Backend::Cpu => PipelineImage::Cpu(image.clone()),
        Backend::Wgpu(ctx) => PipelineImage::new_gpu(ctx, image),
    };

    let modules = &mut pixel_pipeline.pipeline_modules;
    for module in modules.iter_mut() {
        module.process_async(backend, &mut pipeline_image).await;
    }

    *image = match backend {
        Backend::Cpu => match pipeline_image {
            PipelineImage::Cpu(img) => img,
            PipelineImage::Gpu(_, _, _) => unreachable!("CPU backend produced GPU image"),
        },
        Backend::Wgpu(ctx) => match pipeline_image {
            PipelineImage::Cpu(img) => img,
            PipelineImage::Gpu(buf, meta, raw) => {
                let mut img = ctx.download_image_async(&buf, &meta).await;
                img.raw_data = raw;
                ctx.recycle_rgba_buffer(buf);
                img
            }
        },
    };
}

/// Run the pipeline on GPU and present to the configured canvas (no CPU readback).
#[cfg(target_arch = "wasm32")]
pub async fn run_pixel_pipeline_present_async(
    image: &Image,
    pixel_pipeline: &mut config::PipelineConfig,
    backend: &Backend,
) -> Result<(usize, usize), String> {
    let ctx = match backend {
        Backend::Wgpu(ctx) => ctx,
        Backend::Cpu => {
            return Err("GPU present requires Backend::Wgpu".to_string());
        }
    };

    let mut pipeline_image = PipelineImage::new_gpu(ctx, image);
    for module in pixel_pipeline.pipeline_modules.iter_mut() {
        module.process_async(backend, &mut pipeline_image).await;
    }

    match pipeline_image {
        PipelineImage::Gpu(buf, _meta, _raw) => {
            let dims = (buf.width, buf.height);
            ctx.present_image(&buf)?;
            ctx.recycle_rgba_buffer(buf);
            Ok(dims)
        }
        PipelineImage::Cpu(_) => Err("Pipeline ended on CPU unexpectedly".to_string()),
    }
}
