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

    let ctx = match backend {
        Backend::Cpu => None,
        Backend::Wgpu(ctx) => Some(ctx.as_ref()),
    };

    *image = pipeline_image.to_cpu(ctx);
}
