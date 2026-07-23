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

pub async fn run_pixel_pipeline_with_backend_async(
    image: &mut Image,
    pixel_pipeline: &mut config::PipelineConfig,
    backend: &Backend,
) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&"[Pichromatic WASM] Creating GPU PipelineImage...".into());

    let mut pipeline_image = match backend {
        Backend::Cpu => PipelineImage::Cpu(image.clone()),
        Backend::Wgpu(ctx) => PipelineImage::new_gpu(ctx, image),
    };

    let modules = &mut pixel_pipeline.pipeline_modules;
    for (i, module) in modules.iter_mut().enumerate() {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("[Pichromatic WASM] Starting GPU module {}: {}", i, module.schema().name).into());

        module.process_async(backend, &mut pipeline_image).await;

        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("[Pichromatic WASM] Finished GPU module {}: {}", i, module.schema().name).into());
    }

    let ctx = match backend {
        Backend::Cpu => None,
        Backend::Wgpu(ctx) => Some(ctx.as_ref()),
    };

    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&"[Pichromatic WASM] Downloading GPU rendered result to CPU...".into());

    *image = pipeline_image.to_cpu_async(ctx).await;

    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&"[Pichromatic WASM] Downloaded GPU rendered result!".into());
}
