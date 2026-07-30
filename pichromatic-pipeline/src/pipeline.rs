use pichromatic::pixel::Image;
use crate::config;
use crate::backend::{Backend, PipelineImage};

pub extern "C" fn run_pixel_pipeline(
    image: &mut Image,
    pixel_pipeline: &mut config::PipelineConfig,
) {
    let t_start = std::time::Instant::now();
    let mut gpu_success = false;
    if let Some(gpu_ctx) = pichromatic::gpu::GpuContext::try_new_sync() {
        let mut work_image = image.clone();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_pixel_pipeline_with_backend(&mut work_image, pixel_pipeline, &Backend::Wgpu(gpu_ctx));
        }));
        if res.is_ok() {
            *image = work_image;
            gpu_success = true;
            let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[Pichromatic Pipeline] GPU execution succeeded in {:.2} ms", elapsed_ms);
        } else {
            eprintln!("[Pichromatic Pipeline WARNING] GPU execution panicked; falling back to CPU execution");
        }
    } else {
        eprintln!("[Pichromatic Pipeline WARNING] GpuContext::try_new_sync() returned None; using CPU fallback");
    }
    if !gpu_success {
        let t_cpu = std::time::Instant::now();
        run_pixel_pipeline_with_backend(image, pixel_pipeline, &Backend::Cpu);
        let elapsed_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[Pichromatic Pipeline] CPU execution succeeded in {:.2} ms", elapsed_ms);
    }
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
        let t_mod = std::time::Instant::now();
        module.process(backend, &mut pipeline_image);
        println!("  [Module] {:?} took {:.2?}", module.schema().name, t_mod.elapsed());
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

#[derive(Debug, Clone, Copy)]
pub struct PipelineRenderResult {
    pub width: usize,
    pub height: usize,
    pub submit_ms: f64,
    pub fence_wait_ms: f64,
    pub total_ms: f64,
}

/// Run the pipeline on GPU and present to the configured canvas (no CPU readback).
#[cfg(target_arch = "wasm32")]
pub async fn run_pixel_pipeline_present_async(
    image: &Image,
    pixel_pipeline: &mut config::PipelineConfig,
    backend: &Backend,
) -> Result<PipelineRenderResult, String> {
    let t_start = js_sys::Date::now();
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
            let width = buf.width;
            let height = buf.height;
            ctx.present_image(&buf)?;
            let t_submit = js_sys::Date::now();
            if let Err(e) = ctx.end_of_render_fence_async(&buf.buffer).await {
                return Err(e);
            }
            let t_fence = js_sys::Date::now();
            ctx.recycle_rgba_buffer(buf);

            let total_ms = (t_fence - t_start).max(0.0);
            let submit_ms = (t_submit - t_start).max(0.0);
            let fence_wait_ms = (t_fence - t_submit).max(0.0);

            Ok(PipelineRenderResult {
                width,
                height,
                submit_ms,
                fence_wait_ms,
                total_ms,
            })
        }
        PipelineImage::Cpu(_) => Err("Pipeline ended on CPU unexpectedly".to_string()),
    }
}

/// Run the pipeline and return display RGBA8 (GPU encode + u8 readback, or CPU clamp).
/// For preview/editing when WebGPU canvas present is unavailable (e.g. iOS Safari).
pub async fn run_pixel_pipeline_readback_u8_async(
    image: &Image,
    pixel_pipeline: &mut config::PipelineConfig,
    backend: &Backend,
) -> Result<(usize, usize, Vec<u8>), String> {
    match backend {
        Backend::Wgpu(ctx) => {
            let mut pipeline_image = PipelineImage::new_gpu(ctx, image);
            for module in pixel_pipeline.pipeline_modules.iter_mut() {
                module.process_async(backend, &mut pipeline_image).await;
            }

            match pipeline_image {
                PipelineImage::Gpu(buf, _meta, _raw) => {
                    let width = buf.width;
                    let height = buf.height;
                    let rgba8 = ctx.download_rgba8_async(&buf).await;
                    ctx.recycle_rgba_buffer(buf);
                    Ok((width, height, rgba8))
                }
                PipelineImage::Cpu(img) => {
                    let width = img.metadata.width;
                    let height = img.metadata.height;
                    let rgba8 = pichromatic::gpu::GpuContext::image_to_rgba8(&img);
                    Ok((width, height, rgba8))
                }
            }
        }
        Backend::Cpu => {
            let mut work = image.clone();
            run_pixel_pipeline_with_backend_async(&mut work, pixel_pipeline, backend).await;
            let width = work.metadata.width;
            let height = work.metadata.height;
            let rgba8 = pichromatic::gpu::GpuContext::image_to_rgba8(&work);
            Ok((width, height, rgba8))
        }
    }
}
