use pichromatic::pixel::Image;
use crate::config;

pub fn run_pixel_pipeline(
    image: &mut Image,
    pixel_pipeline: &mut config::PipelineConfig,
){
    let modules = &mut pixel_pipeline.pipeline_modules;

    for module in modules.into_iter() {
        module.process(image);
    }
}
