use std::time::Instant;
use pichromatic::pixel::Image;

use crate::config;

pub fn run_pixel_pipeline(
    image: Image,
    pixel_pipeline: &mut config::PipelineConfig,
) -> Image {
    let modules = &mut pixel_pipeline.pipeline_modules;

    let mut pipeline_image = image.clone();


    for module in modules.into_iter() {
        let now = Instant::now();
        module.process(&mut pipeline_image);
        println!("{:} execution time: {:.2?}",module.get_name(), now.elapsed());
    }
    println!("\n");
    return pipeline_image;
}
