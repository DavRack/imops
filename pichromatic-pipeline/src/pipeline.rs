use std::time::Instant;
use pichromatic::pixel::Image;

use crate::config;

pub fn run_pixel_pipeline(
    mut image: Image,
    pixel_pipeline: &mut config::PipelineConfig,
) -> Image {
    let modules = &mut pixel_pipeline.pipeline_modules;

    let t1 = Instant::now();
    for module in modules.into_iter() {
        let now = Instant::now();
        module.process(&mut image);
        println!("{:} execution time: {:.2?}",module.get_name(), now.elapsed());
    }
    println!("tot exec pipeline: {:.2?}", t1.elapsed());
    println!("\n");
    return image;
}
