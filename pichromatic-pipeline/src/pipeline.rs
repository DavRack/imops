use std::time::Instant;
use pichromatic::pixel::Image;

use crate::config;

pub fn run_pixel_pipeline(
    image: Image,
    pixel_pipeline: config::PipelineConfig,
) -> Image {
    let modules = pixel_pipeline.pipeline_modules;

    let mut pipeline_image = image.clone();

    let set_cache = true;

    // let mut last_image = match modules[0].get_cache(){
    //     Some(image) => image,
    //     None => pipeline_image.clone(),
    // };
    for mut module in modules {
        let now = Instant::now();
        
        module.process(&mut pipeline_image);

        // if set_cache {
        //     match module.get_mask(){
        //         Some(mask) => {
        //             let mask_value = mask.create(&pipeline_image, &raw_image);
        //             pipeline_image.data.par_iter_mut().zip(last_image.data).zip(mask_value).for_each(
        //                 |((new_pixel, old_pixel), pixel_mask_value)|{
        //                     let [r, g, b] = *new_pixel;
        //                     let [or, og, ob] = old_pixel;
        //                     *new_pixel = [
        //                         (or*(1.0-pixel_mask_value)) + (r * pixel_mask_value),
        //                         (og*(1.0-pixel_mask_value)) + (g * pixel_mask_value),
        //                         (ob*(1.0-pixel_mask_value)) + (b * pixel_mask_value),
        //                     ];
        //                 }
        //             );
        //         },
        //         None => ()
        //     }
        //     module.set_cache(pipeline_image.clone());
        //     last_image = pipeline_image.clone();
        // }
        println!("{:} execution time: {:.2?}",module.get_name(), now.elapsed());
    }
    println!("\n");
    return pipeline_image;
}
