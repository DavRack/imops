use pichromatic::{cfa::CFA, demosaic::{Dim2, Point, Rect, crop_and_normalize}, image::ImageMetadata, pixel::Image};
use rawler::{RawImageData, decoders::{RawDecodeParams}, imgop::xyz::Illuminant, rawsource::RawSource};
use wasm_bindgen::prelude::*;
use std::slice;
use crate::{config::{self, PipelineConfig}, pipeline::run_pixel_pipeline};

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[no_mangle]
pub extern "C" fn get_raw_img(
    file_bytes_ptr: *const u8,
    file_bytes_len: usize,
) -> *mut Image {
    let file_bytes = unsafe { slice::from_raw_parts(file_bytes_ptr, file_bytes_len) };
    let img = get_raw_img_internal(file_bytes);
    Box::into_raw(Box::new(img))
}

#[wasm_bindgen]
pub fn get_image_rgb_data(image: *mut Image) -> Vec<u8> {
    let data = unsafe { &mut *image };
    let rgb_data = &data.rgb_data;
    return rgb_data.iter().flatten().map(|sub_pixel|{
        (sub_pixel.clamp(0.0, 1.0) * 255.0) as u8
    }).collect();
}

#[wasm_bindgen]
pub fn get_image_metadata(image: *const Image) -> String {
    let raw_metadata = unsafe { &*image };
    let metadata = serde_json::to_string(&raw_metadata.metadata).unwrap();
    return metadata
}

#[wasm_bindgen]
pub fn get_raw_img_js(file_bytes: Vec<u8>) -> *const Image {
    let img = get_raw_img_internal(&file_bytes);
    return Box::leak(Box::new(img))
}

#[wasm_bindgen]
pub fn get_pixel_pipeline(pixel_pipeline_config: String) -> *const PipelineConfig {
    let pipeline = config::parse_config(pixel_pipeline_config);
    let pipeline = Box::leak(Box::new(pipeline));
    return pipeline
}

#[wasm_bindgen]
pub fn run_pixel_pipeline_js(image: *mut Image, pixel_pipeline: *mut PipelineConfig) -> *const Image {
    let mut image_obj = unsafe {&mut *image};
    let mut pipeline_obj = unsafe {&mut *pixel_pipeline};
    run_pixel_pipeline(&mut image_obj, &mut pipeline_obj);
    return image
}

#[no_mangle]
#[wasm_bindgen]
pub extern "C" fn crop_bayer_center(
    image: *const Image,
    crop_factor: usize,
) -> *const Image {
    if crop_factor <= 1 {
        return image;
    }
    let image_obj = unsafe {&*image};
    // leak image_obj because we need to return a pointer to a section of memory that outlives 
    // this function
    let image_obj = Box::leak(Box::new(image_obj));
    let factor = crop_factor;
    let width = image_obj.metadata.width;
    let height = image_obj.metadata.height;

    // 2. Calculate new dimensions
    let new_width = (width / factor) & !1;
    let new_height = (height / factor) & !1;

    let mut result = Vec::with_capacity(new_width * new_height);

    // 2. Iterate over the *destination* coordinates
    for y in 0..new_height {
        // Determine if we are in the top row (0) or bottom row (1) of a 2x2 block
        let row_offset = y % 2;
        // Determine which 2x2 block row we are pulling from in the source
        // Example: If factor is 2, Dest Y 0->Src Y 0, Dest Y 1->Src Y 1, Dest Y 2->Src Y 4
        let src_block_y = (y / 2) * factor;
        let src_y = (src_block_y * 2) + row_offset;

        // Pre-calculate row start to avoid multiplication in inner loop
        let src_row_start_idx = src_y * width;

        for x in 0..new_width {
            // Determine if we are in the left col (0) or right col (1) of a 2x2 block
            let col_offset = x % 2;

            // Determine which 2x2 block col we are pulling from
            let src_block_x = (x / 2) * factor;
            let src_x = (src_block_x * 2) + col_offset;

            // 3. Extract pixel
            // This logic skips (factor-1) 2x2 blocks between every sample
            let idx = src_row_start_idx + src_x;

            // Safety check for production (optional if you trust your math/inputs)
            if idx < image_obj.raw_data.len() {
                result.push(image_obj.raw_data[idx]);
            } else {
                result.push(0.0); // Padding if math drifts at edges
            }
        }
    }
    let mut new_img = Image::default();
    new_img.raw_data = result;
    new_img.metadata = image_obj.metadata.clone();
    new_img.metadata.width = new_width;
    new_img.metadata.height = new_height;
    return Box::leak(Box::new(new_img))
}

fn is_identity_matrix(matrix: &Option<Vec<f32>>) -> bool {
    if let Some(ref m) = matrix {
        m.len() == 9 && 
        (m[0] - 1.0).abs() < 1e-5 && (m[1] - 0.0).abs() < 1e-5 && (m[2] - 0.0).abs() < 1e-5 &&
        (m[3] - 0.0).abs() < 1e-5 && (m[4] - 1.0).abs() < 1e-5 && (m[5] - 0.0).abs() < 1e-5 &&
        (m[6] - 0.0).abs() < 1e-5 && (m[7] - 0.0).abs() < 1e-5 && (m[8] - 1.0).abs() < 1e-5
    } else {
        true
    }
}

pub fn consolidate_dng_metadata(image: &mut Image, dng_meta: &crate::dng_metadata::DngMetadata) {
    image.metadata.baseline_exposure = dng_meta.baseline_exposure;
    image.metadata.opcode_list1 = dng_meta.opcode_list1.clone();
    image.metadata.opcode_list2 = dng_meta.opcode_list2.clone();
    image.metadata.opcode_list3 = dng_meta.opcode_list3.clone();
    image.metadata.dng_version = dng_meta.dng_version;
    image.metadata.dng_backward_version = dng_meta.dng_backward_version;
    image.metadata.unique_camera_model = dng_meta.unique_camera_model.clone();
    image.metadata.color_matrix1 = dng_meta.color_matrix1.clone();
    image.metadata.color_matrix2 = dng_meta.color_matrix2.clone();
    image.metadata.camera_calibration1 = dng_meta.camera_calibration1.clone();
    image.metadata.camera_calibration2 = dng_meta.camera_calibration2.clone();
    image.metadata.analog_balance = dng_meta.analog_balance.clone();
    image.metadata.as_shot_neutral = dng_meta.as_shot_neutral.clone();
    image.metadata.linear_response_limit = dng_meta.linear_response_limit;
    image.metadata.shadow_scale = dng_meta.shadow_scale;
    image.metadata.noise_profile = dng_meta.noise_profile.clone();
    image.metadata.profile_name = dng_meta.profile_name.clone();
    image.metadata.profile_tone_curve = dng_meta.profile_tone_curve.clone();
    image.metadata.lens_info = dng_meta.lens_info.clone();
    image.metadata.camera_serial_number = dng_meta.camera_serial_number.clone();

    // Fallback consolidation for active calibration matrix:
    if is_identity_matrix(&image.metadata.calibration_matrix_d65) {
        if let Some(ref cm2) = dng_meta.color_matrix2 {
            image.metadata.calibration_matrix_d65 = Some(cm2.clone());
        } else if let Some(ref cm1) = dng_meta.color_matrix1 {
            image.metadata.calibration_matrix_d65 = Some(cm1.clone());
        }
    }
}

pub fn get_raw_img_internal(file_bytes: &[u8]) -> Image {
    // 1. Decode the RAW bytes using rawloader
    let decode_params = RawDecodeParams::default();
    let mut file = RawSource::new_from_slice(file_bytes);
    let raw_image = rawler::decode(&mut file, &decode_params).expect(
        "error decoding file"
    );
    let mut image = parse_raw_image(raw_image);
    
    // 2. Extract DNG metadata directly from the bytes and consolidate
    if let Some(parser) = crate::dng_metadata::DngMetadataParser::new(file_bytes) {
        let dng_meta = parser.parse();
        consolidate_dng_metadata(&mut image, &dng_meta);
    }
    
    image
}
pub fn parse_raw_image(mut raw_image: rawler::RawImage) -> Image {
    let wb_coeffs = raw_image.wb_coeffs.map(|v| if v.is_nan() {0.0} else {v});
    let calibration_matrix_d65 = if let Some(matrix1) = raw_image.camera.color_matrix.get(&Illuminant::A) {
        if let Some(matrix2) = raw_image.camera.color_matrix.get(&Illuminant::D65) {
            interpolate_matrices(matrix1, matrix2, &wb_coeffs)
        } else {
            matrix1.clone()
        }
    } else if let Some(matrix2) = raw_image.camera.color_matrix.get(&Illuminant::D65) {
        matrix2.clone()
    } else if let Some((_, matrix)) = raw_image.camera.color_matrix.iter().next() {
        matrix.clone()
    } else {
        vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
    };
    let raw_image_dimensions = raw_image.dim();
    let raw_image_crop_area = raw_image.crop_area.unwrap();
    println!("DEBUG RAW: dim = {}x{}, crop = ({},{}) {}x{}", 
             raw_image_dimensions.w, raw_image_dimensions.h, 
             raw_image_crop_area.p.x, raw_image_crop_area.p.y, 
             raw_image_crop_area.d.w, raw_image_crop_area.d.h);
    let raw_image_white_level = raw_image.whitelevel.as_bayer_array()[0];
    let raw_image_black_level = raw_image.blacklevel.as_bayer_array()[0];
    let raw_image_cfa = raw_image.camera.cfa.to_string();
    let _ = raw_image.apply_scaling();
    let raw_image_data = match raw_image.data {
        RawImageData::Float(data) => data,
        _ => panic!(""),
    };
    let image_metadata = ImageMetadata{
        width: raw_image_dimensions.w,
        height: raw_image_dimensions.h,
        crop_area: Some(Rect{
            p: Point {
                x: raw_image_crop_area.p.x,
                y: raw_image_crop_area.p.y
            },
            d: Dim2{
                w: raw_image_crop_area.d.w,
                h: raw_image_crop_area.d.h
            },
        }),
        black_level: Some(raw_image_black_level),
        white_level: Some(raw_image_white_level),
        wb_coeffs: Some(wb_coeffs),
        cfa: Some(CFA::new(&raw_image_cfa)),
        calibration_matrix_d65: Some(calibration_matrix_d65),
        color_space: None,
        baseline_exposure: None,
        opcode_list1: None,
        opcode_list2: None,
        opcode_list3: None,
        dng_version: None,
        dng_backward_version: None,
        unique_camera_model: None,
        color_matrix1: None,
        color_matrix2: None,
        camera_calibration1: None,
        camera_calibration2: None,
        analog_balance: None,
        as_shot_neutral: None,
        linear_response_limit: None,
        shadow_scale: None,
        noise_profile: None,
        profile_name: None,
        profile_tone_curve: None,
        lens_info: None,
        camera_serial_number: None,
    };

    let mut image = Image{
        raw_data: raw_image_data,
        rgb_data: vec![],
        metadata: image_metadata,
    };
    let normalized_raw_data = crop_and_normalize(
        &image,
    );
    image.raw_data = normalized_raw_data;
    image.metadata.width = image.metadata.crop_area.unwrap().d.w;
    image.metadata.height = image.metadata.crop_area.unwrap().d.h;
    image
}

#[no_mangle]
pub extern "C" fn get_pixel_pipeline_c(
    config_ptr: *const u8,
    config_len: usize,
) -> *mut PipelineConfig {
    let config_bytes = unsafe { slice::from_raw_parts(config_ptr, config_len) };
    let config_str = std::str::from_utf8(config_bytes).unwrap_or("");
    println!("DEBUG RUST: get_pixel_pipeline_c config_str (len {}):\n{}", config_len, config_str);
    let pipeline = config::parse_config(config_str.to_string());
    Box::into_raw(Box::new(pipeline))
}

#[no_mangle]
pub extern "C" fn run_pixel_pipeline_c(
    image: *mut Image,
    pixel_pipeline: *mut PipelineConfig,
) -> *mut Image {
    let mut image_obj = unsafe { &mut *image };
    let mut pipeline_obj = unsafe { &mut *pixel_pipeline };
    println!("DEBUG PIPELINE: Input image dim = {}x{}, raw_data len = {}", 
             image_obj.metadata.width, image_obj.metadata.height, image_obj.raw_data.len());
    run_pixel_pipeline(&mut image_obj, &mut pipeline_obj);
    image
}

#[no_mangle]
pub extern "C" fn get_image_rgb_data_c(
    image: *mut Image,
    out_width: *mut usize,
    out_height: *mut usize,
    out_len: *mut usize,
) -> *const u8 {
    let data = unsafe { &mut *image };
    let width = data.metadata.width;
    let height = data.metadata.height;
    
    let rgb_u8: Vec<u8> = data.rgb_data.iter().flatten().map(|sub_pixel| {
        (sub_pixel.clamp(0.0, 1.0) * 255.0) as u8
    }).collect();
    
    unsafe {
        if !out_width.is_null() { *out_width = width; }
        if !out_height.is_null() { *out_height = height; }
        if !out_len.is_null() { *out_len = rgb_u8.len(); }
    }
    
    Box::into_raw(rgb_u8.into_boxed_slice()) as *const u8
}

#[no_mangle]
pub extern "C" fn free_image_c(image: *mut Image) {
    if !image.is_null() {
        unsafe { let _ = Box::from_raw(image); }
    }
}

#[no_mangle]
pub extern "C" fn free_pipeline_c(pipeline: *mut PipelineConfig) {
    if !pipeline.is_null() {
        unsafe { let _ = Box::from_raw(pipeline); }
    }
}

#[no_mangle]
pub extern "C" fn free_rgb_buffer_c(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, len, len);
        }
    }
}

fn interpolate_matrices(
    matrix1: &[f32],
    matrix2: &[f32],
    wb_coeffs: &[f32; 4],
) -> Vec<f32> {
    let xyz_a = [1.1507, 1.0, 0.2867];
    let xyz_d65 = [0.9642, 1.0, 0.8251];
    
    let mut s1 = [0.0; 3];
    let mut s2 = [0.0; 3];
    for i in 0..3 {
        s1[i] = matrix1[i * 3] * xyz_a[0] + matrix1[i * 3 + 1] * xyz_a[1] + matrix1[i * 3 + 2] * xyz_a[2];
        s2[i] = matrix2[i * 3] * xyz_d65[0] + matrix2[i * 3 + 1] * xyz_d65[1] + matrix2[i * 3 + 2] * xyz_d65[2];
    }
    
    let ratio1 = if s1[2] != 0.0 { s1[0] / s1[2] } else { 1.0 };
    let ratio2 = if s2[2] != 0.0 { s2[0] / s2[2] } else { 1.0 };
    
    let ratio_s = if wb_coeffs[0] > 0.0 && wb_coeffs[2] > 0.0 {
        wb_coeffs[2] / wb_coeffs[0]
    } else {
        ratio2
    };
    
    let w = if ratio1 != ratio2 && ratio1 > 0.0 && ratio2 > 0.0 && ratio_s > 0.0 {
        let val = (ratio_s.ln() - ratio2.ln()) / (ratio1.ln() - ratio2.ln());
        val.clamp(0.0, 1.0)
    } else {
        0.0
    };
    
    println!("DEBUG COLOR: ratio1 = {}, ratio2 = {}, ratio_s = {}, w = {}", ratio1, ratio2, ratio_s, w);
    
    let mut interpolated = vec![0.0; 9];
    for i in 0..9 {
        let val1 = if i < matrix1.len() { matrix1[i] } else { 0.0 };
        let val2 = if i < matrix2.len() { matrix2[i] } else { 0.0 };
        interpolated[i] = w * val1 + (1.0 - w) * val2;
    }
    interpolated
}

