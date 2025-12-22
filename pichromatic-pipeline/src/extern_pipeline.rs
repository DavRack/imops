use pichromatic::{cfa::CFA, demosaic::{Dim2, Point, Rect, crop_and_normalize}, image::ImageMetadata, pixel::Image};
use rawler::{RawImageData, decoders::{RawDecodeParams}, imgop::xyz::Illuminant, rawsource::RawSource};
use wasm_bindgen::prelude::*;
use std::ffi::{c_float, c_uchar, c_char, CString, CStr};
use std::slice;
use std::mem::ManuallyDrop;

use crate::config::{parse_config};
use crate::pipeline::run_pixel_pipeline;

#[repr(C)]
#[wasm_bindgen]
pub struct ExternalRGBImage {
    pub data: *mut c_uchar,
    pub len: usize,
    pub width: usize,
    pub height: usize,
}

#[repr(C)]
#[wasm_bindgen]
pub struct ExternalRawImage {
    pub data: *mut c_float,
    pub len: usize,
    pub metadata: *mut c_char,
}

#[wasm_bindgen]
impl ExternalRGBImage {
    pub fn get_image_data(&self) -> Vec<u8>{
        let raw_slice = unsafe { slice::from_raw_parts(self.data, self.len) };
        return raw_slice.to_vec()
    }
}

#[wasm_bindgen]
impl ExternalRawImage {
    pub fn metadata_json(&self) -> String {
        unsafe {
            CStr::from_ptr(self.metadata).to_string_lossy().into_owned()
        }
    }
}

// Helper to safely prepare vector for FFI
fn vec_to_raw_parts<T>(v: Vec<T>) -> (*mut T, usize) {
    let mut v = v;
    v.shrink_to_fit(); // Ensure len == cap so we can reconstruct safely with just len
    let mut v = ManuallyDrop::new(v);
    (v.as_mut_ptr(), v.len())
}

impl From<Image> for ExternalRawImage {
    fn from(value: Image) -> ExternalRawImage{
        let metadata = serde_json::to_string(&value.metadata).expect(
            "can't serialize image metadata"
        );
        let c_metadata = CString::new(metadata).expect("CString::new failed");
        
        let (data, len) = vec_to_raw_parts(value.raw_data);

        ExternalRawImage{
            data,
            len,
            metadata: c_metadata.into_raw()
        }
    }
}

impl From<&ExternalRawImage> for Image {
    fn from(value: &ExternalRawImage) -> Image{
        // We don't take ownership here, just read
        let c_str = unsafe { CStr::from_ptr(value.metadata) };
        let metadata_str = c_str.to_str().expect("Invalid UTF-8");
        
        let metadata: ImageMetadata = match serde_json::from_str(metadata_str) {
            Ok(data) => data,
            Err(e) => {log(&e.to_string()); panic!()},
        };
        
        let raw_slice = unsafe { slice::from_raw_parts(value.data, value.len) };

        return Image{
            raw_data: raw_slice.to_vec(), // Clone data
            rgb_data: vec![],
            metadata: metadata,
        }
    }
}

#[no_mangle]
pub extern "C" fn get_raw_img(
    file_bytes_ptr: *const u8,
    file_bytes_len: usize,
) -> ExternalRawImage {
    let file_bytes = unsafe { slice::from_raw_parts(file_bytes_ptr, file_bytes_len) };
    get_raw_img_internal(file_bytes)
}

#[wasm_bindgen]
pub fn get_raw_img_js(file_bytes: Vec<u8>) -> ExternalRawImage {
    get_raw_img_internal(&file_bytes)
}

fn get_raw_img_internal(file_bytes: &[u8]) -> ExternalRawImage {
    // 1. Decode the RAW bytes using rawloader
    let decode_params = RawDecodeParams::default();
    let mut file = RawSource::new_from_slice(file_bytes);
    let mut raw_image = rawler::decode(&mut file, &decode_params).expect(
        "error decoding file"
    );
    
    let calibration_matrix_d65 = raw_image.camera.color_matrix[&Illuminant::D65].clone();
    let wb_coeffs = raw_image.wb_coeffs.map(|v| if v.is_nan() {0.0} else {v});
    let raw_image_dimentions = raw_image.dim();
    let raw_image_crop_area = raw_image.crop_area.unwrap();
    let raw_image_white_level = raw_image.whitelevel.as_bayer_array()[0];
    let raw_image_black_level = raw_image.blacklevel.as_bayer_array()[0];
    let raw_image_cfa = raw_image.camera.cfa.to_string();
    let _ = raw_image.apply_scaling();
    let raw_image_data = match raw_image.data {
        RawImageData::Float(data) => data,
        _ => panic!(""),
    };
    let image_metadata = ImageMetadata{
        width: raw_image_dimentions.w,
        height: raw_image_dimentions.h,
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
    log(&format!("{:?}",image.metadata.wb_coeffs));

    let web_raw_image: ExternalRawImage = image.into();
    
    return web_raw_image
}

#[wasm_bindgen]
extern "C" {
    // Use `js_namespace` here to bind `console.log(..)` instead of just
    // `log(..)`
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[no_mangle]
pub extern "C" fn run_extern_pixel_pipeline(
    web_raw_image: &ExternalRawImage,
    pixel_pipeline: *const c_char,
) -> ExternalRGBImage {
    // decode_buffer takes a reference to the bytes
    let mut image: Image = Image::from(web_raw_image);
    
    let c_str = unsafe { CStr::from_ptr(pixel_pipeline) };
    let pipeline_str = c_str.to_str().expect("Invalid UTF-8");
    
    let mut pipeline = parse_config(pipeline_str.to_string());
    run_pixel_pipeline(&mut image, &mut pipeline);


    
    let out_image_data: Vec<c_uchar> = image.rgb_data.iter().flatten().map(|p| {
        (p*255.0) as c_uchar
    }).collect();
    
    let width = image.metadata.width;
    let height = image.metadata.height;
    let (data, len) = vec_to_raw_parts(out_image_data);
    
    return ExternalRGBImage{
        data,
        width,
        height,
        len,
    }
}

#[wasm_bindgen]
pub fn run_extern_pixel_pipeline_js(
    web_raw_image: &ExternalRawImage,
    pixel_pipeline: String,
) -> ExternalRGBImage {
    let c_pixel_pipeline = CString::new(pixel_pipeline).expect("CString::new failed");
    log(&format!("{:?}", web_raw_image.data));
    run_extern_pixel_pipeline(web_raw_image, c_pixel_pipeline.as_ptr())
}

#[wasm_bindgen]
#[no_mangle]
pub extern "C" fn crop_bayer_center(
    image: &ExternalRawImage,
    crop_factor: usize,
) -> ExternalRawImage {
    // 1. Handle base case (no crop)
    if crop_factor <= 1 {
        // Clone manually because struct contains pointers
        // We need to clone the UNDERLYING data because the return value will be owned by caller
        // and they will likely free it.
        let img: Image = image.into(); // Image::from(&ExternalRawImage) clones the data
        return img.into(); // Convert back to ExternalRawImage (allocates new pointers)
    }
    let mut image_obj: Image = image.into();
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
    image_obj.raw_data = result;
    image_obj.metadata.width = new_width;
    image_obj.metadata.height = new_height;
    return image_obj.into()
}

#[wasm_bindgen]
#[no_mangle]
pub unsafe extern "C" fn free_external_raw_image(img: ExternalRawImage) {
    if !img.data.is_null() {
        let _ = Vec::from_raw_parts(img.data, img.len, img.len);
    }
    if !img.metadata.is_null() {
        let _ = CString::from_raw(img.metadata);
    }
}

#[wasm_bindgen]
#[no_mangle]
pub unsafe extern "C" fn free_rgb_image(img: ExternalRGBImage) {
    if !img.data.is_null() {
        let _ = Vec::from_raw_parts(img.data, img.len, img.len);
    }
}
