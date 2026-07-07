use crate::pixel::{ImageBuffer};

pub fn apply_vignette_radial_correction(
    rgb_data: &mut ImageBuffer,
    width: usize,
    height: usize,
    opcode_list3: &[u8],
    strength: f32,
) {
    if opcode_list3.len() < 4 {
        return;
    }
    // All opcodes lists are stored in big-endian byte order
    let count = u32::from_be_bytes([opcode_list3[0], opcode_list3[1], opcode_list3[2], opcode_list3[3]]) as usize;
    let mut offset = 4;

    for _ in 0..count {
        if offset + 16 > opcode_list3.len() {
            break;
        }
        let opcode_id = u32::from_be_bytes([opcode_list3[offset], opcode_list3[offset+1], opcode_list3[offset+2], opcode_list3[offset+3]]);
        let _version = u32::from_be_bytes([opcode_list3[offset+4], opcode_list3[offset+5], opcode_list3[offset+6], opcode_list3[offset+7]]);
        let _flags = u32::from_be_bytes([opcode_list3[offset+8], opcode_list3[offset+9], opcode_list3[offset+10], opcode_list3[offset+11]]);
        let parameter_size = u32::from_be_bytes([opcode_list3[offset+12], opcode_list3[offset+13], opcode_list3[offset+14], opcode_list3[offset+15]]) as usize;
        
        offset += 16;
        if offset + parameter_size > opcode_list3.len() {
            break;
        }

        let params = &opcode_list3[offset..offset + parameter_size];
        offset += parameter_size;

        if opcode_id == 3 { // FixVignetteRadial
            if parameter_size < 56 {
                continue;
            }
            // Parse 7 doubles (f64), stored in big-endian
            let read_double = |idx: usize| -> f64 {
                let bytes = [
                    params[idx], params[idx+1], params[idx+2], params[idx+3],
                    params[idx+4], params[idx+5], params[idx+6], params[idx+7]
                ];
                f64::from_be_bytes(bytes)
            };

            let k0 = read_double(0) as f32;
            let k1 = read_double(8) as f32;
            let k2 = read_double(16) as f32;
            let k3 = read_double(24) as f32;
            let k4 = read_double(32) as f32;
            let cx = read_double(40) as f32;
            let cy = read_double(48) as f32;

            // Apply vignette correction in parallel
            use rayon::prelude::*;

            let w_f32 = (width - 1) as f32;
            let h_f32 = (height - 1) as f32;

            // Furthest corner distance calculation
            // Corners: (0,0), (1,0), (0,1), (1,1) in normalized coords
            let d_00 = (cx * cx + cy * cy).sqrt();
            let d_10 = ((1.0 - cx).powi(2) + cy * cy).sqrt();
            let d_01 = (cx * cx + (1.0 - cy).powi(2)).sqrt();
            let d_11 = ((1.0 - cx).powi(2) + (1.0 - cy).powi(2)).sqrt();
            let d_max = d_00.max(d_10).max(d_01).max(d_11);

            if d_max > 0.0 {
                rgb_data.par_iter_mut().enumerate().for_each(|(idx, pixel)| {
                    let py = (idx / width) as f32;
                    let px = (idx % width) as f32;

                    let u = px / w_f32;
                    let v = py / h_f32;

                    let du = u - cx;
                    let dv = v - cy;
                    let d = (du * du + dv * dv).sqrt();
                    let r = d / d_max;

                    let r2 = r * r;
                    let r4 = r2 * r2;
                    let r6 = r4 * r2;
                    let r8 = r4 * r4;
                    let r10 = r8 * r2;

                    let correction = k0 * r2 + k1 * r4 + k2 * r6 + k3 * r8 + k4 * r10;
                    let gain = 1.0 + strength * correction;
                    let gain_clamped = gain.max(0.0);

                    pixel[0] *= gain_clamped;
                    pixel[1] *= gain_clamped;
                    pixel[2] *= gain_clamped;
                });
            }
        }
    }
}
