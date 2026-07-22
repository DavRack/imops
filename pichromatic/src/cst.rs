pub use color::ColorSpaceTag;
use crate::gpu::{GpuContext, GpuImageBuffer};
use crate::pixel::{ImageBuffer, Pixel};
use rayon::prelude::*;


pub fn cst(image_buffer: &mut ImageBuffer, source_cs: ColorSpaceTag, target_cs: ColorSpaceTag){
    image_buffer.par_iter_mut().for_each(|pixel|{
        *pixel = source_cs.convert(target_cs, *pixel);
    });
}

pub fn camera_cst(image_buffer: &mut ImageBuffer, target_cs: ColorSpaceTag, calibration_matrix_d65: &[f32]){
    let components = calibration_matrix_d65.len() / 3;
    let mut xyz2cam: [Pixel; 3] = [[0.0; 3]; 3];
    
    // Compute the reference camera response under D65: [0.95042, 1.0, 1.08890]
    let xyz_d65 = [0.95042, 1.0, 1.08890];
    let mut rgb_d65 = [0.0; 3];
    for i in 0..3 {
        rgb_d65[i] = calibration_matrix_d65[i * 3] * xyz_d65[0]
                   + calibration_matrix_d65[i * 3 + 1] * xyz_d65[1]
                   + calibration_matrix_d65[i * 3 + 2] * xyz_d65[2];
    }
    
    let mut multipliers = [1.0; 3];
    for i in 0..3 {
        if rgb_d65[i] > 0.0 {
            multipliers[i] = 1.0 / rgb_d65[i];
        }
    }
    
    for i in 0..components {
        for j in 0..3 {
            xyz2cam[i][j] = calibration_matrix_d65[i * 3 + j] * multipliers[i];
        }
    }
    let foward_matrix = pseudo_inverse_matrix(xyz2cam);
    image_buffer.par_iter_mut().for_each(|pixel|{
        let [r, g, b] = *pixel;
        let xyzd65_pixel = [
            foward_matrix[0][0] * r + foward_matrix[0][1] * g + foward_matrix[0][2] * b,
            foward_matrix[1][0] * r + foward_matrix[1][1] * g + foward_matrix[1][2] * b,
            foward_matrix[2][0] * r + foward_matrix[2][1] * g + foward_matrix[2][2] * b,
        ];
        *pixel = ColorSpaceTag::XyzD65.convert(target_cs, xyzd65_pixel)

    });
}

/// Calculate pseudo-inverse of a given matrix
pub fn pseudo_inverse_matrix<const N: usize>(matrix: [[f32; 3]; N]) -> [[f32; N]; 3] {
  let mut tmp: [[f32; 3]; N] = [Default::default(); N];
  let mut result: [[f32; N]; 3] = [[Default::default(); N]; 3];

  let mut work: [[f32; 6]; 3] = [Default::default(); 3];
  for i in 0..3 {
    for j in 0..6 {
      work[i][j] = if j == i + 3 { 1.0 } else { 0.0 };
    }
    for j in 0..3 {
      for k in 0..N {
        work[i][j] += matrix[k][i] * matrix[k][j];
      }
    }
  }
  for i in 0..3 {
    let mut num = work[i][i];
    for j in 0..6 {
      work[i][j] /= num;
    }
    for k in 0..3 {
      if k == i {
        continue;
      }
      num = work[k][i];
      for j in 0..6 {
        work[k][j] -= work[i][j] * num;
      }
    }
  }
  for i in 0..N {
    for j in 0..3 {
      tmp[i][j] = 0.0;
      for k in 0..3 {
        tmp[i][j] += work[j][k + 3] * matrix[i][k];
      }
    }
  }
  for i in 0..3 {
    for j in 0..N {
      result[i][j] = tmp[j][i];
    }
  }
  result
}

/// Normalize a matrix so that the sum of each row equals to 1.0
pub fn normalize_matrix<const N: usize, const M: usize>(rgb2cam: [[f32; N]; M]) -> [[f32; N]; M] {
  let mut result = [[0.0; N]; M];
  for m in 0..M {
    let sum: f32 = rgb2cam[m].iter().sum();
    if sum.abs() != 0.0 {
      for n in 0..N {
        result[m][n] = rgb2cam[m][n] / sum;
      }
    }
  }
  result
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CstParamsGpu {
    pub r_row: [f32; 4],
    pub g_row: [f32; 4],
    pub b_row: [f32; 4],
    pub flags: [u32; 4],
}

pub fn transform_matrix_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, mat: [[f32; 3]; 3]) {
    let params = CstParamsGpu {
        r_row: [mat[0][0], mat[0][1], mat[0][2], 0.0],
        g_row: [mat[1][0], mat[1][1], mat[1][2], 0.0],
        b_row: [mat[2][0], mat[2][1], mat[2][2], 0.0],
        flags: [0, 0, 0, 0],
    };

    let shader_source = r#"
        struct Params {
            r_row: vec4<f32>,
            g_row: vec4<f32>,
            b_row: vec4<f32>,
            flags: vec4<u32>,
        };

        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<uniform> params: Params;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&pixels)) {
                return;
            }
            let p = pixels[index];
            let r = dot(p.rgb, params.r_row.rgb);
            let g = dot(p.rgb, params.g_row.rgb);
            let b = dot(p.rgb, params.b_row.rgb);
            pixels[index] = vec4<f32>(r, g, b, p.a);
        }
    "#;

    ctx.dispatch_compute_shader(
        "cst_matrix",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}

pub fn cst_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, source_cs: ColorSpaceTag, target_cs: ColorSpaceTag) {
    let is_source_srgb = matches!(source_cs, ColorSpaceTag::Srgb);
    let is_target_srgb = matches!(target_cs, ColorSpaceTag::Srgb);

    let intermediate_source = if is_source_srgb { ColorSpaceTag::LinearSrgb } else { source_cs };
    let intermediate_target = if is_target_srgb { ColorSpaceTag::LinearSrgb } else { target_cs };

    let c0 = intermediate_source.convert(intermediate_target, [1.0, 0.0, 0.0]);
    let c1 = intermediate_source.convert(intermediate_target, [0.0, 1.0, 0.0]);
    let c2 = intermediate_source.convert(intermediate_target, [0.0, 0.0, 1.0]);

    let params = CstParamsGpu {
        r_row: [c0[0], c1[0], c2[0], 0.0],
        g_row: [c0[1], c1[1], c2[1], 0.0],
        b_row: [c0[2], c1[2], c2[2], 0.0],
        flags: [if is_source_srgb { 1 } else { 0 }, if is_target_srgb { 1 } else { 0 }, 0, 0],
    };

    let shader_source = r#"
        struct Params {
            r_row: vec4<f32>,
            g_row: vec4<f32>,
            b_row: vec4<f32>,
            flags: vec4<u32>,
        };

        @group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
        @group(0) @binding(1) var<uniform> params: Params;

        fn srgb_eotf(c: f32) -> f32 {
            let abs_c = abs(c);
            let decoded = select(
                pow((abs_c + 0.055) / 1.055, 2.4),
                abs_c / 12.92,
                abs_c <= 0.04045
            );
            return select(-decoded, decoded, c >= 0.0);
        }

        fn srgb_oetf(c: f32) -> f32 {
            let abs_c = abs(c);
            let encoded = select(
                1.055 * pow(abs_c, 1.0 / 2.4) - 0.055,
                12.92 * abs_c,
                abs_c <= 0.0031308
            );
            return select(-encoded, encoded, c >= 0.0);
        }

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&pixels)) {
                return;
            }
            let p = pixels[index];
            var rgb = p.rgb;

            if (params.flags.x == 1u) {
                rgb = vec3<f32>(srgb_eotf(rgb.r), srgb_eotf(rgb.g), srgb_eotf(rgb.b));
            }

            let r = dot(rgb, params.r_row.rgb);
            let g = dot(rgb, params.g_row.rgb);
            let b = dot(rgb, params.b_row.rgb);
            var out_rgb = vec3<f32>(r, g, b);

            if (params.flags.y == 1u) {
                out_rgb = vec3<f32>(srgb_oetf(out_rgb.r), srgb_oetf(out_rgb.g), srgb_oetf(out_rgb.b));
            }

            pixels[index] = vec4<f32>(out_rgb, p.a);
        }
    "#;

    ctx.dispatch_compute_shader(
        "cst_full",
        shader_source,
        storage_buffer,
        bytemuck::bytes_of(&params),
    );
}

pub fn camera_cst_gpu(ctx: &GpuContext, storage_buffer: &GpuImageBuffer, target_cs: ColorSpaceTag, calibration_matrix_d65: &[f32]) {
    let components = calibration_matrix_d65.len() / 3;
    let mut xyz2cam: [[f32; 3]; 3] = [[0.0; 3]; 3];
    let xyz_d65 = [0.95042, 1.0, 1.08890];
    let mut rgb_d65 = [0.0; 3];
    for i in 0..3 {
        rgb_d65[i] = calibration_matrix_d65[i * 3] * xyz_d65[0]
                   + calibration_matrix_d65[i * 3 + 1] * xyz_d65[1]
                   + calibration_matrix_d65[i * 3 + 2] * xyz_d65[2];
    }
    let mut multipliers = [1.0; 3];
    for i in 0..3 {
        if rgb_d65[i] > 0.0 {
            multipliers[i] = 1.0 / rgb_d65[i];
        }
    }
    for i in 0..components {
        for j in 0..3 {
            xyz2cam[i][j] = calibration_matrix_d65[i * 3 + j] * multipliers[i];
        }
    }
    let forward_matrix = pseudo_inverse_matrix(xyz2cam);

    let convert_cam_to_target = |cam_pixel: [f32; 3]| -> [f32; 3] {
        let [r, g, b] = cam_pixel;
        let xyz_pixel = [
            forward_matrix[0][0] * r + forward_matrix[0][1] * g + forward_matrix[0][2] * b,
            forward_matrix[1][0] * r + forward_matrix[1][1] * g + forward_matrix[1][2] * b,
            forward_matrix[2][0] * r + forward_matrix[2][1] * g + forward_matrix[2][2] * b,
        ];
        ColorSpaceTag::XyzD65.convert(target_cs, xyz_pixel)
    };

    let c0 = convert_cam_to_target([1.0, 0.0, 0.0]);
    let c1 = convert_cam_to_target([0.0, 1.0, 0.0]);
    let c2 = convert_cam_to_target([0.0, 0.0, 1.0]);

    let mat = [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ];

    transform_matrix_gpu(ctx, storage_buffer, mat);
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_cst_identity_and_roundtrip() {
    let mut pixels = vec![[0.5, 0.3, 0.8]];
    cst(&mut pixels, ColorSpaceTag::AcesCg, ColorSpaceTag::Srgb);
    cst(&mut pixels, ColorSpaceTag::Srgb, ColorSpaceTag::AcesCg);
    
    let diff_r = (pixels[0][0] - 0.5).abs();
    let diff_g = (pixels[0][1] - 0.3).abs();
    let diff_b = (pixels[0][2] - 0.8).abs();
    
    assert!(diff_r < 1e-4, "r diff is {}", diff_r);
    assert!(diff_g < 1e-4, "g diff is {}", diff_g);
    assert!(diff_b < 1e-4, "b diff is {}", diff_b);
  }

  #[test]
  fn test_normalize_matrix() {
    let mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let normalized = normalize_matrix(mat);
    for row in normalized {
      let sum: f32 = row.iter().sum();
      assert!((sum - 1.0).abs() < 1e-6);
    }
  }
}
