pub use color::ColorSpaceTag;
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
    for i in 0..components {
        for j in 0..3 {
            xyz2cam[i][j] = calibration_matrix_d65[i * 3 + j];
        }
    }
    let xyz2cam_normalized = normalize_matrix(xyz2cam);
    let foward_matrix = pseudo_inverse_matrix(xyz2cam_normalized);
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
