use crate::wavelets::{Kernel, WaveletDecompose};
use crate::conditional_paralell::prelude::*;
use ndarray::{Array2, ArrayView2, Zip};

#[derive(Clone)]
pub struct ATrousTransform {
    pub input: Vec<[f32; 3]>,
    pub width: usize,
    pub height: usize,
    pub levels: usize,
    pub kernel: Kernel,
    pub current_level: usize,
}
#[derive(Clone)]
pub struct WaveletLayer {
    pub buffer: Vec<[f32; 3]>,
    pub pixel_scale: Option<usize>,
}

impl Iterator for ATrousTransform {
    type Item = WaveletLayer;

    fn next(&mut self) -> Option<Self::Item> {
        let pixel_scale = self.current_level;
        self.current_level += 1;

        if pixel_scale > self.levels {
            return None;
        }

        if pixel_scale == self.levels {
            return Some(WaveletLayer {
                buffer: self.input.clone(),
                pixel_scale: None,
            });
        }

        let layer_buffer = self.input.wavelet_decompose(self.height, self.width, self.kernel, pixel_scale);
        Some(WaveletLayer {
            pixel_scale: Some(pixel_scale),
            buffer: layer_buffer,
        })
    }
}

pub struct GuidedFilterParams {
    pub radius: usize,
    pub eps: f32,
    pub subsampling_ratio: usize,
}

impl Default for GuidedFilterParams {
    fn default() -> Self {
        Self {
            radius: 8,
            eps: 0.02 * 0.02,
            subsampling_ratio: 4,
        }
    }
}

/// Fast Guided Filter Implementation
fn fast_guided_filter(
    guide: &ArrayView2<f32>,
    src: &ArrayView2<f32>,
    params: &GuidedFilterParams,
) -> Array2<f32> {
    let (h, w) = guide.dim();
    let r = params.radius / params.subsampling_ratio;
    let r = if r < 1 { 1 } else { r };

    let guide_sub = subsample(guide, params.subsampling_ratio);
    let src_sub = subsample(src, params.subsampling_ratio);

    let mean_i = box_filter(&guide_sub.view(), r);
    let mean_p = box_filter(&src_sub.view(), r);
    let mean_ip = box_filter(&(&guide_sub * &src_sub).view(), r);
    let mean_ii = box_filter(&(&guide_sub * &guide_sub).view(), r);

    let mut a = Array2::<f32>::zeros(mean_i.dim());
    let mut b = Array2::<f32>::zeros(mean_i.dim());

    Zip::from(&mut a)
       .and(&mut b)
       .and(&mean_ip)
       .and(&mean_i)
       .and(&mean_p)
       .and(&mean_ii)
       .par_for_each(|a_val, b_val, &m_ip, &m_i, &m_p, &m_ii| {
            let cov_ip = m_ip - m_i * m_p;
            let var_i = m_ii - m_i * m_i;
            *a_val = cov_ip / (var_i + params.eps);
            *b_val = m_p - (*a_val * m_i);
        });

    let mean_a = box_filter(&a.view(), r);
    let mean_b = box_filter(&b.view(), r);

    let mean_a_up = upsample(&mean_a.view(), (h, w));
    let mean_b_up = upsample(&mean_b.view(), (h, w));

    let mut q = Array2::<f32>::zeros((h, w));
    Zip::from(&mut q)
       .and(guide)
       .and(&mean_a_up)
       .and(&mean_b_up)
       .par_for_each(|q_val, &i, &ma, &mb| {
            *q_val = ma * i + mb;
        });

    q
}

fn box_filter(input: &ArrayView2<f32>, radius: usize) -> Array2<f32> {
    let (h, w) = input.dim();
    let mut output = Array2::<f32>::zeros((h, w));
    let mut temp = Array2::<f32>::zeros((h, w));
    let r = radius as isize;

    temp.axis_iter_mut(ndarray::Axis(0)).into_par_iter().enumerate().for_each(|(y, mut row)| {
        let input_row = input.row(y);
        for x in 0..w {
            let mut sum = 0.0;
            let mut count = 0;
            for i in -r..=r {
                let idx = (x as isize) + i;
                if idx >= 0 && idx < w as isize {
                    sum += input_row[idx as usize];
                    count += 1;
                }
            }
            row[x] = sum / count as f32;
        }
    });

    output.axis_iter_mut(ndarray::Axis(1)).into_par_iter().enumerate().for_each(|(x, mut col)| {
        let temp_col = temp.column(x);
        for y in 0..h {
            let mut sum = 0.0;
            let mut count = 0;
            for i in -r..=r {
                let idx = (y as isize) + i;
                if idx >= 0 && idx < h as isize {
                    sum += temp_col[idx as usize];
                    count += 1;
                }
            }
            col[y] = sum / count as f32;
        }
    });

    output
}

fn subsample(input: &ArrayView2<f32>, ratio: usize) -> Array2<f32> {
    let (h, w) = input.dim();
    let new_h = h / ratio;
    let new_w = w / ratio;
    let mut out = Array2::<f32>::zeros((new_h, new_w));
    out.indexed_iter_mut().for_each(|((y, x), val)| {
        *val = input[[y * ratio, x * ratio]];
    });
    out
}

fn upsample(input: &ArrayView2<f32>, target_shape: (usize, usize)) -> Array2<f32> {
    let (h, w) = input.dim();
    let (target_h, target_w) = target_shape;
    let mut out = Array2::<f32>::zeros((target_h, target_w));

    out.indexed_iter_mut().par_bridge().for_each(|((y, x), val)| {
        let src_y = y as f32 / target_h as f32 * h as f32;
        let src_x = x as f32 / target_w as f32 * w as f32;
        let y0 = src_y.floor() as usize;
        let x0 = src_x.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let x1 = (x0 + 1).min(w - 1);
        let dy = src_y - y0 as f32;
        let dx = src_x - x0 as f32;
        let v00 = input[[y0, x0]];
        let v01 = input[[y0, x1]];
        let v10 = input[[y1, x0]];
        let v11 = input[[y1, x1]];
        *val = (1.0 - dy) * ((1.0 - dx) * v00 + dx * v01) + dy * ((1.0 - dx) * v10 + dx * v11);
    });
    out
}
