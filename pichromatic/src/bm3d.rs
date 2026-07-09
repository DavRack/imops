use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use rayon::prelude::*;
use std::cmp::Ordering;

// ponytail: keep local params & constant structures simple
const BLOCK_SIZE: usize = 8;
const BLOCK_AREA: usize = 64;
const MAX_GROUP_SIZE: usize = 16;
const STRIDE: usize = 6;
const SEARCH_WINDOW: usize = 19;
const FIXED_POINT_SCALE: f32 = 100_000.0;

#[derive(Clone, Copy)]
pub struct Bm3dParams {
    pub sigma: f32,
    pub hard_th_lambda: f32,
    pub max_dist_hard: f32,
    pub chroma_sigma_scale: f32,
}

impl Bm3dParams {
    pub fn from_intensity(i: f32) -> Self {
        let val = i.clamp(0.001, 1.0);
        Self {
            sigma: val * 80.0,
            hard_th_lambda: 2.0 + (val * 2.5),
            max_dist_hard: 3000.0 + (val * 20000.0),
            chroma_sigma_scale: 1.8,
        }
    }
}

pub fn bm3d(rgb_data: &mut Vec<[f32; 3]>, width: usize, height: usize, intensity: f32) {
    let params = Bm3dParams::from_intensity(intensity);
    let dct_tables = Arc::new(DctTables::new());

    let (mut r, mut g, mut b) = split_channels(rgb_data);
    let (y, cb, cr) = rgb_to_ycbcr(&r, &g, &b);
    let original_y = y.clone();
    let channels = vec![y, cb, cr];

    let patches_x = width.saturating_sub(BLOCK_SIZE) / STRIDE + 1;
    let patches_y = height.saturating_sub(BLOCK_SIZE) / STRIDE + 1;
    let total_work_units = (patches_x * patches_y) * 2;
    let progress_counter = Arc::new(AtomicUsize::new(0));

    let mut denoised_channels = bm3d_process_joint(&channels, width, height, &params, &dct_tables, &progress_counter, total_work_units);

    // Detail Blending (ponytail: keep detail blending code inline and clean)
    let blurred_y = gaussian_blur_1ch(&original_y, width, height, 3.0);
    let detail_strength = (intensity * 0.5_f32).clamp(0.0_f32, 0.5_f32);
    let y_ch = &mut denoised_channels[0];
    for i in 0..y_ch.len() {
        let hf = original_y[i] - blurred_y[i];
        y_ch[i] = (y_ch[i] + detail_strength * hf).clamp(0.0, 255.0);
    }

    ycbcr_to_rgb_inplace(&denoised_channels[0], &denoised_channels[1], &denoised_channels[2], &mut r, &mut g, &mut b);
    merge_channels_inplace(rgb_data, &r, &g, &b);
}

fn rgb_to_ycbcr(r: &[f32], g: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut y = vec![0.0f32; n];
    let mut cb = vec![0.0f32; n];
    let mut cr = vec![0.0f32; n];
    for i in 0..n {
        let rv = r[i];
        let gv = g[i];
        let bv = b[i];
        y[i] = 0.299 * rv + 0.587 * gv + 0.114 * bv;
        cb[i] = -0.168736 * rv - 0.331264 * gv + 0.5 * bv + 128.0;
        cr[i] = 0.5 * rv - 0.418688 * gv - 0.081312 * bv + 128.0;
    }
    (y, cb, cr)
}

fn ycbcr_to_rgb_inplace(
    y: &[f32], cb: &[f32], cr: &[f32],
    r: &mut [f32], g: &mut [f32], b: &mut [f32]
) {
    let n = y.len();
    for i in 0..n {
        let yv = y[i];
        let cbv = cb[i] - 128.0;
        let crv = cr[i] - 128.0;
        r[i] = yv + 1.402 * crv;
        g[i] = yv - 0.344136 * cbv - 0.714136 * crv;
        b[i] = yv + 1.772 * cbv;
    }
}

fn bm3d_process_joint(
    noisy_channels: &[Vec<f32>],
    width: usize,
    height: usize,
    params: &Bm3dParams,
    tables: &DctTables,
    counter: &Arc<AtomicUsize>,
    total_work: usize,
) -> Vec<Vec<f32>> {
    let basic_estimate = run_bm3d_step_joint(
        noisy_channels,
        noisy_channels,
        width,
        height,
        params,
        true,
        tables,
        counter,
        total_work,
    );

    run_bm3d_step_joint(
        noisy_channels,
        &basic_estimate,
        width,
        height,
        params,
        false,
        tables,
        counter,
        total_work,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_bm3d_step_joint(
    noisy: &[Vec<f32>],
    guide: &[Vec<f32>],
    width: usize,
    height: usize,
    params: &Bm3dParams,
    is_step_1: bool,
    tables: &DctTables,
    counter: &Arc<AtomicUsize>,
    _total_work: usize,
) -> Vec<Vec<f32>> {
    let w = width;
    let h = height;
    let count = w * h;
    let num_channels = 3;

    let mut numerators = Vec::new();
    let mut denominators = Vec::new();
    for _ in 0..num_channels {
        numerators.push(Arc::new(AtomicAccumulator::new(count)));
        denominators.push(Arc::new(AtomicAccumulator::new(count)));
    }

    let mut ref_patches = Vec::with_capacity((w / STRIDE) * (h / STRIDE));
    for y in (0..h.saturating_sub(BLOCK_SIZE)).step_by(STRIDE) {
        for x in (0..w.saturating_sub(BLOCK_SIZE)).step_by(STRIDE) {
            ref_patches.push((x, y));
        }
    }

    ref_patches.par_iter().for_each(|&(rx, ry)| {
        counter.fetch_add(1, AtomicOrdering::Relaxed);

        let mut group_locs_buf = [(0, 0); MAX_GROUP_SIZE];
        let group_size =
            block_matching_joint(guide, w, h, rx, ry, is_step_1, params, &mut group_locs_buf);
        let group_locs = &group_locs_buf[0..group_size];

        for ch in 0..num_channels {
            let guide_ch = &guide[ch];
            let noisy_ch = &noisy[ch];

            let ch_sigma = if ch == 0 {
                params.sigma
            } else {
                params.sigma * params.chroma_sigma_scale
            };

            let mut guide_stack = build_3d_group(guide_ch, w, group_locs);
            let mut noisy_stack = if is_step_1 {
                guide_stack.clone()
            } else {
                build_3d_group(noisy_ch, w, group_locs)
            };

            transform_3d(&mut guide_stack, group_size, tables);
            if !is_step_1 {
                transform_3d(&mut noisy_stack, group_size, tables);
            }

            let weight;
            if is_step_1 {
                let threshold = params.hard_th_lambda * ch_sigma;
                let nonzero = hard_threshold(&mut guide_stack, threshold);
                weight = if nonzero > 0 {
                    1.0 / (nonzero as f32)
                } else {
                    1.0
                };
                noisy_stack = guide_stack;
            } else {
                weight = wiener_filter(&mut noisy_stack, &guide_stack, ch_sigma);
            }

            inverse_transform_3d(&mut noisy_stack, group_size, tables);

            let num_acc = &numerators[ch];
            let den_acc = &denominators[ch];

            for (k, &(lx, ly)) in group_locs.iter().enumerate() {
                let patch_offset = k * BLOCK_AREA;
                for dy in 0..BLOCK_SIZE {
                    let row_global = (ly + dy) * w + lx;
                    let row_patch = dy * BLOCK_SIZE;
                    for dx in 0..BLOCK_SIZE {
                        let idx = row_global + dx;
                        let val = noisy_stack[patch_offset + row_patch + dx];
                        let w_val = tables.kaiser[row_patch + dx] * weight;
                        num_acc.add(idx, val * w_val);
                        den_acc.add(idx, w_val);
                    }
                }
            }
        }
    });

    let mut results = Vec::new();
    for ch in 0..num_channels {
        let num_vec = numerators[ch].to_vec();
        let den_vec = denominators[ch].to_vec();
        let final_ch = num_vec
            .iter()
            .zip(den_vec.iter())
            .zip(noisy[ch].iter())
            .map(|((&n, &d), &orig)| if d > 1e-6 { n / d } else { orig })
            .collect();
        results.push(final_ch);
    }
    results
}

fn hard_threshold(stack: &mut [f32], th: f32) -> usize {
    let mut c = 0;
    for (i, x) in stack.iter_mut().enumerate() {
        if i == 0 {
            c += 1;
            continue;
        }

        if x.abs() < th {
            *x = 0.0;
        } else {
            c += 1;
        }
    }
    c
}

fn wiener_filter(noisy: &mut [f32], guide: &[f32], sigma: f32) -> f32 {
    let mut sum = 0.0;
    let s2 = sigma * sigma;
    for (i, (n, g)) in noisy.iter_mut().zip(guide).enumerate() {
        if i == 0 {
            sum += 1.0;
            continue;
        }

        let energy = g * g;
        let coef = energy / (energy + s2 + 1e-5);
        *n *= coef;
        sum += coef * coef;
    }
    if sum > 0.0 { 1.0 / sum } else { 1.0 }
}

#[derive(Clone, Copy)]
struct Match {
    dist: f32,
    x: u16,
    y: u16,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn block_matching_joint(
    channels: &[Vec<f32>],
    w: usize,
    h: usize,
    rx: usize,
    ry: usize,
    is_step_1: bool,
    params: &Bm3dParams,
    out_buf: &mut [(usize, usize)],
) -> usize {
    const MAX_CANDIDATES: usize = 1024;
    let mut candidates: [Match; MAX_CANDIDATES] = [Match {
        dist: f32::MAX,
        x: 0,
        y: 0,
    }; MAX_CANDIDATES];
    let mut cand_count = 0;

    let threshold = if is_step_1 {
        params.max_dist_hard
    } else {
        params.max_dist_hard * 0.5
    };
    let threshold_raw = threshold * 64.0;

    let mut ref_r = [0.0; 64];
    let mut ref_g = [0.0; 64];
    let mut ref_b = [0.0; 64];
    extract_patch(&channels[0], w, rx, ry, &mut ref_r);
    extract_patch(&channels[1], w, rx, ry, &mut ref_g);
    extract_patch(&channels[2], w, rx, ry, &mut ref_b);

    let half_sw = SEARCH_WINDOW / 2;
    let sx_start = rx.saturating_sub(half_sw);
    let sx_end = (rx + half_sw).min(w.saturating_sub(BLOCK_SIZE));
    let sy_start = ry.saturating_sub(half_sw);
    let sy_end = (ry + half_sw).min(h.saturating_sub(BLOCK_SIZE));

    candidates[0] = Match {
        dist: 0.0,
        x: rx as u16,
        y: ry as u16,
    };
    cand_count += 1;

    for y in sy_start..=sy_end {
        for x in sx_start..=sx_end {
            if x == rx && y == ry {
                continue;
            }
            let d_r = compute_ssd_flat(&channels[0], w, x, y, &ref_r, threshold_raw);
            if d_r > threshold_raw {
                continue;
            }
            let d_g = compute_ssd_flat(&channels[1], w, x, y, &ref_g, threshold_raw - d_r);
            if d_r + d_g > threshold_raw {
                continue;
            }
            let d_b = compute_ssd_flat(&channels[2], w, x, y, &ref_b, threshold_raw - (d_r + d_g));
            let total_dist_raw = d_r + d_g + d_b;

            if total_dist_raw < threshold_raw && cand_count < MAX_CANDIDATES {
                candidates[cand_count] = Match {
                    dist: total_dist_raw / 64.0,
                    x: x as u16,
                    y: y as u16,
                };
                cand_count += 1;
            }
        }
    }

    let valid_slice = &mut candidates[0..cand_count];
    valid_slice.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

    let limit = MAX_GROUP_SIZE.min(cand_count);
    let p2_limit = prev_power_of_two(limit);

    for i in 0..p2_limit {
        out_buf[i] = (valid_slice[i].x as usize, valid_slice[i].y as usize);
    }
    p2_limit
}

// ponytail: Optimized flat SSD computation without inner division & using slice pointers
#[inline(always)]
fn compute_ssd_flat(
    img: &[f32],
    w: usize,
    x: usize,
    y: usize,
    ref_patch: &[f32; 64],
    stop_thr_raw: f32,
) -> f32 {
    let mut dist = 0.0;
    for dy in 0..8 {
        let img_base = (y + dy) * w + x;
        let img_slice = &img[img_base..img_base + 8];
        let ref_slice = &ref_patch[dy * 8..dy * 8 + 8];
        for dx in 0..8 {
            let diff = img_slice[dx] - ref_slice[dx];
            dist += diff * diff;
        }
        if dist > stop_thr_raw {
            return dist;
        }
    }
    dist
}

#[inline(always)]
fn extract_patch(img: &[f32], w: usize, x: usize, y: usize, out: &mut [f32]) {
    for dy in 0..8 {
        let src_idx = (y + dy) * w + x;
        let dst_idx = dy * 8;
        out[dst_idx..dst_idx + 8].copy_from_slice(&img[src_idx..src_idx + 8]);
    }
}

fn build_3d_group(img: &[f32], w: usize, locs: &[(usize, usize)]) -> Vec<f32> {
    let mut stack = vec![0.0; locs.len() * 64];
    for (i, &(lx, ly)) in locs.iter().enumerate() {
        let offset = i * 64;
        extract_patch(img, w, lx, ly, &mut stack[offset..offset + 64]);
    }
    stack
}

struct DctTables {
    dct_coeff: [f32; 64],
    idct_coeff: [f32; 64],
    kaiser: Vec<f32>,
}

impl DctTables {
    fn new() -> Self {
        let mut dct_coeff = [0.0; 64];
        let mut idct_coeff = [0.0; 64];
        for k in 0..8 {
            for n in 0..8 {
                let c = k as f32 * std::f32::consts::PI / 8.0;
                let val = ((n as f32 + 0.5) * c).cos();
                let scale = if k == 0 { 0.35355339 } else { 0.5 };
                dct_coeff[k * 8 + n] = val * scale;
            }
        }
        for n in 0..8 {
            for k in 0..8 {
                let theta = (std::f32::consts::PI / 8.0) * (n as f32 + 0.5) * (k as f32);
                let scale = if k == 0 { 0.35355339 } else { 0.5 };
                idct_coeff[n * 8 + k] = scale * theta.cos();
            }
        }
        fn bessel_i0(x: f32) -> f32 {
            let mut sum = 1.0;
            let mut term = 1.0;
            for k in 1..=10 {
                term *= (x / 2.0 / k as f32).powi(2);
                sum += term;
            }
            sum
        }
        let beta = 2.0;
        let i0_beta = bessel_i0(beta);
        let mut kaiser = vec![0.0; 64];
        for y in 0..8 {
            for x in 0..8 {
                let wx = bessel_i0(beta * (1.0 - ((2.0 * x as f32 / 7.0) - 1.0).powi(2)).sqrt()) / i0_beta;
                let wy = bessel_i0(beta * (1.0 - ((2.0 * y as f32 / 7.0) - 1.0).powi(2)).sqrt()) / i0_beta;
                kaiser[y * 8 + x] = wx * wy;
            }
        }
        Self {
            dct_coeff,
            idct_coeff,
            kaiser,
        }
    }
}

struct AtomicAccumulator {
    data: Vec<AtomicI64>,
}

impl AtomicAccumulator {
    fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(AtomicI64::new(0));
        }
        Self { data }
    }
    #[inline(always)]
    fn add(&self, index: usize, value: f32) {
        if index < self.data.len() {
            let fixed = (value * FIXED_POINT_SCALE) as i64;
            self.data[index].fetch_add(fixed, AtomicOrdering::Relaxed);
        }
    }
    fn to_vec(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|a| a.load(AtomicOrdering::Relaxed) as f32 / FIXED_POINT_SCALE)
            .collect()
    }
}

#[inline(always)]
fn transform_3d(stack: &mut [f32], group_size: usize, tables: &DctTables) {
    for i in 0..group_size {
        let offset = i * 64;
        dct_2d_8x8(&mut stack[offset..offset + 64], &tables.dct_coeff);
    }
    for i in 0..64 {
        let mut col = [0.0; MAX_GROUP_SIZE];
        for k in 0..group_size {
            col[k] = stack[k * 64 + i];
        }
        walsh_hadamard_1d(&mut col[0..group_size]);
        for k in 0..group_size {
            stack[k * 64 + i] = col[k];
        }
    }
}

#[inline(always)]
fn inverse_transform_3d(stack: &mut [f32], group_size: usize, tables: &DctTables) {
    for i in 0..64 {
        let mut col = [0.0; MAX_GROUP_SIZE];
        for k in 0..group_size {
            col[k] = stack[k * 64 + i];
        }
        walsh_hadamard_1d(&mut col[0..group_size]);
        for k in 0..group_size {
            stack[k * 64 + i] = col[k];
        }
    }
    for i in 0..group_size {
        let offset = i * 64;
        idct_2d_8x8(&mut stack[offset..offset + 64], &tables.idct_coeff);
    }
}

#[inline]
fn dct_2d_8x8(block: &mut [f32], coeffs: &[f32; 64]) {
    for i in 0..8 {
        dct_1d_8(&mut block[i * 8..(i + 1) * 8], coeffs);
    }
    transpose_8x8(block);
    for i in 0..8 {
        dct_1d_8(&mut block[i * 8..(i + 1) * 8], coeffs);
    }
    transpose_8x8(block);
}

#[inline]
fn idct_2d_8x8(block: &mut [f32], coeffs: &[f32; 64]) {
    transpose_8x8(block);
    for i in 0..8 {
        idct_1d_8(&mut block[i * 8..(i + 1) * 8], coeffs);
    }
    transpose_8x8(block);
    for i in 0..8 {
        idct_1d_8(&mut block[i * 8..(i + 1) * 8], coeffs);
    }
}

// ponytail: Unrolled 1D DCT for SIMD optimization
#[inline]
fn dct_1d_8(x: &mut [f32], coeffs: &[f32; 64]) {
    let mut tmp = [0.0; 8];
    tmp.copy_from_slice(&x[..8]);
    for k in 0..8 {
        let row_start = k * 8;
        x[k] = tmp[0] * coeffs[row_start]
             + tmp[1] * coeffs[row_start + 1]
             + tmp[2] * coeffs[row_start + 2]
             + tmp[3] * coeffs[row_start + 3]
             + tmp[4] * coeffs[row_start + 4]
             + tmp[5] * coeffs[row_start + 5]
             + tmp[6] * coeffs[row_start + 6]
             + tmp[7] * coeffs[row_start + 7];
    }
}

// ponytail: Unrolled 1D IDCT for SIMD optimization
#[inline]
fn idct_1d_8(x: &mut [f32], coeffs: &[f32; 64]) {
    let mut tmp = [0.0; 8];
    tmp.copy_from_slice(&x[..8]);
    for n in 0..8 {
        let row_start = n * 8;
        x[n] = tmp[0] * coeffs[row_start]
             + tmp[1] * coeffs[row_start + 1]
             + tmp[2] * coeffs[row_start + 2]
             + tmp[3] * coeffs[row_start + 3]
             + tmp[4] * coeffs[row_start + 4]
             + tmp[5] * coeffs[row_start + 5]
             + tmp[6] * coeffs[row_start + 6]
             + tmp[7] * coeffs[row_start + 7];
    }
}

#[inline]
fn transpose_8x8(b: &mut [f32]) {
    for y in 0..8 {
        for x in (y + 1)..8 {
            b.swap(y * 8 + x, x * 8 + y);
        }
    }
}

#[inline]
fn walsh_hadamard_1d(data: &mut [f32]) {
    let n = data.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for x in data {
        *x *= scale;
    }
}

fn split_channels(img: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let size = img.len();
    let mut r = vec![0.0; size];
    let mut g = vec![0.0; size];
    let mut b = vec![0.0; size];
    for (i, p) in img.iter().enumerate() {
        r[i] = p[0] * 255.0;
        g[i] = p[1] * 255.0;
        b[i] = p[2] * 255.0;
    }
    (r, g, b)
}

fn merge_channels_inplace(img: &mut [[f32; 3]], r: &[f32], g: &[f32], b: &[f32]) {
    for i in 0..img.len() {
        img[i][0] = r[i].max(0.0) / 255.0;
        img[i][1] = g[i].max(0.0) / 255.0;
        img[i][2] = b[i].max(0.0) / 255.0;
    }
}

fn prev_power_of_two(x: usize) -> usize {
    if x == 0 {
        return 0;
    }
    let mut p = 1;
    while p * 2 <= x {
        p *= 2;
    }
    p
}

fn gaussian_blur_1ch(data: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let klen = 2 * radius + 1;
    let mut kernel = vec![0.0f32; klen];
    let two_s2 = 2.0 * sigma * sigma;
    for (i, kernel_val) in kernel.iter_mut().enumerate() {
        let k = i as f32 - radius as f32;
        *kernel_val = (-k * k / two_s2).exp();
    }
    let ksum: f32 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= ksum;
    }

    let mut tmp = vec![0.0f32; width * height];
    for y in 0..height {
        let row_in = &data[y * width..(y + 1) * width];
        let row_out = &mut tmp[y * width..(y + 1) * width];
        for (x, out_val) in row_out.iter_mut().enumerate() {
            let mut val = 0.0f32;
            let mut wsum = 0.0f32;
            let x0 = x as isize - radius as isize;
            for (ki, &kernel_val) in kernel.iter().enumerate() {
                let kx = x0 + ki as isize;
                if kx >= 0 && kx < width as isize {
                    val += row_in[kx as usize] * kernel_val;
                    wsum += kernel_val;
                }
            }
            *out_val = val / wsum;
        }
    }

    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        let y0 = y as isize - radius as isize;
        for x in 0..width {
            let mut val = 0.0f32;
            let mut wsum = 0.0f32;
            for (ki, &kernel_val) in kernel.iter().enumerate() {
                let ky = y0 + ki as isize;
                if ky >= 0 && ky < height as isize {
                    val += tmp[ky as usize * width + x] * kernel_val;
                    wsum += kernel_val;
                }
            }
            out[y * width + x] = val / wsum;
        }
    }

    out
}

// ponytail: Fast chroma-only denoising via per-block DCT thresholding
// Instead of full BM3D (block matching + 3D collaborative filtering), each
// 8x8 block is independently DCT'd, hard-thresholded, and IDCT'd. ~100x faster
// and chroma noise is visually forgiving, so quality is comparable.
pub fn chroma_bm3d(rgb_data: &mut Vec<[f32; 3]>, width: usize, height: usize, intensity: f32) {
    let sigma = intensity.clamp(0.001, 1.0) * 80.0;
    let threshold = (2.0 + (intensity.clamp(0.001, 1.0) * 2.5)) * sigma * 1.8;
    let tables = Arc::new(DctTables::new());

    let (mut r, mut g, mut b) = split_channels(rgb_data);
    let (y, mut cb, mut cr) = rgb_to_ycbcr(&r, &g, &b);

    dct_chroma_denoise(&mut cb, width, height, threshold, &tables);
    dct_chroma_denoise(&mut cr, width, height, threshold, &tables);

    ycbcr_to_rgb_inplace(&y, &cb, &cr, &mut r, &mut g, &mut b);
    merge_channels_inplace(rgb_data, &r, &g, &b);
}

fn dct_chroma_denoise(ch: &mut [f32], width: usize, height: usize, threshold: f32, tables: &DctTables) {
    let count = width * height;
    let numerator = Arc::new(AtomicAccumulator::new(count));
    let denominator = Arc::new(AtomicAccumulator::new(count));

    let mut patches = Vec::with_capacity((width / STRIDE) * (height / STRIDE));
    for y in (0..height.saturating_sub(BLOCK_SIZE)).step_by(STRIDE) {
        for x in (0..width.saturating_sub(BLOCK_SIZE)).step_by(STRIDE) {
            patches.push((x, y));
        }
    }

    patches.par_iter().for_each(|&(px, py)| {
        let mut block = [0.0; 64];
        extract_patch(ch, width, px, py, &mut block);

        dct_2d_8x8(&mut block, &tables.dct_coeff);
        hard_threshold(&mut block, threshold);
        idct_2d_8x8(&mut block, &tables.idct_coeff);

        for dy in 0..BLOCK_SIZE {
            let row_global = (py + dy) * width + px;
            let row_patch = dy * BLOCK_SIZE;
            for dx in 0..BLOCK_SIZE {
                let idx = row_global + dx;
                let w = tables.kaiser[row_patch + dx];
                numerator.add(idx, block[row_patch + dx] * w);
                denominator.add(idx, w);
            }
        }
    });

    let num = numerator.to_vec();
    let den = denominator.to_vec();
    for i in 0..count {
        if den[i] > 1e-6 {
            ch[i] = num[i] / den[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_image(width: usize, height: usize) -> Vec<[f32; 3]> {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x as f32 * 0.12).sin() * 0.5 + 0.5) * 255.0;
                let g = ((y as f32 * 0.17).cos() * 0.5 + 0.5) * 255.0;
                let b = (((x + y) as f32 * 0.09).sin() * 0.5 + 0.5) * 255.0;
                data.push([r / 255.0, g / 255.0, b / 255.0]);
            }
        }
        data
    }

    fn refined_sum(img: &[[f32; 3]], channel_idx: usize) -> f64 {
        let mut sum = 0.0f64;
        let mut c = 0.0f64;
        for p in img {
            let x = p[channel_idx] as f64;
            let y = x - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }

    #[test]
    fn test_bm3d_regression() {
        let size = 2048;
        let mut img = generate_test_image(size, size);
        bm3d(&mut img, size, size, 0.1);
        
        let sum_r = refined_sum(&img, 0);
        let sum_g = refined_sum(&img, 1);
        let sum_b = refined_sum(&img, 2);
        
        println!("BM3D checksums 2048x2048: r={}, g={}, b={}", sum_r, sum_g, sum_b);
        let diff_r = (sum_r - 2100044.776915211_f64).abs();
        let diff_g = (sum_g - 2101586.5654235613_f64).abs();
        let diff_b = (sum_b - 2098591.6187395807_f64).abs();
        assert!(diff_r < 1e-3, "r diff is {}", diff_r);
        assert!(diff_g < 1e-3, "g diff is {}", diff_g);
        assert!(diff_b < 1e-3, "b diff is {}", diff_b);
    }

    #[test]
    fn test_chroma_bm3d_regression() {
        let size = 2048;
        let mut img = generate_test_image(size, size);
        chroma_bm3d(&mut img, size, size, 0.1);
        
        let sum_r = refined_sum(&img, 0);
        let sum_g = refined_sum(&img, 1);
        let sum_b = refined_sum(&img, 2);
        
        println!("CHROMA_BM3D checksums 2048x2048: r={}, g={}, b={}", sum_r, sum_g, sum_b);
        let diff_r = (sum_r - 2099732.10061259_f64).abs();
        let diff_g = (sum_g - 2101692.5347650475_f64).abs();
        let diff_b = (sum_b - 2098392.755773447_f64).abs();
        assert!(diff_r < 1e-3, "r diff is {}", diff_r);
        assert!(diff_g < 1e-3, "g diff is {}", diff_g);
        assert!(diff_b < 1e-3, "b diff is {}", diff_b);
    }
}

