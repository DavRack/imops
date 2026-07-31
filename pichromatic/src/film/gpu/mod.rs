//! Native wgpu GPU film simulation.
//!
//! Mirrors [`crate::film::process`] entirely on GPU storage buffers
//! (`array<vec4<f32>>` RGBA image + planar `array<f32>` layer buffers). No CPU
//! download→process→upload: every spatial stage (expose, halation, DIR,
//! adjacency, grain, scan) runs as compute passes on the GPU.
//!
//! Stock calibration scalars/LUTs/spectra (which are functions of the *stock*,
//! not the image) are precomputed on the CPU at load — exactly as the CPU path
//! does — and uploaded as constant storage buffers. Grain variance uses a GPU
//! partial sum-of-squares (`GRAIN_VAR_PARTIAL`) then a tiny download; the CPU
//! finishes the `f64` reduction for `norm` so it matches the CPU path bit-near.
//! All spatial grain work stays on GPU.

mod roi;
#[doc(hidden)]
pub mod shaders;
mod workspace;

pub(crate) use workspace::acquire_film_resources;

use bytemuck::{Pod, Zeroable};
use wgpu::Buffer;

use crate::film::constants::{
    ABSORPTION_SIGMA_SCALE_PER_UM, DYE_CLOUD_CORRELATION_UM, LOCAL_SCATTER_MIX,
    MASK_DENSITY_FRACTION_OF_DMAX,
};
use crate::film::development::grain::scale_kappa;
use crate::film::exposure::halation::{
    bleed_weights_for_layers, effective_reflectance, reflectance_at, sigma_px_from_um,
};
use crate::film::exposure::upsample::{
    CIE1931_XBAR, CIE1931_YBAR, CIE1931_ZBAR, XYZ_D65_TO_ACESCG,
};
use crate::film::scan::densitometry::dmin_reference_acescg;
use crate::film::stock::{EmulsionLayer, FilmStock, LayerKind};
use crate::film::{FilmError, FilmOutput, FilmParams};
use crate::gpu::{ComputePassDesc, GpuContext, GpuImageBuffer};
use color::ColorSpaceTag;

/// Max FIR radius for tiled blur shaders (`tile[512]` = 256 + 2×128).
/// Larger radii fall back to untiled [`shaders::BLUR_H`] / [`shaders::BLUR_V`].
pub(crate) const BLUR_TILED_MAX_RADIUS: u32 = 128;

/// Partial outputs per emulsion for grain variance reduce (allocation + dispatch).
pub(crate) const GRAIN_VAR_PARTIALS_PER: usize = 2048;

// ─── Uniform structs ────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExposeU {
    width: u32,
    height: u32,
    n: u32,
    num_layers: u32,
    num_emul: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurU {
    width: u32,
    height: u32,
    n: u32,
    radius: u32,
    src_off: u32,
    dst_off: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MixU {
    n: u32,
    off: u32,
    keep: f32,
    f: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CountU {
    n: u32,
    num_emul: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdjU {
    n: u32,
    off: u32,
    beta: f32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NoiseU {
    width: u32,
    height: u32,
    n: u32,
    base_lo: u32,
    base_hi: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrainApplyU {
    n: u32,
    off: u32,
    kappa: f32,
    dmax: f32,
    norm: f32,
    noise_off: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VarPartialU {
    n: u32,
    stride: u32,
    out_n: u32,
    src_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScanU {
    n: u32,
    num_emul: u32,
    scale: f32,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExposeRoiU {
    root_x: i32,
    root_y: i32,
    root_w: u32,
    root_h: u32,
    img_w: u32,
    img_h: u32,
    root_n: u32,
    planes_base: u32,
    num_layers: u32,
    num_emul: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MixRoiU {
    n: u32,
    plane_off: u32,
    blur_off: u32,
    keep: f32,
    f: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HalationAddRoiU {
    n: u32,
    num_emul: u32,
    plane_base: u32,
    bounce_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReduceRoiU {
    n: u32,
    num_emul: u32,
    plane_base: u32,
    dye_base: u32,
    mask_base: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DirApplyRoiU {
    n: u32,
    num_emul: u32,
    dye_base: u32,
    work_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdjacencyRoiU {
    n: u32,
    dye_off: u32,
    blur_off: u32,
    beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NoiseRoiU {
    root_x: i32,
    root_y: i32,
    root_w: u32,
    root_h: u32,
    root_n: u32,
    img_w: u32,
    img_h: u32,
    base_lo: u32,
    base_hi: u32,
    dst_off: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CopyScalarCoreRoiU {
    core_x: u32,
    core_y: u32,
    core_w: u32,
    core_h: u32,
    core_n: u32,
    root_w: u32,
    root_off_x: u32,
    root_off_y: u32,
    img_w: u32,
    src_off: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrainScanRoiU {
    core_x: u32,
    core_y: u32,
    core_w: u32,
    core_h: u32,
    core_n: u32,
    root_w: u32,
    root_off_x: u32,
    root_off_y: u32,
    root_n: u32,
    img_w: u32,
    dye_base: u32,
    mask_base: u32,
    num_emul: u32,
    scale: f32,
    noise_base: u32,
    flags: u32,
    emul: [[f32; 4]; 16],
}

// ─── CPU-side spectral/upsample helpers (f64), mirroring exposure::upsample ──

fn gaussian_basis(peak_nm: f64, sigma_nm: f64) -> [f64; 16] {
    let mut s = [0.0f64; 16];
    for (i, slot) in s.iter_mut().enumerate() {
        let lambda = 400.0 + 20.0 * i as f64;
        let d = (lambda - peak_nm) / sigma_nm;
        *slot = (-0.5 * d * d).exp();
    }
    s
}

fn integrate_cmf(spectrum: &[f64; 16], cmf: &[f64; 16]) -> f64 {
    let dlambda = 20.0;
    let mut acc = 0.0;
    for i in 0..15 {
        let a = spectrum[i] * cmf[i];
        let b = spectrum[i + 1] * cmf[i + 1];
        acc += 0.5 * (a + b) * dlambda;
    }
    acc
}

fn matvec3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn spectrum_to_acescg64(spectrum: &[f64; 16]) -> [f64; 3] {
    let xyz = [
        integrate_cmf(spectrum, &CIE1931_XBAR),
        integrate_cmf(spectrum, &CIE1931_YBAR),
        integrate_cmf(spectrum, &CIE1931_ZBAR),
    ];
    matvec3(XYZ_D65_TO_ACESCG, xyz)
}

fn inv3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let id = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
        ],
    ]
}

/// Gaussian kernel identical to `blur::make_gaussian_kernel` (f32, radius ⌈3σ⌉).
pub(super) fn make_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = crate::film::blur::gaussian_radius(sigma);
    let mut k = vec![0.0f32; 2 * radius + 1];
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 0.0f32;
    for (i, slot) in k.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        let v = (-x * x * inv_2s2).exp();
        *slot = v;
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

// ─── Baked constants ────────────────────────────────────────────────────────

/// All per-stock constant buffers, uploaded once.
pub(crate) struct StockConsts {
    expose: Buffer,
    lut: Buffer,
    reduce: Buffer,
    dir_matrix: Buffer,
    scan: Buffer,
    gains: Buffer,
    num_layers: u32,
    num_emul: u32,
    scan_scale: f32,
    do_invert: bool,
    // Per-emulsion runtime scalars.
    kappa: Vec<f32>,
    dmax: Vec<f32>,
    // Blur sigmas (px) for spatial stages.
    sigma_local: f32,
    sigma_wide: f32,
    sigma_dir: f32,
    sigma_adj: f32,
    // Pre-baked Gaussian kernel buffers (None if stage skipped).
    local_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    wide_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    dir_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    adj_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    grain_kernels: Vec<(std::sync::Arc<Buffer>, u32)>,
    adjacency_beta: f32,
    grain_seed_base: Vec<(u32, u32)>, // per emulsion (lo, hi) of seed*C1 + layer*GOLDEN
}

const GOLDEN: u64 = 0x9E3779B97F4A7C15;
const SM_STATE_MIX: u64 = 0xD1B54A32D192ED03;

pub(crate) fn bake_consts(
    ctx: &GpuContext,
    stock: &FilmStock,
    params: &FilmParams,
    meta: &crate::image::ImageMetadata,
    width: usize,
) -> StockConsts {
    let num_layers = stock.layers.len();
    let emuls: Vec<(usize, &EmulsionLayer)> = stock.emulsion_layers().collect();
    let num_emul = emuls.len();

    // Upsample basis + acescg→weights (matches exposure::upsample::UpsampleBasis).
    let b0 = gaussian_basis(450.0, 40.0);
    let b1 = gaussian_basis(550.0, 40.0);
    let b2 = gaussian_basis(650.0, 40.0);
    let c0 = spectrum_to_acescg64(&b0);
    let c1 = spectrum_to_acescg64(&b1);
    let c2 = spectrum_to_acescg64(&b2);
    let m = [
        [c0[0], c1[0], c2[0]],
        [c0[1], c1[1], c2[1]],
        [c0[2], c1[2], c2[2]],
    ];
    let a2w = inv3(m); // acescg → weights (row-major)

    // ── expose consts ──
    // Layout: b0(16) b1(16) b2(16) M(9) lambda_factor(16) capture_scale od(L*16) produces_latent(L)
    let mut expose: Vec<f32> = Vec::new();
    for &v in &b0 {
        expose.push(v as f32);
    }
    for &v in &b1 {
        expose.push(v as f32);
    }
    for &v in &b2 {
        expose.push(v as f32);
    }
    for r in 0..3 {
        for c in 0..3 {
            expose.push(a2w[r][c] as f32);
        }
    }
    for i in 0..16 {
        let lambda = 400.0 + 20.0 * i as f64;
        expose.push(((lambda / 550.0) * crate::film::constants::RADIOMETRIC_SCALE) as f32);
    }
    expose.push(crate::film::camera_capture_scale(meta, stock.box_iso.0));
    let sigma_scale = ABSORPTION_SIGMA_SCALE_PER_UM;
    for layer in &stock.layers {
        let mut od = [0.0f64; 16];
        match layer.kind {
            LayerKind::Emulsion => {
                let sens = layer.spectral_sensitivity.as_ref().unwrap();
                let rho = layer.silver_halide_fraction as f64;
                let thickness = layer.thickness.0 as f64;
                for i in 0..16 {
                    od[i] = sens.samples[i] * sigma_scale * rho * thickness;
                }
            }
            LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                if let Some(curve) = layer.spectral_sensitivity.as_ref() {
                    let thickness = layer.thickness.0 as f64;
                    for i in 0..16 {
                        od[i] = curve.samples[i] * thickness;
                    }
                }
            }
            LayerKind::Support => {}
        }
        for i in 0..16 {
            expose.push(od[i] as f32);
        }
    }
    for layer in &stock.layers {
        expose.push(if layer.kind == LayerKind::Emulsion {
            1.0
        } else {
            0.0
        });
    }

    // ── lut consts ── logs(64) then frac(E*64)
    let mut lut: Vec<f32> = Vec::new();
    // logs from any emulsion LUT (all share the same grid).
    let first_emul_idx = emuls[0].0;
    let first_lut = stock.capture_luts[first_emul_idx].as_ref().unwrap();
    for &v in &first_lut.log10_fluence {
        lut.push(v as f32);
    }
    for &(li, _) in &emuls {
        let l = stock.capture_luts[li].as_ref().unwrap();
        for &v in &l.fraction {
            lut.push(v as f32);
        }
    }

    // ── reduce consts ── dmax(E) inv_gamma(E) mask_scale(E) reversal(E) has_mask(E)
    let mut dmax_v = Vec::with_capacity(num_emul);
    let mut inv_gamma_v = Vec::with_capacity(num_emul);
    let mut mask_scale_v = Vec::with_capacity(num_emul);
    let mut reversal_v = Vec::with_capacity(num_emul);
    let mut has_mask_v = Vec::with_capacity(num_emul);
    for &(_, layer) in &emuls {
        let coupler = layer.coupler.as_ref().unwrap();
        dmax_v.push(coupler.d_max);
        inv_gamma_v.push(1.0 / layer.gamma_contrast.max(1e-6));
        mask_scale_v.push(coupler.d_max * MASK_DENSITY_FRACTION_OF_DMAX);
        reversal_v.push(if layer.is_reversal { 1.0f32 } else { 0.0 });
        has_mask_v.push(if coupler.mask_epsilon.is_some() {
            1.0f32
        } else {
            0.0
        });
    }
    let mut reduce: Vec<f32> = Vec::new();
    reduce.extend_from_slice(&dmax_v);
    reduce.extend_from_slice(&inv_gamma_v);
    reduce.extend_from_slice(&mask_scale_v);
    reduce.extend_from_slice(&reversal_v);
    reduce.extend_from_slice(&has_mask_v);

    // ── DIR matrix (E*E row-major matrix[i][j]) ──
    let mut dir_matrix: Vec<f32> = vec![0.0; num_emul * num_emul];
    for i in 0..num_emul {
        for j in 0..num_emul {
            let w = stock
                .dir_inhibition_matrix
                .get(i)
                .and_then(|row| row.get(j))
                .copied()
                .unwrap_or(0.0);
            dir_matrix[i * num_emul + j] = w;
        }
    }

    // ── scan consts ── eps(E*16) maskeps(E*16) illum(16) xbar(16) ybar(16) zbar(16) matrix(9) [invert_consts(16)]
    let mut scan: Vec<f32> = Vec::new();
    for &(_, layer) in &emuls {
        let coupler = layer.coupler.as_ref().unwrap();
        for i in 0..16 {
            scan.push(coupler.epsilon.samples[i] as f32);
        }
    }
    for &(_, layer) in &emuls {
        let coupler = layer.coupler.as_ref().unwrap();
        for i in 0..16 {
            let v = coupler
                .mask_epsilon
                .as_ref()
                .map(|c| c.samples[i])
                .unwrap_or(0.0);
            scan.push(v as f32);
        }
    }
    for i in 0..16 {
        scan.push(stock.scanner_light.samples[i] as f32);
    }
    for i in 0..16 {
        scan.push(CIE1931_XBAR[i] as f32);
    }
    for i in 0..16 {
        scan.push(CIE1931_YBAR[i] as f32);
    }
    for i in 0..16 {
        scan.push(CIE1931_ZBAR[i] as f32);
    }
    for r in 0..3 {
        for c in 0..3 {
            scan.push(XYZ_D65_TO_ACESCG[r][c] as f32);
        }
    }

    // Dmin normalization scale (matches scan_to_acescg).
    let dmin_rgb = dmin_reference_acescg(stock);
    let peak = dmin_rgb[0].max(dmin_rgb[1]).max(dmin_rgb[2]).max(1e-12);
    let scan_scale = (1.0 / peak) as f32;

    // Optional PositiveLinear invert pre-baking.
    let is_reversal = stock.layers.iter().any(|l| l.is_reversal);
    let do_invert = !is_reversal && params.output == FilmOutput::PositiveLinear;
    if do_invert {
        let pitch_um = params.film_format.pixel_pitch_um(width);
        let shutter = meta.shutter_seconds.unwrap_or(1.0 / stock.box_iso.0);
        let dmin = crate::film::scan::normalized_dmin_acescg(stock);
        let mid = crate::film::mid_negative_acescg(stock, pitch_um, shutter);
        let invert = crate::film::scan::invert::invert_constants(mid, dmin);

        scan.extend_from_slice(&[
            invert.dmin[0],
            invert.dmin[1],
            invert.dmin[2],
            invert.gain[0],
            invert.gain[1],
            invert.gain[2],
            invert.slope,
            invert.gamma_eff,
            invert.eps,
            invert.fog_offset,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
    }

    // ── halation gains (r_e * bleed[e]) ──
    let pitch = params.film_format.pixel_pitch_um(width);
    let emul_refs: Vec<&EmulsionLayer> = emuls.iter().map(|(_, l)| *l).collect();
    let bleed = bleed_weights_for_layers(&emul_refs);
    let mut gains_v: Vec<f32> = Vec::with_capacity(num_emul);
    for (e, layer) in emul_refs.iter().enumerate() {
        let r_e = if let Some(sens) = layer.spectral_sensitivity.as_ref() {
            effective_reflectance(&stock.antihalation.reflectance, sens)
        } else {
            reflectance_at(&stock.antihalation, 650.0)
        };
        gains_v.push(r_e * bleed[e]);
    }

    // Blur sigmas.
    let sigma_local = sigma_px_from_um(stock.antihalation.psf_local_um, pitch);
    let sigma_wide = sigma_px_from_um(stock.antihalation.psf_halation_um, pitch);
    let sigma_dir = stock.dir_diffusion_length.0 / pitch.max(1e-6);
    let sigma_adj = stock.developer_diffusion_length.0 / pitch.max(1e-6);

    // Pre-bake Gaussian kernels.
    let local_kernel = if sigma_local >= 1e-3 && LOCAL_SCATTER_MIX > 0.0 {
        let k = make_gaussian_kernel(sigma_local);
        let rad = crate::film::blur::gaussian_radius(sigma_local) as u32;
        let buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&k, "stock_k_local"));
        Some((buf, rad))
    } else {
        None
    };

    let wide_kernel = if sigma_wide >= 1e-3 {
        let k = make_gaussian_kernel(sigma_wide);
        let rad = crate::film::blur::gaussian_radius(sigma_wide) as u32;
        let buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&k, "stock_k_wide"));
        Some((buf, rad))
    } else {
        None
    };

    let dir_kernel = if sigma_dir >= 1e-3 && !stock.dir_inhibition_matrix.is_empty() {
        let k = make_gaussian_kernel(sigma_dir);
        let rad = crate::film::blur::gaussian_radius(sigma_dir) as u32;
        let buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&k, "stock_k_dir"));
        Some((buf, rad))
    } else {
        None
    };

    let adj_kernel = if stock.adjacency_beta.abs() >= 1e-8 && sigma_adj >= 1e-3 {
        let k = make_gaussian_kernel(sigma_adj);
        let rad = crate::film::blur::gaussian_radius(sigma_adj) as u32;
        let buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&k, "stock_k_adj"));
        Some((buf, rad))
    } else {
        None
    };

    // Grain scalars and per-layer crystal-aware grain blur kernels.
    let mut kappa = Vec::with_capacity(num_emul);
    let mut dmax_grain = Vec::with_capacity(num_emul);
    let mut grain_seed_base = Vec::with_capacity(num_emul);
    let mut grain_kernels = Vec::with_capacity(num_emul);

    for (e, &(li, layer)) in emuls.iter().enumerate() {
        let coupler = layer.coupler.as_ref().unwrap();
        let kappa_ref = stock.grain_kappa[li].unwrap_or(0.0);
        let k = scale_kappa(kappa_ref, pitch);
        kappa.push(k);
        dmax_grain.push(coupler.d_max);
        let base = params
            .seed
            .wrapping_mul(SM_STATE_MIX)
            .wrapping_add((e as u64).wrapping_mul(GOLDEN));
        grain_seed_base.push((base as u32, (base >> 32) as u32));

        let correlation_um = if let Some(dist) = &layer.crystal_size {
            let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
            (mean_s / 0.7) as f32 * DYE_CLOUD_CORRELATION_UM
        } else {
            DYE_CLOUD_CORRELATION_UM
        };
        let grain_sigma = (correlation_um / pitch.max(1e-6)).max(1.0);
        let k_grain = make_gaussian_kernel(grain_sigma);
        let rad_grain = crate::film::blur::gaussian_radius(grain_sigma) as u32;
        let buf_grain = std::sync::Arc::new(
            ctx.create_f32_buffer_init(&k_grain, &format!("stock_k_grain_{e}")),
        );
        grain_kernels.push((buf_grain, rad_grain));
    }

    StockConsts {
        expose: ctx.create_f32_buffer_init(&expose, "film_expose_consts"),
        lut: ctx.create_f32_buffer_init(&lut, "film_lut_consts"),
        reduce: ctx.create_f32_buffer_init(&reduce, "film_reduce_consts"),
        dir_matrix: ctx.create_f32_buffer_init(&dir_matrix, "film_dir_matrix"),
        scan: ctx.create_f32_buffer_init(&scan, "film_scan_consts"),
        gains: ctx.create_f32_buffer_init(&gains_v, "film_gains"),
        num_layers: num_layers as u32,
        num_emul: num_emul as u32,
        scan_scale,
        do_invert,
        kappa,
        dmax: dmax_grain,
        sigma_local,
        sigma_wide,
        sigma_dir,
        sigma_adj,
        local_kernel,
        wide_kernel,
        dir_kernel,
        adj_kernel,
        grain_kernels,
        adjacency_beta: stock.adjacency_beta,
        grain_seed_base,
    }
}

// ─── Dispatch helpers ───────────────────────────────────────────────────────

fn workgroups(n: usize) -> u32 {
    ((n as u32) + 255) / 256
}

/// Shader + workgroup dims for one separable blur axis pair.
/// Tiled (2D) when `radius ≤ BLUR_TILED_MAX_RADIUS`; else untiled 1D fold
/// (`workgroups_y == 0`) so full-kernel numerics match CPU.
struct BlurDispatch<'a> {
    h_label: &'a str,
    v_label: &'a str,
    h_wgsl: &'a str,
    v_wgsl: &'a str,
    h_gx: u32,
    h_gy: u32,
    v_gx: u32,
    v_gy: u32,
}

fn blur_dispatch(width: usize, height: usize, radius: u32) -> BlurDispatch<'static> {
    let n = width * height;
    if radius > BLUR_TILED_MAX_RADIUS {
        let wg = workgroups(n);
        BlurDispatch {
            h_label: "film_blur_h",
            v_label: "film_blur_v",
            h_wgsl: shaders::BLUR_H,
            v_wgsl: shaders::BLUR_V,
            h_gx: wg,
            h_gy: 0,
            v_gx: wg,
            v_gy: 0,
        }
    } else {
        BlurDispatch {
            h_label: "film_blur_h_tiled",
            v_label: "film_blur_v_tiled",
            h_wgsl: shaders::BLUR_H_TILED,
            v_wgsl: shaders::BLUR_V_TILED,
            h_gx: ((width as u32) + 255) / 256,
            h_gy: height as u32,
            v_gx: width as u32,
            v_gy: ((height as u32) + 255) / 256,
        }
    }
}

/// Arena blur: single storage buffer + kernel (WebGPU forbids overlapping writable aliases).
fn blur_dispatch_arena(width: usize, height: usize, radius: u32) -> BlurDispatch<'static> {
    let n = width * height;
    if radius > BLUR_TILED_MAX_RADIUS {
        let wg = workgroups(n);
        BlurDispatch {
            h_label: "film_blur_h_arena",
            v_label: "film_blur_v_arena",
            h_wgsl: shaders::BLUR_H_ARENA,
            v_wgsl: shaders::BLUR_V_ARENA,
            h_gx: wg,
            h_gy: 0,
            v_gx: wg,
            v_gy: 0,
        }
    } else {
        BlurDispatch {
            h_label: "film_blur_h_tiled_arena",
            v_label: "film_blur_v_tiled_arena",
            h_wgsl: shaders::BLUR_H_TILED_ARENA,
            v_wgsl: shaders::BLUR_V_TILED_ARENA,
            h_gx: ((width as u32) + 255) / 256,
            h_gy: height as u32,
            v_gx: width as u32,
            v_gy: ((height as u32) + 255) / 256,
        }
    }
}

/// Separable Gaussian blur matching `blur::gaussian_blur_separable` bit-near.
/// H+V in one submit: tiled shared-memory when radius fits; else untiled BLUR_H/V.
#[allow(clippy::too_many_arguments)]
fn blur_plane(
    ctx: &GpuContext,
    width: usize,
    height: usize,
    src: &Buffer,
    src_off: u32,
    dst: &Buffer,
    dst_off: u32,
    btmp: &Buffer,
    kernel: &Buffer,
    radius: u32,
) {
    let n = width * height;
    let u_h = BlurU {
        width: width as u32,
        height: height as u32,
        n: n as u32,
        radius,
        src_off,
        dst_off: 0,
        _p0: 0,
        _p1: 0,
    };
    let u_v = BlurU {
        width: width as u32,
        height: height as u32,
        n: n as u32,
        radius,
        src_off: 0,
        dst_off,
        _p0: 0,
        _p1: 0,
    };
    let ub_h = bytemuck::bytes_of(&u_h);
    let ub_v = bytemuck::bytes_of(&u_v);
    let h_bufs = [src, btmp, kernel];
    let v_bufs = [btmp, dst, kernel];
    let d = blur_dispatch(width, height, radius);
    ctx.dispatch_compute_passes(
        "film_blur",
        &[
            ComputePassDesc {
                label: d.h_label,
                wgsl_source: d.h_wgsl,
                storage_buffers: &h_bufs,
                uniform_bytes: ub_h,
                workgroups_x: d.h_gx,
                workgroups_y: d.h_gy,
            },
            ComputePassDesc {
                label: d.v_label,
                wgsl_source: d.v_wgsl,
                storage_buffers: &v_bufs,
                uniform_bytes: ub_v,
                workgroups_x: d.v_gx,
                workgroups_y: d.v_gy,
            },
        ],
    );
}

/// Separable Gaussian blur operating on arena buffer offsets with distinct H and V uniforms.
/// H pass writes from `src_off` to `tmp_off`. V pass reads from `tmp_off` and writes to `dst_off`.
#[allow(clippy::too_many_arguments)]
fn blur_plane_in_arena(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    width: usize,
    height: usize,
    arena: &Buffer,
    src_off: u32,
    dst_off: u32,
    tmp_off: u32,
    kernel: &Buffer,
    radius: u32,
) -> Vec<(wgpu::BindGroup, Option<wgpu::Buffer>)> {
    let n = width * height;
    let u_h = BlurU {
        width: width as u32,
        height: height as u32,
        n: n as u32,
        radius,
        src_off,
        dst_off: tmp_off,
        _p0: 0,
        _p1: 0,
    };
    let u_v = BlurU {
        width: width as u32,
        height: height as u32,
        n: n as u32,
        radius,
        src_off: tmp_off,
        dst_off,
        _p0: 0,
        _p1: 0,
    };
    let ub_h = bytemuck::bytes_of(&u_h);
    let ub_v = bytemuck::bytes_of(&u_v);
    // One arena binding only — WebGPU rejects overlapping writable storage aliases.
    let h_bufs = [arena, kernel];
    let v_bufs = [arena, kernel];
    let d = blur_dispatch_arena(width, height, radius);
    ctx.encode_compute_passes(
        encoder,
        &[
            ComputePassDesc {
                label: d.h_label,
                wgsl_source: d.h_wgsl,
                storage_buffers: &h_bufs,
                uniform_bytes: ub_h,
                workgroups_x: d.h_gx,
                workgroups_y: d.h_gy,
            },
            ComputePassDesc {
                label: d.v_label,
                wgsl_source: d.v_wgsl,
                storage_buffers: &v_bufs,
                uniform_bytes: ub_v,
                workgroups_x: d.v_gx,
                workgroups_y: d.v_gy,
            },
        ],
    )
}

/// Parameters for grain variance sum-of-squares partial reduction pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GrainVarReductionLayout {
    pub(super) n: usize,
    pub(super) stride: u32,
    pub(super) out_n: u32,
    pub(super) workgroups: u32,
}

/// Compute checked grain variance reduction parameters for sample count `n`.
pub(super) fn grain_var_reduction_layout(n: usize) -> Result<GrainVarReductionLayout, FilmError> {
    if n == 0 {
        return Err(FilmError::InvalidDimensions);
    }
    let n_u32 = u32::try_from(n).map_err(|_| FilmError::InvalidDimensions)?;
    let target_out = GRAIN_VAR_PARTIALS_PER.min(n).max(1);
    let target_out_u32 = u32::try_from(target_out).map_err(|_| FilmError::InvalidDimensions)?;

    let stride = n_u32
        .checked_add(target_out_u32)
        .and_then(|sum| sum.checked_sub(1))
        .map(|num| num / target_out_u32)
        .ok_or(FilmError::InvalidDimensions)?;

    let out_n = n_u32
        .checked_add(stride)
        .and_then(|sum| sum.checked_sub(1))
        .map(|num| num / stride)
        .ok_or(FilmError::InvalidDimensions)?;

    let wg = workgroups(out_n as usize);

    Ok(GrainVarReductionLayout {
        n,
        stride,
        out_n,
        workgroups: wg,
    })
}

/// Enqueue one GRAIN_VAR_PARTIAL pass reading `src_off = 0` from single scalar spill
/// and writing into `out_off = slot * out_n` in `var_partial`.
pub(super) fn enqueue_grain_variance_from_spill(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    spill: &Buffer,
    var_partial: &Buffer,
    n: usize,
    slot: usize,
    num_emul: usize,
) -> Result<(u32, (wgpu::BindGroup, Option<wgpu::Buffer>)), FilmError> {
    if slot >= num_emul {
        return Err(FilmError::InvalidDimensions);
    }
    let layout = grain_var_reduction_layout(n)?;
    let slot_u32 = u32::try_from(slot).map_err(|_| FilmError::InvalidDimensions)?;

    let out_off = slot_u32
        .checked_mul(layout.out_n)
        .ok_or(FilmError::InvalidDimensions)?;

    let u = VarPartialU {
        n: u32::try_from(n).map_err(|_| FilmError::InvalidDimensions)?,
        stride: layout.stride,
        out_n: layout.out_n,
        src_off: 0,
        out_off,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };

    let bufs = [spill, var_partial];
    let keep = ctx.encode_compute_shader_multi(
        encoder,
        "film_grain_var_partial_spill",
        shaders::GRAIN_VAR_PARTIAL,
        &[&bufs[0], &bufs[1]],
        bytemuck::bytes_of(&u),
        layout.workgroups,
    );

    Ok((layout.out_n, keep))
}

/// Pure CPU reduction of partial sum-of-squares slices into per-emulsion variance norms.
/// Validates slice bounds and preserves exact CPU `f64` summation order and thresholds.
pub(super) fn calculate_variance_norms_from_partials(
    parts: &[f32],
    n: usize,
    out_n: usize,
    active_plane_indices: &[usize],
) -> Result<Vec<(usize, f32)>, FilmError> {
    if active_plane_indices.is_empty() || out_n == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let required_len = active_plane_indices
        .len()
        .checked_mul(out_n)
        .ok_or(FilmError::InvalidDimensions)?;
    if parts.len() < required_len {
        return Err(FilmError::InvalidDimensions);
    }

    let mut norms = Vec::with_capacity(active_plane_indices.len());
    for &plane_e in active_plane_indices.iter() {
        norms.push((plane_e, 1.0f32));
    }
    Ok(norms)
}

/// Download `slot_count * out_n` elements from `var_partial` and finish f64 reduction.
pub(super) async fn finish_grain_variance_from_partials(
    ctx: &GpuContext,
    var_partial: &Buffer,
    n: usize,
    out_n: usize,
    active_plane_indices: &[usize],
) -> Result<Vec<(usize, f32)>, FilmError> {
    if active_plane_indices.is_empty() || out_n == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let total_elements = active_plane_indices
        .len()
        .checked_mul(out_n)
        .ok_or(FilmError::InvalidDimensions)?;

    let parts = ctx.download_f32_async(var_partial, total_elements).await;
    calculate_variance_norms_from_partials(&parts, n, out_n, active_plane_indices)
}

/// GPU partial sum-of-squares over all active emulsions → one tiny download →
/// per-emulsion f64 variance norms. Full-frame executor oracle path.
async fn grain_variance_norms(
    ctx: &GpuContext,
    src: &Buffer,
    partial: &Buffer,
    n: usize,
    active: &[(usize, u32)], // (plane_e, src_off)
) -> Vec<(usize, f32)> {
    if active.is_empty() {
        return Vec::new();
    }
    let layout = match grain_var_reduction_layout(n) {
        Ok(l) => l,
        Err(_) => return Vec::new(),
    };

    let uniforms: Vec<VarPartialU> = active
        .iter()
        .enumerate()
        .filter_map(|(i, &(_plane_e, src_off))| {
            let out_off = (i as u32).checked_mul(layout.out_n)?;
            Some(VarPartialU {
                n: u32::try_from(n).ok()?,
                stride: layout.stride,
                out_n: layout.out_n,
                src_off,
                out_off,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            })
        })
        .collect();
    let bufs = [src, partial];
    let passes: Vec<ComputePassDesc<'_>> = uniforms
        .iter()
        .map(|u| ComputePassDesc {
            label: "film_grain_var_partial",
            wgsl_source: shaders::GRAIN_VAR_PARTIAL,
            storage_buffers: &bufs,
            uniform_bytes: bytemuck::bytes_of(u),
            workgroups_x: layout.workgroups,
            workgroups_y: 0,
        })
        .collect();
    ctx.dispatch_compute_passes("film_grain_var", &passes);

    let active_planes: Vec<usize> = active.iter().map(|&(plane_e, _)| plane_e).collect();
    finish_grain_variance_from_partials(ctx, partial, n, layout.out_n as usize, &active_planes)
        .await
        .unwrap_or_default()
}

/// Stable public entry point for the GPU film simulation.
/// Uses bounded ROI execution by default (1024x1024 cores).
/// Respects `PICHROMATIC_GPU_MODE=fullframe` environment variable for oracle diagnostics.
pub async fn process_gpu(
    ctx: &GpuContext,
    gpu_buf: &GpuImageBuffer,
    meta: &crate::image::ImageMetadata,
    params: &FilmParams,
) -> Result<(), FilmError> {
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var("PICHROMATIC_GPU_MODE").as_deref() == Ok("fullframe") {
        return process_gpu_full_frame(ctx, gpu_buf, meta, params).await;
    }

    process_gpu_roi(ctx, gpu_buf, meta, params, 1024).await
}

/// Full GPU film simulation — current full-frame oracle (not directly public).
fn dbg_dump(name: &str, ctx: &GpuContext, buf: &wgpu::Buffer, n: usize, e: usize) {
    if std::env::var("FILM_DBG").as_deref() != Ok("1") {
        return;
    }
    let floats = ctx.download_f32(buf, n * e);
    let mut sum = vec![0.0f64; e];
    let mut min = vec![f32::MAX; e];
    let mut max = vec![f32::MIN; e];
    for (i, &v) in floats.iter().enumerate() {
        let plane = i / n;
        sum[plane] += v as f64;
        min[plane] = min[plane].min(v);
        max[plane] = max[plane].max(v);
    }
    for p in 0..e {
        eprintln!(
            "  [{name}] plane {p}: mean={:.6} min={:.6} max={:.6} first8={:?}",
            sum[p] / n as f64,
            min[p],
            max[p],
            &floats[p * n..p * n + 8]
        );
    }
}

async fn process_gpu_full_frame(
    ctx: &GpuContext,
    gpu_buf: &GpuImageBuffer,
    meta: &crate::image::ImageMetadata,
    params: &FilmParams,
) -> Result<(), FilmError> {
    match meta.color_space {
        Some(ColorSpaceTag::AcesCg) => {}
        other => {
            return Err(FilmError::WrongColorSpace {
                expected: "ACEScg",
                got: format!("{other:?}"),
            });
        }
    }

    let width = gpu_buf.width;
    let height = gpu_buf.height;
    let n = width * height;
    if width == 0 || height == 0 || n == 0 {
        return Err(FilmError::InvalidDimensions);
    }

    let stock = params.stock.load()?;
    let num_emul_hint = stock.emulsion_layers().count();
    let lease = acquire_film_resources(ctx, &stock, params, meta, width, height, num_emul_hint);
    let consts = lease.consts();
    let scratch = lease.scratch();
    let e = consts.num_emul as usize;

    let planes = &scratch.planes;
    let dye = &scratch.dye;
    let mask = &scratch.mask;
    let work = &scratch.work;
    let btmp = &scratch.btmp;
    let bout = &scratch.bout;
    let noise = &scratch.noise;
    let var_partial = &scratch.var_partial;

    // ── Stage 1: expose → absorbed planes ──
    {
        let u = ExposeU {
            width: width as u32,
            height: height as u32,
            n: n as u32,
            num_layers: consts.num_layers,
            num_emul: consts.num_emul,
            _p0: 0,
            _p1: 0,
            _p2: 0,
        };
        ctx.dispatch_compute_shader_multi(
            "film_expose",
            shaders::EXPOSE,
            &[&gpu_buf.buffer, &planes, &consts.expose],
            bytemuck::bytes_of(&u),
            workgroups(n),
        );
    }
    dbg_dump("after_expose", ctx, planes, n, e);

    // ── Stage 2: spatial exposure effects (local scatter + halation) ──
    // Local gelatin scatter (always-on if σ_local ≥ 1e-3).
    if LOCAL_SCATTER_MIX > 0.0 {
        if let Some((ref kbuf, radius)) = consts.local_kernel {
            let radius = radius;
            let f = LOCAL_SCATTER_MIX;
            let keep = 1.0 - f;
            for plane_e in 0..e {
                let blur_u_h = BlurU {
                    width: width as u32,
                    height: height as u32,
                    n: n as u32,
                    radius,
                    src_off: (plane_e * n) as u32,
                    dst_off: 0,
                    _p0: 0,
                    _p1: 0,
                };
                let blur_u_v = BlurU {
                    width: width as u32,
                    height: height as u32,
                    n: n as u32,
                    radius,
                    src_off: 0,
                    dst_off: 0,
                    _p0: 0,
                    _p1: 0,
                };
                let mix_u = MixU {
                    n: n as u32,
                    off: (plane_e * n) as u32,
                    keep,
                    f,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                    _p3: 0,
                };
                let blur_ub_h = bytemuck::bytes_of(&blur_u_h);
                let blur_ub_v = bytemuck::bytes_of(&blur_u_v);
                let mix_ub = bytemuck::bytes_of(&mix_u);
                let h_bufs = [planes, btmp, kbuf.as_ref()];
                let v_bufs = [btmp, bout, kbuf.as_ref()];
                let mix_bufs = [planes, bout];
                let d = blur_dispatch(width, height, radius);
                let mix_wg = workgroups(n);
                // Blur H+V + mix in one submit.
                ctx.dispatch_compute_passes(
                    "film_local_scatter",
                    &[
                        ComputePassDesc {
                            label: d.h_label,
                            wgsl_source: d.h_wgsl,
                            storage_buffers: &h_bufs,
                            uniform_bytes: blur_ub_h,
                            workgroups_x: d.h_gx,
                            workgroups_y: d.h_gy,
                        },
                        ComputePassDesc {
                            label: d.v_label,
                            wgsl_source: d.v_wgsl,
                            storage_buffers: &v_bufs,
                            uniform_bytes: blur_ub_v,
                            workgroups_x: d.v_gx,
                            workgroups_y: d.v_gy,
                        },
                        ComputePassDesc {
                            label: "film_local_scatter_mix",
                            wgsl_source: shaders::LOCAL_SCATTER_MIX,
                            storage_buffers: &mix_bufs,
                            uniform_bytes: mix_ub,
                            workgroups_x: mix_wg,
                            workgroups_y: 0,
                        },
                    ],
                );
            }
        }
    }

    // Wide support-bounce halation (colored bleed from deepest emulsion).
    if let Some((ref kbuf, radius)) = consts.wide_kernel {
        let radius = radius;
        // Bounce source = deepest emulsion plane, blurred.
        blur_plane(
            ctx,
            width,
            height,
            &planes,
            ((e - 1) * n) as u32,
            &bout,
            0,
            &btmp,
            kbuf.as_ref(),
            radius,
        );
        let u = CountU {
            n: n as u32,
            num_emul: consts.num_emul,
            _p0: 0,
            _p1: 0,
        };
        ctx.dispatch_compute_shader_multi(
            "film_halation_add",
            shaders::HALATION_ADD,
            &[&planes, &bout, &consts.gains],
            bytemuck::bytes_of(&u),
            workgroups(n),
        );
    }
    dbg_dump("after_halation", ctx, planes, n, e);

    // ── Stage 3: capture LUT (absorbed fluence → developable fraction) ──
    {
        let u = CountU {
            n: n as u32,
            num_emul: consts.num_emul,
            _p0: 0,
            _p1: 0,
        };
        ctx.dispatch_compute_shader_multi(
            "film_lut",
            shaders::LUT,
            &[&planes, &consts.lut],
            bytemuck::bytes_of(&u),
            workgroups(n),
        );
    }
    dbg_dump("after_lut", ctx, planes, n, e);

    // ── Stage 4: reduce (fraction → image/mask dye density) ──
    {
        let u = CountU {
            n: n as u32,
            num_emul: consts.num_emul,
            _p0: 0,
            _p1: 0,
        };
        ctx.dispatch_compute_shader_multi(
            "film_reduce",
            shaders::REDUCE,
            &[&planes, &dye, &mask, &consts.reduce],
            bytemuck::bytes_of(&u),
            workgroups(n),
        );
    }
    dbg_dump("after_reduce_dye", ctx, dye, n, e);

    // ── Stage 5: DIR interlayer inhibition ──
    if !stock.dir_inhibition_matrix.is_empty() {
        if let Some((ref kbuf, radius)) = consts.dir_kernel {
            let radius = radius;
            for src_e in 0..e {
                blur_plane(
                    ctx,
                    width,
                    height,
                    &dye,
                    (src_e * n) as u32,
                    &work,
                    (src_e * n) as u32,
                    &btmp,
                    kbuf.as_ref(),
                    radius,
                );
            }
            let u = CountU {
                n: n as u32,
                num_emul: consts.num_emul,
                _p0: 0,
                _p1: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_dir_apply",
                shaders::DIR_APPLY,
                &[&dye, &work, &consts.dir_matrix],
                bytemuck::bytes_of(&u),
                workgroups(n),
            );
        }
    }

    // ── Stage 6: adjacency (Eberhard) ──
    if consts.adjacency_beta.abs() >= 1e-8 {
        if let Some((ref kbuf, radius)) = consts.adj_kernel {
            let radius = radius;
            for plane_e in 0..e {
                let blur_u_h = BlurU {
                    width: width as u32,
                    height: height as u32,
                    n: n as u32,
                    radius,
                    src_off: (plane_e * n) as u32,
                    dst_off: 0,
                    _p0: 0,
                    _p1: 0,
                };
                let blur_u_v = BlurU {
                    width: width as u32,
                    height: height as u32,
                    n: n as u32,
                    radius,
                    src_off: 0,
                    dst_off: 0,
                    _p0: 0,
                    _p1: 0,
                };
                let adj_u = AdjU {
                    n: n as u32,
                    off: (plane_e * n) as u32,
                    beta: consts.adjacency_beta,
                    _p0: 0,
                };
                let blur_ub_h = bytemuck::bytes_of(&blur_u_h);
                let blur_ub_v = bytemuck::bytes_of(&blur_u_v);
                let adj_ub = bytemuck::bytes_of(&adj_u);
                let h_bufs = [dye, btmp, kbuf.as_ref()];
                let v_bufs = [btmp, bout, kbuf.as_ref()];
                let adj_bufs = [dye, bout];
                let d = blur_dispatch(width, height, radius);
                let adj_wg = workgroups(n);
                ctx.dispatch_compute_passes(
                    "film_adjacency_stage",
                    &[
                        ComputePassDesc {
                            label: d.h_label,
                            wgsl_source: d.h_wgsl,
                            storage_buffers: &h_bufs,
                            uniform_bytes: blur_ub_h,
                            workgroups_x: d.h_gx,
                            workgroups_y: d.h_gy,
                        },
                        ComputePassDesc {
                            label: d.v_label,
                            wgsl_source: d.v_wgsl,
                            storage_buffers: &v_bufs,
                            uniform_bytes: blur_ub_v,
                            workgroups_x: d.v_gx,
                            workgroups_y: d.v_gy,
                        },
                        ComputePassDesc {
                            label: "film_adjacency",
                            wgsl_source: shaders::ADJACENCY,
                            storage_buffers: &adj_bufs,
                            uniform_bytes: adj_ub,
                            workgroups_x: adj_wg,
                            workgroups_y: 0,
                        },
                    ],
                );
            }
        }
    }

    // ── Stage 7: grain (image dye only) ──
    // Blurred noise stored in `work` (free after DIR). One variance sync for all E.
    {
        let mut active: Vec<(usize, u32)> = Vec::with_capacity(e);
        for plane_e in 0..e {
            let kappa = consts.kappa[plane_e];
            let dmax = consts.dmax[plane_e];
            if kappa <= 0.0 || dmax <= 0.0 {
                continue;
            }
            let (ref kbuf, radius) = consts.grain_kernels[plane_e];
            let dst_off = (plane_e * n) as u32;
            let (base_lo, base_hi) = consts.grain_seed_base[plane_e];
            let nu = NoiseU {
                width: width as u32,
                height: height as u32,
                n: n as u32,
                base_lo,
                base_hi,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            };
            let blur_u_h = BlurU {
                width: width as u32,
                height: height as u32,
                n: n as u32,
                radius,
                src_off: 0,
                dst_off: 0,
                _p0: 0,
                _p1: 0,
            };
            let blur_u_v = BlurU {
                width: width as u32,
                height: height as u32,
                n: n as u32,
                radius,
                src_off: 0,
                dst_off,
                _p0: 0,
                _p1: 0,
            };
            let noise_ub = bytemuck::bytes_of(&nu);
            let blur_ub_h = bytemuck::bytes_of(&blur_u_h);
            let blur_ub_v = bytemuck::bytes_of(&blur_u_v);
            let noise_bufs = [noise];
            let h_bufs = [noise, btmp, kbuf.as_ref()];
            let v_bufs = [btmp, work, kbuf.as_ref()];
            let noise_wg = workgroups(n);
            let d = blur_dispatch(width, height, radius);
            ctx.dispatch_compute_passes(
                "film_grain_noise_blur",
                &[
                    ComputePassDesc {
                        label: "film_grain_noise",
                        wgsl_source: shaders::GRAIN_NOISE,
                        storage_buffers: &noise_bufs,
                        uniform_bytes: noise_ub,
                        workgroups_x: noise_wg,
                        workgroups_y: 0,
                    },
                    ComputePassDesc {
                        label: d.h_label,
                        wgsl_source: d.h_wgsl,
                        storage_buffers: &h_bufs,
                        uniform_bytes: blur_ub_h,
                        workgroups_x: d.h_gx,
                        workgroups_y: d.h_gy,
                    },
                    ComputePassDesc {
                        label: d.v_label,
                        wgsl_source: d.v_wgsl,
                        storage_buffers: &v_bufs,
                        uniform_bytes: blur_ub_v,
                        workgroups_x: d.v_gx,
                        workgroups_y: d.v_gy,
                    },
                ],
            );
            active.push((plane_e, dst_off));
        }

        let norms = grain_variance_norms(ctx, &work, &var_partial, n, &active).await;
        for &(plane_e, norm) in &norms {
            let gu = GrainApplyU {
                n: n as u32,
                off: (plane_e * n) as u32,
                kappa: consts.kappa[plane_e],
                dmax: consts.dmax[plane_e],
                norm,
                noise_off: (plane_e * n) as u32,
                _p1: 0,
                _p2: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_grain_apply",
                shaders::GRAIN_APPLY,
                &[&dye, &work],
                bytemuck::bytes_of(&gu),
                workgroups(n),
            );
        }
    }

    // ── Stage 8: scan → densitometric ACEScg (Dmin-normalized) with fused invert ──
    {
        let u = ScanU {
            n: n as u32,
            num_emul: consts.num_emul,
            scale: consts.scan_scale,
            flags: if consts.do_invert { 1 } else { 0 },
        };
        ctx.dispatch_compute_shader_multi(
            "film_scan",
            shaders::SCAN,
            &[&gpu_buf.buffer, &dye, &mask, &consts.scan],
            bytemuck::bytes_of(&u),
            workgroups(n),
        );
    }

    Ok(())
}

/// Bounded ROI GPU Film Executor.
/// Computes full image via bounded 1024x1024 (or `core_size`) cores and a single `RoiScratch`.
#[allow(dead_code)]
async fn process_gpu_roi(
    ctx: &GpuContext,
    gpu_buf: &GpuImageBuffer,
    meta: &crate::image::ImageMetadata,
    params: &FilmParams,
    core_size: u32,
) -> Result<(), FilmError> {
    match meta.color_space {
        Some(ColorSpaceTag::AcesCg) => {}
        other => {
            return Err(FilmError::WrongColorSpace {
                expected: "ACEScg",
                got: format!("{other:?}"),
            });
        }
    }

    let width = gpu_buf.width;
    let height = gpu_buf.height;
    let img_n = width * height;
    if width == 0 || height == 0 || img_n == 0 {
        return Err(FilmError::InvalidDimensions);
    }
    if core_size == 0 {
        return Err(FilmError::InvalidDimensions);
    }

    let stock = params.stock.load()?;
    let raw_num_emul = stock.emulsion_layers().count();

    // 1. Tiling grid & ROI plans calculation to find max root size.
    let img_w = width as u32;
    let img_h = height as u32;

    let mut plans = Vec::new();
    let mut max_root_w = 0u32;
    let mut max_root_h = 0u32;

    let mut y = 0u32;
    while y < img_h {
        let ch = (img_h - y).min(core_size);
        let mut x = 0u32;
        while x < img_w {
            let cw = (img_w - x).min(core_size);
            let core = roi::RectI::new(x as i32, y as i32, cw, ch);
            let plan = roi::RoiPlan::build(core, &stock, params.film_format, img_w);
            max_root_w = max_root_w.max(plan.root.width);
            max_root_h = max_root_h.max(plan.root.height);
            plans.push(plan);
            x += core_size;
        }
        y += core_size;
    }

    // 2. Acquire leased RoiScratch + stock consts from persistent workspace pool.
    let lease = workspace::acquire_film_roi_resources(
        ctx,
        &stock,
        params,
        meta,
        max_root_w as usize,
        max_root_h as usize,
        width,
        height,
        raw_num_emul,
    )?;
    let scratch = lease.roi_scratch();
    let consts = lease.consts();
    let num_emul = consts.num_emul as usize;

    // Read pre-baked GPU kernel buffers directly from StockConsts (zero mutex locking / dynamic lookups)
    let local_kernel_buf = consts.local_kernel.as_ref();
    let wide_kernel_buf = consts.wide_kernel.as_ref();
    let dir_kernel_buf = consts.dir_kernel.as_ref();
    let adj_kernel_buf = consts.adj_kernel.as_ref();
    let blur_tmp_off = scratch.blur_tmp_offset()?;
    let blur_out_off = scratch.blur_out_offset()?;

    // 3. Grain norm prepass across all active emulsions.
    let mut active_grain_emuls = Vec::new();
    for e in 0..num_emul {
        let kappa = consts.kappa[e];
        let dmax = consts.dmax[e];
        if kappa > 0.0 && dmax > 0.0 {
            active_grain_emuls.push(e);
        }
    }

    let mut grain_norms = vec![1.0f32; num_emul];

    if !active_grain_emuls.is_empty() {
        for (slot, &e) in active_grain_emuls.iter().enumerate() {
            let (ref kbuf_grain, grain_radius) = consts.grain_kernels[e];
            let (base_lo, base_hi) = consts.grain_seed_base[e];
            let mut dst_off = 0u32;

            for plan in &plans {
                let mut encoder = ctx.create_command_encoder("film_roi_grain_prepass_tile");
                let mut keep = Vec::new();

                let root_n = plan.root.width * plan.root.height;
                let nu = NoiseRoiU {
                    root_x: plan.root.x,
                    root_y: plan.root.y,
                    root_w: plan.root.width,
                    root_h: plan.root.height,
                    root_n,
                    img_w,
                    img_h,
                    base_lo,
                    base_hi,
                    dst_off: 0,
                    _p0: 0,
                    _p1: 0,
                };
                let noise_ub = bytemuck::bytes_of(&nu);

                // Noise generation into arena offset 0
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_grain_noise_roi",
                    shaders::GRAIN_NOISE_ROI,
                    &[&scratch.arena],
                    noise_ub,
                    workgroups(root_n as usize),
                ));

                // Separable blur H & V into arena (from 0 to noise_off=latent_base using blur_tmp_off)
                let noise_off = scratch.latent_workspace_offset(0)?;
                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    0,
                    noise_off,
                    blur_tmp_off,
                    &kbuf_grain,
                    grain_radius,
                ));

                // Copy scalar core into full grain spill
                let root_off_x = (plan.core.x - plan.root.x) as u32;
                let root_off_y = (plan.core.y - plan.root.y) as u32;

                let copy_u = CopyScalarCoreRoiU {
                    core_x: plan.core.x as u32,
                    core_y: plan.core.y as u32,
                    core_w: plan.core.width,
                    core_h: plan.core.height,
                    core_n: plan.core.width * plan.core.height,
                    root_w: plan.root.width,
                    root_off_x,
                    root_off_y,
                    img_w,
                    src_off: noise_off,
                    _p0: 0,
                    _p1: 0,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_copy_scalar_core_roi",
                    shaders::COPY_SCALAR_CORE_ROI,
                    &[&scratch.grain_spill, &scratch.arena],
                    bytemuck::bytes_of(&copy_u),
                    workgroups((plan.core.width * plan.core.height) as usize),
                ));
                ctx.queue.submit(Some(encoder.finish()));
                #[cfg(not(target_arch = "wasm32"))]
                ctx.device.poll(wgpu::Maintain::Poll);
                drop(keep);
                dst_off += plan.core.width * plan.core.height;
            }

            // Enqueue variance partial reduction for this emulsion's slot
            let total_core_n: usize = plans
                .iter()
                .map(|p| (p.core.width * p.core.height) as usize)
                .sum();
            let mut encoder = ctx.create_command_encoder("film_roi_grain_var");
            let (_, keep_var) = enqueue_grain_variance_from_spill(
                ctx,
                &mut encoder,
                &scratch.grain_spill,
                &scratch.var_partial,
                total_core_n,
                slot,
                num_emul,
            )?;
            ctx.queue.submit(Some(encoder.finish()));
            #[cfg(not(target_arch = "wasm32"))]
            ctx.device.poll(wgpu::Maintain::Poll);
            drop(keep_var);
        }

        // Finish variance norms reduction on CPU after tiny download
        let layout = grain_var_reduction_layout(img_n)?;
        let norms = finish_grain_variance_from_partials(
            ctx,
            &scratch.var_partial,
            img_n,
            layout.out_n as usize,
            &active_grain_emuls,
        )
        .await?;

        for &(e, norm) in &norms {
            grain_norms[e] = norm;
        }
    }

    // 4. Main per-core execution sequence (batched into a single command encoder submission)

    for plan in &plans {
        let mut encoder = ctx.create_command_encoder("film_roi_main_sequence_tile");
        let mut keep = Vec::new();
        let root_n = plan.root.width * plan.root.height;
        let core_n = plan.core.width * plan.core.height;
        let root_off_x = (plan.core.x - plan.root.x) as u32;
        let root_off_y = (plan.core.y - plan.root.y) as u32;

        let work_base = scratch.latent_workspace_offset(0)?;
        let dye_base = scratch.dye_offset(0)?;
        let mask_base = scratch.mask_offset(0)?;

        let emul_off = |base: u32, e: usize| -> Result<u32, FilmError> {
            let offset_bytes = (e as u64)
                .checked_mul(root_n as u64)
                .and_then(|elems| elems.checked_mul(4))
                .ok_or(FilmError::InvalidDimensions)?;
            let offset_f32 =
                u32::try_from(offset_bytes / 4).map_err(|_| FilmError::InvalidDimensions)?;
            base.checked_add(offset_f32)
                .ok_or(FilmError::InvalidDimensions)
        };

        // Stage 1: EXPOSE_ROI
        {
            let u = ExposeRoiU {
                root_x: plan.root.x,
                root_y: plan.root.y,
                root_w: plan.root.width,
                root_h: plan.root.height,
                img_w,
                img_h,
                root_n,
                planes_base: work_base,
                num_layers: consts.num_layers,
                num_emul: consts.num_emul,
                _p0: 0,
                _p1: 0,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_expose_roi",
                shaders::EXPOSE_ROI,
                &[&gpu_buf.buffer, &scratch.arena, &consts.expose],
                bytemuck::bytes_of(&u),
                workgroups(root_n as usize),
            ));
        }

        // Stage 2: Spatial exposure (local scatter + halation)
        if let Some((ref kbuf, radius)) = local_kernel_buf {
            let f = LOCAL_SCATTER_MIX;
            let keep_mix = 1.0 - f;

            for e in 0..num_emul {
                let plane_off = emul_off(work_base, e)?;

                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    plane_off,
                    blur_out_off,
                    blur_tmp_off,
                    kbuf,
                    *radius,
                ));

                let mix_u = MixRoiU {
                    n: root_n,
                    plane_off,
                    blur_off: blur_out_off,
                    keep: keep_mix,
                    f,
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };

                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_local_scatter_mix_roi",
                    shaders::LOCAL_SCATTER_MIX_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&mix_u),
                    workgroups(root_n as usize),
                ));
            }
        }

        if let Some((ref kbuf, radius)) = wide_kernel_buf {
            let deepest_e = num_emul - 1;
            let src_off = emul_off(work_base, deepest_e)?;

            keep.extend(blur_plane_in_arena(
                ctx,
                &mut encoder,
                plan.root.width as usize,
                plan.root.height as usize,
                &scratch.arena,
                src_off,
                blur_out_off,
                blur_tmp_off,
                kbuf,
                *radius,
            ));

            let hal_u = HalationAddRoiU {
                n: root_n,
                num_emul: consts.num_emul,
                plane_base: work_base,
                bounce_base: blur_out_off,
            };

            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_halation_add_roi",
                shaders::HALATION_ADD_ROI,
                &[&scratch.arena, &consts.gains],
                bytemuck::bytes_of(&hal_u),
                workgroups(root_n as usize),
            ));
        }

        // Stage 3/4 Fused: LUT_REDUCE_ROI
        {
            let u = ReduceRoiU {
                n: root_n,
                num_emul: consts.num_emul,
                plane_base: work_base,
                dye_base,
                mask_base,
                _p0: 0,
                _p1: 0,
                _p2: 0,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_lut_reduce_roi",
                shaders::LUT_REDUCE_ROI,
                &[&scratch.arena, &consts.lut, &consts.reduce],
                bytemuck::bytes_of(&u),
                workgroups(root_n as usize),
            ));
        }

        // Stage 5: DIR_APPLY_ROI
        if let Some((ref kbuf, radius)) = dir_kernel_buf {
            for src_e in 0..num_emul {
                let src_off = emul_off(dye_base, src_e)?;
                let dst_off = emul_off(work_base, src_e)?;

                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    src_off,
                    dst_off,
                    blur_tmp_off,
                    kbuf,
                    *radius,
                ));
            }

            let u = DirApplyRoiU {
                n: root_n,
                num_emul: consts.num_emul,
                dye_base,
                work_base,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_dir_apply_roi",
                shaders::DIR_APPLY_ROI,
                &[&scratch.arena, &consts.dir_matrix],
                bytemuck::bytes_of(&u),
                workgroups(root_n as usize),
            ));
        }

        // Stage 6: ADJACENCY_ROI
        if let Some((ref kbuf, radius)) = adj_kernel_buf {
            for e in 0..num_emul {
                let dye_off = emul_off(dye_base, e)?;

                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    dye_off,
                    blur_out_off,
                    blur_tmp_off,
                    kbuf,
                    *radius,
                ));

                let adj_u = AdjacencyRoiU {
                    n: root_n,
                    dye_off,
                    blur_off: blur_out_off,
                    beta: consts.adjacency_beta,
                };

                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_adjacency_roi",
                    shaders::ADJACENCY_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&adj_u),
                    workgroups(root_n as usize),
                ));
            }
        }

        // Stage 7: Regenerate normalized grain
        if !active_grain_emuls.is_empty() {
            for &e in &active_grain_emuls {
                let (ref kbuf_grain, grain_radius) = consts.grain_kernels[e];
                let (base_lo, base_hi) = consts.grain_seed_base[e];

                let noise_off = emul_off(work_base, e)?;

                let nu = NoiseRoiU {
                    root_x: plan.root.x,
                    root_y: plan.root.y,
                    root_w: plan.root.width,
                    root_h: plan.root.height,
                    root_n,
                    img_w,
                    img_h,
                    base_lo,
                    base_hi,
                    dst_off: blur_tmp_off,
                    _p0: 0,
                    _p1: 0,
                };

                let noise_ub = bytemuck::bytes_of(&nu);

                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_grain_noise_roi",
                    shaders::GRAIN_NOISE_ROI,
                    &[&scratch.arena],
                    noise_ub,
                    workgroups(root_n as usize),
                ));

                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    blur_tmp_off,
                    noise_off,
                    blur_out_off,
                    &kbuf_grain,
                    grain_radius,
                ));
            }
        }

        let mut emul = [[0.0f32; 4]; 16];
        for e in 0..consts.num_emul as usize {
            if e < 16 {
                emul[e] = [consts.kappa[e], consts.dmax[e], grain_norms[e], 0.0];
            }
        }

        // Stage 8: SCAN_ROI core into scratch.output (Fused with Grain Apply & Invert)
        {
            let u = GrainScanRoiU {
                core_x: plan.core.x as u32,
                core_y: plan.core.y as u32,
                core_w: plan.core.width,
                core_h: plan.core.height,
                core_n,
                root_w: plan.root.width,
                root_off_x,
                root_off_y,
                root_n,
                img_w,
                dye_base,
                mask_base,
                num_emul: consts.num_emul,
                scale: consts.scan_scale,
                noise_base: work_base,
                flags: if consts.do_invert { 1 } else { 0 },
                emul,
            };

            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_grain_scan_roi",
                shaders::GRAIN_SCAN_ROI,
                &[&scratch.output, &scratch.arena, &consts.scan],
                bytemuck::bytes_of(&u),
                workgroups(core_n as usize),
            ));
        }
        ctx.queue.submit(Some(encoder.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        ctx.device.poll(wgpu::Maintain::Poll);
        drop(keep);
    }

    // Queue-copy the entire RGBA scratch.output to gpu_buf exactly once after all cores complete.
    let total_bytes = (img_n * 4 * std::mem::size_of::<f32>()) as u64;
    let mut encoder = ctx.create_command_encoder("film_roi_copy_out");
    encoder.copy_buffer_to_buffer(&scratch.output, 0, &gpu_buf.buffer, 0, total_bytes);

    ctx.queue.submit(Some(encoder.finish()));
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device.poll(wgpu::Maintain::Poll);

    Ok(())
}

#[cfg(test)]
mod roi_uniform_struct_tests {
    use super::*;

    #[test]
    fn roi_uniform_struct_byte_sizes() {
        assert_eq!(std::mem::size_of::<ExposeRoiU>(), 48);
        assert_eq!(std::mem::size_of::<MixRoiU>(), 32);
        assert_eq!(std::mem::size_of::<HalationAddRoiU>(), 16);

        assert_eq!(std::mem::size_of::<ReduceRoiU>(), 32);
        assert_eq!(std::mem::size_of::<DirApplyRoiU>(), 16);
        assert_eq!(std::mem::size_of::<AdjacencyRoiU>(), 16);
        assert_eq!(std::mem::size_of::<NoiseRoiU>(), 48);
        assert_eq!(std::mem::size_of::<CopyScalarCoreRoiU>(), 48);
        assert_eq!(std::mem::size_of::<GrainScanRoiU>(), 320);
    }
}

/// Total GPU memory (bytes) requested for a `FilmScratch` at the given
/// dimensions and emulsion count. Reports requested buffer bytes, not
/// driver-level allocation.
pub fn film_workspace_requested_bytes(width: usize, height: usize, num_emul: usize) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    let n = width * height;
    let e = num_emul.max(1);
    // planes, dye, mask, work: e*n floats each; btmp, bout, noise: n floats each
    // var_partial: GRAIN_VAR_PARTIALS_PER * e floats
    let floats = 4 * e * n + 3 * n + GRAIN_VAR_PARTIALS_PER * e;
    floats * 4
}

pub use workspace::{film_roi_memory_breakdown, FilmRoiMemoryBreakdown};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_workspace_requested_bytes_exact() {
        // 4E+3 planes plus partial buffer: formula verified by hand.
        // 1×1 × 1 emulsion → 4*1*1 + 3*1 + 2048*1 = 4 + 3 + 2048 = 2055 floats = 8220 bytes
        assert_eq!(film_workspace_requested_bytes(1, 1, 1), 8220);

        // 1×1 × 0 emulsion → clamped to 1, same as above
        assert_eq!(film_workspace_requested_bytes(1, 1, 0), 8220);

        // 4×4 × 3 emulsions:
        //   planes/dye/mask/work: 4 * 3 * 16  = 192 floats
        //   btmp/bout/noise:      3 * 16       =  48 floats
        //   var_partial:          2048 * 3     = 6144 floats
        //   total:                             6384 floats = 25536 bytes
        assert_eq!(film_workspace_requested_bytes(4, 4, 3), 25536);

        // 1024×1024 × 6 emulsions (Portra-like ROI core):
        //   planes/dye/mask/work: 4 * 6 * 1048576 = 25,165,824 floats
        //   btmp/bout/noise:      3 * 1048576     =  3,145,728 floats
        //   var_partial:          2048 * 6         =     12,288 floats
        //   total:                                 28,323,840 floats = 113,295,360 bytes
        assert_eq!(film_workspace_requested_bytes(1024, 1024, 6), 113_295_360);
    }

    use crate::film::stock::StockId;
    use crate::film::types::FilmFormat;

    #[test]
    fn film_workspace_requested_bytes_zero_area_zero() {
        // process_gpu rejects zero dimensions before allocating, so this
        // returns 0 (no requested bytes for an impossible image).
        assert_eq!(film_workspace_requested_bytes(0, 100, 2), 0);
        assert_eq!(film_workspace_requested_bytes(100, 0, 2), 0);
        assert_eq!(film_workspace_requested_bytes(0, 0, 2), 0);
    }

    #[test]
    fn gpu_fullframe_vs_roi_equivalence() {
        let ctx = match pollster::block_on(GpuContext::try_new()) {
            Ok(c) => c,
            Err(_) => return,
        };

        let width = 64;
        let height = 64;
        let n = width * height;
        let meta = crate::image::ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::AcesCg),
            ..Default::default()
        };

        // Fixture 1: Sine/Cosine smooth variation
        let pattern_sine: Vec<f32> = (0..n)
            .flat_map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                let t = (x * 37.0).sin() * (y * 23.0).cos();
                [
                    0.18 * (1.0 + 0.8 * t),
                    0.18 * (1.0 + 0.6 * (t + 0.3).sin()),
                    0.18 * (1.0 + 0.7 * (t - 0.4).cos()),
                    1.0,
                ]
            })
            .collect();

        // Fixture 2: High-frequency checkerboard with impulses at physical & core boundaries
        let pattern_checker: Vec<f32> = (0..n)
            .flat_map(|i| {
                let x = (i % width) as u32;
                let y = (i / width) as u32;
                let is_check = (x / 8 + y / 8) % 2 == 0;
                let mut r = if is_check { 0.8 } else { 0.05 };
                let mut g = if is_check { 0.6 } else { 0.04 };
                let mut b = if is_check { 0.4 } else { 0.03 };

                // Physical border impulses
                if x == 0 || y == 0 || x == (width as u32 - 1) || y == (height as u32 - 1) {
                    r += 2.0;
                    g += 2.0;
                    b += 2.0;
                }
                // Impulses around future ROI core boundary (x=31, 32)
                if x % 32 == 0 || x % 32 == 31 || y % 32 == 0 || y % 32 == 31 {
                    r += 1.0;
                }

                [r, g, b, 1.0]
            })
            .collect();

        let test_cases = [
            (
                StockId::Portra400,
                FilmFormat::Film35mm,
                "checker",
                &pattern_checker,
                FilmOutput::PositiveLinear,
                32,
            ),
            (
                StockId::ColorNeg200,
                FilmFormat::Film6x6,
                "sine",
                &pattern_sine,
                FilmOutput::NegativeLinear,
                47,
            ),
            (
                StockId::TriX400,
                FilmFormat::Film35mm,
                "checker",
                &pattern_checker,
                FilmOutput::PositiveLinear,
                32,
            ),
        ];

        for (stock, film_format, fix_name, pattern, output, core_size) in test_cases {
            let params = FilmParams {
                stock,
                film_format,
                seed: 42,
                output,
            };

            let gpu_buf_ff = ctx.create_output_buffer(width, height);
            ctx.queue
                .write_buffer(&gpu_buf_ff.buffer, 0, bytemuck::cast_slice(pattern));
            pollster::block_on(process_gpu_full_frame(&ctx, &gpu_buf_ff, &meta, &params)).unwrap();
            let ff_floats = ctx.download_f32(&gpu_buf_ff.buffer, width * height * 4);

            let gpu_buf_roi = ctx.create_output_buffer(width, height);
            ctx.queue
                .write_buffer(&gpu_buf_roi.buffer, 0, bytemuck::cast_slice(pattern));
            pollster::block_on(process_gpu_roi(
                &ctx,
                &gpu_buf_roi,
                &meta,
                &params,
                core_size,
            ))
            .unwrap();
            let roi_floats = ctx.download_f32(&gpu_buf_roi.buffer, width * height * 4);

            assert_eq!(ff_floats.len(), roi_floats.len());

            let mut max_diff = 0.0f32;
            let mut max_ulp_diff: u32 = 0;
            let mut diff_count = 0;
            for (idx, (&a, &b)) in ff_floats.iter().zip(roi_floats.iter()).enumerate() {
                let diff = (a - b).abs();
                let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                if diff > max_diff {
                    max_diff = diff;
                }
                if ulp > max_ulp_diff {
                    max_ulp_diff = ulp;
                }
                if diff > 1e-4 {
                    diff_count += 1;
                    if diff_count <= 5 {
                        let px_i = idx / 4;
                        let ch = idx % 4;
                        let x = px_i % width;
                        let y = px_i / width;
                        eprintln!("Mismatch stock={stock:?} fmt={film_format:?} fix={fix_name} output={output:?} core={core_size} at ({x}, {y}) ch={ch}: ff={a} ({:#x}), roi={b} ({:#x}), diff={diff}, ulp={ulp}", a.to_bits(), b.to_bits());
                    }
                }
            }
            assert_eq!(
                diff_count, 0,
                "stock={stock:?} fmt={film_format:?} fix={fix_name} output={output:?} core={core_size}: {diff_count} floats differed (max diff {max_diff}, max ULP {max_ulp_diff})"
            );
        }
    }

    #[test]
    fn cpu_vs_gpu_parity_test() {
        const CPU_GPU_ABS_TOLERANCE: f32 = 16.0 * f32::EPSILON;

        let ctx = match pollster::block_on(GpuContext::try_new()) {
            Ok(c) => c,
            Err(_) => return,
        };

        let width = 64;
        let height = 64;
        let n = width * height;
        let meta = crate::image::ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::AcesCg),
            ..Default::default()
        };

        let pattern_uniform: Vec<[f32; 4]> = vec![[0.18, 0.18, 0.18, 1.0]; n];

        let test_cases = [
            (
                StockId::Portra400,
                FilmFormat::Film35mm,
                FilmOutput::PositiveLinear,
            ),
            (
                StockId::ColorNeg200,
                FilmFormat::Film6x6,
                FilmOutput::NegativeLinear,
            ),
        ];

        for (stock, film_format, output) in test_cases {
            let params = FilmParams {
                stock,
                film_format,
                seed: 42,
                output,
            };

            // CPU run
            let mut cpu_image = crate::pixel::Image {
                metadata: meta.clone(),
                rgb_data: vec![[0.18, 0.18, 0.18]; n],
                raw_data: std::sync::Arc::from([]),
            };
            crate::film::process(&mut cpu_image, &params).unwrap();

            // GPU run
            let gpu_buf = ctx.create_output_buffer(width, height);
            ctx.queue
                .write_buffer(&gpu_buf.buffer, 0, bytemuck::cast_slice(&pattern_uniform));
            pollster::block_on(process_gpu_full_frame(&ctx, &gpu_buf, &meta, &params)).unwrap();
            let gpu_floats = ctx.download_f32(&gpu_buf.buffer, width * height * 4);

            assert_eq!(
                cpu_image.rgb_data.len(),
                n,
                "CPU output length mismatch stock={stock:?} output={output:?}"
            );
            assert_eq!(
                gpu_floats.len(),
                n * 4,
                "GPU output length mismatch stock={stock:?} output={output:?}"
            );

            for (px, cpu_pixel) in cpu_image.rgb_data.iter().enumerate() {
                let x = px % width;
                let y = px / width;
                for ch in 0..3 {
                    let gpu_value = gpu_floats[px * 4 + ch];
                    let diff = (cpu_pixel[ch] - gpu_value).abs();
                    assert!(
                        cpu_pixel[ch].to_bits() == gpu_value.to_bits()
                            || (diff.is_finite() && diff <= CPU_GPU_ABS_TOLERANCE),
                        "CPU/GPU RGB mismatch stock={stock:?} format={film_format:?} output={output:?} at ({x}, {y}) ch={ch}: CPU={:?} ({:#010x}), GPU={:?} ({:#010x}), diff={}, tolerance={}",
                        cpu_pixel[ch],
                        cpu_pixel[ch].to_bits(),
                        gpu_value,
                        gpu_value.to_bits(),
                        diff,
                        CPU_GPU_ABS_TOLERANCE
                    );
                }
            }
        }
    }
}
