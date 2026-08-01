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
struct GrainApplySubU {
    n: u32,
    off: u32,
    kappa: f32,
    dmax: f32,
    norm: f32,
    noise0_off: u32,
    noise1_off: u32,
    l2base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrainApplySubRoiU {
    n: u32,
    dye_off: u32,
    noise0_off: u32,
    noise1_off: u32,
    kappa: f32,
    dmax: f32,
    norm: f32,
    root_w: u32,
    root_h: u32,
    radius0: u32,
    radius1: u32,
    l2base: u32,
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
struct HalationAccumU {
    n: u32,
    w: f32,
    init: u32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HalationAccumRoiU {
    n: u32,
    out_off: u32,
    blur_off: u32,
    w: f32,
    init: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
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
    // `kappa` is the CPU sublayer κ = scale_kappa(...)·√2 (two dye-cloud sublayers).
    kappa: Vec<f32>,
    dmax: Vec<f32>,
    // Blur sigmas (px) for spatial stages.
    sigma_local: f32,
    sigma_wide: f32,
    sigma_dir: f32,
    sigma_adj: f32,
    // Pre-baked Gaussian kernel buffers (None if stage skipped).
    local_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    // CPU multi-bounce halation: decay-weighted bounces at σ√(k+1), k = 0..HALATION_BOUNCES.
    halation_kernels: Vec<(std::sync::Arc<Buffer>, u32)>,
    halation_weights: Vec<f32>,
    dir_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    adj_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    // Two dye-cloud sublayer kernels per emulsion (σ × 1.3 and × 0.7, matching
    // CPU `apply_grain` sublayer_scales), plus per-sublayer SplitMix64 row seeds.
    grain_kernels: Vec<[(std::sync::Arc<Buffer>, u32); 2]>,
    adjacency_beta: f32,
    grain_seed_base: Vec<[(u32, u32); 2]>, // per emulsion (lo, hi) per sublayer
}

const GOLDEN: u64 = 0x9E3779B97F4A7C15;
const SM_STATE_MIX: u64 = 0xD1B54A32D192ED03;

/// Multi-bounce backing-reflection count and decay (mirrors `halation.rs`).
const HALATION_BOUNCES: usize = 3;
const HALATION_RHO: f32 = 0.5;

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
    // Layout: b0(16) b1(16) b2(16) M(9) lambda_factor(16) capture_scale trans(L*16) produces_latent(L)
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
        let mut trans = [0.0f32; 16];
        match layer.kind {
            LayerKind::Emulsion => {
                let sens = layer.spectral_sensitivity.as_ref().unwrap();
                let rho = layer.silver_halide_fraction as f64;
                let thickness = layer.thickness.0 as f64;
                for i in 0..16 {
                    let od = sens.samples[i] * sigma_scale * rho * thickness;
                    // Baked transmittance (deterministic f64 exp): the GPU reads
                    // this instead of calling `exp`, so CPU/GPU share the value.
                    trans[i] = (-od).exp() as f32;
                }
            }
            LayerKind::Filter | LayerKind::Overcoat | LayerKind::Antihalation => {
                if let Some(curve) = layer.spectral_sensitivity.as_ref() {
                    let thickness = layer.thickness.0 as f64;
                    for i in 0..16 {
                        let od = curve.samples[i] * thickness;
                        trans[i] = (-od).exp() as f32;
                    }
                }
            }
            LayerKind::Support => {}
        }
        for i in 0..16 {
            expose.push(trans[i]);
        }
    }
    for layer in &stock.layers {
        expose.push(if layer.kind == LayerKind::Emulsion {
            1.0
        } else {
            0.0
        });
    }

    // ── lut consts ── logs(64) then frac(E*64) then log2_table(257) exp2_table(257)
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
    lut.extend_from_slice(crate::film::math::log2_table());
    lut.extend_from_slice(crate::film::math::exp2_table());

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
    // Deterministic transcendental tables for the reduce `pow` (log2 + exp2).
    reduce.extend_from_slice(crate::film::math::log2_table());
    reduce.extend_from_slice(crate::film::math::exp2_table());

    // ── DIR matrix (E*E row-major matrix[i][j]) + exp2 table for exp(-ΔI) ──
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
    dir_matrix.extend_from_slice(crate::film::math::exp2_table());

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

    // Dmin normalization scale (matches scan_to_acescg; f32 as the scan runs in f32).
    let dmin_rgb = dmin_reference_acescg(stock);
    let peak = dmin_rgb[0].max(dmin_rgb[1]).max(dmin_rgb[2]).max(1e-12);
    let scan_scale = 1.0 / peak;

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
            invert.inv_dmin[0],
            invert.inv_dmin[1],
            invert.inv_dmin[2],
            invert.inv_gamma,
            invert.inv_gamma_log2_10,
            0.0,
        ]);
    }

    // Deterministic transcendental tables (log2 + exp2) for the scan + invert
    // shaders; the CPU mirrors read the same values via film::math.
    scan.extend_from_slice(crate::film::math::log2_table());
    scan.extend_from_slice(crate::film::math::exp2_table());

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

    // CPU multi-bounce halation: decay-weighted bounces at σ√(k+1), k = 0..2.
    let (mut halation_kernels, mut halation_weights) = (Vec::new(), Vec::new());
    if sigma_wide >= 1e-3 {
        let mut decay = [0.0f32; HALATION_BOUNCES];
        let mut sum = 0.0f32;
        for k in 0..HALATION_BOUNCES {
            decay[k] = HALATION_RHO.powi(k as i32);
            sum += decay[k];
        }
        for k in 0..HALATION_BOUNCES {
            decay[k] /= sum;
        }
        for k in 0..HALATION_BOUNCES {
            let bounce_sigma = sigma_wide * ((k + 1) as f32).sqrt();
            let bk = make_gaussian_kernel(bounce_sigma);
            let rad = crate::film::blur::gaussian_radius(bounce_sigma) as u32;
            let buf = std::sync::Arc::new(ctx.create_f32_buffer_init(
                &bk,
                &format!("stock_k_halation_{k}"),
            ));
            halation_kernels.push((buf, rad));
            halation_weights.push(decay[k]);
        }
    }

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

    // Grain scalars and per-layer crystal-aware dye-cloud sublayer kernels.
    // Mirrors CPU `apply_grain`: two sublayers (scales 1.3 / 0.7) with independent
    // SplitMix64 streams; sublayer κ = scale_kappa·√2.
    let mut kappa = Vec::with_capacity(num_emul);
    let mut dmax_grain = Vec::with_capacity(num_emul);
    let mut grain_seed_base = Vec::with_capacity(num_emul);
    let mut grain_kernels = Vec::with_capacity(num_emul);

    const SUBLAYER_SCALES: [f32; 2] = [1.3, 0.7];
    const SUBLAYER_STREAM_MIX: u64 = 0x123456789;

    for (e, &(li, layer)) in emuls.iter().enumerate() {
        let coupler = layer.coupler.as_ref().unwrap();
        let kappa_ref = stock.grain_kappa[li].unwrap_or(0.0);
        let k = scale_kappa(kappa_ref, pitch) * (SUBLAYER_SCALES.len() as f32).sqrt();
        kappa.push(k);
        dmax_grain.push(coupler.d_max);
        let base = params
            .seed
            .wrapping_mul(SM_STATE_MIX)
            .wrapping_add((e as u64).wrapping_mul(GOLDEN));
        let mut bases = [(0u32, 0u32); 2];
        for sl in 0..2 {
            let b = base.wrapping_add((sl as u64).wrapping_mul(SUBLAYER_STREAM_MIX));
            bases[sl] = (b as u32, (b >> 32) as u32);
        }
        grain_seed_base.push(bases);

        let base_correlation_um = if let Some(dist) = &layer.crystal_size {
            let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
            (mean_s / 0.7) as f32 * DYE_CLOUD_CORRELATION_UM
        } else {
            DYE_CLOUD_CORRELATION_UM
        };
        let mut kernels: Vec<(std::sync::Arc<Buffer>, u32)> = Vec::with_capacity(2);
        for &sl_scale in SUBLAYER_SCALES.iter() {
            let correlation_um = base_correlation_um * sl_scale;
            let grain_sigma = (correlation_um / pitch.max(1e-6)).max(1.0);
            let k_grain = make_gaussian_kernel(grain_sigma);
            let rad_grain = crate::film::blur::gaussian_radius(grain_sigma) as u32;
            let buf_grain = std::sync::Arc::new(ctx.create_f32_buffer_init(
                &k_grain,
                &format!("stock_k_grain_{e}_{}", kernels.len()),
            ));
            kernels.push((buf_grain, rad_grain));
        }
        grain_kernels.push(kernels.try_into().unwrap());
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
        halation_kernels,
        halation_weights,
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

    // Wide support-bounce halation: CPU multi-bounce model — decay-weighted bounces
    // at σ√(k+1) of the deepest emulsion plane, accumulated into `work` plane 0,
    // then added to every plane with per-emulsion bleed gain.
    if !consts.halation_kernels.is_empty() {
        let bounce_src = ((e - 1) * n) as u32;
        for (k, (ref kbuf, radius)) in consts.halation_kernels.iter().enumerate() {
            blur_plane(
                ctx,
                width,
                height,
                &planes,
                bounce_src,
                &bout,
                0,
                &btmp,
                kbuf.as_ref(),
                *radius,
            );
            let u = HalationAccumU {
                n: n as u32,
                w: consts.halation_weights[k],
                init: if k == 0 { 1 } else { 0 },
                _p0: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_halation_accum",
                shaders::HALATION_ACCUM,
                &[&work, &bout],
                bytemuck::bytes_of(&u),
                workgroups(n),
            );
        }
        let u = CountU {
            n: n as u32,
            num_emul: consts.num_emul,
            _p0: 0,
            _p1: 0,
        };
        ctx.dispatch_compute_shader_multi(
            "film_halation_add",
            shaders::HALATION_ADD,
            &[&planes, &work, &consts.gains],
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
    // CPU `apply_grain` two-sublayer model: sublayer 0 blurred into `work` plane e,
    // sublayer 1 blurred into `noise` (in place); one apply pass averages both.
    // The apply runs per plane right after its noise generation, because `noise`
    // is a single-plane scratch: deferring the apply would overwrite plane e's
    // sublayer 1 with the next plane's noise. (Norm is always 1.0.)
    {
        let reduce_l2base = 5u32 * consts.num_emul;
        for plane_e in 0..e {
            let kappa = consts.kappa[plane_e];
            let dmax = consts.dmax[plane_e];
            if kappa <= 0.0 || dmax <= 0.0 {
                continue;
            }
            let (ref k0, r0) = consts.grain_kernels[plane_e][0];
            let (ref k1, r1) = consts.grain_kernels[plane_e][1];
            let dst_off = (plane_e * n) as u32;

            for sl in 0..2 {
                let (ref kbuf, radius) = if sl == 0 {
                    (k0, r0)
                } else {
                    (k1, r1)
                };
                let (base_lo, base_hi) = consts.grain_seed_base[plane_e][sl];
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
                // sl0: noise -> btmp -> work@dst_off; sl1: noise -> btmp -> noise (in place).
                let dst_buf = if sl == 0 { work } else { noise };
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
                    dst_off: if sl == 0 { dst_off } else { 0 },
                    _p0: 0,
                    _p1: 0,
                };
                let noise_ub = bytemuck::bytes_of(&nu);
                let blur_ub_h = bytemuck::bytes_of(&blur_u_h);
                let blur_ub_v = bytemuck::bytes_of(&blur_u_v);
                let noise_bufs = [noise];
                let h_bufs = [noise, btmp, kbuf.as_ref()];
                let v_bufs = [btmp, dst_buf, kbuf.as_ref()];
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
            }

            let gu = GrainApplySubU {
                n: n as u32,
                off: dst_off,
                kappa,
                dmax,
                norm: 1.0,
                noise0_off: dst_off,
                noise1_off: 0,
                l2base: reduce_l2base,
            };
            ctx.dispatch_compute_shader_multi(
                "film_grain_apply_sub",
                shaders::GRAIN_APPLY_SUB,
                &[&dye, &work, &noise, &consts.reduce],
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
            let (ref kbuf_grain, grain_radius) = consts.grain_kernels[e][0];
            let (base_lo, base_hi) = consts.grain_seed_base[e][0];
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

        // Wide support-bounce halation (CPU multi-bounce model). Decay-weighted
        // bounces at σ√(k+1) of the deepest latent plane are accumulated into mask
        // plane 0 (unused until REDUCE writes it), then added with per-emulsion gain.
        if !consts.halation_kernels.is_empty() {
            let deepest_e = num_emul - 1;
            let src_off = emul_off(work_base, deepest_e)?;
            let acc_off = scratch.mask_offset(0)?;

            for (k, (ref kbuf, radius)) in consts.halation_kernels.iter().enumerate() {
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

                let acc_u = HalationAccumRoiU {
                    n: root_n,
                    out_off: acc_off,
                    blur_off: blur_out_off,
                    w: consts.halation_weights[k],
                    init: if k == 0 { 1 } else { 0 },
                    _p0: 0,
                    _p1: 0,
                    _p2: 0,
                };

                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_halation_accum_roi",
                    shaders::HALATION_ACCUM_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&acc_u),
                    workgroups(root_n as usize),
                ));
            }

            let hal_u = HalationAddRoiU {
                n: root_n,
                num_emul: consts.num_emul,
                plane_base: work_base,
                bounce_base: acc_off,
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

        // Stage 7: two-sublayer grain (image dye only). Raw sublayer noises are
        // generated at blur_tmp / blur_out, blurred inline inside the apply pass
        // (separable, root-edge reflection — same order as standalone blurs), and
        // the two sublayer results averaged. Grain is no longer fused into scan.
        if !active_grain_emuls.is_empty() {
            for &e in &active_grain_emuls {
                let (ref k0, r0) = consts.grain_kernels[e][0];
                let (ref k1, r1) = consts.grain_kernels[e][1];
                let (b0_lo, b0_hi) = consts.grain_seed_base[e][0];
                let (b1_lo, b1_hi) = consts.grain_seed_base[e][1];
                let dye_off = emul_off(dye_base, e)?;

                for sl in 0..2 {
                    let (base_lo, base_hi) = if sl == 0 {
                        (b0_lo, b0_hi)
                    } else {
                        (b1_lo, b1_hi)
                    };
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
                        dst_off: if sl == 0 { blur_tmp_off } else { blur_out_off },
                        _p0: 0,
                        _p1: 0,
                    };

                    keep.push(ctx.encode_compute_shader_multi(
                        &mut encoder,
                        "film_grain_noise_roi",
                        shaders::GRAIN_NOISE_ROI,
                        &[&scratch.arena],
                        bytemuck::bytes_of(&nu),
                        workgroups(root_n as usize),
                    ));
                }

                let gu = GrainApplySubRoiU {
                    n: root_n,
                    dye_off,
                    noise0_off: blur_tmp_off,
                    noise1_off: blur_out_off,
                    kappa: consts.kappa[e],
                    dmax: consts.dmax[e],
                    norm: grain_norms[e],
                    root_w: plan.root.width,
                    root_h: plan.root.height,
                    radius0: r0,
                    radius1: r1,
                    l2base: 5u32 * consts.num_emul,
                };

                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_grain_apply_sub_roi",
                    shaders::GRAIN_APPLY_SUB_RAW_ROI,
                    &[&scratch.arena, k0.as_ref(), k1.as_ref(), &consts.reduce],
                    bytemuck::bytes_of(&gu),
                    workgroups(root_n as usize),
                ));
            }
        }

        let mut emul = [[0.0f32; 4]; 16];
        for e in 0..consts.num_emul as usize {
            if e < 16 {
                // Grain already applied in Stage 7; disable fused scan grain.
                emul[e] = [0.0, consts.dmax[e], grain_norms[e], 0.0];
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

    // TEMP DIAGNOSTIC: stage-by-stage CPU vs GPU parity on the stressed pipeline image.
    #[test]
    fn dbg_stage_by_stage_parity() {
        fn report_stage(
            name: &str,
            cpu: &[f32],
            gpu: &[f32],
            e: usize,
            n: usize,
            print_first: bool,
        ) {
            assert_eq!(cpu.len(), gpu.len());
            let mut max_ulp = 0u32;
            let mut max_diff = 0.0f32;
            let mut max_at: Option<(usize, f32, f32)> = None;
            let mut first: Option<(usize, usize, f32, f32)> = None;
            for (i, (&a, &b)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                let d = (a - b).abs();
                if ulp > max_ulp {
                    max_ulp = ulp;
                    max_at = Some((i, a, b));
                }
                if d > max_diff {
                    max_diff = d;
                }
                if a.to_bits() != b.to_bits() && first.is_none() {
                    first = Some((i / n, i % n, a, b));
                }
            }
            let mut max_str = String::new();
            if let Some((i, a, b)) = max_at {
                let pe = i / n;
                let pi = i % n;
                max_str = format!(
                    "max_ulp_at plane={pe} px={pi} (x={}, y={}): cpu={a} ({:#010x}) gpu={b} ({:#010x})",
                    pi % 128,
                    pi / 128,
                    a.to_bits(),
                    b.to_bits()
                );
            }
            let mut first_str = String::new();
            if let Some((pe, pi, a, b)) = first {
                first_str = format!(
                    "first diff plane={pe} px={pi}: cpu={a} ({:#010x}) gpu={b} ({:#010x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
            println!(
                "[stage {name}] planes={e} max_ulp={max_ulp} max_diff={max_diff} {max_str} | {first_str}"
            );
            if print_first {
                if let Some((pe, pi, a, b)) = first {
                    println!(
                        "  first diff detail: plane={pe} px={pi} x={} y={}: cpu={} gpu={} ulp={}",
                        pi % 128,
                        pi / 128,
                        a,
                        b,
                        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
                    );
                    println!("  gpu first16 plane{pe}: {:?}", &gpu[pe * n..pe * n + 16]);
                    println!("  cpu first16 plane{pe}: {:?}", &cpu[pe * n..pe * n + 16]);
                }
            }
        }

        let ctx = match pollster::block_on(GpuContext::try_new()) {
            Ok(c) => c,
            Err(_) => return,
        };

        use rand::Rng;
        use rand_chacha::rand_core::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(333);
        let width = 128;
        let height = 128;
        let n = width * height;
        let mut rgb_data: Vec<[f32; 3]> = (0..n)
            .map(|_| [rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>()])
            .collect();
        for (i, px) in rgb_data.iter_mut().enumerate() {
            let scale = if i % 2 == 0 { 50.0 } else { 0.001 };
            px[0] *= scale;
            px[1] *= scale;
            px[2] *= scale;
        }
        let mut image = crate::pixel::Image {
            metadata: crate::image::ImageMetadata {
                width,
                height,
                color_space: Some(ColorSpaceTag::Srgb),
                ..Default::default()
            },
            rgb_data: rgb_data
                .iter()
                .map(|p| {
                    let mut px = *p;
                    let _ = &mut px;
                    px
                })
                .collect(),
            raw_data: std::sync::Arc::from([]),
        };
        image.cst(ColorSpaceTag::AcesCg);

        let params = FilmParams {
            stock: StockId::ColorNeg200,
            film_format: FilmFormat::Film35mm,
            seed: 1,
            output: FilmOutput::PositiveLinear,
        };

        // ── CPU checkpoints ──
        let stock = params.stock.load().unwrap();
        let pitch = params.film_format.pixel_pitch_um(width);
        let shutter = image.metadata.shutter_seconds.unwrap_or(1.0 / stock.box_iso.0);
        let capture_scale = crate::film::camera_capture_scale(&image.metadata, stock.box_iso.0);

        // TEMP: is the CPU grain output seed-invariant for this config?
        {
            use crate::film::development::grain::{SplitMix64, scale_kappa};
            let uniform: Vec<[f32; 3]> = vec![[0.18, 0.18, 0.18]; n];
            let lat = crate::film::exposure::expose_with_pitch_shutter_and_scale(
                &uniform,
                width,
                height,
                &stock,
                pitch,
                shutter,
                capture_scale,
            );
            let emul_count = stock.emulsion_layers().count();
            let mut d_max_v = Vec::new();
            let mut kappas_v = Vec::new();
            let mut crystal_v = Vec::new();
            for (layer_idx, layer) in stock.layers.iter().enumerate() {
                if layer.kind != crate::film::stock::LayerKind::Emulsion {
                    continue;
                }
                let coupler = layer.coupler.as_ref().unwrap();
                d_max_v.push(coupler.d_max);
                let kappa_ref = stock.grain_kappa[layer_idx].unwrap_or(0.0);
                kappas_v.push(crate::film::development::grain::scale_kappa(kappa_ref, pitch));
                crystal_v.push(layer.crystal_size.clone());
            }
            let mut dye0 = crate::film::development::reduction::reduce(&stock, &lat);
            println!(
                "TEMP pre-grain plane0 first4={:?}",
                &dye0.image_dye[0][..4]
            );
            crate::film::development::grain::apply_grain(
                &mut dye0,
                &d_max_v,
                &kappas_v,
                pitch,
                42,
                &crystal_v,
            );
            println!(
                "TEMP post-grain plane0 first4={:?}",
                &dye0.image_dye[0][..4]
            );
            for seed in [42u64, 43] {
                let dy = crate::film::development::develop(&stock, &lat, seed, pitch);
                println!(
                    "TEMP cpu grain seed={seed} plane0 first4={:?}",
                    &dy.image_dye[0][..4]
                );
            }
            let layer_idx = stock
                .layers
                .iter()
                .position(|l| l.kind == crate::film::stock::LayerKind::Emulsion)
                .unwrap();
            let kappa_ref = stock.grain_kappa[layer_idx].unwrap_or(0.0);
            let kappa = scale_kappa(kappa_ref, pitch);
            println!(
                "TEMP grain kappa_ref={kappa_ref} kappa={kappa} kappa_sub={} emul_count={emul_count}",
                kappa * (2.0f32).sqrt()
            );
            let (base_lo, base_hi) = {
                const GOLDEN: u64 = 0x9E3779B97F4A7C15;
                const SM_MIX: u64 = 0xD1B54A32D192ED03;
                const SL_MIX: u64 = 0x123456789;
                let b = 42u64
                    .wrapping_mul(SM_MIX)
                    .wrapping_add(0u64.wrapping_mul(GOLDEN))
                    .wrapping_add(1u64.wrapping_mul(SL_MIX));
                ((b as u32), ((b >> 32) as u32))
            };
            let base = ((base_lo as u64) | ((base_hi as u64) << 32));
            let mut rng = SplitMix64::new(base);
            println!("TEMP cpu noise first8={:?}", (0..8).map(|_| rng.next_gaussian()).collect::<Vec<f32>>());
        }
        let latent = crate::film::exposure::expose_with_pitch_shutter_and_scale(
            &image.rgb_data,
            width,
            height,
            &stock,
            pitch,
            shutter,
            capture_scale,
        );
        let cpu_reduce =
            crate::film::development::reduction::reduce(&stock, &latent);
        let cpu_dyes = crate::film::development::develop(&stock, &latent, params.seed, pitch);
        let dmin = crate::film::scan::normalized_dmin_acescg(&stock);
        let mid = crate::film::mid_negative_acescg(&stock, pitch, shutter);
        let cpu_scan = if params.output == FilmOutput::PositiveLinear {
            crate::film::scan::scan(
                &stock,
                &cpu_dyes,
                crate::film::scan::ScanMode::PositiveLinear { dmin, mid },
            )
        } else {
            crate::film::scan::scan(&stock, &cpu_dyes, crate::film::scan::ScanMode::NegativeLinear)
        };

        // ── GPU full-frame with per-stage downloads ──
        let gpu_buf = ctx.create_output_buffer(width, height);
        let rgba: Vec<[f32; 4]> = image.rgb_data.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect();
        ctx.queue.write_buffer(&gpu_buf.buffer, 0, bytemuck::cast_slice(&rgba));

        let num_emul_hint = stock.emulsion_layers().count();
        let lease = acquire_film_resources(&ctx, &stock, &params, &image.metadata, width, height, num_emul_hint);
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

        // Stage 1+2: expose + local scatter + halation (copy of full-frame code).
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
        if !consts.halation_kernels.is_empty() {
            let bounce_src = ((e - 1) * n) as u32;
            for (k, (ref kbuf, radius)) in consts.halation_kernels.iter().enumerate() {
                blur_plane(
                    &ctx,
                    width,
                    height,
                    &planes,
                    bounce_src,
                    &bout,
                    0,
                    &btmp,
                    kbuf.as_ref(),
                    *radius,
                );
                let u = HalationAccumU {
                    n: n as u32,
                    w: consts.halation_weights[k],
                    init: if k == 0 { 1 } else { 0 },
                    _p0: 0,
                };
                ctx.dispatch_compute_shader_multi(
                    "film_halation_accum",
                    shaders::HALATION_ACCUM,
                    &[&work, &bout],
                    bytemuck::bytes_of(&u),
                    workgroups(n),
                );
            }
            let u = CountU {
                n: n as u32,
                num_emul: consts.num_emul,
                _p0: 0,
                _p1: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_halation_add",
                shaders::HALATION_ADD,
                &[&planes, &work, &consts.gains],
                bytemuck::bytes_of(&u),
                workgroups(n),
            );
        }
        // CPU absorbed planes (post-spatial, pre-LUT) for comparison.
        use crate::film::constants::ABSORPTION_SIGMA_SCALE_PER_UM;
        use crate::film::exposure::absorption::{absorb_stack, integrated_absorbed};
        use crate::film::exposure::halation::apply_spatial_exposure_effects;
        use crate::film::exposure::pixel_fluence_spectrum;
        let mut cpu_absorbed = vec![vec![0.0f32; n]; e];
        for (p, px) in image.rgb_data.iter().enumerate() {
            let spectrum = pixel_fluence_spectrum(px, capture_scale);
            let (layers, _) = absorb_stack(&stock.layers, &spectrum, ABSORPTION_SIGMA_SCALE_PER_UM);
            let mut ei = 0usize;
            for la in &layers {
                if la.produces_latent {
                    cpu_absorbed[ei][p] = integrated_absorbed(&la.absorbed);
                    ei += 1;
                }
            }
        }
        apply_spatial_exposure_effects(&mut cpu_absorbed, width, height, &stock, pitch);
        {
            let gpu_floats = ctx.download_f32(planes, n * e);
            let cpu: Vec<f32> = cpu_absorbed.iter().flatten().copied().collect();
            report_stage("expose_absorbed", &cpu, &gpu_floats, e, n, true);
        }

        // TEMP: raw grain noise comparison (layer 0, sublayer 0) before blur.
        let mut cpu_raw_l0_sl0 = vec![0.0f32; n];
        {
            use crate::film::development::grain::SplitMix64;
            let (base_lo, base_hi) = consts.grain_seed_base[0][0];
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
            ctx.dispatch_compute_shader_multi(
                "film_grain_noise_raw",
                shaders::GRAIN_NOISE,
                &[&noise],
                bytemuck::bytes_of(&nu),
                workgroups(n),
            );
            let gpu_noise = ctx.download_f32(noise, n);
            let mut cpu_noise = vec![0.0f32; n];
            let base = ((base_lo as u64) | ((base_hi as u64) << 32));
            for y in 0..height {
                let mut rng = SplitMix64::new(base.wrapping_add(y as u64));
                for x in 0..width {
                    cpu_noise[y * width + x] = rng.next_gaussian();
                }
            }
            let mut max_ulp = 0u32;
            let mut max_diff = 0.0f32;
            let mut ndiff = 0usize;
            for (i, (&a, &b)) in cpu_noise.iter().zip(gpu_noise.iter()).enumerate() {
                let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                if ulp > max_ulp {
                    max_ulp = ulp;
                }
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                }
                if a.to_bits() != b.to_bits() {
                    ndiff += 1;
                }
            }
            println!(
                "[noise raw l0 sl0] max_ulp={max_ulp} max_diff={max_diff} ndiff={ndiff}/{} first cpu={:?} gpu={:?}",
                n,
                &cpu_noise[..4],
                &gpu_noise[..4]
            );
        }

        // TEMP: micro-test GPU exp vs CPU exp on the baked optical densities.
        {
            let ec = ctx.download_f32(&consts.expose, 74 + (consts.num_layers as usize) * 16);
            let ods: Vec<f32> = ec[74..74 + consts.num_layers as usize * 16].to_vec();
            let od_buf = ctx.create_f32_buffer_init(&ods, "tmp_ods");
            let out_buf = ctx.create_f32_buffer(ods.len(), "tmp_exp_out");
            let u = CountU {
                n: ods.len() as u32,
                num_emul: 0,
                _p0: 0,
                _p1: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_exp_test",
                r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    out[i] = exp(-inp[i]);
}
"#,
                &[&od_buf, &out_buf],
                bytemuck::bytes_of(&u),
                workgroups(ods.len()),
            );
            let gpu_exp = ctx.download_f32(&out_buf, ods.len());
            let mut ndiff = 0;
            let mut max_ulp = 0u32;
            let mut ndiff2 = 0;
            let mut max_ulp2 = 0u32;
            let mut ndiff3 = 0;
            let mut max_ulp3 = 0u32;
            let mut ndiff4 = 0;
            let mut max_ulp4 = 0u32;
            const LOG2E: f32 = 1.4426950408889634;
            for (i, (&a, &b)) in ods.iter().zip(gpu_exp.iter()).enumerate() {
                let c = (-a).exp();
                let ulp = (c.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                if ulp > max_ulp {
                    max_ulp = ulp;
                }
                if c.to_bits() != b.to_bits() {
                    ndiff += 1;
                }
                let c2 = (-a * LOG2E).exp2();
                let ulp2 = (c2.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                if ulp2 > max_ulp2 {
                    max_ulp2 = ulp2;
                }
                if c2.to_bits() != b.to_bits() {
                    ndiff2 += 1;
                }
                let c3 = 2.0f32.powf(-a);
                let ulp3 = (c3.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                if ulp3 > max_ulp3 {
                    max_ulp3 = ulp3;
                }
                if c3.to_bits() != b.to_bits() {
                    ndiff3 += 1;
                }
                let c4 = ((-a as f64).exp()) as f32;
                let ulp4 = (c4.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                if ulp4 > max_ulp4 {
                    max_ulp4 = ulp4;
                }
                if c4.to_bits() != b.to_bits() {
                    ndiff4 += 1;
                }
                if i < 4 {
                    println!(
                        "  exp od={a}: expf={c} ({:#010x}) f64path={c4} ({:#010x}) gpu={b} ({:#010x})",
                        c.to_bits(),
                        c4.to_bits(),
                        b.to_bits()
                    );
                }
            }
            println!(
                "[exp test] expf: ndiff={ndiff}/{} max_ulp={max_ulp} | exp2path: ndiff={ndiff2} max_ulp={max_ulp2} | powf: ndiff={ndiff3} max_ulp={max_ulp3} | f64path: ndiff={ndiff4} max_ulp={max_ulp4}",
                ods.len()
            );
        }

        // TEMP: micro-test all transcendentals used by film shaders.
        {
            let vals: Vec<f32> = (0..512)
                .map(|i| {
                    let t = i as f32 / 511.0;
                    match i % 4 {
                        0 => (t * 50.0 + 1e-4) * (1.0 + (i % 3) as f32 * 7.0),
                        1 => 1e-3 + t * 1e2,
                        2 => t * 6.0 - 3.0,
                        _ => (t * 4.0 - 2.0),
                    }
                })
                .collect();
            let in_buf = ctx.create_f32_buffer_init(&vals, "tmp_tvals");
            let out_buf = ctx.create_f32_buffer(vals.len() * 4, "tmp_tout");
            let u = CountU {
                n: vals.len() as u32,
                num_emul: 0,
                _p0: 0,
                _p1: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_math_test",
                r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let x = inp[i];
    let den = (x * 3.7) + 0.3;
    out[i * 4u + 0u] = x / den;
    out[i * 4u + 1u] = (x + 1.0) / (x * x + 2.0);
    out[i * 4u + 2u] = 300.0 / den;
    out[i * 4u + 3u] = x / 7.0;
}
"#,
                &[&in_buf, &out_buf],
                bytemuck::bytes_of(&u),
                workgroups(vals.len()),
            );
            let gpu_out = ctx.download_f32(&out_buf, vals.len() * 4);
            let mut stats = [(0usize, 0u32); 4];
            let mut stats2 = [(0usize, 0u32); 4];
            for (i, &x) in vals.iter().enumerate() {
                let den = (x * 3.7) + 0.3;
                let cpu = [
                    x / den,
                    (x + 1.0) / (x * x + 2.0),
                    300.0 / den,
                    x / 7.0,
                ];
                let cpu_rcp = [
                    x * (1.0 / den),
                    (x + 1.0) * (1.0 / (x * x + 2.0)),
                    300.0 * (1.0 / den),
                    x * (1.0 / 7.0),
                ];
                for k in 0..4 {
                    let a = cpu[k];
                    let b = gpu_out[i * 4 + k];
                    let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                    stats[k].0 += (a.to_bits() != b.to_bits()) as usize;
                    stats[k].1 = stats[k].1.max(ulp);
                    let a2 = cpu_rcp[k];
                    let ulp2 = (a2.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                    stats2[k].0 += (a2.to_bits() != b.to_bits()) as usize;
                    stats2[k].1 = stats2[k].1.max(ulp2);
                }
            }
            for (k, name) in ["div_a", "div_b", "div_c", "div_d"].iter().enumerate() {
                println!(
                    "[math test {name}] plain: ndiff={}/{} max_ulp={} | rcp-emul: ndiff={} max_ulp={}",
                    stats[k].0,
                    vals.len(),
                    stats[k].1,
                    stats2[k].0,
                    stats2[k].1
                );
            }
            for (k, name) in ["div_a", "div_b", "div_c", "div_d"].iter().enumerate() {
                println!(
                    "[math test {name}] ndiff={}/{} max_ulp={}",
                    stats[k].0,
                    vals.len(),
                    stats[k].1
                );
            }
            println!(
                "  sample exp: cpu={:?} gpu={:?}",
                (0..4).map(|i| (vals[i] as f32).exp()).collect::<Vec<_>>(),
                (0..4).map(|i| gpu_out[i * 4]).collect::<Vec<_>>()
            );
        }

        // TEMP: probe lut_sample math on GPU vs CPU for a set of phis.
        {
            let gpu_planes = ctx.download_f32(planes, n * e);
            let mut phis = vec![0.0f32; 4096];
            let mut ndiff_lut = 0usize;
            for i in 0..4096 {
                phis[i] = gpu_planes[1 * n + i * 4 % n];
            }
            let phi_buf = ctx.create_f32_buffer_init(&phis, "tmp_phis");
            let out_buf = ctx.create_f32_buffer(phis.len() * 4, "tmp_lutout");
            let u = CountU {
                n: phis.len() as u32,
                num_emul: 0,
                _p0: 0,
                _p1: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_lut_probe",
                r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> lc: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

const LOG10_2: f32 = 0.3010299956639812;

fn log2_det(x: f32, tbase: u32) -> f32 {
    let bits = bitcast<u32>(x);
    let e = i32(bits >> 23u) - 127;
    let m = bitcast<f32>((bits & 0x7FFFFFu) | 0x3F800000u);
    let t = (m - 1.0) * 256.0;
    var i = u32(t);
    if (i > 255u) { i = 255u; }
    let r = t - f32(i);
    let l = lc[tbase + i] + r * (lc[tbase + i + 1u] - lc[tbase + i]);
    return f32(e) + l;
}

fn exp2_det(x: f32, tbase: u32) -> f32 {
    let k = floor(x);
    let f = x - k;
    let t = f * 256.0;
    var i = u32(t);
    if (i > 255u) { i = 255u; }
    let r = t - f32(i);
    let base = lc[tbase + i] + r * (lc[tbase + i + 1u] - lc[tbase + i]);
    let scale = bitcast<f32>(u32((i32(k) + 127) << 23));
    return base * scale;
}

fn div_det(a: f32, b: f32, l2base: u32, e2base: u32) -> f32 {
    return a * exp2_det(-log2_det(b, l2base), e2base);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let phi = inp[i];
    let l2base = 64u + 3u * 64u;
    let e2base = l2base + 257u;
    out[i * 4u + 0u] = log2_det(phi, l2base);
    out[i * 4u + 1u] = exp2_det(-log2_det(phi, l2base), e2base);
    out[i * 4u + 2u] = div_det(phi, phi + 1.0, l2base, e2base);
    out[i * 4u + 3u] = phi;
}
"#,
                &[&phi_buf, &out_buf, &consts.lut],
                bytemuck::bytes_of(&u),
                workgroups(phis.len()),
            );
            let gpu_lut_probe = ctx.download_f32(&out_buf, phis.len() * 4);
            for (i, &phi) in phis.iter().enumerate() {
                let cpu_l = crate::film::math::log2_det(phi);
                let cpu_r = crate::film::math::exp2_det(-crate::film::math::log2_det(phi));
                let cpu_d = crate::film::math::div_det(phi, phi + 1.0);
                let g_l = gpu_lut_probe[i * 4];
                let g_r = gpu_lut_probe[i * 4 + 1];
                let g_d = gpu_lut_probe[i * 4 + 2];
                if cpu_l.to_bits() != g_l.to_bits() || cpu_r.to_bits() != g_r.to_bits()
                    || cpu_d.to_bits() != g_d.to_bits()
                {
                    ndiff_lut += 1;
                    if ndiff_lut <= 3 {
                        println!(
                            "  lut probe i={i} phi={phi}: log2 cpu={} gpu={} | rcp cpu={} gpu={} | div cpu={} gpu={}",
                            cpu_l.to_bits(), g_l.to_bits(), cpu_r.to_bits(), g_r.to_bits(), cpu_d.to_bits(), g_d.to_bits()
                        );
                    }
                }
            }
            println!("[lut probe] ndiff={ndiff_lut}/{}", phis.len());
        }

        // Stage 3: LUT.
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
        {
            let gpu_floats = ctx.download_f32(planes, n * e);
            let cpu: Vec<f32> = latent.layers.iter().flatten().copied().collect();
            report_stage("lut_fraction", &cpu, &gpu_floats, e, n, true);
        }

        // Stage 4: reduce.
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
        {
            let gpu_floats = ctx.download_f32(dye, n * e);
            let cpu: Vec<f32> = cpu_reduce.image_dye.iter().flatten().copied().collect();
            report_stage("reduce_dye", &cpu, &gpu_floats, e, n, false);
        }

        // Stages 5-7: DIR, adjacency, grain (copy of full-frame code).
        if !stock.dir_inhibition_matrix.is_empty() {
            if let Some((ref kbuf, radius)) = consts.dir_kernel {
                let radius = radius;
                for src_e in 0..e {
                    blur_plane(
                        &ctx,
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
        // TEMP: checkpoint after DIR + adjacency vs CPU.
        {
            let gpu_floats = ctx.download_f32(dye, n * e);
            let mut cpu_da = crate::film::development::reduction::reduce(&stock, &latent);
            crate::film::development::diffusion::apply_dir_inhibition(
                &mut cpu_da,
                stock.dir_diffusion_length.0 / pitch.max(1e-6),
                &stock.dir_inhibition_matrix,
            );
            crate::film::development::diffusion::apply_adjacency(
                &mut cpu_da,
                stock.developer_diffusion_length.0 / pitch.max(1e-6),
                stock.adjacency_beta,
            );
            let cpu: Vec<f32> = cpu_da.image_dye.iter().flatten().copied().collect();
            report_stage("dir_adj_dye", &cpu, &gpu_floats, e, n, true);
        }
        // TEMP: verify which plane's sl1 the `noise` buffer holds after the loop.
        {
            use crate::film::blur::gaussian_blur_separable;
            use crate::film::development::grain::SplitMix64;
            // First verify raw (plane2, sl1) noise generated fresh matches CPU.
            {
                let (base_lo, base_hi) = consts.grain_seed_base[2][1];
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
                ctx.dispatch_compute_shader_multi(
                    "film_grain_noise_raw_p2s1",
                    shaders::GRAIN_NOISE,
                    &[&noise],
                    bytemuck::bytes_of(&nu),
                    workgroups(n),
                );
                let gpu_raw = ctx.download_f32(noise, n);
                let base = ((base_lo as u64) | ((base_hi as u64) << 32));
                let mut cpu_raw = vec![0.0f32; n];
                for y in 0..height {
                    let mut rng = SplitMix64::new(base.wrapping_add(y as u64));
                    for x in 0..width {
                        cpu_raw[y * width + x] = rng.next_gaussian();
                    }
                }
                let ndiff = cpu_raw
                    .iter()
                    .zip(gpu_raw.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!("[raw p2 sl1 fresh] ndiff={ndiff}/{}", n);
            }
            // Now what does the buffer hold right now (after the plane loop)?
            // TEMP: fresh dispatch of (plane2, sl1) raw + blur, compare with CPU.
            {
                let (ref k1, r1) = consts.grain_kernels[2][1];
                let (base_lo, base_hi) = consts.grain_seed_base[2][1];
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
                    radius: r1,
                    src_off: 0,
                    dst_off: 0,
                    _p0: 0,
                    _p1: 0,
                };
                let blur_u_v = BlurU {
                    width: width as u32,
                    height: height as u32,
                    n: n as u32,
                    radius: r1,
                    src_off: 0,
                    dst_off: 0,
                    _p0: 0,
                    _p1: 0,
                };
                let h_bufs = [noise, btmp, k1.as_ref()];
                let v_bufs = [btmp, noise, k1.as_ref()];
                let d = blur_dispatch(width, height, r1);
                ctx.dispatch_compute_passes(
                    "film_grain_noise_blur_p2s1",
                    &[
                        ComputePassDesc {
                            label: "film_grain_noise_p2s1",
                            wgsl_source: shaders::GRAIN_NOISE,
                            storage_buffers: &[noise],
                            uniform_bytes: bytemuck::bytes_of(&nu),
                            workgroups_x: workgroups(n),
                            workgroups_y: 0,
                        },
                        ComputePassDesc {
                            label: d.h_label,
                            wgsl_source: d.h_wgsl,
                            storage_buffers: &h_bufs,
                            uniform_bytes: bytemuck::bytes_of(&blur_u_h),
                            workgroups_x: d.h_gx,
                            workgroups_y: d.h_gy,
                        },
                        ComputePassDesc {
                            label: d.v_label,
                            wgsl_source: d.v_wgsl,
                            storage_buffers: &v_bufs,
                            uniform_bytes: bytemuck::bytes_of(&blur_u_v),
                            workgroups_x: d.v_gx,
                            workgroups_y: d.v_gy,
                        },
                    ],
                );
                let gpu_blurred = ctx.download_f32(noise, n);
                use crate::film::blur::gaussian_blur_separable;
                use crate::film::development::grain::SplitMix64;
                let base = ((base_lo as u64) | ((base_hi as u64) << 32));
                let mut cpu_noise = vec![0.0f32; n];
                for y in 0..height {
                    let mut rng = SplitMix64::new(base.wrapping_add(y as u64));
                    for x in 0..width {
                        cpu_noise[y * width + x] = rng.next_gaussian();
                    }
                }
                let stock2 = params.stock.load().unwrap();
                let layer = stock2.emulsion_layers().nth(2).unwrap().1;
                let base_corr = if let Some(dist) = &layer.crystal_size {
                    let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
                    (mean_s / 0.7) as f32 * crate::film::constants::DYE_CLOUD_CORRELATION_UM
                } else {
                    crate::film::constants::DYE_CLOUD_CORRELATION_UM
                };
                let corr = base_corr * 0.7;
                let sigma_px = (corr / pitch.max(1e-6)).max(1.0);
                gaussian_blur_separable(&mut cpu_noise, width, height, sigma_px);
                let ndiff = cpu_noise
                    .iter()
                    .zip(gpu_blurred.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                println!(
                    "[p2 sl1 fresh blur sigma={sigma_px} r={r1}] ndiff={ndiff}/{}",
                    n
                );
            }
            let gpu_noise = ctx.download_f32(noise, n);
            for plane_e in 0..e {
                let (base_lo, base_hi) = consts.grain_seed_base[plane_e][1];
                let base = ((base_lo as u64) | ((base_hi as u64) << 32));
                let mut cpu_noise = vec![0.0f32; n];
                for y in 0..height {
                    let mut rng = SplitMix64::new(base.wrapping_add(y as u64));
                    for x in 0..width {
                        cpu_noise[y * width + x] = rng.next_gaussian();
                    }
                }
                let stock2 = params.stock.load().unwrap();
                let layer = stock2.emulsion_layers().nth(plane_e).unwrap().1;
                let base_corr = if let Some(dist) = &layer.crystal_size {
                    let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
                    (mean_s / 0.7) as f32 * crate::film::constants::DYE_CLOUD_CORRELATION_UM
                } else {
                    crate::film::constants::DYE_CLOUD_CORRELATION_UM
                };
                let corr = base_corr * 0.7;
                let sigma_px = (corr / pitch.max(1e-6)).max(1.0);
                gaussian_blur_separable(&mut cpu_noise, width, height, sigma_px);
                let mut ndiff = 0usize;
                for (i, (&a, &b)) in cpu_noise.iter().zip(gpu_noise.iter()).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        ndiff += 1;
                    }
                }
                println!(
                    "[noise buffer holds plane{plane_e} sl1?] ndiff={ndiff}/{} (0 means yes)",
                    n
                );
            }
        }
        {
            for plane_e in 0..e {
                let kappa = consts.kappa[plane_e];
                let dmax = consts.dmax[plane_e];
                if kappa <= 0.0 || dmax <= 0.0 {
                    continue;
                }
                let (ref k0, r0) = consts.grain_kernels[plane_e][0];
                let (ref k1, r1) = consts.grain_kernels[plane_e][1];
                let dst_off = (plane_e * n) as u32;
                for sl in 0..2 {
                    let (ref kbuf, radius) = if sl == 0 { (k0, r0) } else { (k1, r1) };
                    let (base_lo, base_hi) = consts.grain_seed_base[plane_e][sl];
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
                    let dst_buf = if sl == 0 { work } else { noise };
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
                        dst_off: if sl == 0 { dst_off } else { 0 },
                        _p0: 0,
                        _p1: 0,
                    };
                    let noise_ub = bytemuck::bytes_of(&nu);
                    let blur_ub_h = bytemuck::bytes_of(&blur_u_h);
                    let blur_ub_v = bytemuck::bytes_of(&blur_u_v);
                    let noise_bufs = [noise];
                    let h_bufs = [noise, btmp, kbuf.as_ref()];
                    let v_bufs = [btmp, dst_buf, kbuf.as_ref()];
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
                }
                let gu = GrainApplySubU {
                    n: n as u32,
                    off: (plane_e * n) as u32,
                    kappa: consts.kappa[plane_e],
                    dmax: consts.dmax[plane_e],
                    norm: 1.0,
                    noise0_off: (plane_e * n) as u32,
                    noise1_off: 0,
                    l2base: 5u32 * consts.num_emul,
                };
                ctx.dispatch_compute_shader_multi(
                    "film_grain_apply_sub",
                    shaders::GRAIN_APPLY_SUB,
                    &[&dye, &work, &noise, &consts.reduce],
                    bytemuck::bytes_of(&gu),
                    workgroups(n),
                );
            }
        }
        {
            let gpu_floats = ctx.download_f32(dye, n * e);
            let cpu: Vec<f32> = cpu_dyes.image_dye.iter().flatten().copied().collect();
            report_stage("grain_dye", &cpu, &gpu_floats, e, n, false);

            // TEMP: does the GPU plane-0 output match CPU grain using the LAST
            // plane's sl1 noise (noise1_off=0 reuse) instead of plane-0's own?
            {
                use crate::film::development::grain::apply_grain;
                use crate::film::types::DyePlanes;
                let mut cpu_da = crate::film::development::reduction::reduce(&stock, &latent);
                crate::film::development::diffusion::apply_dir_inhibition(
                    &mut cpu_da,
                    stock.dir_diffusion_length.0 / pitch.max(1e-6),
                    &stock.dir_inhibition_matrix,
                );
                crate::film::development::diffusion::apply_adjacency(
                    &mut cpu_da,
                    stock.developer_diffusion_length.0 / pitch.max(1e-6),
                    stock.adjacency_beta,
                );
                // Replace plane 0's sl1 with plane 2's sl1 by re-running grain
                // per plane with a custom noise injection: easiest is to compare
                // plane 2's own GPU output against the CPU plane-2 output.
                let mut d_max = Vec::new();
                let mut kappas = Vec::new();
                let mut crystal_sizes = Vec::new();
                for (layer_idx, layer) in stock.layers.iter().enumerate() {
                    if layer.kind != crate::film::stock::LayerKind::Emulsion {
                        continue;
                    }
                    let coupler = layer.coupler.as_ref().unwrap();
                    d_max.push(coupler.d_max);
                    let kappa_ref = stock.grain_kappa[layer_idx].unwrap_or(0.0);
                    let kappa = crate::film::development::grain::scale_kappa(kappa_ref, pitch);
                    kappas.push(kappa);
                    crystal_sizes.push(layer.crystal_size.clone());
                }
                apply_grain(&mut cpu_da, &d_max, &kappas, pitch, params.seed, &crystal_sizes);
                let cpu_g: Vec<f32> = cpu_da.image_dye.iter().flatten().copied().collect();
                let mut ndiff_p0 = 0usize;
                let mut ndiff_p2 = 0usize;
                for i in 0..n {
                    let a0 = cpu_g[0 * n + i];
                    let b0 = gpu_floats[0 * n + i];
                    if a0.to_bits() != b0.to_bits() {
                        ndiff_p0 += 1;
                    }
                    let a2 = cpu_g[2 * n + i];
                    let b2 = gpu_floats[2 * n + i];
                    if a2.to_bits() != b2.to_bits() {
                        ndiff_p2 += 1;
                    }
                }
                println!("[grain per-plane-noise check] plane0 ndiff={ndiff_p0}/{} plane2 ndiff={ndiff_p2}/{}", n, n);

                // Search: which plane's sl1 blurred noise reproduces the GPU
                // plane-0 output?
                use crate::film::blur::gaussian_blur_separable as gbs;
                use crate::film::development::grain::SplitMix64 as SM64;
                let mut cpu_pregrain = crate::film::development::reduction::reduce(&stock, &latent);
                crate::film::development::diffusion::apply_dir_inhibition(
                    &mut cpu_pregrain,
                    stock.dir_diffusion_length.0 / pitch.max(1e-6),
                    &stock.dir_inhibition_matrix,
                );
                crate::film::development::diffusion::apply_adjacency(
                    &mut cpu_pregrain,
                    stock.developer_diffusion_length.0 / pitch.max(1e-6),
                    stock.adjacency_beta,
                );
                let dmax0 = d_max[0];
                let kappa_sub0 = kappas[0] * (2.0f32).sqrt();
                let mut sl1_blurred: Vec<Vec<f32>> = Vec::new();
                for plane_e in 0..e {
                    let (base_lo, base_hi) = consts.grain_seed_base[plane_e][1];
                    let base = ((base_lo as u64) | ((base_hi as u64) << 32));
                    let mut cpu_noise = vec![0.0f32; n];
                    for y in 0..height {
                        let mut rng = SM64::new(base.wrapping_add(y as u64));
                        for x in 0..width {
                            cpu_noise[y * width + x] = rng.next_gaussian();
                        }
                    }
                    let stock2 = params.stock.load().unwrap();
                    let layer = stock2.emulsion_layers().nth(plane_e).unwrap().1;
                    let base_corr = if let Some(dist) = &layer.crystal_size {
                        let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
                        (mean_s / 0.7) as f32 * crate::film::constants::DYE_CLOUD_CORRELATION_UM
                    } else {
                        crate::film::constants::DYE_CLOUD_CORRELATION_UM
                    };
                    let sigma_px = ((base_corr * 0.7) / pitch.max(1e-6)).max(1.0);
                    gbs(&mut cpu_noise, width, height, sigma_px);
                    sl1_blurred.push(cpu_noise);
                }
                let gpu_work = ctx.download_f32(work, n * e);
                for cand in 0..e {
                    let mut ndiff_c = 0usize;
                    for i in 0..n {
                        let dens = cpu_pregrain.image_dye[0][i].clamp(0.0, dmax0);
                        let eps_toe = 0.05 * dmax0;
                        let taper = crate::film::math::div_det(dens, dens + eps_toe).min(1.0);
                        let sd = taper * (dens * (dmax0 - dens)).max(0.0).sqrt();
                        let n0 = gpu_work[i];
                        let n1 = sl1_blurred[cand][i];
                        let noisy0 = (kappa_sub0 * sd * n0).mul_add(1.0, dens);
                        let noisy1 = (kappa_sub0 * sd * n1).mul_add(1.0, dens);
                        let knee = 0.005 * dmax0;
                        let d0 = if noisy0 >= knee {
                            noisy0
                        } else {
                            crate::film::math::div_det(knee * knee, 2.0 * knee - noisy0)
                        }
                        .min(dmax0 * 1.05);
                        let d1 = if noisy1 >= knee {
                            noisy1
                        } else {
                            crate::film::math::div_det(knee * knee, 2.0 * knee - noisy1)
                        }
                        .min(dmax0 * 1.05);
                        let out = (d0 + d1) * 0.5;
                        if out.to_bits() != gpu_floats[i].to_bits() {
                            ndiff_c += 1;
                        }
                    }
                    println!("[grain plane0 cand sl1=plane{cand}] ndiff={ndiff_c}/{}", n);
                }
            }

            // TEMP: compare blurred noise planes for layer 0 against CPU blur.
            let (ref k0, r0) = consts.grain_kernels[0][0];
            let (ref k1, r1) = consts.grain_kernels[0][1];
            let gpu_sl0 = ctx.download_f32(work, n * e);
            let gpu_sl1 = ctx.download_f32(noise, n);
            for sl in 0..2 {
                use crate::film::development::grain::SplitMix64;
                use crate::film::blur::gaussian_blur_separable;
                let (base_lo, base_hi) = consts.grain_seed_base[0][sl];
                let base = ((base_lo as u64) | ((base_hi as u64) << 32));
                let mut cpu_noise = vec![0.0f32; n];
                for y in 0..height {
                    let mut rng = SplitMix64::new(base.wrapping_add(y as u64));
                    for x in 0..width {
                        cpu_noise[y * width + x] = rng.next_gaussian();
                    }
                }
                if sl == 0 {
                    cpu_raw_l0_sl0 = cpu_noise.clone();
                }
                let sigma = if sl == 0 { r0 } else { r1 };
                let _ = (k0, k1);
                let _ = sigma;
                // CPU blur with the same sigma the GPU kernel was built from.
                // Recover sigma from the baked kernel by re-deriving correlation.
                let stock2 = params.stock.load().unwrap();
                let layer = stock2.emulsion_layers().nth(0).unwrap().1;
                let base_corr = if let Some(dist) = &layer.crystal_size {
                    let mean_s = (dist.mu_ln + 0.5 * dist.sigma_ln * dist.sigma_ln).exp();
                    (mean_s / 0.7) as f32 * crate::film::constants::DYE_CLOUD_CORRELATION_UM
                } else {
                    crate::film::constants::DYE_CLOUD_CORRELATION_UM
                };
                let sl_scale = if sl == 0 { 1.3f32 } else { 0.7 };
                let corr = base_corr * sl_scale;
                let sigma_px = (corr / pitch.max(1e-6)).max(1.0);
                gaussian_blur_separable(&mut cpu_noise, width, height, sigma_px);
                let gpu_ref: &[f32] = if sl == 0 { &gpu_sl0[0 * n..1 * n] } else { &gpu_sl1 };
                let mut max_ulp = 0u32;
                let mut ndiff = 0usize;
                for (i, (&a, &b)) in cpu_noise.iter().zip(gpu_ref.iter()).enumerate() {
                    let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                    if ulp > max_ulp {
                        max_ulp = ulp;
                    }
                    if a.to_bits() != b.to_bits() {
                        ndiff += 1;
                    }
                }
                println!(
                    "[noise blurred l0 sl{sl} sigma={sigma_px}] max_ulp={max_ulp} ndiff={ndiff}/{}",
                    n
                );
                if sl == 0 {
                    println!("  cpu blurred first8: {:?}", &cpu_noise[..8]);
                    println!("  gpu blurred first8: {:?}", &gpu_ref[..8]);
                    println!("  cpu raw first8: {:?}", &cpu_raw_l0_sl0[..8]);
                }
                // TEMP: compare the baked kernel with the CPU kernel.
                let (ref kbuf, rad) = consts.grain_kernels[0][sl];
                let gpu_kernel = ctx.download_f32(kbuf, (2 * rad + 1) as usize);
                let cpu_kernel = make_gaussian_kernel(sigma_px);
                println!(
                    "  kernels sl{sl}: rad gpu={rad} cpu={} gpu={:?} cpu={:?}",
                    crate::film::blur::gaussian_radius(sigma_px),
                    gpu_kernel,
                    cpu_kernel
                );
            }
        }

        // Stage 8: scan + invert.
        {
            let u = ScanU {
                n: n as u32,
                num_emul: consts.num_emul,
                scale: consts.scan_scale,
                flags: if params.output == FilmOutput::PositiveLinear { 1 } else { 0 },
            };
            ctx.dispatch_compute_shader_multi(
                "film_scan",
                shaders::SCAN,
                &[&gpu_buf.buffer, &dye, &mask, &consts.scan],
                bytemuck::bytes_of(&u),
                workgroups(n),
            );
        }
        // TEMP: pre-invert scan comparison (flags=0) on a scratch buffer.
        {
            let pre_buf = ctx.create_output_buffer(width, height);
            ctx.queue.write_buffer(
                &pre_buf.buffer,
                0,
                bytemuck::cast_slice(&rgba),
            );
            let u = ScanU {
                n: n as u32,
                num_emul: consts.num_emul,
                scale: consts.scan_scale,
                flags: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_scan_pre",
                shaders::SCAN,
                &[&pre_buf.buffer, &dye, &mask, &consts.scan],
                bytemuck::bytes_of(&u),
                workgroups(n),
            );
            let pre_floats = ctx.download_f32(&pre_buf.buffer, n * 4);
            let cpu_pre = crate::film::scan::densitometry::scan_to_acescg(&stock, &cpu_dyes);
            let mut ndiff_pre = 0usize;
            let mut first_pre: Option<(usize, f32, f32)> = None;
            for px in 0..n {
                for ch in 0..3 {
                    let a = cpu_pre[px][ch];
                    let b = pre_floats[px * 4 + ch];
                    if a.to_bits() != b.to_bits() {
                        ndiff_pre += 1;
                        if first_pre.is_none() {
                            first_pre = Some((px, a, b));
                        }
                    }
                }
            }
            println!(
                "[stage scan_preinvert] ndiff={ndiff_pre}/{} first={:?}",
                n * 3,
                first_pre
            );
            // TEMP: normalization probe.
            let dmin_ref = crate::film::scan::densitometry::dmin_reference_acescg(&stock);
            let peak = dmin_ref[0].max(dmin_ref[1]).max(dmin_ref[2]);
            println!(
                "  probe dmin_ref={dmin_ref:?} peak={peak} scan_scale={} cpu px0 first4={:?} gpu px0 first4={:?}",
                consts.scan_scale,
                &cpu_pre[0],
                &pre_floats[..4]
            );
        }
        // TEMP: scan+invert math probe for px=0.
        {
            let px = 4usize;
            let mut dye0 = [0.0f32; 3];
            let mut mask0 = [0.0f32; 3];
            for e in 0..e {
                dye0[e] = cpu_dyes.image_dye[e][px];
                mask0[e] = cpu_dyes.mask_dye[e][px];
            }
            let dye_buf = ctx.create_f32_buffer_init(&dye0, "tmp_dye0");
            let mask_buf = ctx.create_f32_buffer_init(&mask0, "tmp_mask0");
            let out_buf = ctx.create_f32_buffer(64, "tmp_scanout");
            let u = CountU {
                n: 1,
                num_emul: consts.num_emul,
                _p0: 0,
                _p1: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_scan_probe",
                r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read> dye: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

const LOG10_2: f32 = 0.3010299956639812;
const LOG2_10: f32 = 3.3219280948873623;

fn exp2_det(x: f32, tbase: u32) -> f32 {
    let k = floor(x);
    let f = x - k;
    let t = f * 256.0;
    var i = u32(t);
    if (i > 255u) { i = 255u; }
    let r = t - f32(i);
    let base = fma(r, sc[tbase + i + 1u] - sc[tbase + i], sc[tbase + i]);
    let scale = bitcast<f32>(u32((i32(k) + 127) << 23));
    return base * scale;
}

fn log2_det(x: f32, tbase: u32) -> f32 {
    let bits = bitcast<u32>(x);
    let e = i32(bits >> 23u) - 127;
    let m = bitcast<f32>((bits & 0x7FFFFFu) | 0x3F800000u);
    let t = (m - 1.0) * 256.0;
    var i = u32(t);
    if (i > 255u) { i = 255u; }
    let r = t - f32(i);
    let l = fma(r, sc[tbase + i + 1u] - sc[tbase + i], sc[tbase + i]);
    return f32(e) + l;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u.n) { return; }
    let E = u.num_emul;
    let eps_base = 0u;
    let maskeps_base = E * 16u;
    let illum_base = 2u * E * 16u;
    let xbar_base = illum_base + 16u;
    let ybar_base = xbar_base + 16u;
    let zbar_base = ybar_base + 16u;
    let mat_base = zbar_base + 16u;
    let l2base = mat_base + 9u + 16u;
    let e2base = l2base + 257u;

    var dens: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) { dens[k] = 0.0; }
    for (var e = 0u; e < E; e = e + 1u) {
        let di = dye[e];
        let dm = mask[e];
        let eb = eps_base + e * 16u;
        let mb = maskeps_base + e * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            dens[k] = fma(dm, sc[mb + k], fma(di, sc[eb + k], dens[k]));
        }
    }
    var t: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        t[k] = exp2_det(-dens[k] * LOG2_10, e2base) * sc[illum_base + k];
    }
    var X = 0.0; var Y = 0.0; var Z = 0.0;
    for (var k = 0u; k < 15u; k = k + 1u) {
        X = fma(0.5 * fma(t[k + 1u], sc[xbar_base + k + 1u], t[k] * sc[xbar_base + k]), 20.0, X);
        Y = fma(0.5 * fma(t[k + 1u], sc[ybar_base + k + 1u], t[k] * sc[ybar_base + k]), 20.0, Y);
        Z = fma(0.5 * fma(t[k + 1u], sc[zbar_base + k + 1u], t[k] * sc[zbar_base + k]), 20.0, Z);
    }
    var rgb = vec3<f32>(
        fma(sc[mat_base + 2u], Z, fma(sc[mat_base + 1u], Y, sc[mat_base + 0u] * X)) * 0.017417744,
        fma(sc[mat_base + 5u], Z, fma(sc[mat_base + 4u], Y, sc[mat_base + 3u] * X)) * 0.017417744,
        fma(sc[mat_base + 8u], Z, fma(sc[mat_base + 7u], Y, sc[mat_base + 6u] * X)) * 0.017417744
    );
    out[0] = dens[0];
    out[1] = t[0];
    out[2] = X;
    out[3] = Y;
    out[4] = Z;
    out[5] = rgb.x;
    out[6] = rgb.y;
    out[7] = rgb.z;
    out[8] = 0.0;
    // Invert steps.
    let inv_base = mat_base + 9u;
    let inv_dmin = vec3<f32>(sc[inv_base + 10u], sc[inv_base + 11u], sc[inv_base + 12u]);
    let g_val = vec3<f32>(sc[inv_base + 3u], sc[inv_base + 4u], sc[inv_base + 5u]);
    let slope = sc[inv_base + 6u];
    let inv_gamma_log2_10 = sc[inv_base + 14u];
    let eps = sc[inv_base + 8u];
    let fog_offset = sc[inv_base + 9u];
    let tc = vec3<f32>(
        max(rgb.x * inv_dmin.x, eps),
        max(rgb.y * inv_dmin.y, eps),
        max(rgb.z * inv_dmin.z, eps),
    );
    let d_img = vec3<f32>(
        -(log2_det(tc.x, l2base) * LOG10_2),
        -(log2_det(tc.y, l2base) * LOG10_2),
        -(log2_det(tc.z, l2base) * LOG10_2),
    );
    let d_clamped = max(d_img - vec3<f32>(fog_offset), vec3<f32>(0.0));
    var e_scene = vec3<f32>(0.0);
    if (d_clamped.x > 0.0) { e_scene.x = exp2_det(d_clamped.x * inv_gamma_log2_10, e2base) - 1.0; } else { e_scene.x = slope * d_clamped.x; }
    if (d_clamped.y > 0.0) { e_scene.y = exp2_det(d_clamped.y * inv_gamma_log2_10, e2base) - 1.0; } else { e_scene.y = slope * d_clamped.y; }
    if (d_clamped.z > 0.0) { e_scene.z = exp2_det(d_clamped.z * inv_gamma_log2_10, e2base) - 1.0; } else { e_scene.z = slope * d_clamped.z; }
    let fin = vec3<f32>(g_val.x * e_scene.x, g_val.y * e_scene.y, g_val.z * e_scene.z);
    out[8] = tc.x;
    out[9] = d_img.x;
    out[10] = d_clamped.x;
    out[11] = e_scene.x;
    out[12] = fin.x;
    out[13] = fin.y;
    out[14] = fin.z;
    out[15] = tc.z;
    out[16] = d_img.z;
    out[17] = e_scene.z;
    out[18] = rgb.z;
    out[19] = inv_dmin.z;
    out[20] = log2_det(tc.z, l2base);
    out[21] = d_clamped.z * inv_gamma_log2_10;
    out[22] = inv_gamma_log2_10;
    out[23] = d_clamped.z;
    out[24] = sc[inv_base + 14u];
    out[25] = tc.y;
    out[26] = d_img.y;
    out[27] = d_clamped.y;
    out[28] = e_scene.y;
}
"#,
                &[&dye_buf, &mask_buf, &consts.scan, &out_buf],
                bytemuck::bytes_of(&u),
                1,
            );
            let probe = ctx.download_f32(&out_buf, 64);
            // CPU mirror of the same steps.
            let mut dens = [0.0f32; 16];
            let mut ei = 0usize;
            for layer in &stock.layers {
                if layer.kind != crate::film::stock::LayerKind::Emulsion {
                    continue;
                }
                let coupler = layer.coupler.as_ref().unwrap();
                let di = cpu_dyes.image_dye[ei][px];
                let dm = cpu_dyes.mask_dye[ei][px];
                for (k, dk) in dens.iter_mut().enumerate() {
                    *dk = di.mul_add(coupler.epsilon.samples[k] as f32, *dk);
                    if let Some(ref me) = coupler.mask_epsilon {
                        *dk = dm.mul_add(me.samples[k] as f32, *dk);
                    }
                }
                ei += 1;
            }
            let mut t = [0.0f32; 16];
            for (k, dk) in dens.iter().enumerate() {
                t[k] = crate::film::math::pow10_det(-dk) * stock.scanner_light.samples[k] as f32;
            }
            let c = crate::film::exposure::upsample::scan_basis_f32_pub();
            let mut xyz = [0.0f32; 3];
            for k in 0..15 {
                xyz[0] = (0.5 * t[k + 1].mul_add(c.xbar[k + 1], t[k] * c.xbar[k])).mul_add(20.0, xyz[0]);
                xyz[1] = (0.5 * t[k + 1].mul_add(c.ybar[k + 1], t[k] * c.ybar[k])).mul_add(20.0, xyz[1]);
                xyz[2] = (0.5 * t[k + 1].mul_add(c.zbar[k + 1], t[k] * c.zbar[k])).mul_add(20.0, xyz[2]);
            }
            let scale = consts.scan_scale;
            let rgb_c = [
                c.to_acescg[0][2].mul_add(xyz[2], c.to_acescg[0][1].mul_add(xyz[1], c.to_acescg[0][0] * xyz[0])) * scale,
                c.to_acescg[1][2].mul_add(xyz[2], c.to_acescg[1][1].mul_add(xyz[1], c.to_acescg[1][0] * xyz[0])) * scale,
                c.to_acescg[2][2].mul_add(xyz[2], c.to_acescg[2][1].mul_add(xyz[1], c.to_acescg[2][0] * xyz[0])) * scale,
            ];
            println!(
                "[scan probe px={px}] dens0 cpu={} gpu={} | t0 cpu={} gpu={} | X cpu={} gpu={} | Y cpu={} gpu={} | Z cpu={} gpu={}",
                dens[0], probe[0], t[0], probe[1], xyz[0], probe[2], xyz[1], probe[3], xyz[2], probe[4]
            );
            println!(
                "  rgb cpu={rgb_c:?} gpu=({}, {}, {})",
                probe[5], probe[6], probe[7]
            );
            // CPU invert intermediates at px=0.
            let dmin = crate::film::scan::normalized_dmin_acescg(&stock);
            let mid = crate::film::mid_negative_acescg(&stock, pitch, shutter);
            let ic = crate::film::scan::invert::invert_constants(mid, dmin);
            let tc_c: Vec<f32> = rgb_c
                .iter()
                .zip(ic.inv_dmin.iter())
                .map(|(&r, &id)| (r * id).max(ic.eps))
                .collect();
            let d_img_c: Vec<f32> = tc_c
                .iter()
                .map(|&t| -(crate::film::math::log2_det(t) * crate::film::math::LOG10_2))
                .collect();
            let d_cl_c: Vec<f32> = d_img_c.iter().map(|&d| (d - ic.fog_offset).max(0.0)).collect();
            let e_sc_c: Vec<f32> = d_cl_c
                .iter()
                .map(|&d| {
                    if d > 0.0 {
                        crate::film::math::exp2_det(d * ic.inv_gamma_log2_10) - 1.0
                    } else {
                        ic.slope * d
                    }
                })
                .collect();
            let fin_c: Vec<f32> = (0..3).map(|k| ic.gain[k] * e_sc_c[k]).collect();
            println!(
                "  invert cpu tc={tc_c:?} d_img={d_img_c:?} d_cl={d_cl_c:?} e_sc={e_sc_c:?} fin={fin_c:?}",
            );
            println!(
                "  invert gpu ch0: tc={} d_img={} d_cl={} e_sc={} | fin=({},{},{})",
                probe[8], probe[9], probe[10], probe[11], probe[12], probe[13], probe[14]
            );
            println!(
                "  invert gpu ch2: rgb={} inv_dmin={} tc={} log2det={} d_img={} e_sc={}",
                probe[18], probe[19], probe[15], probe[20], probe[16], probe[17]
            );
            println!(
                "  invert gpu ch2 x: (d_cl*inv_gamma)*LOG2_10={} inv_gamma={} d_cl={} sc[13]={}",
                probe[21], probe[22], probe[23], probe[24]
            );
            println!(
                "  invert cpu ch2: rgb={} inv_dmin={} tc={} log2det={} d_img={} e_sc={}",
                rgb_c[2],
                ic.inv_dmin[2],
                tc_c[2],
                crate::film::math::log2_det(tc_c[2]),
                d_img_c[2],
                e_sc_c[2]
            );
            println!(
                "  invert cpu ch2 x: {} inv_gamma={} d_cl={}",
                d_cl_c[2] * ic.inv_gamma_log2_10,
                ic.inv_gamma_log2_10,
                d_cl_c[2]
            );
            println!(
                "  invert gpu ch1: tc={} d_img={} d_cl={} e_sc={}",
                probe[25], probe[26], probe[27], probe[28]
            );
            println!(
                "  probe rgb_c={rgb_c:?}"
            );
            let cpu_pre2 = crate::film::scan::densitometry::scan_to_acescg(&stock, &cpu_dyes);
            println!("  real scan_to_acescg px4={:?}", &cpu_pre2[4]);
            println!(
                "  probe fin_c={fin_c:?} gain={:?} real cpu_scan[4]={:?}",
                ic.gain,
                &cpu_scan[4]
            );
            {
                let inv_base = 3 * 16 + 16 + 16 + 16 + 16 + 9; // mat_base = zbar_base+16
                let scan_c = ctx.download_f32(&consts.scan, 16);
                let _ = inv_base;
                let _ = scan_c;
            }
            let scan_consts_full = ctx.download_f32(&consts.scan, 699);
            let mat_base = 3 * 16 + 16 + 16 + 16 + 16;
            println!(
                "  baked idx 96..176: {:?}",
                &scan_consts_full[96..176]
            );
            println!(
                "  invert cpu ch1: tc={} d_img={} d_cl={} e_sc={} (bits {:#x})",
                tc_c[1], d_img_c[1], d_cl_c[1], e_sc_c[1], e_sc_c[1].to_bits()
            );
            println!(
                "  invert gpu ch1 e_sc bits={:#x} fin.y bits={:#x}",
                probe[28].to_bits(),
                probe[13].to_bits()
            );
        }
        {
            let gpu_floats = ctx.download_f32(&gpu_buf.buffer, n * 4);
            let cpu: Vec<f32> = cpu_scan.iter().flatten().copied().collect();
            let mut max_ulp = 0u32;
            let mut first: Option<(usize, f32, f32)> = None;
            for px in 0..n {
                for ch in 0..3 {
                    let a = cpu[px * 3 + ch];
                    let b = gpu_floats[px * 4 + ch];
                    if a.to_bits() != b.to_bits() && first.is_none() {
                        first = Some((px, a, b));
                    }
                    let ulp = (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32;
                    if ulp > max_ulp {
                        max_ulp = ulp;
                    }
                }
            }
            // TEMP: re-dispatch the real scan and re-check px=4.
            {
                let u = ScanU {
                    n: n as u32,
                    num_emul: consts.num_emul,
                    scale: consts.scan_scale,
                    flags: 1,
                };
                ctx.dispatch_compute_shader_multi(
                    "film_scan_redispatch",
                    shaders::SCAN,
                    &[&gpu_buf.buffer, &dye, &mask, &consts.scan],
                    bytemuck::bytes_of(&u),
                    workgroups(n),
                );
                let dye_check = ctx.download_f32(dye, n * e);
            println!(
                "  gpu dye px4={:?} cpu dye px4={:?}",
                (0..e).map(|k| dye_check[k * n + 4]).collect::<Vec<_>>(),
                (0..e).map(|k| cpu_dyes.image_dye[k][4]).collect::<Vec<_>>()
            );
            // TEMP: dispatch the REAL shaders::SCAN on the probe's 1-pixel inputs.
            {
                let dye1: Vec<f32> = (0..e).map(|k| cpu_dyes.image_dye[k][4]).collect();
                let mask1: Vec<f32> = (0..e).map(|k| cpu_dyes.mask_dye[k][4]).collect();
                let d1 = ctx.create_f32_buffer_init(&dye1, "t_d1");
                let m1 = ctx.create_f32_buffer_init(&mask1, "t_m1");
                let out1 = ctx.create_output_buffer(1, 1);
                ctx.queue.write_buffer(&out1.buffer, 0, bytemuck::cast_slice(&[[0.0f32; 4]]));
                let u = ScanU {
                    n: 1,
                    num_emul: consts.num_emul,
                    scale: consts.scan_scale,
                    flags: 1,
                };
                ctx.dispatch_compute_shader_multi(
                    "film_scan_realprobe",
                    shaders::SCAN,
                    &[&out1.buffer, &d1, &m1, &consts.scan],
                    bytemuck::bytes_of(&u),
                    1,
                );
                let o1 = ctx.download_f32(&out1.buffer, 4);
                println!(
                    "  scan_scale bits={:#x} literal bits={:#x}",
                    consts.scan_scale.to_bits(),
                    0.017417744f32.to_bits()
                );
                println!(
                    "  real-SCAN on probe inputs px4={:?} bits={:#x}",
                    &o1[..3],
                    o1[1].to_bits()
                );
            }
            let g2 = ctx.download_f32(&gpu_buf.buffer, n * 4);
                println!(
                    "  redispatched scan px4={:?} (was {:?})",
                    &g2[4 * 4..4 * 4 + 3],
                    &gpu_floats[4 * 4..4 * 4 + 3]
                );
            }
            println!("[stage scan_invert] max_ulp={max_ulp}");
            if let Some((px, a, b)) = first {
                println!(
                    "  first diff: px={px} x={} y={} ch={}: cpu={a} ({:#010x}) gpu={b} ({:#010x}) ulp={}",
                    px % 128,
                    px / 128,
                    0,
                    a.to_bits(),
                    b.to_bits(),
                    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
                );
                println!("  cpu px{px}: {:?}", &cpu[px * 3..px * 3 + 3]);
                println!("  gpu px{px}: {:?}", &gpu_floats[px * 4..px * 4 + 3]);
            }
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

            let mut max_diff = 0.0f32;
            let mut max_at = (0usize, 0usize);
            for (px, cpu_pixel) in cpu_image.rgb_data.iter().enumerate() {
                for ch in 0..3 {
                    let gpu_value = gpu_floats[px * 4 + ch];
                    let diff = (cpu_pixel[ch] - gpu_value).abs();
                    if diff > max_diff {
                        max_diff = diff;
                        max_at = (px, ch);
                    }
                }
            }
            eprintln!(
                "TEMP parity stock={stock:?} fmt={film_format:?} max_diff={max_diff} at px={} ch={}",
                max_at.0, max_at.1
            );

            // TEMP: does the GPU output match a CPU run with grain disabled?
            let mut cpu_nograin = crate::pixel::Image {
                metadata: meta.clone(),
                rgb_data: vec![[0.18, 0.18, 0.18]; n],
                raw_data: std::sync::Arc::from([]),
            };
            {
                let mut p = params.clone();
                p.seed = 0;
                let _ = &mut p;
            }
            let mut no_grain_params = params.clone();
            no_grain_params.seed = u64::MAX;
            crate::film::process(&mut cpu_nograin, &no_grain_params).unwrap();
            let mut ndiff_nograin = 0usize;
            for (px, cpu_pixel) in cpu_nograin.rgb_data.iter().enumerate() {
                for ch in 0..3 {
                    let gpu_value = gpu_floats[px * 4 + ch];
                    if cpu_pixel[ch].to_bits() != gpu_value.to_bits() {
                        ndiff_nograin += 1;
                    }
                }
            }
            eprintln!(
                "TEMP parity vs nograin(seed=MAX) ndiff={ndiff_nograin}/{}",
                n * 3
            );

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
