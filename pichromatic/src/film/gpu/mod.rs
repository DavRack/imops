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

mod shaders;
mod workspace;

pub(crate) use workspace::acquire_film_resources;

use bytemuck::{Pod, Zeroable};
use wgpu::Buffer;

use crate::film::constants::{
    ABSORPTION_SIGMA_SCALE_PER_UM, CHROMOGENIC_DYE_GRAIN_SCALE, DYE_CLOUD_CORRELATION_UM,
    LOCAL_SCATTER_MIX, MASK_DENSITY_FRACTION_OF_DMAX,
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
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InvertU {
    n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
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
fn make_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil().max(1.0) as usize;
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
    // Per-emulsion runtime scalars.
    kappa: Vec<f32>,
    dmax: Vec<f32>,
    // Blur sigmas (px) for the spatial stages (None → stage skipped).
    sigma_local: f32,
    sigma_wide: f32,
    sigma_dir: f32,
    sigma_adj: f32,
    adjacency_beta: f32,
    grain_seed_base: Vec<(u32, u32)>, // per emulsion (lo, hi) of seed*C1 + layer*GOLDEN
}

const GOLDEN: u64 = 0x9E3779B97F4A7C15;
const SM_STATE_MIX: u64 = 0xD1B54A32D192ED03;

pub(crate) fn bake_consts(ctx: &GpuContext, stock: &FilmStock, params: &FilmParams, width: usize) -> StockConsts {
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
    // Layout: b0(16) b1(16) b2(16) M(9) lambda_factor(16) od(L*16) produces_latent(L)
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
        expose.push((lambda / 550.0) as f32);
    }
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

    // ── scan consts ── eps(E*16) maskeps(E*16) illum(16) xbar(16) ybar(16) zbar(16) matrix(9)
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

    // Grain scalars.
    let mut kappa = Vec::with_capacity(num_emul);
    let mut dmax_grain = Vec::with_capacity(num_emul);
    let mut grain_seed_base = Vec::with_capacity(num_emul);
    for (e, &(li, layer)) in emuls.iter().enumerate() {
        let coupler = layer.coupler.as_ref().unwrap();
        let kappa_ref = stock.grain_kappa[li].unwrap_or(0.0);
        let k = scale_kappa(kappa_ref, pitch) * CHROMOGENIC_DYE_GRAIN_SCALE;
        kappa.push(k);
        dmax_grain.push(coupler.d_max);
        let base = params
            .seed
            .wrapping_mul(SM_STATE_MIX)
            .wrapping_add((e as u64).wrapping_mul(GOLDEN));
        grain_seed_base.push((base as u32, (base >> 32) as u32));
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
        kappa,
        dmax: dmax_grain,
        sigma_local,
        sigma_wide,
        sigma_dir,
        sigma_adj,
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
    let u = BlurU {
        width: width as u32,
        height: height as u32,
        n: n as u32,
        radius,
        src_off,
        dst_off,
        _p0: 0,
        _p1: 0,
    };
    let ub = bytemuck::bytes_of(&u);
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
                uniform_bytes: ub,
                workgroups_x: d.h_gx,
                workgroups_y: d.h_gy,
            },
            ComputePassDesc {
                label: d.v_label,
                wgsl_source: d.v_wgsl,
                storage_buffers: &v_bufs,
                uniform_bytes: ub,
                workgroups_x: d.v_gx,
                workgroups_y: d.v_gy,
            },
        ],
    );
}

/// GPU partial sum-of-squares over all active emulsions → one tiny download →
/// per-emulsion f64 variance norms. Single sync for the whole grain stage.
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
    let target_out = GRAIN_VAR_PARTIALS_PER.min(n).max(1);
    let stride = ((n + target_out - 1) / target_out) as u32;
    let out_n = ((n as u32) + stride - 1) / stride;
    let wg = workgroups(out_n as usize);

    // One batched submit for all emulsions; download drains them.
    let uniforms: Vec<VarPartialU> = active
        .iter()
        .enumerate()
        .map(|(i, &(_plane_e, src_off))| VarPartialU {
            n: n as u32,
            stride,
            out_n,
            src_off,
            out_off: (i as u32) * out_n,
            _p0: 0,
            _p1: 0,
            _p2: 0,
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
            workgroups_x: wg,
            workgroups_y: 0,
        })
        .collect();
    ctx.dispatch_compute_passes("film_grain_var", &passes);

    let parts = ctx.download_f32_async(partial, active.len() * out_n as usize).await;
    let mut norms = Vec::with_capacity(active.len());
    for (i, &(plane_e, _)) in active.iter().enumerate() {
        let start = i * out_n as usize;
        let end = start + out_n as usize;
        let sum_sq: f64 = parts[start..end].iter().map(|&v| v as f64).sum();
        let var = sum_sq / n as f64;
        let norm = if var > 1e-12 {
            (1.0 / var.sqrt()) as f32
        } else {
            1.0
        };
        norms.push((plane_e, norm));
    }
    norms
}

/// Full GPU film simulation, mirroring [`crate::film::process`].
pub async fn process_gpu(
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
    let lease = acquire_film_resources(ctx, &stock, params, width, height, num_emul_hint);
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

    // ── Stage 2: spatial exposure effects (local scatter + halation) ──
    // Local gelatin scatter (always-on if σ_local ≥ 1e-3).
    if consts.sigma_local >= 1e-3 && LOCAL_SCATTER_MIX > 0.0 {
        let kernel = make_gaussian_kernel(consts.sigma_local);
        let radius = (kernel.len() / 2) as u32;
        let kbuf = ctx.create_f32_buffer_init(&kernel, "film_k_local");
        let f = LOCAL_SCATTER_MIX;
        let keep = 1.0 - f;
        for plane_e in 0..e {
            let blur_u = BlurU {
                width: width as u32,
                height: height as u32,
                n: n as u32,
                radius,
                src_off: (plane_e * n) as u32,
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
            let blur_ub = bytemuck::bytes_of(&blur_u);
            let mix_ub = bytemuck::bytes_of(&mix_u);
            let h_bufs = [planes, btmp, &kbuf];
            let v_bufs = [btmp, bout, &kbuf];
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
                        uniform_bytes: blur_ub,
                        workgroups_x: d.h_gx,
                        workgroups_y: d.h_gy,
                    },
                    ComputePassDesc {
                        label: d.v_label,
                        wgsl_source: d.v_wgsl,
                        storage_buffers: &v_bufs,
                        uniform_bytes: blur_ub,
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

    // Wide support-bounce halation (colored bleed from deepest emulsion).
    if consts.sigma_wide >= 1e-3 {
        let kernel = make_gaussian_kernel(consts.sigma_wide);
        let radius = (kernel.len() / 2) as u32;
        let kbuf = ctx.create_f32_buffer_init(&kernel, "film_k_wide");
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
            &kbuf,
            radius
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

    // ── Stage 5: DIR interlayer inhibition ──
    if consts.sigma_dir >= 1e-3 && !stock.dir_inhibition_matrix.is_empty() {
        let kernel = make_gaussian_kernel(consts.sigma_dir);
        let radius = (kernel.len() / 2) as u32;
        let kbuf = ctx.create_f32_buffer_init(&kernel, "film_k_dir");
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
                &kbuf,
                radius
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

    // ── Stage 6: adjacency (Eberhard) ──
    if consts.adjacency_beta.abs() >= 1e-8 && consts.sigma_adj >= 1e-3 {
        let kernel = make_gaussian_kernel(consts.sigma_adj);
        let radius = (kernel.len() / 2) as u32;
        let kbuf = ctx.create_f32_buffer_init(&kernel, "film_k_adj");
        for plane_e in 0..e {
            let blur_u = BlurU {
                width: width as u32,
                height: height as u32,
                n: n as u32,
                radius,
                src_off: (plane_e * n) as u32,
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
            let blur_ub = bytemuck::bytes_of(&blur_u);
            let adj_ub = bytemuck::bytes_of(&adj_u);
            let h_bufs = [dye, btmp, &kbuf];
            let v_bufs = [btmp, bout, &kbuf];
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
                        uniform_bytes: blur_ub,
                        workgroups_x: d.h_gx,
                        workgroups_y: d.h_gy,
                    },
                    ComputePassDesc {
                        label: d.v_label,
                        wgsl_source: d.v_wgsl,
                        storage_buffers: &v_bufs,
                        uniform_bytes: blur_ub,
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

    // ── Stage 7: grain (image dye only) ──
    // Blurred noise stored in `work` (free after DIR). One variance sync for all E.
    {
        let grain_sigma = (DYE_CLOUD_CORRELATION_UM / {
            let pitch = params.film_format.pixel_pitch_um(width);
            pitch.max(1e-6)
        })
        .max(1.0);
        let kernel = make_gaussian_kernel(grain_sigma);
        let radius = (kernel.len() / 2) as u32;
        let kbuf = ctx.create_f32_buffer_init(&kernel, "film_k_grain");
        let mut active: Vec<(usize, u32)> = Vec::with_capacity(e);
        for plane_e in 0..e {
            let kappa = consts.kappa[plane_e];
            let dmax = consts.dmax[plane_e];
            if kappa <= 0.0 || dmax <= 0.0 {
                continue;
            }
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
            let blur_u = BlurU {
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
            let blur_ub = bytemuck::bytes_of(&blur_u);
            let noise_bufs = [noise];
            let h_bufs = [noise, btmp, &kbuf];
            let v_bufs = [btmp, work, &kbuf];
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
                        uniform_bytes: blur_ub,
                        workgroups_x: d.h_gx,
                        workgroups_y: d.h_gy,
                    },
                    ComputePassDesc {
                        label: d.v_label,
                        wgsl_source: d.v_wgsl,
                        storage_buffers: &v_bufs,
                        uniform_bytes: blur_ub,
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

    // ── Stage 8: scan → densitometric ACEScg (Dmin-normalized) ──
    {
        let u = ScanU {
            n: n as u32,
            num_emul: consts.num_emul,
            scale: consts.scan_scale,
            _p0: 0,
        };
        ctx.dispatch_compute_shader_multi(
            "film_scan",
            shaders::SCAN,
            &[&gpu_buf.buffer, &dye, &mask, &consts.scan],
            bytemuck::bytes_of(&u),
            workgroups(n),
        );
    }

    // ── Optional invert → PositiveLinear ──
    let is_reversal = stock.layers.iter().any(|l| l.is_reversal);
    if !is_reversal && params.output == FilmOutput::PositiveLinear {
        apply_invert(ctx, gpu_buf, &stock, params, width, height)?;
    }

    Ok(())
}

/// PositiveLinear invert, mirroring `scan::invert::invert_negative`.
fn apply_invert(
    ctx: &GpuContext,
    gpu_buf: &GpuImageBuffer,
    stock: &FilmStock,
    params: &FilmParams,
    width: usize,
    height: usize,
) -> Result<(), FilmError> {
    use crate::film::exposure::expose_with_pitch_and_shutter;
    use crate::film::scan::{scan, ScanMode};
    use crate::pixel::{
        B_RELATIVE_LUMINANCE, G_RELATIVE_LUMINANCE, MIDDLE_GRAY, R_RELATIVE_LUMINANCE,
    };

    let pitch = params.film_format.pixel_pitch_um(width);
    let shutter = 1.0 / stock.box_iso.0;

    // Dmin (normalized) and mid-gray negative, computed on CPU (stock calibration).
    let dmin = crate::film::scan::normalized_dmin_acescg(stock);
    let mid = {
        const N: usize = 32;
        let g = {
            use crate::film::exposure::radiance::{
                relative_to_absolute_luminance, sunny16_exposure,
            };
            use crate::film::units::IsoSpeed;
            let ex = sunny16_exposure(IsoSpeed(stock.box_iso.0));
            relative_to_absolute_luminance(
                MIDDLE_GRAY as f64,
                ex.shutter_seconds as f64,
                ex.f_number as f64,
                ex.iso as f64,
            ) as f32
        };
        let rgb = vec![[g, g, g]; N * N];
        let latent = expose_with_pitch_and_shutter(&rgb, N, N, stock, pitch, shutter);
        let dyes = crate::film::development::develop(stock, &latent, 0, pitch);
        let buf = scan(stock, &dyes, ScanMode::NegativeLinear);
        crate::film::scan::mean_rgb(&buf)
    };

    // Precompute invert coefficients (mirrors invert_negative setup).
    const CHROMA_KEEP: f32 = 0.55;
    const HEADROOM: f32 = 8.0;
    const SHADOW_CHROMA_FADE_Y: f32 = 0.02;
    let eps = 1e-6f32;
    let dmin_c = [dmin[0].max(eps), dmin[1].max(eps), dmin[2].max(eps)];
    let soft_inv = |inv: f32, shoulder: f32| -> f32 {
        let inv = inv.max(0.0);
        let s = shoulder.max(1e-6);
        inv / (1.0 + inv / s)
    };
    let mut shoulder = [0.0f32; 3];
    let mut g = [0.0f32; 3];
    for c in 0..3 {
        let mid_t = (mid[c] / dmin_c[c]).clamp(eps, 1.0 - eps);
        let inv_mid = (1.0 / mid_t - 1.0).max(0.0);
        shoulder[c] = (inv_mid * HEADROOM).max(eps);
        let soft_mid = soft_inv(inv_mid, shoulder[c]).max(eps);
        g[c] = MIDDLE_GRAY / soft_mid;
    }

    // Constant buffer for invert: dmin(3) g(3) shoulder(3) lum(3) [CHROMA_KEEP, FADE_Y, eps]
    let ic: Vec<f32> = vec![
        dmin_c[0],
        dmin_c[1],
        dmin_c[2],
        g[0],
        g[1],
        g[2],
        shoulder[0],
        shoulder[1],
        shoulder[2],
        R_RELATIVE_LUMINANCE,
        G_RELATIVE_LUMINANCE,
        B_RELATIVE_LUMINANCE,
        CHROMA_KEEP,
        SHADOW_CHROMA_FADE_Y,
        eps,
    ];
    let icbuf = ctx.create_f32_buffer_init(&ic, "film_invert_consts");
    let n = width * height;
    let u = InvertU {
        n: n as u32,
        _p0: 0,
        _p1: 0,
        _p2: 0,
    };
    ctx.dispatch_compute_shader_multi(
        "film_invert",
        shaders::INVERT,
        &[&gpu_buf.buffer, &icbuf],
        bytemuck::bytes_of(&u),
        workgroups(n),
    );
    Ok(())
}
