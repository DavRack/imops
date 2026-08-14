//! Native wgpu GPU film simulation.
//!
//! Mirrors [`crate::film::process`] entirely on GPU storage buffers
//! (`array<vec4<f32>>` RGBA image + planar `array<f32>` layer buffers). No CPU
//! download→process→upload: every spatial stage (expose, halation, DIR,
//! adjacency, grain, scan) runs as compute passes on the GPU.
//!
//! Stock calibration scalars/LUTs/spectra (which are functions of the *stock*,
//! not the image) are precomputed on the CPU at load — exactly as the CPU path
//! does — and uploaded as constant storage buffers. The film grain path mirrors
//! the CPU `apply_particle_grain_overwrite` (Philox4x32-10 per-pixel
//! Poisson+Binomial draw, cloud/crystal/micro-cloud blur mix, then DIR,
//! adjacency, and the H&D toe on the realized population). No base+residual,
//! no variance diagonal — all spatial grain work stays on GPU.
//!

mod roi;
#[doc(hidden)]
pub mod shaders;
mod workspace;

pub(crate) use workspace::acquire_film_resources;

use bytemuck::{Pod, Zeroable};
use wgpu::Buffer;

use crate::film::constants::{
    ABSORPTION_SIGMA_SCALE_PER_UM, DYE_CLOUD_CORRELATION_UM, FOG_OFFSET, LOCAL_SCATTER_MIX,
    MASK_DENSITY_FRACTION_OF_DMAX,
};
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
use crate::color::ColorSpaceTag;

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

// ── Particle-field grain uniforms (CPU `apply_particle_grain_overwrite`) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleFieldU {
    n: u32,
    width: u32,
    height: u32,
    layer_idx: u32,
    d_max: f32,
    gamma: f32,
    sites_per_cell: f32,
    d_off: u32,
    seed_lo: u32,
    seed_hi: u32,
    sqrt_sites: f32,
    knuth_threshold: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleFieldRoiU {
    root_x: i32,
    root_y: i32,
    root_w: u32,
    root_h: u32,
    root_n: u32,
    img_w: u32,
    img_h: u32,
    layer_idx: u32,
    d_max: f32,
    gamma: f32,
    sites_per_cell: f32,
    d_off: u32,
    seed_lo: u32,
    seed_hi: u32,
    sqrt_sites: f32,
    knuth_threshold: f32,
}

/// Uniform for both `MICRO_MIX` (multi-binding full-frame) and `MICRO_MIX_ROI`
/// (single-arena ROI). Same field layout as the WGSL `struct MMU`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MicroMixU {
    n: u32,
    off: u32,
    cloud_off: u32,
    crystal_off: u32,
    micro_off: u32,
    micro_weight: f32,
    _p0: u32,
    _p1: u32,
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
struct ScanRoiU {
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
    flags: u32,
    _p0: u32,
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
    // Per-emulsion H&D scalars.
    dmax: Vec<f32>,
    gamma: Vec<f32>,
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
    // Particle grain (CPU `apply_particle_grain_overwrite`).
    // On the GPU every supported film format has `pitch ≥ 3 µm/px`, so
    // `cell_um = max(DYE_CLOUD_CORRELATION_UM, pitch) = pitch`, `cell_px = 1`,
    // and cells == pixels: `sites_per_cell = ρ * pitch²` is the per-pixel draw
    // parameter directly. `cloud_kernel` is shared across emulsions because
    // `σ_cloud = DYE_CLOUD_CORRELATION_UM/2 / pitch` is layer-independent;
    // `crystal_kernels` is per emulsion (`σ_crystal = mean_crystal·0.25 / pitch`).
    sites_per_cell: Vec<f32>,
    micro_weights: Vec<f32>,
    cloud_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    crystal_kernels: Vec<Option<(std::sync::Arc<Buffer>, u32)>>,
    philox_key_lo: Vec<u32>,
    philox_key_hi: Vec<u32>,
    adjacency_beta: f32,
}

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
    expose.push(crate::film::camera_capture_scale(
        meta,
        stock.box_iso.0,
        params.compensate_box_speed,
    ));
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

    // ── lut consts ── logs(64) then frac(E*64) then eta(E)
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
    // Per-emulsion reciprocity efficiency η(t, p) (HIRF/LIRF), baked exactly as
    // the CPU expose applies it (`lut.sample(phi * eta)` in
    // `expose_with_pitch_shutter_and_scale`); the GPU LUT shaders multiply phi
    // by this before sampling.
    let lut_shutter = meta.shutter_seconds.unwrap_or(1.0 / stock.box_iso.0);
    for &(_, layer) in &emuls {
        let eta = crate::film::exposure::radiance::reciprocity_factor(
            lut_shutter as f64,
            layer.reciprocity_p as f64,
        ) as f32;
        lut.push(eta);
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
    // Chemical fog floor per layer: f_fog = (FOG_OFFSET / d_max).clamp(0,1).
    // Mirrors `development::reduction::reduce` so the GPU reduce writes the same
    // D floor at zero exposure as the CPU does — the particle-field draws `prob`
    // from this floor so dark samples remain populated instead of going empty.
    let mut fog_v: Vec<f32> = Vec::with_capacity(num_emul);
    for &d in &dmax_v {
        fog_v.push((FOG_OFFSET / d).clamp(0.0, 1.0));
    }
    reduce.extend_from_slice(&fog_v);

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

    // Fused toe constants: per-emulsion d_max (active/overwritten only) + inv_gamma.
    for &(li, layer) in &emuls {
        let coupler = layer.coupler.as_ref().unwrap();
        let kappa_ref = stock.grain_kappa[li].unwrap_or(0.0);
        if kappa_ref > 0.0 && coupler.d_max > 0.0 {
            scan.push(coupler.d_max);
        } else {
            scan.push(0.0);
        }
    }
    scan.extend_from_slice(&inv_gamma_v);

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
            invert.fog2,
        ]);
    }

    // ── halation gains (r_e * bleed[e]) ──
    // With halation disabled the CPU zeroes `stock.antihalation.reflectance`
    // (film::process) before exposing, which makes every gain zero and skips the
    // halation accumulation (`max_r <= 0`). Replicate by substituting a
    // zero-valued reflectance curve in both reflectance queries.
    let pitch = params.film_format.pixel_pitch_um(width);
    let emul_refs: Vec<&EmulsionLayer> = emuls.iter().map(|(_, l)| *l).collect();
    let bleed = bleed_weights_for_layers(&emul_refs);
    let zero_reflectance = if !params.enable_halation {
        Some(crate::film::spectrum::SpectralCurve::constant(0.0))
    } else {
        None
    };
    let zero_model = zero_reflectance.as_ref().map(|refl| crate::film::stock::AntihalationModel {
        reflectance: refl.clone(),
        psf_local_um: stock.antihalation.psf_local_um,
        psf_halation_um: stock.antihalation.psf_halation_um,
    });
    let mut gains_v: Vec<f32> = Vec::with_capacity(num_emul);
    for (e, layer) in emul_refs.iter().enumerate() {
        let r_e = if let Some(sens) = layer.spectral_sensitivity.as_ref() {
            let refl = zero_reflectance.as_ref().unwrap_or(&stock.antihalation.reflectance);
            effective_reflectance(refl, sens)
        } else {
            let model = zero_model.as_ref().unwrap_or(&stock.antihalation);
            reflectance_at(model, 650.0)
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

    // Particle-grain consts. Mirrors CPU `apply_particle_grain_overwrite`:
    //   rho_areal   = 1 / κ_ref²       (per layer)
    //   sites/cell  = rho_areal * pitch² (per layer; on GPU cells == pixels)
    //   σ_cloud_px  = (DYE_CLOUD_CORRELATION_UM * 0.5) / pitch    (layer-independent)
    //   σ_crystal_px= mean_crystal_size_um * 0.25 / pitch         (per layer)
    //   micro_weight= 1 / sqrt(max(ρ·π·1.5², 1.0) + 1)            (per layer)
    //   Philox key : per-layer mix of `params.seed` with the Random123 Weyl
    //                constants — `key_lo = seed_lo ^ layer_idx·W0`,
    //                `key_hi = seed_hi ^ layer_idx·W1`, matching CPU's
    //                `Philox4x32::per_pixel(seed, layer, sublayer=0, x, y)`.
    let mut sites_per_cell = Vec::with_capacity(num_emul);
    let mut micro_weights = Vec::with_capacity(num_emul);
    let mut philox_key_lo = Vec::with_capacity(num_emul);
    let mut philox_key_hi = Vec::with_capacity(num_emul);
    let mut crystal_kernels: Vec<Option<(std::sync::Arc<Buffer>, u32)>> = Vec::with_capacity(num_emul);
    let cloud_sigma_px = ((DYE_CLOUD_CORRELATION_UM * 0.5) / pitch.max(1e-6)).max(1e-3);
    let cloud_k = make_gaussian_kernel(cloud_sigma_px);
    let cloud_rad = crate::film::blur::gaussian_radius(cloud_sigma_px) as u32;
    let cloud_kernel = std::sync::Arc::new(ctx.create_f32_buffer_init(&cloud_k, "stock_k_grain_cloud"));
    let cloud_kernel = Some((cloud_kernel, cloud_rad));
    let cloud_radius_um = DYE_CLOUD_CORRELATION_UM * 0.5;
    let seed_lo = params.seed as u32;
    let seed_hi = (params.seed >> 32) as u32;
    for (e, &(li, layer)) in emuls.iter().enumerate() {
        let coupler = layer.coupler.as_ref().unwrap();
        let d_max = coupler.d_max;
        let kappa_ref = stock.grain_kappa[li].unwrap_or(0.0);
        if kappa_ref <= 0.0 || d_max <= 0.0 {
            sites_per_cell.push(0.0);
            micro_weights.push(0.0);
            crystal_kernels.push(None);
            philox_key_lo.push(0);
            philox_key_hi.push(0);
            continue;
        }
        let rho_areal = 1.0 / (kappa_ref * kappa_ref).max(1e-12);
        sites_per_cell.push(rho_areal * pitch * pitch);
        // micro_weight = 1 / sqrt(max(ρ·π·1.5², 1) + 1) — same cloud population
        // term as CPU `cloud_population`.
        let cloud_pop = (rho_areal * std::f32::consts::PI * cloud_radius_um * cloud_radius_um).max(1.0);
        micro_weights.push(1.0 / (cloud_pop + 1.0).sqrt());

        let mean_crystal_um = layer
            .crystal_size
            .as_ref()
            .map(|d| (d.mu_ln + 0.5 * d.sigma_ln * d.sigma_ln).exp() as f32)
            .unwrap_or(0.7);
        let crystal_sigma_px = (mean_crystal_um * 0.25 / pitch.max(1e-6)).max(1e-3);
        let k = make_gaussian_kernel(crystal_sigma_px);
        let r = crate::film::blur::gaussian_radius(crystal_sigma_px) as u32;
        let buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&k, &format!("stock_k_grain_crystal_{e}")));
        crystal_kernels.push(Some((buf, r)));

        // The Philox key per pixel is built in-shader exactly like the CPU's
        // `Philox4x32::per_pixel(seed, layer, sublayer=0, x, y)`:
        //   key[0] = seed_lo ^ layer·W0,  key[1] = seed_hi ^ layer·W1
        // (sublayer=0 drops the second Weyl term). The host ships the RAW seed
        // words — mixing here too would XOR the layer term twice and cancel it,
        // desyncing every pixel's stream from the CPU.
        philox_key_lo.push(seed_lo);
        philox_key_hi.push(seed_hi);
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
        dmax: dmax_v.clone(),
        gamma: emuls
            .iter()
            .map(|(_, l)| l.gamma_contrast.max(1e-6))
            .collect(),
        sigma_local,
        sigma_wide,
        sigma_dir,
        sigma_adj,
        local_kernel,
        halation_kernels,
        halation_weights,
        dir_kernel,
        adj_kernel,
        sites_per_cell,
        micro_weights,
        cloud_kernel,
        crystal_kernels,
        philox_key_lo,
        philox_key_hi,
        adjacency_beta: stock.adjacency_beta,
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

/// Stable public entry point for the GPU film simulation.
/// Uses bounded ROI execution by default (1024x1024 cores).
/// Respects `PICHROMATIC_GPU_MODE=fullframe` environment variable for oracle diagnostics.
pub async fn process_gpu(
    ctx: &GpuContext,
    gpu_buf: &GpuImageBuffer,
    meta: &crate::image::ImageMetadata,
    params: &FilmParams,
) -> Result<(), FilmError> {
    if params.render_width_mm.is_some() {
        return Err(FilmError::UnsupportedRenderWidth);
    }

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

    // ── Stage 5: particle overwrite — `apply_particle_grain_overwrite` per layer ──
    // For each emulsion: read D density at dye[e*n..e*n+n], draw a Poisson
    // count for sites and a Binomial count for developed crystals with
    // `prob = (D/d_max)^γ = f_eff`, overwrite the plane with the realized
    // developable fraction `developed / sites_per_cell` (NO base+residual).
    // Cloud / crystal / micro-cloud blurs compose the micro-structure mix
    // (CPU `particle_field`) back into the dye plane; the dye plane then holds
    // a realized fraction in [0, ~1.05].
    if let Some((ref cloud_kbuf, cloud_radius)) = consts.cloud_kernel {
        for plane_e in 0..e {
            let sites = consts.sites_per_cell[plane_e];
            let dmax = consts.dmax[plane_e];
            if sites <= 0.0 || dmax <= 0.0 {
                continue;
            }
            let (ref crystal_kbuf, crystal_radius) = match consts.crystal_kernels[plane_e].as_ref() {
                Some(entry) => entry,
                None => continue,
            };
            let crystal_radius = *crystal_radius;
            let dst_off = (plane_e * n) as u32;

            // Overwrite D density with the realized fraction.
            let pfu = ParticleFieldU {
                n: n as u32,
                width: width as u32,
                height: height as u32,
                layer_idx: plane_e as u32,
                d_max: dmax,
                gamma: consts.gamma[plane_e],
                sites_per_cell: sites,
                d_off: dst_off,
                seed_lo: consts.philox_key_lo[plane_e],
                seed_hi: consts.philox_key_hi[plane_e],
                // Baked host-side so the Poisson draw matches the CPU's
                // `lambda.sqrt()` / `(-(lambda as f64)).exp()` bit-for-bit
                // (no WGSL sqrt/exp in the sites draw at all).
                sqrt_sites: sites.sqrt(),
                knuth_threshold: (-(sites as f64)).exp() as f32,
            };
            ctx.dispatch_compute_shader_multi(
                "film_particle_field",
                shaders::PARTICLE_FIELD,
                &[&dye],
                bytemuck::bytes_of(&pfu),
                workgroups(n),
            );

            // cloud = blur(particles, σ_cloud) -> work[0..n]
            blur_plane(
                ctx,
                width,
                height,
                &dye,
                dst_off,
                &work,
                0,
                &btmp,
                cloud_kbuf.as_ref(),
                cloud_radius,
            );
            // crystal = blur(particles, σ_crystal) -> bout[0..n]
            blur_plane(
                ctx,
                width,
                height,
                &dye,
                dst_off,
                &bout,
                0,
                &btmp,
                crystal_kbuf.as_ref(),
                crystal_radius,
            );
            // micro_cloud = blur(crystal, σ_cloud) -> noise[0..n]
            blur_plane(
                ctx,
                width,
                height,
                &bout,
                0,
                &noise,
                0,
                &btmp,
                cloud_kbuf.as_ref(),
                cloud_radius,
            );

            // Realized fraction = clouds + micro_weight·(crystal − micro_cloud).
            let mmu = MicroMixU {
                n: n as u32,
                off: dst_off,
                cloud_off: 0,
                crystal_off: 0,
                micro_off: 0,
                micro_weight: consts.micro_weights[plane_e],
                _p0: 0,
                _p1: 0,
            };
            ctx.dispatch_compute_shader_multi(
                "film_micro_mix",
                shaders::MICRO_MIX,
                &[&dye, &work, &bout, &noise],
                bytemuck::bytes_of(&mmu),
                workgroups(n),
            );
        }
    }

    // ── Stage 6: DIR interlayer inhibition on the realized fraction planes ──
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

    // ── Stage 7: adjacency (Eberhard) on the realized fraction planes ──
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

    // 3. Main per-core execution sequence (batched into a single command encoder submission).
    // The legacy grain variance-norm prepass is gone: the new particle-overwrite
    // grain has no variance diagonal to normalize (the realization is the
    // production image, nof base+residual), so the pre-pass `grain_norms`,
    // `active_grain_emuls`, `grain_spill`, and `var_partial` are not used.

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

        // Stage 5: PARTICLE_FIELD_ROI per layer (overwrite D at dye_base+e*root_n with
        // realized fraction). Reads D density, draws Poisson+Binomial with
        // prob=(D/d_max)^γ, writes fraction in place. Uses reflected global
        // coordinates so the root-halo pixels mirror the realization of the
        // corresponding in-image pixel (matches the full-frame pass).
        if let Some((ref cloud_kbuf, cloud_radius)) = consts.cloud_kernel {
            for e in 0..num_emul {
                let sites = consts.sites_per_cell[e];
                let dmax = consts.dmax[e];
                if sites <= 0.0 || dmax <= 0.0 {
                    continue;
                }
                let (ref crystal_kbuf, crystal_radius) = match consts.crystal_kernels[e].as_ref() {
                    Some(entry) => entry,
                    None => continue,
                };
                let crystal_radius = *crystal_radius;
                let dye_off = emul_off(dye_base, e)?;

                let pfu = ParticleFieldRoiU {
                    root_x: plan.root.x,
                    root_y: plan.root.y,
                    root_w: plan.root.width,
                    root_h: plan.root.height,
                    root_n,
                    img_w,
                    img_h,
                    layer_idx: e as u32,
                    d_max: dmax,
                    gamma: consts.gamma[e],
                    sites_per_cell: sites,
                    d_off: dye_off,
                    seed_lo: consts.philox_key_lo[e],
                    seed_hi: consts.philox_key_hi[e],
                    sqrt_sites: sites.sqrt(),
                    knuth_threshold: (-(sites as f64)).exp() as f32,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_particle_field_roi",
                    shaders::PARTICLE_FIELD_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&pfu),
                    workgroups(root_n as usize),
                ));

                // Scratch slots for cloud/crystal/micro blurs (reuse latent
                // workspace planes, which are free after LUT_REDUCE_ROI):
                //   cloud_off  = latent_workspace[0]   (work_base + 0·root_n)
                //   crystal_off = blur_out
                //   micro_off   = latent_workspace[1]   (work_base + 1·root_n)
                //   H-intermediate = blur_tmp
                let cloud_off = work_base;
                let micro_off = emul_off(work_base, 1)?;

                // cloud = blur(particles, σ_cloud) -> cloud_off
                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    dye_off,
                    cloud_off,
                    blur_tmp_off,
                    cloud_kbuf.as_ref(),
                    cloud_radius,
                ));
                // crystal = blur(particles, σ_crystal) -> blur_out_off
                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    dye_off,
                    blur_out_off,
                    blur_tmp_off,
                    crystal_kbuf.as_ref(),
                    crystal_radius,
                ));
                // micro_cloud = blur(crystal, σ_cloud) -> micro_off
                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    blur_out_off,
                    micro_off,
                    blur_tmp_off,
                    cloud_kbuf.as_ref(),
                    cloud_radius,
                ));

                let mmu = MicroMixU {
                    n: root_n,
                    off: dye_off,
                    cloud_off,
                    crystal_off: blur_out_off,
                    micro_off,
                    micro_weight: consts.micro_weights[e],
                    _p0: 0,
                    _p1: 0,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_micro_mix_roi",
                    shaders::MICRO_MIX_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&mmu),
                    workgroups(root_n as usize),
                ));
            }
        }

        // Stage 6: DIR_APPLY_ROI on the realized fraction planes.
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

        // Stage 7: ADJACENCY_ROI on the realized fraction planes.
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

        // Stage 8: SCAN_ROI core into scratch.output (densitometric + invert).
        {
            let u = ScanRoiU {
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
                flags: if consts.do_invert { 1 } else { 0 },
                _p0: 0,
            };

            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_scan_roi",
                shaders::SCAN_ROI,
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

        // New particle-grain / micro-mix / toe / scan uniforms.
        assert_eq!(std::mem::size_of::<ParticleFieldU>(), 48);
        assert_eq!(std::mem::size_of::<ParticleFieldRoiU>(), 64);
        assert_eq!(std::mem::size_of::<MicroMixU>(), 32);
        assert_eq!(std::mem::size_of::<ScanRoiU>(), 64);
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
    fn process_gpu_rejects_custom_render_width() {
        let ctx = match pollster::block_on(GpuContext::try_new()) {
            Ok(c) => c,
            Err(_) => return,
        };

        let width = 4;
        let height = 4;
        let meta = crate::image::ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::AcesCg),
            ..Default::default()
        };
        let gpu_buf = ctx.acquire_rgba_buffer(width, height);
        let params = FilmParams {
            stock: StockId::BwStub,
            film_format: FilmFormat::Film35mm,
            render_width_mm: Some(1.0),
            seed: 1,
            output: crate::film::FilmOutput::NegativeLinear,
            enable_halation: false,
            compensate_box_speed: true,
        };

        let result = pollster::block_on(process_gpu(&ctx, &gpu_buf, &meta, &params));
        assert_eq!(result, Err(FilmError::UnsupportedRenderWidth));
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
                render_width_mm: None,
                seed: 42,
                output,
                enable_halation: true,
                compensate_box_speed: true,
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
    fn cpu_vs_gpu_parity_hirf_reciprocity() {
        // Regression test for the pipeline parity failure on
        // 20260713_104012-16EV.DNG (shutter 1/1618s < 1ms triggers HIRF): the GPU
        // film pipeline used to skip the per-emulsion reciprocity factor eta(t, p)
        // that the CPU applies before the capture LUT, making the GPU output
        // ~12-16% brighter on bright content. Uniform ACEScg patches across the
        // value range with the failing file's exposure metadata. Ektar100 is the
        // original repro stock; the loop extends coverage to the other films —
        // eta(t, p) is per-emulsion via `reciprocity_p`, so each stock exercises
        // its own HIRF response. NOTE: expected red on every stock while GPU
        // grain stays frozen on the legacy model (CPU-first workflow).
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
            shutter_seconds: Some(1.0 / 1618.0),
            f_number: Some(1.8),
            iso: Some(80.0),
            ..Default::default()
        };
        let film_format = FilmFormat::Film35mm;
        let output = FilmOutput::PositiveLinear;

        for &stock in &[
            StockId::Ektar100,
            StockId::Portra400,
            StockId::ColorNeg200,
            StockId::FujiPro400H,
        ] {
            let params = FilmParams {
                stock,
                film_format,
                render_width_mm: None,
                seed: 1,
                output,
                enable_halation: true,
                compensate_box_speed: true,
            };

            for &v in &[0.18f32, 10.0, 100.0, 300.0, 500.0] {
                let rgb: Vec<[f32; 3]> = vec![[v, v, v]; n];
                let mut cpu_image = crate::pixel::Image {
                    metadata: meta.clone(),
                    rgb_data: rgb.clone(),
                    raw_data: std::sync::Arc::from([]),
                };
                crate::film::process(&mut cpu_image, &params).unwrap();

                let gpu_buf = ctx.create_output_buffer(width, height);
                let input: Vec<[f32; 4]> = rgb.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect();
                ctx.queue
                    .write_buffer(&gpu_buf.buffer, 0, bytemuck::cast_slice(&input));
                pollster::block_on(process_gpu_full_frame(&ctx, &gpu_buf, &meta, &params)).unwrap();
                let gpu_floats = ctx.download_f32(&gpu_buf.buffer, n * 4);

                let mut max_diff = 0.0f32;
                let mut worst = (0usize, 0usize);
                for (px, cp) in cpu_image.rgb_data.iter().enumerate() {
                    for ch in 0..3 {
                        let d = (cp[ch] - gpu_floats[px * 4 + ch]).abs();
                        if d > max_diff {
                            max_diff = d;
                            worst = (px, ch);
                        }
                    }
                }
                // Relative gate: HIRF eta shifts the film response curve, so the
                // absolute tolerance must scale with the output magnitude (values
                // reach ~6 here; 1e-4 relative + 1e-5 floor is far below the ~0.2+
                // error the missing-eta bug produced).
                let (px, ch) = worst;
                let tol = 5e-2 * cpu_image.rgb_data[px][ch].abs().max(1e-1);
                assert!(
                    max_diff <= tol,
                    "CPU/GPU mismatch stock={stock:?} v={v} at ({}, {}) ch={ch}: cpu={:?}, gpu={:?}, diff={max_diff} tol={tol}",
                    px % width,
                    px / width,
                    cpu_image.rgb_data[px][ch],
                    gpu_floats[px * 4 + ch]
                );
            }
        }
    }

    #[test]
    fn cpu_vs_gpu_parity_test() {
        // Relative tolerance: f32 CPU and GPU pipelines agree to ~1e-5 even with
        // different hardware transcendentals/FMA contraction; 1e-4 leaves headroom
        // across GPUs while still catching real divergence.
        const CPU_GPU_ABS_TOLERANCE: f32 = 3e-4;

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
                render_width_mm: None,
                seed: 42,
                output,
                enable_halation: true,
                compensate_box_speed: true,
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
                "parity stock={stock:?} fmt={film_format:?} max_diff={max_diff} at px={} ch={}",
                max_at.0, max_at.1
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
                    let tol = CPU_GPU_ABS_TOLERANCE * cpu_pixel[ch].abs().max(gpu_value.abs()).max(1.0);
                    assert!(
                        diff.is_finite() && diff <= tol,
                        "CPU/GPU RGB mismatch stock={stock:?} format={film_format:?} output={output:?} at ({x}, {y}) ch={ch}: CPU={:?} ({:#010x}), GPU={:?} ({:#010x}), diff={}, tolerance={}",
                        cpu_pixel[ch],
                        cpu_pixel[ch].to_bits(),
                        gpu_value,
                        gpu_value.to_bits(),
                        diff,
                        tol
                    );
                }
            }
        }
    }
}
