//! Native wgpu GPU film simulation.
//!
//! Mirrors [`crate::film::process`] entirely on GPU storage buffers
//! (`array<vec4<f32>>` RGBA image + planar `array<f32>` layer buffers). No CPU
//! download->process->upload: every spatial stage (expose, halation,
//! irradiation, adjacency, grain, scan, scanner MTF, invert) runs as compute passes on the GPU.
//!
//! Stock calibration scalars/LUTs/spectra are precomputed on the CPU at load
//! and uploaded as constant storage buffers. The film grain path mirrors
//! the CPU `apply_particle_grain_overwrite` (Philox4x32-10 per-pixel
//! Poisson+Binomial draw, effective dye-cloud blur, then two-component adjacency).

mod roi;
#[doc(hidden)]
pub mod shaders;
mod workspace;

pub(crate) use workspace::acquire_film_resources;

use bytemuck::{Pod, Zeroable};
use wgpu::Buffer;

use crate::color::ColorSpaceTag;
use crate::film::constants::{
    ABSORPTION_SIGMA_SCALE_PER_UM, DYE_CLOUD_CORRELATION_UM, FOG_OFFSET,
    MASK_DENSITY_FRACTION_OF_DMAX,
};
use crate::film::exposure::halation::sigma_px_from_um;
use crate::film::exposure::upsample::{
    CIE1931_XBAR, CIE1931_YBAR, CIE1931_ZBAR, XYZ_D65_TO_ACESCG,
};
use crate::film::scan::densitometry::{
    dmin_reference_acescg, scanner_aperture_sigma_px, scanner_calibration_acescg,
};
use crate::film::stock::{EmulsionLayer, FilmStock, LayerKind};
use crate::film::{FilmError, FilmOutput, FilmParams};
use crate::gpu::{ComputePassDesc, GpuContext, GpuImageBuffer};

/// Max FIR radius for tiled blur shaders (`tile[1024]` = 256 + 2x384).
/// Larger radii fall back to untiled [`shaders::BLUR_H`] / [`shaders::BLUR_V`].
pub(crate) const BLUR_TILED_MAX_RADIUS: u32 = 384;

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
struct ExposeRoiU {
    root_x: i32,
    root_y: i32,
    root_w: u32,
    root_h: u32,
    img_w: u32,
    img_h: u32,
    root_n: u32,
    planes_base: u32,
    bounce_base: u32,
    num_layers: u32,
    num_emul: u32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransposeU {
    width: u32,
    height: u32,
    src_off: u32,
    dst_off: u32,
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
struct IrradiationMixU {
    n: u32,
    plane_off: u32,
    weight: f32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IrradiationMixRoiU {
    n: u32,
    plane_off: u32,
    core_off: u32,
    tail1_off: u32,
    tail2_off: u32,
    weight: f32,
    _p0: u32,
    _p1: u32,
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
struct HalationAddRoiU {
    n: u32,
    num_emul: u32,
    plane_base: u32,
    bounce_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HalationAddEmulU {
    n: u32,
    plane_off: u32,
    _p0: u32,
    _p1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HalationAddEmulRoiU {
    n: u32,
    plane_off: u32,
    acc_off: u32,
    _p0: u32,
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
struct ParticleFieldU {
    n: u32,
    width: u32,
    height: u32,
    in_off: u32,
    out_off: u32,
    d_max: f32,
    sites_per_cell: f32,
    key_lo: u32,
    key_hi: u32,
    sqrt_sites: f32,
    knuth_threshold: f32,
    _p0: u32,
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
    in_off: u32,
    out_off: u32,
    d_max: f32,
    sites_per_cell: f32,
    key_lo: u32,
    key_hi: u32,
    sqrt_sites: f32,
    knuth_threshold: f32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScaleDmaxU {
    n: u32,
    in_off: u32,
    out_off: u32,
    d_max: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScaleDmaxRoiU {
    n: u32,
    in_off: u32,
    out_off: u32,
    d_max: f32,
}

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
struct AdjacencyTwoCompU {
    n: u32,
    num_emul: u32,
    ex_active: u32,
    dir_active: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdjacencyTwoCompRoiU {
    n: u32,
    num_emul: u32,
    dye_base: u32,
    diff_ex_base: u32,
    diff_dir_base: u32,
    ex_active: u32,
    dir_active: u32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScanToAcescgU {
    n: u32,
    num_emul: u32,
    scale: f32,
    _p0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScanToAcescgRoiU {
    root_n: u32,
    num_emul: u32,
    dye_base: u32,
    mask_base: u32,
    r_base: u32,
    g_base: u32,
    b_base: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InvertU {
    n: u32,
    mode: u32,
    eps: f32,
    /// Chroma decode anchor (`crate::pixel::MIDDLE_GRAY`).
    mid_gray: f32,
    /// Per-channel inverse exponents (1/γ), xyz active; matches WGSL vec4<f32>.
    inv_gamma: [f32; 4],
    inv_dmin: [f32; 4],
    exponent: [f32; 4],
    gain: [f32; 4],
    /// Chroma decode matrix rows (row-major, xyz active); matches WGSL vec4<f32>.
    chroma_r: [f32; 4],
    chroma_g: [f32; 4],
    chroma_b: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InvertRoiU {
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
    r_base: u32,
    g_base: u32,
    b_base: u32,
    mode: u32,
    eps: f32,
    /// Chroma decode anchor (`crate::pixel::MIDDLE_GRAY`).
    mid_gray: f32,
    /// Per-channel inverse exponents (1/γ), xyz active; matches WGSL vec4<f32>.
    inv_gamma: [f32; 4],
    inv_dmin: [f32; 4],
    exponent: [f32; 4],
    gain: [f32; 4],
    /// Chroma decode matrix rows (row-major, xyz active); matches WGSL vec4<f32>.
    chroma_r: [f32; 4],
    chroma_g: [f32; 4],
    chroma_b: [f32; 4],
}

/// Packs a row-major 3x3 chroma decode matrix into three vec4-aligned rows
/// (w = 0) matching the WGSL `chroma_r/g/b` uniform members.
fn chroma_rows(m: [[f32; 3]; 3]) -> ([f32; 4], [f32; 4], [f32; 4]) {
    (
        [m[0][0], m[0][1], m[0][2], 0.0],
        [m[1][0], m[1][1], m[1][2], 0.0],
        [m[2][0], m[2][1], m[2][2], 0.0],
    )
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

/// Gaussian kernel identical to `blur::make_gaussian_kernel` (f32, radius ceil(3σ)).
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
    mat_ex: Buffer,
    mat_dir: Buffer,
    scan: Buffer,
    num_layers: u32,
    num_emul: u32,
    scan_scale: f32,
    // Per-emulsion H&D scalars.
    dmax: Vec<f32>,
    sites_per_cell: Vec<f32>,
    philox_key_lo: Vec<u32>,
    philox_key_hi: Vec<u32>,
    // Blur kernels
    irrad_kernels: Option<(
        (std::sync::Arc<Buffer>, u32),
        (std::sync::Arc<Buffer>, u32),
        (std::sync::Arc<Buffer>, u32),
    )>,
    irrad_weights: Vec<f32>,
    halation_kernels: Vec<(std::sync::Arc<Buffer>, u32)>,
    halation_weights: Vec<f32>,
    eff_grain_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    adj_ex_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    adj_dir_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    scanner_mtf_kernel: Option<(std::sync::Arc<Buffer>, u32)>,
    // Invert parameters
    invert_mode: u32,
    inv_dmin: [f32; 4],
    exponent: [f32; 4],
    gain: [f32; 4],
    inv_gamma: [f32; 4],
    eps: f32,
    /// Post-invert cross-channel chroma decode (row-major), from the same
    /// `scanner_calibration_acescg` as the invert constants. Identity unless
    /// PositiveLinear.
    chroma_decode: [[f32; 3]; 3],
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
    let pitch = params.film_format.pixel_pitch_um(width);
    let shutter = meta.shutter_seconds.unwrap_or(1.0 / stock.box_iso.0);

    // Upsample basis + acescg->weights (matches exposure::upsample::UpsampleBasis).
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
    let a2w = inv3(m); // acescg -> weights (row-major)

    // ── expose consts ──
    // Layout: b0(16) b1(16) b2(16) M(9) lambda_factor(16) capture_scale trans(L*16) produces_latent(L) reflectance(16)
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
    if params.enable_halation {
        for i in 0..16 {
            expose.push(stock.antihalation.reflectance.samples[i] as f32);
        }
    } else {
        for _ in 0..16 {
            expose.push(0.0);
        }
    }

    // ── lut consts ── logs(64) then frac(E*64) then eta(E)
    let mut lut: Vec<f32> = Vec::new();
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
    for &(_, layer) in &emuls {
        let eta = crate::film::exposure::radiance::reciprocity_factor(
            shutter as f64,
            layer.reciprocity_p as f64,
        ) as f32;
        lut.push(eta);
    }

    // ── reduce consts ── dmax(E) inv_gamma(E) mask_scale(E) reversal(E) has_mask(E) fog_offset(E)
    let mut dmax_v = Vec::with_capacity(num_emul);
    let mut inv_gamma_v = Vec::with_capacity(num_emul);
    let mut mask_scale_v = Vec::with_capacity(num_emul);
    let mut reversal_v = Vec::with_capacity(num_emul);
    let mut has_mask_v = Vec::with_capacity(num_emul);
    let mut fog_v = Vec::with_capacity(num_emul);
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
        fog_v.push(FOG_OFFSET);
    }
    let mut reduce: Vec<f32> = Vec::new();
    reduce.extend_from_slice(&dmax_v);
    reduce.extend_from_slice(&inv_gamma_v);
    reduce.extend_from_slice(&mask_scale_v);
    reduce.extend_from_slice(&reversal_v);
    reduce.extend_from_slice(&has_mask_v);
    reduce.extend_from_slice(&fog_v);

    // ── Stage 2: Spatial Exposure kernels ──
    // 1. In-emulsion Irradiation Response
    let irrad_kernels = if let Some(resp) = stock.irradiation_response {
        let core_sigma_px = resp.core_sigma_um / pitch.max(1e-6);
        let tail_decay_px = resp.tail_decay_um / pitch.max(1e-6);
        let tail1_sigma_px = 0.9401 * tail_decay_px;
        let tail2_sigma_px = 2.5177 * tail_decay_px;

        let core_k = make_gaussian_kernel(core_sigma_px);
        let core_r = crate::film::blur::gaussian_radius(core_sigma_px) as u32;
        let core_buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&core_k, "stock_k_irrad_core"));

        let t1_k = make_gaussian_kernel(tail1_sigma_px);
        let t1_r = crate::film::blur::gaussian_radius(tail1_sigma_px) as u32;
        let t1_buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&t1_k, "stock_k_irrad_tail1"));

        let t2_k = make_gaussian_kernel(tail2_sigma_px);
        let t2_r = crate::film::blur::gaussian_radius(tail2_sigma_px) as u32;
        let t2_buf = std::sync::Arc::new(ctx.create_f32_buffer_init(&t2_k, "stock_k_irrad_tail2"));

        Some(((core_buf, core_r), (t1_buf, t1_r), (t2_buf, t2_r)))
    } else {
        None
    };

    let irrad_weights: Vec<f32> = if let Some(resp) = stock.irradiation_response {
        (0..num_emul).map(|e| resp.tail_weight_bgr[e / 2]).collect()
    } else {
        vec![0.0; num_emul]
    };

    // 2. Wide backing-reflection halation kernels
    let sigma_wide = sigma_px_from_um(stock.antihalation.psf_halation_um, pitch);
    let max_r = if params.enable_halation {
        stock
            .antihalation
            .reflectance
            .samples
            .iter()
            .copied()
            .fold(0.0f64, f64::max) as f32
    } else {
        0.0
    };

    let (mut halation_kernels, mut halation_weights) = (Vec::new(), Vec::new());
    if sigma_wide >= 1e-3 && max_r > 0.0 {
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
            let buf = std::sync::Arc::new(
                ctx.create_f32_buffer_init(&bk, &format!("stock_k_halation_{k}")),
            );
            halation_kernels.push((buf, rad));
            halation_weights.push(decay[k]);
        }
    }

    // ── Stage 5: Particle Grain Consts ──
    let cloud_sigma_px = (DYE_CLOUD_CORRELATION_UM * 0.25) / pitch.max(1e-6);
    let eff_sigma_px = (cloud_sigma_px * cloud_sigma_px + 1.0 / 12.0).sqrt();
    let eff_k = make_gaussian_kernel(eff_sigma_px);
    let eff_r = crate::film::blur::gaussian_radius(eff_sigma_px) as u32;
    let eff_grain_kernel = Some((
        std::sync::Arc::new(ctx.create_f32_buffer_init(&eff_k, "stock_k_eff_grain")),
        eff_r,
    ));

    let record_sublayers = crate::film::development::stock_record_sublayers(stock);
    let mut sites_per_cell = Vec::with_capacity(num_emul);
    let mut philox_key_lo = Vec::with_capacity(num_emul);
    let mut philox_key_hi = Vec::with_capacity(num_emul);

    const PHILOX_W0: u32 = 0x9E3779B9;
    const PHILOX_W1: u32 = 0xBB67AE85;

    for (e, &(li, layer)) in emuls.iter().enumerate() {
        let coupler = layer.coupler.as_ref().unwrap();
        let d_max = coupler.d_max;
        let kappa_ref = stock.grain_kappa[li].unwrap_or(0.0);
        if kappa_ref <= 0.0 || d_max <= 0.0 {
            sites_per_cell.push(0.0);
            philox_key_lo.push(0);
            philox_key_hi.push(0);
            continue;
        }
        let rho_areal = 1.0 / (kappa_ref * kappa_ref).max(1e-12);
        sites_per_cell.push(rho_areal * pitch * pitch);

        let (rec_i, sub_i) = if e < record_sublayers.len() {
            record_sublayers[e]
        } else {
            (e as u32, 0)
        };

        let seed_lo = params.seed as u32;
        let seed_hi = (params.seed >> 32) as u32;
        let layer_tag = rec_i + 1;
        let sublayer_tag = sub_i + 1;

        let k_lo = seed_lo
            ^ layer_tag.wrapping_mul(PHILOX_W0)
            ^ sublayer_tag.wrapping_mul(PHILOX_W1);
        let k_hi = seed_hi
            ^ layer_tag.wrapping_mul(PHILOX_W1)
            ^ sublayer_tag.wrapping_mul(PHILOX_W0);

        philox_key_lo.push(k_lo);
        philox_key_hi.push(k_hi);
    }

    // ── Stage 6: Two-Component Adjacency Matrices & Kernels ──
    let sigma_ex_px = stock.developer_diffusion_length.0 / pitch.max(1e-6);
    let sigma_dir_px = stock.inhibitor_diffusion_length.0 / pitch.max(1e-6);
    let p_sat = 2.0 * std::f32::consts::PI.sqrt() * 1.25;
    let adj_scale = (pitch / p_sat).clamp(0.15, 1.0);

    let mut matrix_ex = stock.adjacency_exhaustion_matrix();
    let mut matrix_dir = stock.adjacency_inhibitor_matrix();
    for row in &mut matrix_ex {
        for v in row.iter_mut() {
            *v *= adj_scale;
        }
    }
    for row in &mut matrix_dir {
        for v in row.iter_mut() {
            *v *= adj_scale;
        }
    }

    let mut flat_mat_ex = Vec::with_capacity(num_emul * num_emul);
    for row in &matrix_ex {
        flat_mat_ex.extend_from_slice(row);
    }
    let mut flat_mat_dir = Vec::with_capacity(num_emul * num_emul);
    for row in &matrix_dir {
        flat_mat_dir.extend_from_slice(row);
    }

    let adj_ex_kernel = if sigma_ex_px >= 1e-3
        && !matrix_ex.iter().all(|row| row.iter().all(|&v| v.abs() < 1e-8))
    {
        let k = make_gaussian_kernel(sigma_ex_px);
        let r = crate::film::blur::gaussian_radius(sigma_ex_px) as u32;
        Some((std::sync::Arc::new(ctx.create_f32_buffer_init(&k, "stock_k_adj_ex")), r))
    } else {
        None
    };

    let adj_dir_kernel = if sigma_dir_px >= 1e-3
        && !matrix_dir.iter().all(|row| row.iter().all(|&v| v.abs() < 1e-8))
    {
        let k = make_gaussian_kernel(sigma_dir_px);
        let r = crate::film::blur::gaussian_radius(sigma_dir_px) as u32;
        Some((std::sync::Arc::new(ctx.create_f32_buffer_init(&k, "stock_k_adj_dir")), r))
    } else {
        None
    };

    // ── Stage 7: Scan & Scanner MTF & Invert ──
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

    let dmin_rgb = dmin_reference_acescg(stock);
    let peak = dmin_rgb[0].max(dmin_rgb[1]).max(dmin_rgb[2]).max(1e-12);
    let scan_scale = 1.0 / peak;

    let sigma_scanner_px = scanner_aperture_sigma_px(pitch);
    let scanner_mtf_kernel = if sigma_scanner_px >= 1e-3 {
        let k = make_gaussian_kernel(sigma_scanner_px);
        let r = crate::film::blur::gaussian_radius(sigma_scanner_px) as u32;
        Some((std::sync::Arc::new(ctx.create_f32_buffer_init(&k, "stock_k_scanner_mtf")), r))
    } else {
        None
    };

    let is_reversal = stock.layers.iter().any(|l| l.is_reversal);
    let identity_decode = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let (invert_mode, inv_dmin, exponent, gain, inv_gamma, chroma_decode) =
        if is_reversal || params.output == FilmOutput::NegativeLinear {
            (
                0u32,
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0 / crate::film::scan::invert::GAMMA_EFF; 4],
                identity_decode,
            )
        } else {
            let cal = scanner_calibration_acescg(stock, pitch, shutter).unwrap_or(
                crate::film::scan::ScannerCalibration {
                    dmin: [1.0, 1.0, 1.0],
                    mid: [0.18, 0.18, 0.18],
                    gamma_eff: [crate::film::scan::invert::GAMMA_EFF; 3],
                    chroma_decode: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            );
            match params.output {
                #[allow(deprecated)]
                FilmOutput::PositiveLinear | FilmOutput::PositiveInverseHd => {
                    let eps = 1e-6f32;
                    let ig = cal.inv_gamma();
                    let inv_d = [
                        1.0 / cal.dmin[0].max(eps),
                        1.0 / cal.dmin[1].max(eps),
                        1.0 / cal.dmin[2].max(eps),
                    ];
                    let mut g = [0.0f32; 3];
                    for c in 0..3 {
                        let t_mid = (cal.mid[c] * inv_d[c]).max(eps);
                        let e_mid = (t_mid.powf(-ig[c]) - 1.0).max(0.0);
                        g[c] = if e_mid > eps {
                            crate::pixel::MIDDLE_GRAY / e_mid
                        } else {
                            0.0
                        };
                    }
                    (
                        2u32,
                        [inv_d[0], inv_d[1], inv_d[2], 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [g[0], g[1], g[2], 0.0],
                        [ig[0], ig[1], ig[2], 0.0],
                        cal.chroma_decode,
                    )
                }
                FilmOutput::NegativeLinear => (
                    0u32,
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0 / crate::film::scan::invert::GAMMA_EFF; 4],
                    identity_decode,
                ),
            }
        };

    StockConsts {
        expose: ctx.create_f32_buffer_init(&expose, "film_expose_consts"),
        lut: ctx.create_f32_buffer_init(&lut, "film_lut_consts"),
        reduce: ctx.create_f32_buffer_init(&reduce, "film_reduce_consts"),
        mat_ex: ctx.create_f32_buffer_init(&flat_mat_ex, "film_mat_ex"),
        mat_dir: ctx.create_f32_buffer_init(&flat_mat_dir, "film_mat_dir"),
        scan: ctx.create_f32_buffer_init(&scan, "film_scan_consts"),
        num_layers: num_layers as u32,
        num_emul: num_emul as u32,
        scan_scale,
        dmax: dmax_v,
        sites_per_cell,
        philox_key_lo,
        philox_key_hi,
        irrad_kernels,
        irrad_weights,
        halation_kernels,
        halation_weights,
        eff_grain_kernel,
        adj_ex_kernel,
        adj_dir_kernel,
        scanner_mtf_kernel,
        invert_mode,
        inv_dmin,
        exponent,
        gain,
        inv_gamma,
        eps: 1e-6,
        chroma_decode,
    }
}

// ─── Dispatch helpers ───────────────────────────────────────────────────────

fn workgroups(n: usize) -> u32 {
    ((n as u32) + 255) / 256
}

#[allow(clippy::too_many_arguments)]
fn blur_plane(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    width: usize,
    height: usize,
    src: &Buffer,
    src_off: u32,
    dst: &Buffer,
    dst_off: u32,
    btmp: &Buffer,
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

    if radius > BLUR_TILED_MAX_RADIUS {
        let wg = workgroups(n);
        ctx.encode_compute_passes(
            encoder,
            &[
                ComputePassDesc {
                    label: "film_blur_h",
                    wgsl_source: shaders::BLUR_H,
                    storage_buffers: &h_bufs,
                    uniform_bytes: ub_h,
                    workgroups_x: wg,
                    workgroups_y: 0,
                },
                ComputePassDesc {
                    label: "film_blur_v",
                    wgsl_source: shaders::BLUR_V,
                    storage_buffers: &v_bufs,
                    uniform_bytes: ub_v,
                    workgroups_x: wg,
                    workgroups_y: 0,
                },
            ],
        )
    } else {
        ctx.encode_compute_passes(
            encoder,
            &[
                ComputePassDesc {
                    label: "film_blur_h_tiled",
                    wgsl_source: shaders::BLUR_H_TILED,
                    storage_buffers: &h_bufs,
                    uniform_bytes: ub_h,
                    workgroups_x: ((width as u32) + 255) / 256,
                    workgroups_y: height as u32,
                },
                ComputePassDesc {
                    label: "film_blur_v_tiled",
                    wgsl_source: shaders::BLUR_V_TILED,
                    storage_buffers: &v_bufs,
                    uniform_bytes: ub_v,
                    workgroups_x: width as u32,
                    workgroups_y: ((height as u32) + 255) / 256,
                },
            ],
        )
    }
}

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
    let h_bufs = [arena, kernel];
    let v_bufs = [arena, kernel];

    if radius > BLUR_TILED_MAX_RADIUS {
        let wg = workgroups(n);
        ctx.encode_compute_passes(
            encoder,
            &[
                ComputePassDesc {
                    label: "film_blur_h_arena",
                    wgsl_source: shaders::BLUR_H_ARENA,
                    storage_buffers: &h_bufs,
                    uniform_bytes: ub_h,
                    workgroups_x: wg,
                    workgroups_y: 0,
                },
                ComputePassDesc {
                    label: "film_blur_v_arena",
                    wgsl_source: shaders::BLUR_V_ARENA,
                    storage_buffers: &v_bufs,
                    uniform_bytes: ub_v,
                    workgroups_x: wg,
                    workgroups_y: 0,
                },
            ],
        )
    } else {
        ctx.encode_compute_passes(
            encoder,
            &[
                ComputePassDesc {
                    label: "film_blur_h_tiled_arena",
                    wgsl_source: shaders::BLUR_H_TILED_ARENA,
                    storage_buffers: &h_bufs,
                    uniform_bytes: ub_h,
                    workgroups_x: ((width as u32) + 255) / 256,
                    workgroups_y: height as u32,
                },
                ComputePassDesc {
                    label: "film_blur_v_tiled_arena",
                    wgsl_source: shaders::BLUR_V_TILED_ARENA,
                    storage_buffers: &v_bufs,
                    uniform_bytes: ub_v,
                    workgroups_x: width as u32,
                    workgroups_y: ((height as u32) + 255) / 256,
                },
            ],
        )
    }
}

/// Stable public entry point for the GPU film simulation.
/// Uses fast fullframe execution by default on native platforms.
/// Respects `PICHROMATIC_GPU_MODE=roi` environment variable for bounded ROI execution.
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
    if std::env::var("PICHROMATIC_GPU_MODE").as_deref() != Ok("roi") {
        return process_gpu_full_frame(ctx, gpu_buf, meta, params).await;
    }

    process_gpu_roi(ctx, gpu_buf, meta, params, 1024).await
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

    let mut encoder = ctx.create_command_encoder("film_fullframe_main_sequence");
    let mut keep = Vec::new();

    // ── Stage 1: expose -> forward planes + upward bounce planes (in work) ──
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
        keep.push(ctx.encode_compute_shader_multi(
            &mut encoder,
            "film_expose",
            shaders::EXPOSE,
            &[&gpu_buf.buffer, planes, work, &consts.expose],
            bytemuck::bytes_of(&u),
            workgroups(n),
        ));
    }

    // ── Stage 2: Spatial Exposure Effects (Irradiation & Multi-bounce Halation) ──
    // 1. In-emulsion Irradiation Response
    if let Some(((ref core_k, core_r), (ref t1_k, t1_r), (ref t2_k, t2_r))) = consts.irrad_kernels {
        for plane_e in 0..e {
            let plane_off = (plane_e * n) as u32;
            keep.extend(blur_plane(ctx, &mut encoder, width, height, planes, plane_off, bout, 0, btmp, core_k.as_ref(), core_r));
            keep.extend(blur_plane(ctx, &mut encoder, width, height, planes, plane_off, noise, 0, btmp, t1_k.as_ref(), t1_r));
            keep.extend(blur_plane(ctx, &mut encoder, width, height, planes, plane_off, mask, 0, btmp, t2_k.as_ref(), t2_r));

            let u = IrradiationMixU {
                n: n as u32,
                plane_off,
                weight: consts.irrad_weights[plane_e],
                _p0: 0,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_irradiation_mix",
                shaders::IRRADIATION_MIX,
                &[planes, bout, noise, mask],
                bytemuck::bytes_of(&u),
                workgroups(n),
            ));
        }
    }

    // 2. Multi-bounce backing-reflection Halation
    if !consts.halation_kernels.is_empty() {
        for plane_e in 0..e {
            let bounce_off = (plane_e * n) as u32;
            for (k, (ref kbuf, radius)) in consts.halation_kernels.iter().enumerate() {
                keep.extend(blur_plane(
                    ctx,
                    &mut encoder,
                    width,
                    height,
                    work,
                    bounce_off,
                    bout,
                    0,
                    btmp,
                    kbuf.as_ref(),
                    *radius,
                ));
                let u = HalationAccumU {
                    n: n as u32,
                    w: consts.halation_weights[k],
                    init: if k == 0 { 1 } else { 0 },
                    _p0: 0,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_halation_accum",
                    shaders::HALATION_ACCUM,
                    &[noise, bout],
                    bytemuck::bytes_of(&u),
                    workgroups(n),
                ));
            }
            let u = HalationAddEmulU {
                n: n as u32,
                plane_off: (plane_e * n) as u32,
                _p0: 0,
                _p1: 0,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_halation_add_emul",
                shaders::HALATION_ADD_EMUL,
                &[planes, noise],
                bytemuck::bytes_of(&u),
                workgroups(n),
            ));
        }
    }

    // ── Stage 3: capture LUT (absorbed fluence -> developable fraction) ──
    {
        let u = CountU {
            n: n as u32,
            num_emul: consts.num_emul,
            _p0: 0,
            _p1: 0,
        };
        keep.push(ctx.encode_compute_shader_multi(
            &mut encoder,
            "film_lut",
            shaders::LUT,
            &[planes, &consts.lut],
            bytemuck::bytes_of(&u),
            workgroups(n),
        ));
    }

    // ── Stage 4: reduce (fraction -> image/mask dye density) ──
    {
        let u = CountU {
            n: n as u32,
            num_emul: consts.num_emul,
            _p0: 0,
            _p1: 0,
        };
        keep.push(ctx.encode_compute_shader_multi(
            &mut encoder,
            "film_reduce",
            shaders::REDUCE,
            &[planes, dye, mask, &consts.reduce],
            bytemuck::bytes_of(&u),
            workgroups(n),
        ));
    }

    // ── Stage 5: particle overwrite -> effective dye-cloud convolution -> scale d_max ──
    if let Some((ref eff_kbuf, eff_radius)) = consts.eff_grain_kernel {
        for plane_e in 0..e {
            let sites = consts.sites_per_cell[plane_e];
            let dmax = consts.dmax[plane_e];
            if sites <= 0.0 || dmax <= 0.0 {
                continue;
            }
            let in_off = (plane_e * n) as u32;
            let out_off = (plane_e * n) as u32;

            let pfu = ParticleFieldU {
                n: n as u32,
                width: width as u32,
                height: height as u32,
                in_off,
                out_off,
                d_max: dmax,
                sites_per_cell: sites,
                key_lo: consts.philox_key_lo[plane_e],
                key_hi: consts.philox_key_hi[plane_e],
                sqrt_sites: sites.sqrt(),
                knuth_threshold: (-(sites as f64)).exp() as f32,
                _p0: 0,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_particle_field",
                shaders::PARTICLE_FIELD,
                &[dye, planes],
                bytemuck::bytes_of(&pfu),
                workgroups(n),
            ));

            keep.extend(blur_plane(
                ctx,
                &mut encoder,
                width,
                height,
                planes,
                out_off,
                bout,
                0,
                btmp,
                eff_kbuf.as_ref(),
                eff_radius,
            ));

            let sdu = ScaleDmaxU {
                n: n as u32,
                in_off: 0,
                out_off: in_off,
                d_max: dmax,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_scale_dmax",
                shaders::SCALE_DMAX,
                &[bout, dye],
                bytemuck::bytes_of(&sdu),
                workgroups(n),
            ));
        }
    }

    // ── Stage 6: Two-Component Physical Adjacency ──
    let ex_active = consts.adj_ex_kernel.is_some();
    let dir_active = consts.adj_dir_kernel.is_some();

    if ex_active || dir_active {
        if let Some((ref kbuf, radius)) = consts.adj_ex_kernel {
            for plane_e in 0..e {
                let off = (plane_e * n) as u32;
                keep.extend(blur_plane(ctx, &mut encoder, width, height, dye, off, work, off, btmp, kbuf.as_ref(), radius));
            }
        }
        if let Some((ref kbuf, radius)) = consts.adj_dir_kernel {
            for plane_e in 0..e {
                let off = (plane_e * n) as u32;
                keep.extend(blur_plane(ctx, &mut encoder, width, height, dye, off, planes, off, btmp, kbuf.as_ref(), radius));
            }
        }

        let u = AdjacencyTwoCompU {
            n: n as u32,
            num_emul: consts.num_emul,
            ex_active: if ex_active { 1 } else { 0 },
            dir_active: if dir_active { 1 } else { 0 },
        };
        keep.push(ctx.encode_compute_shader_multi(
            &mut encoder,
            "film_adjacency_two_comp",
            shaders::ADJACENCY_TWO_COMPONENT,
            &[dye, work, planes, &consts.mat_ex, &consts.mat_dir],
            bytemuck::bytes_of(&u),
            workgroups(n),
        ));
    }

    // ── Stage 7: Densitometric Scan -> Scanner MTF -> Technical Invert ──
    // 1. Densitometric Integration -> Linear ACEScg R (planes[0]), G (work[0]), B (noise)
    {
        let u = ScanToAcescgU {
            n: n as u32,
            num_emul: consts.num_emul,
            scale: consts.scan_scale,
            _p0: 0,
        };
        keep.push(ctx.encode_compute_shader_multi(
            &mut encoder,
            "film_scan_to_acescg",
            shaders::SCAN_TO_ACESCG,
            &[dye, mask, planes, work, noise, &consts.scan],
            bytemuck::bytes_of(&u),
            workgroups(n),
        ));
    }

    // 2. Scanner Optical + Sensor Aperture MTF Gaussian Blur
    if let Some((ref kbuf, radius)) = consts.scanner_mtf_kernel {
        keep.extend(blur_plane(ctx, &mut encoder, width, height, planes, 0, planes, 0, btmp, kbuf.as_ref(), radius));
        keep.extend(blur_plane(ctx, &mut encoder, width, height, work, 0, work, 0, btmp, kbuf.as_ref(), radius));
        keep.extend(blur_plane(ctx, &mut encoder, width, height, noise, 0, noise, 0, btmp, kbuf.as_ref(), radius));
    }

    // 3. Technical Scanner Invert
    {
        let (chroma_r, chroma_g, chroma_b) = chroma_rows(consts.chroma_decode);
        let u = InvertU {
            n: n as u32,
            mode: consts.invert_mode,
            eps: consts.eps,
            mid_gray: crate::pixel::MIDDLE_GRAY,
            inv_gamma: consts.inv_gamma,
            inv_dmin: consts.inv_dmin,
            exponent: consts.exponent,
            gain: consts.gain,
            chroma_r,
            chroma_g,
            chroma_b,
        };
        keep.push(ctx.encode_compute_shader_multi(
            &mut encoder,
            "film_invert",
            shaders::INVERT,
            &[&gpu_buf.buffer, planes, work, noise],
            bytemuck::bytes_of(&u),
            workgroups(n),
        ));
    }

    ctx.queue.submit(Some(encoder.finish()));
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device.poll(wgpu::Maintain::Poll);
    drop(keep);

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

    let blur_tmp_off = scratch.blur_tmp_offset()?;
    let blur_out_off = scratch.blur_out_offset()?;

    for plan in &plans {
        let mut encoder = ctx.create_command_encoder("film_roi_main_sequence_tile");
        let mut keep = Vec::new();
        let root_n = plan.root.width * plan.root.height;
        let core_n = plan.core.width * plan.core.height;
        let root_off_x = (plan.core.x - plan.root.x) as u32;
        let root_off_y = (plan.core.y - plan.root.y) as u32;

        let latent_base = scratch.latent_workspace_offset(0)?;
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

        // Stage 1: EXPOSE_ROI -> forward absorbed planes at latent_base, upward bounce planes at mask_base
        {
            let u = ExposeRoiU {
                root_x: plan.root.x,
                root_y: plan.root.y,
                root_w: plan.root.width,
                root_h: plan.root.height,
                img_w,
                img_h,
                root_n,
                planes_base: latent_base,
                bounce_base: mask_base,
                num_layers: consts.num_layers,
                num_emul: consts.num_emul,
                _p0: 0,
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

        // Stage 2: Spatial Exposure Effects (Irradiation + Halation)
        // 1. Irradiation Response
        if let Some(((ref core_k, core_r), (ref t1_k, t1_r), (ref t2_k, t2_r))) = consts.irrad_kernels {
            for e in 0..num_emul {
                let plane_off = emul_off(latent_base, e)?;
                let core_off = emul_off(dye_base, 0)?;
                let tail1_off = emul_off(dye_base, 1.min(num_emul - 1))?;
                let tail2_off = blur_out_off;

                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    plane_off,
                    core_off,
                    blur_tmp_off,
                    core_k,
                    core_r,
                ));
                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    plane_off,
                    tail1_off,
                    blur_tmp_off,
                    t1_k,
                    t1_r,
                ));
                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    plane_off,
                    tail2_off,
                    blur_tmp_off,
                    t2_k,
                    t2_r,
                ));

                let u = IrradiationMixRoiU {
                    n: root_n,
                    plane_off,
                    core_off,
                    tail1_off,
                    tail2_off,
                    weight: consts.irrad_weights[e],
                    _p0: 0,
                    _p1: 0,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_irradiation_mix_roi",
                    shaders::IRRADIATION_MIX_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&u),
                    workgroups(root_n as usize),
                ));
            }
        }

        // 2. Multi-bounce Backing-Reflection Halation
        if !consts.halation_kernels.is_empty() {
            let acc_off = emul_off(dye_base, 0)?;
            for e in 0..num_emul {
                let src_off = emul_off(mask_base, e)?;
                let plane_off = emul_off(latent_base, e)?;

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

                let hal_u = HalationAddEmulRoiU {
                    n: root_n,
                    plane_off,
                    acc_off,
                    _p0: 0,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_halation_add_emul_roi",
                    shaders::HALATION_ADD_EMUL_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&hal_u),
                    workgroups(root_n as usize),
                ));
            }
        }

        // Stage 3 & 4 Fused: LUT_REDUCE_ROI
        {
            let u = ReduceRoiU {
                n: root_n,
                num_emul: consts.num_emul,
                plane_base: latent_base,
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

        // Stage 5: Particle Grain Overwrite
        if let Some((ref eff_kbuf, eff_radius)) = consts.eff_grain_kernel {
            for e in 0..num_emul {
                let sites = consts.sites_per_cell[e];
                let dmax = consts.dmax[e];
                if sites <= 0.0 || dmax <= 0.0 {
                    continue;
                }
                let dye_off = emul_off(dye_base, e)?;
                let scratch_off = emul_off(latent_base, 0)?;

                let pfu = ParticleFieldRoiU {
                    root_x: plan.root.x,
                    root_y: plan.root.y,
                    root_w: plan.root.width,
                    root_h: plan.root.height,
                    root_n,
                    img_w,
                    img_h,
                    in_off: dye_off,
                    out_off: scratch_off,
                    d_max: dmax,
                    sites_per_cell: sites,
                    key_lo: consts.philox_key_lo[e],
                    key_hi: consts.philox_key_hi[e],
                    sqrt_sites: sites.sqrt(),
                    knuth_threshold: (-(sites as f64)).exp() as f32,
                    _p0: 0,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_particle_field_roi",
                    shaders::PARTICLE_FIELD_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&pfu),
                    workgroups(root_n as usize),
                ));

                keep.extend(blur_plane_in_arena(
                    ctx,
                    &mut encoder,
                    plan.root.width as usize,
                    plan.root.height as usize,
                    &scratch.arena,
                    scratch_off,
                    blur_out_off,
                    blur_tmp_off,
                    eff_kbuf,
                    eff_radius,
                ));

                let sdu = ScaleDmaxRoiU {
                    n: root_n,
                    in_off: blur_out_off,
                    out_off: dye_off,
                    d_max: dmax,
                };
                keep.push(ctx.encode_compute_shader_multi(
                    &mut encoder,
                    "film_scale_dmax_roi",
                    shaders::SCALE_DMAX_ROI,
                    &[&scratch.arena],
                    bytemuck::bytes_of(&sdu),
                    workgroups(root_n as usize),
                ));
            }
        }

        // Stage 6: Two-Component Physical Adjacency
        let ex_active = consts.adj_ex_kernel.is_some();
        let dir_active = consts.adj_dir_kernel.is_some();

        if ex_active || dir_active {
            let diff_ex_base = latent_base;
            let diff_dir_base = latent_base;

            if let Some((ref kbuf, radius)) = consts.adj_ex_kernel {
                for e in 0..num_emul {
                    let dye_off = emul_off(dye_base, e)?;
                    let dst_off = emul_off(diff_ex_base, e)?;
                    keep.extend(blur_plane_in_arena(
                        ctx,
                        &mut encoder,
                        plan.root.width as usize,
                        plan.root.height as usize,
                        &scratch.arena,
                        dye_off,
                        dst_off,
                        blur_tmp_off,
                        kbuf,
                        radius,
                    ));
                }
            }

            let u = AdjacencyTwoCompRoiU {
                n: root_n,
                num_emul: consts.num_emul,
                dye_base,
                diff_ex_base,
                diff_dir_base,
                ex_active: if ex_active { 1 } else { 0 },
                dir_active: if dir_active { 1 } else { 0 },
                _p0: 0,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_adjacency_two_comp_roi",
                shaders::ADJACENCY_TWO_COMPONENT_ROI,
                &[&scratch.arena, &consts.mat_ex, &consts.mat_dir],
                bytemuck::bytes_of(&u),
                workgroups(root_n as usize),
            ));
        }

        // Stage 7: Densitometric Scan -> Scanner MTF -> Technical Invert
        let r_base = latent_base;
        let g_base = if num_emul >= 2 { emul_off(latent_base, 1)? } else { blur_tmp_off };
        let b_base = if num_emul >= 3 { emul_off(latent_base, 2)? } else { blur_out_off };

        // 1. Densitometric Integration
        {
            let u = ScanToAcescgRoiU {
                root_n,
                num_emul: consts.num_emul,
                dye_base,
                mask_base,
                r_base,
                g_base,
                b_base,
                scale: consts.scan_scale,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_scan_to_acescg_roi",
                shaders::SCAN_TO_ACESCG_ROI,
                &[&scratch.arena, &consts.scan],
                bytemuck::bytes_of(&u),
                workgroups(root_n as usize),
            ));
        }

        // 2. Scanner Optical + Sensor Aperture MTF Gaussian Blur
        if let Some((ref kbuf, radius)) = consts.scanner_mtf_kernel {
            keep.extend(blur_plane_in_arena(
                ctx,
                &mut encoder,
                plan.root.width as usize,
                plan.root.height as usize,
                &scratch.arena,
                r_base,
                r_base,
                blur_tmp_off,
                kbuf,
                radius,
            ));

            keep.extend(blur_plane_in_arena(
                ctx,
                &mut encoder,
                plan.root.width as usize,
                plan.root.height as usize,
                &scratch.arena,
                g_base,
                g_base,
                blur_tmp_off,
                kbuf,
                radius,
            ));

            keep.extend(blur_plane_in_arena(
                ctx,
                &mut encoder,
                plan.root.width as usize,
                plan.root.height as usize,
                &scratch.arena,
                b_base,
                b_base,
                blur_tmp_off,
                kbuf,
                radius,
            ));
        }

        // 3. Technical Scanner Invert -> writes directly into scratch.output
        {
            let (chroma_r, chroma_g, chroma_b) = chroma_rows(consts.chroma_decode);
            let u = InvertRoiU {
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
                r_base,
                g_base,
                b_base,
                mode: consts.invert_mode,
                eps: consts.eps,
                mid_gray: crate::pixel::MIDDLE_GRAY,
                inv_gamma: consts.inv_gamma,
                inv_dmin: consts.inv_dmin,
                exponent: consts.exponent,
                gain: consts.gain,
                chroma_r,
                chroma_g,
                chroma_b,
            };
            keep.push(ctx.encode_compute_shader_multi(
                &mut encoder,
                "film_invert_roi",
                shaders::INVERT_ROI,
                &[&scratch.output, &scratch.arena],
                bytemuck::bytes_of(&u),
                workgroups(core_n as usize),
            ));
        }

        ctx.queue.submit(Some(encoder.finish()));
        #[cfg(not(target_arch = "wasm32"))]
        ctx.device.poll(wgpu::Maintain::Poll);
        drop(keep);
    }

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

        // Particle-grain / micro-mix / scan uniforms.
        assert_eq!(std::mem::size_of::<ParticleFieldU>(), 48);
        assert_eq!(std::mem::size_of::<ParticleFieldRoiU>(), 64);
        assert_eq!(std::mem::size_of::<MicroMixU>(), 32);
        assert_eq!(std::mem::size_of::<ScanRoiU>(), 64);
        assert_eq!(std::mem::size_of::<InvertU>(), 128);
        assert_eq!(std::mem::size_of::<InvertRoiU>(), 176);
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
    let floats = 4 * e * n + 3 * n + GRAIN_VAR_PARTIALS_PER * e;
    floats * 4
}

pub use workspace::{film_roi_memory_breakdown, FilmRoiMemoryBreakdown};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn film_workspace_requested_bytes_exact() {
        assert_eq!(film_workspace_requested_bytes(1, 1, 1), 8220);
        assert_eq!(film_workspace_requested_bytes(1, 1, 0), 8220);
        assert_eq!(film_workspace_requested_bytes(4, 4, 3), 25536);
        assert_eq!(film_workspace_requested_bytes(1024, 1024, 6), 113_295_360);
    }

    use crate::film::stock::StockId;
    use crate::film::types::FilmFormat;

    #[test]
    fn film_workspace_requested_bytes_zero_area_zero() {
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

        let pattern_checker: Vec<f32> = (0..n)
            .flat_map(|i| {
                let x = (i % width) as u32;
                let y = (i / width) as u32;
                let is_check = (x / 8 + y / 8) % 2 == 0;
                let mut r = if is_check { 0.8 } else { 0.05 };
                let mut g = if is_check { 0.6 } else { 0.04 };
                let mut b = if is_check { 0.4 } else { 0.03 };

                if x == 0 || y == 0 || x == (width as u32 - 1) || y == (height as u32 - 1) {
                    r += 2.0;
                    g += 2.0;
                    b += 2.0;
                }
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
    fn cpu_vs_gpu_parity_test() {
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

            let mut cpu_image = crate::pixel::Image {
                metadata: meta.clone(),
                rgb_data: vec![[0.18, 0.18, 0.18]; n],
                raw_data: std::sync::Arc::from([]),
            };
            crate::film::process(&mut cpu_image, &params).unwrap();

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
                    let tol =
                        CPU_GPU_ABS_TOLERANCE * cpu_pixel[ch].abs().max(gpu_value.abs()).max(1.0);
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

    /// CPU-vs-GPU parity through PositiveLinear with strongly chromatic content.
    /// The uniform-gray parity test cannot catch a chroma-decode bug because a
    /// decode matrix maps mid-gray to itself; saturated patches exercise every
    /// cross-channel term of `chroma_decode`.
    #[test]
    fn cpu_vs_gpu_parity_positive_linear_chromatic() {
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

        // AP1 (ACEScg) primaries' luminance row, used only to normalize each
        // patch's brightness to mid-gray so all blocks share exposure while
        // keeping full chroma direction.
        const AP1_LUMA: [f32; 3] = [0.2722, 0.6741, 0.0537];
        let palette_srgb: [[f32; 3]; 8] = [
            [1.0, 0.0, 0.0], // red
            [0.0, 1.0, 0.0], // green
            [0.0, 0.0, 1.0], // blue
            [1.0, 1.0, 0.0], // yellow
            [0.0, 1.0, 1.0], // cyan
            [1.0, 0.0, 1.0], // magenta
            [1.0, 1.0, 1.0], // white
            [0.5, 0.5, 0.5], // gray
        ];
        let palette_acescg: Vec<[f32; 3]> = palette_srgb
            .iter()
            .map(|c| {
                let lin = ColorSpaceTag::Srgb.convert(ColorSpaceTag::AcesCg, *c);
                let y = AP1_LUMA[0] * lin[0] + AP1_LUMA[1] * lin[1] + AP1_LUMA[2] * lin[2];
                let s = if y > 1e-6 {
                    crate::pixel::MIDDLE_GRAY / y
                } else {
                    1.0
                };
                [lin[0] * s, lin[1] * s, lin[2] * s]
            })
            .collect();

        // 8x8 pixel blocks cycling diagonally through the palette.
        let cpu_pattern: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let bx = (i % width) / 8;
                let by = (i / width) / 8;
                palette_acescg[((bx + by) % 8) as usize]
            })
            .collect();
        let gpu_pattern: Vec<[f32; 4]> =
            cpu_pattern.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect();

        let params = FilmParams {
            stock: StockId::Portra400,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 42,
            output: FilmOutput::PositiveLinear,
            enable_halation: true,
            compensate_box_speed: true,
        };

        let mut cpu_image = crate::pixel::Image {
            metadata: meta.clone(),
            rgb_data: cpu_pattern,
            raw_data: std::sync::Arc::from([]),
        };
        crate::film::process(&mut cpu_image, &params).unwrap();

        let gpu_buf = ctx.create_output_buffer(width, height);
        ctx.queue
            .write_buffer(&gpu_buf.buffer, 0, bytemuck::cast_slice(&gpu_pattern));
        pollster::block_on(process_gpu_full_frame(&ctx, &gpu_buf, &meta, &params)).unwrap();
        let gpu_floats = ctx.download_f32(&gpu_buf.buffer, width * height * 4);

        let mut max_diff = 0.0f32;
        let mut max_at = (0usize, 0usize);
        for (px, cpu_pixel) in cpu_image.rgb_data.iter().enumerate() {
            let x = px % width;
            let y = px / width;
            for ch in 0..3 {
                let gpu_value = gpu_floats[px * 4 + ch];
                let diff = (cpu_pixel[ch] - gpu_value).abs();
                let tol =
                    CPU_GPU_ABS_TOLERANCE * cpu_pixel[ch].abs().max(gpu_value.abs()).max(1.0);
                if diff > max_diff {
                    max_diff = diff;
                    max_at = (px, ch);
                }
                assert!(
                    diff.is_finite() && diff <= tol,
                    "CPU/GPU RGB mismatch (chromatic PositiveLinear) at ({x}, {y}) ch={ch}: CPU={} ({:#010x}), GPU={} ({:#010x}), diff={}, tolerance={}",
                    cpu_pixel[ch],
                    cpu_pixel[ch].to_bits(),
                    gpu_value,
                    gpu_value.to_bits(),
                    diff,
                    tol
                );
            }
        }
        eprintln!(
            "chromatic parity max_diff={max_diff} at px=({},{}) ch={}",
            max_at.0 % width,
            max_at.0 / width,
            max_at.1
        );
    }

    /// The baked chroma decode shipped to the GPU must be exactly the one the
    /// CPU calibration computes for the same stock/format/metadata.
    #[test]
    fn baked_chroma_decode_matches_scanner_calibration() {
        let ctx = match pollster::block_on(GpuContext::try_new()) {
            Ok(c) => c,
            Err(_) => return,
        };

        let width = 64usize;
        let meta = crate::image::ImageMetadata {
            width,
            height: width,
            color_space: Some(ColorSpaceTag::AcesCg),
            ..Default::default()
        };
        let params = FilmParams {
            stock: StockId::Portra400,
            film_format: FilmFormat::Film35mm,
            render_width_mm: None,
            seed: 42,
            output: FilmOutput::PositiveLinear,
            enable_halation: true,
            compensate_box_speed: true,
        };
        let stock = params.stock.load().unwrap();
        let consts = bake_consts(&ctx, &stock, &params, &meta, width);

        let pitch = params.film_format.pixel_pitch_um(width);
        let shutter = meta.shutter_seconds.unwrap_or(1.0 / stock.box_iso.0);
        let cal = scanner_calibration_acescg(&stock, pitch, shutter).unwrap();

        eprintln!("baked chroma_decode = {:?}", consts.chroma_decode);
        assert_eq!(consts.chroma_decode, cal.chroma_decode);
    }
}
