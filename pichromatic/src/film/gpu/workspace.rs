//! Persistent film GPU scratch + stock const buffers (Approach A).
//!
//! Large working buffers are allocated once per `(width, height, num_emul)` and
//! reused across runs. **Two size slots are retained** (typically preview + final)
//! so toggling resolutions does not thrash multi‑GB WebGPU allocations (browsers
//! often do not return that memory to the OS, which previously ballooned RSS).
//!
//! Resources are leased (taken) for the duration of a film run so the workspace
//! mutex is not held across GPU awaits. [`FilmGpuLease`] restores them on drop.

use std::sync::Mutex;

use wgpu::Buffer;

use crate::film::stock::{FilmStock, StockId};
use crate::film::types::FilmFormat;
use crate::film::{FilmOutput, FilmParams};
use crate::gpu::GpuContext;

use super::{bake_consts, StockConsts, GRAIN_VAR_PARTIALS_PER};

/// Const buffers cache limit (stock consts).
const MAX_CONSTS_SLOTS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ConstsKey {
    stock: StockId,
    film_format: FilmFormat,
    seed: u64,
    output: FilmOutput,
    width: usize,
}

impl ConstsKey {
    fn from_params(params: &FilmParams, width: usize) -> Self {
        Self {
            stock: params.stock,
            film_format: params.film_format,
            seed: params.seed,
            output: params.output,
            width,
        }
    }
}

pub struct FilmScratch {
    pub width: usize,
    pub height: usize,
    pub num_emul: usize,
    pub planes: Buffer,
    pub dye: Buffer,
    pub mask: Buffer,
    pub work: Buffer,
    pub btmp: Buffer,
    pub bout: Buffer,
    pub noise: Buffer,
    pub var_partial: Buffer,
}

impl FilmScratch {
    fn allocate(ctx: &GpuContext, width: usize, height: usize, num_emul: usize) -> Self {
        let n = width * height;
        let e = num_emul.max(1);
        Self {
            width,
            height,
            num_emul: e,
            planes: ctx.create_f32_buffer(e * n, "film_planes"),
            dye: ctx.create_f32_buffer(e * n, "film_dye"),
            mask: ctx.create_f32_buffer(e * n, "film_mask"),
            work: ctx.create_f32_buffer(e * n, "film_work"),
            btmp: ctx.create_f32_buffer(n, "film_btmp"),
            bout: ctx.create_f32_buffer(n, "film_bout"),
            noise: ctx.create_f32_buffer(n, "film_noise"),
            var_partial: ctx.create_f32_buffer(GRAIN_VAR_PARTIALS_PER * e, "film_var_partial"),
        }
    }
}

/// Layout calculation for bounded ROI arena & workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RoiLayout {
    pub(super) roi_width: usize,
    pub(super) roi_height: usize,
    pub(super) img_width: usize,
    pub(super) img_height: usize,
    pub(super) num_emul: usize,
    pub(super) roi_n: usize,
    pub(super) img_n: usize,
    pub(super) total_arena_floats: usize,
}

impl RoiLayout {
    pub(super) fn new(
        roi_width: usize,
        roi_height: usize,
        img_width: usize,
        img_height: usize,
        num_emul: usize,
    ) -> Result<Self, crate::film::FilmError> {
        let e = num_emul.max(1);
        let roi_n = roi_width.checked_mul(roi_height).ok_or(crate::film::FilmError::InvalidDimensions)?;
        let img_n = img_width.checked_mul(img_height).ok_or(crate::film::FilmError::InvalidDimensions)?;

        let num_planes = e
            .checked_mul(3)
            .and_then(|three_e| three_e.checked_add(2))
            .ok_or(crate::film::FilmError::InvalidDimensions)?;
        let total_arena_floats = num_planes.checked_mul(roi_n).ok_or(crate::film::FilmError::InvalidDimensions)?;

        Ok(Self {
            roi_width,
            roi_height,
            img_width,
            img_height,
            num_emul: e,
            roi_n,
            img_n,
            total_arena_floats,
        })
    }

    /// Offset (in f32 elements) of dye plane `e` in the arena.
    #[inline]
    pub(super) fn dye_offset(&self, e: usize) -> Result<u32, crate::film::FilmError> {
        if e >= self.num_emul {
            return Err(crate::film::FilmError::InvalidDimensions);
        }
        let off = e.checked_mul(self.roi_n).ok_or(crate::film::FilmError::InvalidDimensions)?;
        u32::try_from(off).map_err(|_| crate::film::FilmError::InvalidDimensions)
    }

    /// Offset (in f32 elements) of mask plane `e` in the arena.
    #[inline]
    pub(super) fn mask_offset(&self, e: usize) -> Result<u32, crate::film::FilmError> {
        if e >= self.num_emul {
            return Err(crate::film::FilmError::InvalidDimensions);
        }
        let off = self.num_emul
            .checked_add(e)
            .and_then(|idx| idx.checked_mul(self.roi_n))
            .ok_or(crate::film::FilmError::InvalidDimensions)?;
        u32::try_from(off).map_err(|_| crate::film::FilmError::InvalidDimensions)
    }

    /// Offset (in f32 elements) of latent workspace plane `e` in the arena.
    /// Aliased lifetime: latent planes -> REDUCE -> inhibition / grain workspace.
    #[inline]
    pub(super) fn latent_workspace_offset(&self, e: usize) -> Result<u32, crate::film::FilmError> {
        if e >= self.num_emul {
            return Err(crate::film::FilmError::InvalidDimensions);
        }
        let off = self.num_emul
            .checked_mul(2)
            .and_then(|two_e| two_e.checked_add(e))
            .and_then(|idx| idx.checked_mul(self.roi_n))
            .ok_or(crate::film::FilmError::InvalidDimensions)?;
        u32::try_from(off).map_err(|_| crate::film::FilmError::InvalidDimensions)
    }

    /// Offset (in f32 elements) of `blur_tmp` plane in the arena.
    #[inline]
    pub(super) fn blur_tmp_offset(&self) -> Result<u32, crate::film::FilmError> {
        let off = self.num_emul
            .checked_mul(3)
            .and_then(|three_e| three_e.checked_mul(self.roi_n))
            .ok_or(crate::film::FilmError::InvalidDimensions)?;
        u32::try_from(off).map_err(|_| crate::film::FilmError::InvalidDimensions)
    }

    /// Offset (in f32 elements) of `blur_out` plane in the arena.
    #[inline]
    pub(super) fn blur_out_offset(&self) -> Result<u32, crate::film::FilmError> {
        let off = self.num_emul
            .checked_mul(3)
            .and_then(|three_e| three_e.checked_add(1))
            .and_then(|idx| idx.checked_mul(self.roi_n))
            .ok_or(crate::film::FilmError::InvalidDimensions)?;
        u32::try_from(off).map_err(|_| crate::film::FilmError::InvalidDimensions)
    }
}

/// Detailed memory breakdown of ROI Film execution allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilmRoiMemoryBreakdown {
    pub arena_bytes: usize,
    pub grain_spill_bytes: usize,
    pub output_bytes: usize,
    pub var_partial_bytes: usize,
    pub internal_scratch_bytes: usize,
    pub total_film_owned_bytes: usize,
}

/// Calculate exact byte breakdown for bounded ROI execution. Returns `Result` with overflow protection.
pub fn film_roi_memory_breakdown(
    roi_width: usize,
    roi_height: usize,
    img_width: usize,
    img_height: usize,
    num_emul: usize,
) -> Result<FilmRoiMemoryBreakdown, crate::film::FilmError> {
    if roi_width == 0 || roi_height == 0 || img_width == 0 || img_height == 0 {
        return Ok(FilmRoiMemoryBreakdown {
            arena_bytes: 0,
            grain_spill_bytes: 0,
            output_bytes: 0,
            var_partial_bytes: 0,
            internal_scratch_bytes: 0,
            total_film_owned_bytes: 0,
        });
    }

    let layout = RoiLayout::new(roi_width, roi_height, img_width, img_height, num_emul)?;

    let arena_bytes = layout
        .total_arena_floats
        .checked_mul(4)
        .ok_or(crate::film::FilmError::InvalidDimensions)?;
    let grain_spill_bytes = layout
        .img_n
        .checked_mul(4)
        .ok_or(crate::film::FilmError::InvalidDimensions)?;
    let output_bytes = layout
        .img_n
        .checked_mul(16)
        .ok_or(crate::film::FilmError::InvalidDimensions)?;

    let var_partial_floats = GRAIN_VAR_PARTIALS_PER
        .checked_mul(layout.num_emul)
        .ok_or(crate::film::FilmError::InvalidDimensions)?;
    let var_partial_bytes = var_partial_floats
        .checked_mul(4)
        .ok_or(crate::film::FilmError::InvalidDimensions)?;

    let internal_scratch_bytes = arena_bytes
        .checked_add(grain_spill_bytes)
        .and_then(|b| b.checked_add(var_partial_bytes))
        .ok_or(crate::film::FilmError::InvalidDimensions)?;

    let total_film_owned_bytes = internal_scratch_bytes
        .checked_add(output_bytes)
        .ok_or(crate::film::FilmError::InvalidDimensions)?;

    Ok(FilmRoiMemoryBreakdown {
        arena_bytes,
        grain_spill_bytes,
        output_bytes,
        var_partial_bytes,
        internal_scratch_bytes,
        total_film_owned_bytes,
    })
}

/// Bounded ROI Scratch Arena & execution buffers (Approach B for Epic 3/4).
pub(super) struct RoiScratch {
    pub(super) layout: RoiLayout,
    /// One contiguous arena storing (3E + 2) ROI planes.
    pub(super) arena: Buffer,
    /// Single scalar spill for grain variance normalization prepass (full image size).
    pub(super) grain_spill: Buffer,
    /// Full RGBA32F output buffer (full image size).
    pub(super) output: Buffer,
    /// Variance partial reduction buffer.
    pub(super) var_partial: Buffer,
}

impl RoiScratch {
    pub(super) fn matches(
        &self,
        roi_width: usize,
        roi_height: usize,
        img_width: usize,
        img_height: usize,
        num_emul: usize,
    ) -> bool {
        let img_fits = (self.layout.img_width >= img_width && self.layout.img_height >= img_height)
            || (self.layout.img_width >= img_height && self.layout.img_height >= img_width);
        let roi_fits = (self.layout.roi_width >= roi_width && self.layout.roi_height >= roi_height)
            || (self.layout.roi_width >= roi_height && self.layout.roi_height >= roi_width);
        roi_fits && img_fits && self.layout.num_emul == num_emul
    }

    pub(super) fn allocate(
        ctx: &GpuContext,
        roi_width: usize,
        roi_height: usize,
        img_width: usize,
        img_height: usize,
        num_emul: usize,
    ) -> Result<Self, crate::film::FilmError> {
        let layout = RoiLayout::new(roi_width, roi_height, img_width, img_height, num_emul)?;

        let arena = ctx.create_f32_buffer(layout.total_arena_floats, "film_roi_arena");
        let grain_spill = ctx.create_f32_buffer(layout.img_n, "film_roi_grain_spill");
        let output = ctx.create_f32_buffer(
            layout
                .img_n
                .checked_mul(4)
                .ok_or(crate::film::FilmError::InvalidDimensions)?,
            "film_roi_output",
        );
        let var_partial = ctx.create_f32_buffer(
            GRAIN_VAR_PARTIALS_PER
                .checked_mul(layout.num_emul)
                .ok_or(crate::film::FilmError::InvalidDimensions)?,
            "film_roi_var_partial",
        );

        Ok(Self {
            layout,
            arena,
            grain_spill,
            output,
            var_partial,
        })
    }

    #[inline]
    pub(super) fn dye_offset(&self, e: usize) -> Result<u32, crate::film::FilmError> {
        self.layout.dye_offset(e)
    }

    #[inline]
    pub(super) fn mask_offset(&self, e: usize) -> Result<u32, crate::film::FilmError> {
        self.layout.mask_offset(e)
    }

    #[inline]
    pub(super) fn latent_workspace_offset(&self, e: usize) -> Result<u32, crate::film::FilmError> {
        self.layout.latent_workspace_offset(e)
    }

    #[inline]
    pub(super) fn blur_tmp_offset(&self) -> Result<u32, crate::film::FilmError> {
        self.layout.blur_tmp_offset()
    }

    #[inline]
    pub(super) fn blur_out_offset(&self) -> Result<u32, crate::film::FilmError> {
        self.layout.blur_out_offset()
    }
}

struct FilmGpuWorkspace {
    generation: u64,
    /// Single bounded ROI arena for Epic 3 memory limits. No preview/final caching.
    roi_scratch: Option<RoiScratch>,
    /// LRU const buffers keyed by stock fingerprint + width.
    consts: Vec<(ConstsKey, StockConsts)>,
}

impl FilmGpuWorkspace {
    const fn new() -> Self {
        Self {
            generation: 0,
            roi_scratch: None,
            consts: Vec::new(),
        }
    }

    fn check_generation(&mut self, ctx: &GpuContext) {
        if self.generation != ctx.generation {
            self.roi_scratch = None;
            self.consts.clear();
            self.generation = ctx.generation;
        }
    }

    fn take_scratch(
        &mut self,
        ctx: &GpuContext,
        width: usize,
        height: usize,
        num_emul: usize,
    ) -> FilmScratch {
        self.check_generation(ctx);
        let e = num_emul.max(1);
        FilmScratch::allocate(ctx, width, height, e)
    }

    fn take_roi_scratch(
        &mut self,
        ctx: &GpuContext,
        roi_width: usize,
        roi_height: usize,
        img_width: usize,
        img_height: usize,
        num_emul: usize,
    ) -> Result<RoiScratch, crate::film::FilmError> {
        self.check_generation(ctx);
        if let Some(s) = self.roi_scratch.take() {
            if s.matches(roi_width, roi_height, img_width, img_height, num_emul) {
                #[cfg(not(target_arch = "wasm32"))]
                ctx.poll_wait();
                return Ok(s);
            }
        }
        RoiScratch::allocate(ctx, roi_width, roi_height, img_width, img_height, num_emul)
    }

    fn restore_roi_scratch(&mut self, scratch: RoiScratch, generation: u64) {
        if self.generation == generation {
            self.roi_scratch = Some(scratch);
        }
    }

    fn take_consts(
        &mut self,
        ctx: &GpuContext,
        stock: &FilmStock,
        params: &FilmParams,
        meta: &crate::image::ImageMetadata,
        width: usize,
    ) -> (ConstsKey, StockConsts) {
        self.check_generation(ctx);
        let key = ConstsKey::from_params(params, width);
        if let Some(i) = self.consts.iter().position(|(k, _)| *k == key) {
            return self.consts.remove(i);
        }
        (key, bake_consts(ctx, stock, params, meta, width))
    }

    fn restore_consts(&mut self, key: ConstsKey, consts: StockConsts, generation: u64) {
        if self.generation != generation {
            return;
        }
        self.consts.retain(|(k, _)| *k != key);
        self.consts.push((key, consts));
        while self.consts.len() > MAX_CONSTS_SLOTS {
            self.consts.remove(0);
        }
    }
}

static WORKSPACE: SyncWorkspace = SyncWorkspace(Mutex::new(FilmGpuWorkspace::new()));

/// wgpu::Buffer is !Send on wasm; match GpuContext's unsafe Sync/Send.
struct SyncWorkspace(Mutex<FilmGpuWorkspace>);
unsafe impl Sync for SyncWorkspace {}
unsafe impl Send for SyncWorkspace {}

/// Leased film GPU resources. Restored to the global workspace on drop.
pub struct FilmGpuLease {
    scratch: Option<FilmScratch>,
    consts: Option<StockConsts>,
    consts_key: ConstsKey,
    generation: u64,
}

impl FilmGpuLease {
    pub fn scratch(&self) -> &FilmScratch {
        self.scratch.as_ref().expect("FilmGpuLease scratch")
    }

    pub fn consts(&self) -> &StockConsts {
        self.consts.as_ref().expect("FilmGpuLease consts")
    }
}

impl Drop for FilmGpuLease {
    fn drop(&mut self) {
        let scratch = self.scratch.take();
        let consts = self.consts.take();
        drop(scratch); // Full-frame scratch is dropped immediately (not retained in workspace).
        if let Some(consts) = consts {
            let mut ws = WORKSPACE.0.lock().unwrap();
            ws.restore_consts(self.consts_key, consts, self.generation);
        }
    }
}

/// Leased ROI film GPU resources. Restored to global workspace on drop.
pub(super) struct FilmRoiLease {
    roi_scratch: Option<RoiScratch>,
    consts: Option<StockConsts>,
    consts_key: ConstsKey,
    generation: u64,
}

impl FilmRoiLease {
    pub(super) fn roi_scratch(&self) -> &RoiScratch {
        self.roi_scratch.as_ref().expect("FilmRoiLease roi_scratch")
    }

    pub(super) fn consts(&self) -> &StockConsts {
        self.consts.as_ref().expect("FilmRoiLease consts")
    }
}

impl Drop for FilmRoiLease {
    fn drop(&mut self) {
        let roi_scratch = self.roi_scratch.take();
        let consts = self.consts.take();
        if roi_scratch.is_none() && consts.is_none() {
            return;
        }
        let mut ws = WORKSPACE.0.lock().unwrap();
        if let Some(roi_scratch) = roi_scratch {
            ws.restore_roi_scratch(roi_scratch, self.generation);
        }
        if let Some(consts) = consts {
            ws.restore_consts(self.consts_key, consts, self.generation);
        }
    }
}

/// Acquire persistent scratch + consts for a film run (exclusive lease).
pub fn acquire_film_resources(
    ctx: &GpuContext,
    stock: &FilmStock,
    params: &FilmParams,
    meta: &crate::image::ImageMetadata,
    width: usize,
    height: usize,
    num_emul: usize,
) -> FilmGpuLease {
    let mut ws = WORKSPACE.0.lock().unwrap();
    let scratch = ws.take_scratch(ctx, width, height, num_emul);
    let (consts_key, consts) = ws.take_consts(ctx, stock, params, meta, width);
    FilmGpuLease {
        scratch: Some(scratch),
        consts: Some(consts),
        consts_key,
        generation: ctx.generation,
    }
}

pub(super) fn acquire_film_roi_resources(
    ctx: &GpuContext,
    stock: &FilmStock,
    params: &FilmParams,
    meta: &crate::image::ImageMetadata,
    roi_width: usize,
    roi_height: usize,
    img_width: usize,
    img_height: usize,
    num_emul: usize,
) -> Result<FilmRoiLease, crate::film::FilmError> {
    let mut ws = WORKSPACE.0.lock().unwrap();
    let roi_scratch = ws.take_roi_scratch(ctx, roi_width, roi_height, img_width, img_height, num_emul)?;
    let (consts_key, consts) = ws.take_consts(ctx, stock, params, meta, img_width);
    Ok(FilmRoiLease {
        roi_scratch: Some(roi_scratch),
        consts: Some(consts),
        consts_key,
        generation: ctx.generation,
    })
}

/// Acquire persistent consts only (for ROI rendering which allocates scratch on demand).
#[allow(dead_code)]
pub fn acquire_film_consts(
    ctx: &GpuContext,
    stock: &FilmStock,
    params: &FilmParams,
    meta: &crate::image::ImageMetadata,
    width: usize,
) -> FilmGpuLease {
    let mut ws = WORKSPACE.0.lock().unwrap();
    let (consts_key, consts) = ws.take_consts(ctx, stock, params, meta, width);
    FilmGpuLease {
        scratch: None,
        consts: Some(consts),
        consts_key,
        generation: ctx.generation,
    }
}

#[cfg(test)]
mod roi_workspace_tests {
    use super::*;
    use crate::film::gpu::roi::{RectI, RoiPlan};
    use crate::film::stock::StockId;
    use crate::film::types::FilmFormat;

    #[test]
    fn roi_memory_breakdown_12mp_portra_exact_radii_within_caps() {
        // 12 MP (4032 x 3024), Portra 400 (6 emulsions), 1024x1024 core
        let stock = StockId::Portra400.load().unwrap();
        let core = RectI::new(0, 0, 1024, 1024);
        let plan = RoiPlan::build(core, &stock, FilmFormat::Film35mm, 4032);

        // Compute real plan.root dimensions (1024 + 2 * total_halo)
        let roi_w = plan.root.width as usize;
        let roi_h = plan.root.height as usize;
        // Portra 400 halo radius: local(2) + wide(41) + dir(6) + adj(3) = 52.
        // Wide covers the CPU multi-bounce halation's widest kernel (σ·√3).
        // 1024 + 104 = 1128.
        assert_eq!(roi_w, 1128);
        assert_eq!(roi_h, 1128);

        let breakdown = film_roi_memory_breakdown(roi_w, roi_h, 4032, 3024, 6).unwrap();

        // Exact byte verification for 1128x1128 root:
        // Arena: (3*6 + 2) * 1128 * 1128 * 4 = 20 * 1,272,384 * 4 = 101,790,720 bytes (~97.07 MiB)
        // Grain Spill: 4032 * 3024 * 4 = 12,192,768 * 4 = 48,771,072 bytes (~46.51 MiB)
        // Partial: 2048 * 6 * 4 = 49,152 bytes (~0.05 MiB)
        // Output: 4032 * 3024 * 16 = 195,084,288 bytes (~186.05 MiB)
        assert_eq!(breakdown.arena_bytes, 101_790_720);
        assert_eq!(breakdown.grain_spill_bytes, 48_771_072);
        assert_eq!(breakdown.var_partial_bytes, 49_152);
        assert_eq!(breakdown.output_bytes, 195_084_288);

        let internal_scratch_mb = breakdown.internal_scratch_bytes as f64 / (1024.0 * 1024.0);
        let total_owned_mb = breakdown.total_film_owned_bytes as f64 / (1024.0 * 1024.0);

        assert_eq!(breakdown.internal_scratch_bytes, 150_610_944); // ~143.63 MiB
        assert_eq!(breakdown.total_film_owned_bytes, 345_695_232); // ~329.68 MiB

        assert!(
            internal_scratch_mb <= 150.0,
            "Internal ROI scratch ({internal_scratch_mb:.2} MiB) exceeds 150 MiB ceiling"
        );
        assert!(
            total_owned_mb <= 350.0,
            "Total Film-owned target ({total_owned_mb:.2} MiB) exceeds 350 MiB ceiling"
        );
    }

    #[test]
    fn roi_memory_breakdown_zero_area_returns_zero() {
        let b = film_roi_memory_breakdown(0, 100, 4032, 3024, 6).unwrap();
        assert_eq!(b.total_film_owned_bytes, 0);
        let b2 = film_roi_memory_breakdown(100, 100, 0, 3024, 6).unwrap();
        assert_eq!(b2.total_film_owned_bytes, 0);
    }

    #[test]
    fn roi_layout_plane_offsets_disjoint_and_contiguous() {
        let num_emul = 6;
        let roi_w = 100;
        let roi_h = 100;
        let layout = RoiLayout::new(roi_w, roi_h, 4000, 3000, num_emul).unwrap();
        let roi_n = layout.roi_n as u32;

        let mut prev_end = 0u32;
        // 1. Dye planes (0..E)
        for e in 0..num_emul {
            let off = layout.dye_offset(e).unwrap();
            assert_eq!(off, prev_end);
            prev_end = off + roi_n;
        }

        // 2. Mask planes (E..2E)
        for e in 0..num_emul {
            let off = layout.mask_offset(e).unwrap();
            assert_eq!(off, prev_end);
            prev_end = off + roi_n;
        }

        // 3. Latent workspace planes (2E..3E)
        for e in 0..num_emul {
            let off = layout.latent_workspace_offset(e).unwrap();
            assert_eq!(off, prev_end);
            prev_end = off + roi_n;
        }

        // 4. Blur tmp plane (3E)
        let blur_tmp = layout.blur_tmp_offset().unwrap();
        assert_eq!(blur_tmp, prev_end);
        prev_end = blur_tmp + roi_n;

        // 5. Blur out plane (3E + 1)
        let blur_out = layout.blur_out_offset().unwrap();
        assert_eq!(blur_out, prev_end);
        prev_end = blur_out + roi_n;

        // Verify total arena size is exactly (3E + 2) * roi_n
        assert_eq!(prev_end, layout.total_arena_floats as u32);

        // Check out-of-bounds error handling
        assert!(layout.dye_offset(num_emul).is_err());
        assert!(layout.mask_offset(num_emul).is_err());
        assert!(layout.latent_workspace_offset(num_emul).is_err());
    }

    #[test]
    fn grain_variance_norm_pure_reduction_helper() {
        use crate::film::gpu::calculate_variance_norms_from_partials;

        // Test reduction of partial sum-of-squares with known values.
        // Emulsion 0: 4 partials of value 1.0 -> sum_sq = 4.0, n = 4 -> var = 1.0 -> norm = 1.0
        // Emulsion 1: 4 partials of value 4.0 -> sum_sq = 16.0, n = 4 -> var = 4.0 -> norm = 0.5
        let parts = vec![1.0f32, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0];
        let active_indices = vec![0, 1];
        let norms = calculate_variance_norms_from_partials(&parts, 4, 4, &active_indices).unwrap();

        assert_eq!(norms.len(), 2);
        assert_eq!(norms[0], (0, 1.0f32));
        assert_eq!(norms[1], (1, 1.0f32));
    }

    #[test]
    fn grain_variance_reduction_layout_test() {
        use crate::film::gpu::grain_var_reduction_layout;

        let layout = grain_var_reduction_layout(10000).unwrap();
        assert_eq!(layout.n, 10000);
        assert_eq!(layout.stride, 5);
        assert_eq!(layout.out_n, 2000);
        assert_eq!(layout.workgroups, 8);

        // Check zero dimension error
        assert!(grain_var_reduction_layout(0).is_err());
    }
}
