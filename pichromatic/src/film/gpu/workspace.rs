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

/// Keep at most this many resolution-specific scratch sets (preview + final).
const MAX_SCRATCH_SLOTS: usize = 2;
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

    fn matches(&self, width: usize, height: usize, num_emul: usize) -> bool {
        self.width == width && self.height == height && self.num_emul == num_emul.max(1)
    }
}

struct FilmGpuWorkspace {
    /// LRU: front = oldest, back = newest. Cap [`MAX_SCRATCH_SLOTS`].
    scratches: Vec<FilmScratch>,
    /// LRU const buffers keyed by stock fingerprint + width.
    consts: Vec<(ConstsKey, StockConsts)>,
}

impl FilmGpuWorkspace {
    const fn new() -> Self {
        Self {
            scratches: Vec::new(),
            consts: Vec::new(),
        }
    }

    fn take_scratch(
        &mut self,
        ctx: &GpuContext,
        width: usize,
        height: usize,
        num_emul: usize,
    ) -> FilmScratch {
        let e = num_emul.max(1);
        if let Some(i) = self
            .scratches
            .iter()
            .position(|s| s.matches(width, height, e))
        {
            #[cfg(not(target_arch = "wasm32"))]
            ctx.poll_wait();
            return self.scratches.remove(i);
        }
        FilmScratch::allocate(ctx, width, height, e)
    }

    fn restore_scratch(&mut self, scratch: FilmScratch) {
        self.scratches
            .retain(|s| !s.matches(scratch.width, scratch.height, scratch.num_emul));
        self.scratches.push(scratch);
        while self.scratches.len() > MAX_SCRATCH_SLOTS {
            self.scratches.remove(0);
        }
    }

    fn take_consts(
        &mut self,
        ctx: &GpuContext,
        stock: &FilmStock,
        params: &FilmParams,
        width: usize,
    ) -> (ConstsKey, StockConsts) {
        let key = ConstsKey::from_params(params, width);
        if let Some(i) = self.consts.iter().position(|(k, _)| *k == key) {
            return self.consts.remove(i);
        }
        (key, bake_consts(ctx, stock, params, width))
    }

    fn restore_consts(&mut self, key: ConstsKey, consts: StockConsts) {
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
        if scratch.is_none() && consts.is_none() {
            return;
        }
        let mut ws = WORKSPACE.0.lock().unwrap();
        if let Some(scratch) = scratch {
            ws.restore_scratch(scratch);
        }
        if let Some(consts) = consts {
            ws.restore_consts(self.consts_key, consts);
        }
    }
}

/// Acquire persistent scratch + consts for a film run (exclusive lease).
pub fn acquire_film_resources(
    ctx: &GpuContext,
    stock: &FilmStock,
    params: &FilmParams,
    width: usize,
    height: usize,
    num_emul: usize,
) -> FilmGpuLease {
    let mut ws = WORKSPACE.0.lock().unwrap();
    let scratch = ws.take_scratch(ctx, width, height, num_emul);
    let (consts_key, consts) = ws.take_consts(ctx, stock, params, width);
    FilmGpuLease {
        scratch: Some(scratch),
        consts: Some(consts),
        consts_key,
    }
}
