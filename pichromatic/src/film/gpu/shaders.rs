//! WGSL compute shaders for the GPU film path. Each mirrors a CPU stage.
//!
//! All storage bindings are declared `read_write` to match the bind-group layout
//! produced by [`crate::gpu::GpuContext::dispatch_compute_shader_multi`] (which
//! always uses non-read-only storage), even where the shader only reads.

/// Horizontal separable-blur pass. Mirrors `blur::convolve_1d_reflect`.
/// Used when radius > [`super::BLUR_TILED_MAX_RADIUS`] (tiled path cannot cover full kernel).
pub const BLUR_H: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(2) var<storage, read> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let w = u.width;
    let x = i % w;
    let y = i / w;
    let len = 2u * u.radius + 1u;
    var acc = 0.0;
    for (var k = 0u; k < len; k = k + 1u) {
        let xx = reflect_index(i32(x) + i32(k) - i32(u.radius), i32(w));
        acc = acc + src[u.src_off + y * w + u32(xx)] * ker[k];
    }
    tmp[u.dst_off + y * w + x] = acc;
}
"#;

/// Vertical separable-blur pass.
/// Used when radius > [`super::BLUR_TILED_MAX_RADIUS`] (tiled path cannot cover full kernel).
pub const BLUR_V: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let w = u.width;
    let x = i % w;
    let y = i / w;
    let len = 2u * u.radius + 1u;
    var acc = 0.0;
    for (var k = 0u; k < len; k = k + 1u) {
        let yy = reflect_index(i32(y) + i32(k) - i32(u.radius), i32(u.height));
        acc = acc + tmp[u.src_off + u32(yy) * w + x] * ker[k];
    }
    dst[u.dst_off + y * w + x] = acc;
}
"#;

/// Tiled horizontal blur: one workgroup = one row segment with shared-memory halo.
/// Dispatch: gx = ceil(width/256), gy = height.
///
/// Contract: host must only dispatch this when `radius ≤ BLUR_TILED_MAX_RADIUS` (128).
/// For larger radii the host falls back to [`BLUR_H`] so full-kernel numerics match CPU.
/// The `min(radius, 128)` below is only a defensive tile bound (tile[512] = 256+2*128);
/// it is **not** a correct FIR truncation — do not rely on it for radius > 128.
pub const BLUR_H_TILED: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(2) var<storage, read> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> tile: array<f32, 512>;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let y = wid.y;
    if (y >= u.height) { return; }
    let w = i32(u.width);
    let radius = i32(u.radius);
    // Defensive tile bound only (host must fall back for radius > 128).
    let r = min(radius, 128);
    let x0 = i32(wid.x * 256u);
    let lx = i32(lid.x);
    let base = u.src_off + y * u.width;

    // Cooperative load of [x0-r, x0+256+r) into tile[0 .. 256+2r).
    let tile_w = 256 + 2 * r;
    var t = lx;
    loop {
        if (t >= tile_w) { break; }
        let xx = reflect_index(x0 + t - r, w);
        tile[t] = src[base + u32(xx)];
        t = t + 256;
    }
    workgroupBarrier();

    let x = x0 + lx;
    if (x >= w) { return; }
    let len = 2u * u32(r) + 1u;
    var acc = 0.0;
    let center = lx + r;
    for (var k = 0u; k < len; k = k + 1u) {
        acc = acc + tile[u32(center) + k - u32(r)] * ker[k];
    }
    tmp[u.dst_off + y * u.width + u32(x)] = acc;
}
"#;

/// Tiled vertical blur: one workgroup = one column segment with shared-memory halo.
/// Dispatch: gx = width, gy = ceil(height/256).
///
/// Same contract as [`BLUR_H_TILED`]: host falls back to [`BLUR_V`] when radius > 128.
pub const BLUR_V_TILED: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> tile: array<f32, 512>;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let x = wid.x;
    if (x >= u.width) { return; }
    let h = i32(u.height);
    let radius = i32(u.radius);
    // Defensive tile bound only (host must fall back for radius > 128).
    let r = min(radius, 128);
    let y0 = i32(wid.y * 256u);
    let ly = i32(lid.x);
    let w = u.width;

    let tile_h = 256 + 2 * r;
    var t = ly;
    loop {
        if (t >= tile_h) { break; }
        let yy = reflect_index(y0 + t - r, h);
        tile[t] = tmp[u.src_off + u32(yy) * w + x];
        t = t + 256;
    }
    workgroupBarrier();

    let y = y0 + ly;
    if (y >= h) { return; }
    let len = 2u * u32(r) + 1u;
    var acc = 0.0;
    let center = ly + r;
    for (var k = 0u; k < len; k = k + 1u) {
        acc = acc + tile[u32(center) + k - u32(r)] * ker[k];
    }
    dst[u.dst_off + u32(y) * w + x] = acc;
}
"#;

/// Arena blur H: single storage buffer (WebGPU forbids overlapping writable aliases).
/// Reads `data[src_off..]`, writes `data[dst_off..]`. Bindings: data, ker, uniform.
pub const BLUR_H_ARENA: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> ker: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let w = u.width;
    let x = i % w;
    let y = i / w;
    let len = 2u * u.radius + 1u;
    var acc = 0.0;
    for (var k = 0u; k < len; k = k + 1u) {
        let xx = reflect_index(i32(x) + i32(k) - i32(u.radius), i32(w));
        acc = acc + data[u.src_off + y * w + u32(xx)] * ker[k];
    }
    data[u.dst_off + y * w + x] = acc;
}
"#;

/// Arena blur V: single storage buffer. Bindings: data, ker, uniform.
pub const BLUR_V_ARENA: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> ker: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let w = u.width;
    let x = i % w;
    let y = i / w;
    let len = 2u * u.radius + 1u;
    var acc = 0.0;
    for (var k = 0u; k < len; k = k + 1u) {
        let yy = reflect_index(i32(y) + i32(k) - i32(u.radius), i32(u.height));
        acc = acc + data[u.src_off + u32(yy) * w + x] * ker[k];
    }
    data[u.dst_off + y * w + x] = acc;
}
"#;

/// Arena tiled blur H: single storage buffer. Bindings: data, ker, uniform.
pub const BLUR_H_TILED_ARENA: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> ker: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

var<workgroup> tile: array<f32, 512>;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let y = wid.y;
    if (y >= u.height) { return; }
    let w = i32(u.width);
    let radius = i32(u.radius);
    let r = min(radius, 128);
    let x0 = i32(wid.x * 256u);
    let lx = i32(lid.x);
    let base = u.src_off + y * u.width;

    let tile_w = 256 + 2 * r;
    var t = lx;
    loop {
        if (t >= tile_w) { break; }
        let xx = reflect_index(x0 + t - r, w);
        tile[t] = data[base + u32(xx)];
        t = t + 256;
    }
    workgroupBarrier();

    let x = x0 + lx;
    if (x >= w) { return; }
    let len = 2u * u32(r) + 1u;
    var acc = 0.0;
    let center = lx + r;
    for (var k = 0u; k < len; k = k + 1u) {
        acc = acc + tile[u32(center) + k - u32(r)] * ker[k];
    }
    data[u.dst_off + y * u.width + u32(x)] = acc;
}
"#;

/// Arena tiled blur V: single storage buffer. Bindings: data, ker, uniform.
pub const BLUR_V_TILED_ARENA: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> ker: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

var<workgroup> tile: array<f32, 512>;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let x = wid.x;
    if (x >= u.width) { return; }
    let h = i32(u.height);
    let radius = i32(u.radius);
    let r = min(radius, 128);
    let y0 = i32(wid.y * 256u);
    let ly = i32(lid.x);
    let w = u.width;

    let tile_h = 256 + 2 * r;
    var t = ly;
    loop {
        if (t >= tile_h) { break; }
        let yy = reflect_index(y0 + t - r, h);
        tile[t] = data[u.src_off + u32(yy) * w + x];
        t = t + 256;
    }
    workgroupBarrier();

    let y = y0 + ly;
    if (y >= h) { return; }
    let len = 2u * u32(r) + 1u;
    var acc = 0.0;
    let center = ly + r;
    for (var k = 0u; k < len; k = k + 1u) {
        acc = acc + tile[u32(center) + k - u32(r)] * ker[k];
    }
    data[u.dst_off + u32(y) * w + x] = acc;
}
"#;

/// Expose: ACEScg pixel → per-emulsion absorbed mean fluence plane.
/// Mirrors `exposure::expose_with_pitch_and_shutter` (spectral part).
pub const EXPOSE: &str = r#"
struct U { width:u32, height:u32, n:u32, num_layers:u32, num_emul:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> planes: array<f32>;
@group(0) @binding(2) var<storage, read_write> ec: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let px = pixels[i];
    let r = px.x; let g = px.y; let b = px.z;

    // Upsample: w = M * rgb, clamp negatives (Meng-style).
    var w0 = ec[48] * r + ec[49] * g + ec[50] * b;
    var w1 = ec[51] * r + ec[52] * g + ec[53] * b;
    var w2 = ec[54] * r + ec[55] * g + ec[56] * b;
    w0 = max(w0, 0.0); w1 = max(w1, 0.0); w2 = max(w2, 0.0);

    // Fluence spectrum: Φ(λ) = spectrum(λ) * (λ / 550).
    var phi: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let s = w0 * ec[k] + w1 * ec[16u + k] + w2 * ec[32u + k];
        phi[k] = s * ec[57u + k] * ec[73u];
    }

    let off_od = 74u;
    let off_pl = 74u + u.num_layers * 16u;
    var emul = 0u;
    for (var l = 0u; l < u.num_layers; l = l + 1u) {
        var absorbed: array<f32, 16>;
        let ob = off_od + l * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            // Transmittance baked on the host (f64 exp rounded to f32), so the
            // GPU never calls `exp` here — CPU and GPU share the value.
            let trans = ec[ob + k];
            let pt = phi[k] * trans;
            absorbed[k] = phi[k] - pt;
            phi[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            // integrated_absorbed = trapz(400..700, Δ20) / 300.
            var acc = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc = acc + (absorbed[k] + absorbed[k + 1u]) * (1.0 / 30.0);
            }
            planes[emul * u.n + i] = acc;
            emul = emul + 1u;
        }
    }
}
"#;

/// ROI Expose: reads immutable full-frame input via signed global coordinate reflection
/// and writes into root ROI planes buffer starting at `planes_base`.
pub const EXPOSE_ROI: &str = r#"
struct U {
    root_x: i32, root_y: i32, root_w: u32, root_h: u32,
    img_w: u32, img_h: u32, root_n: u32, planes_base: u32,
    num_layers: u32, num_emul: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> planes: array<f32>;
@group(0) @binding(2) var<storage, read> ec: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.root_n) { return; }

    let lx = i % u.root_w;
    let ly = i / u.root_w;

    let gx = u.root_x + i32(lx);
    let gy = u.root_y + i32(ly);

    let px = reflect_index(gx, i32(u.img_w));
    let py = reflect_index(gy, i32(u.img_h));

    let img_idx = u32(py) * u.img_w + u32(px);
    let pix = pixels[img_idx];
    let r = pix.x; let g = pix.y; let b = pix.z;

    // Upsample: w = M * rgb, clamp negatives (Meng-style).
    var w0 = ec[48] * r + ec[49] * g + ec[50] * b;
    var w1 = ec[51] * r + ec[52] * g + ec[53] * b;
    var w2 = ec[54] * r + ec[55] * g + ec[56] * b;
    w0 = max(w0, 0.0); w1 = max(w1, 0.0); w2 = max(w2, 0.0);

    var phi: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        let s = w0 * ec[k] + w1 * ec[16u + k] + w2 * ec[32u + k];
        phi[k] = s * ec[57u + k] * ec[73u];
    }

    let off_od = 74u;
    let off_pl = 74u + u.num_layers * 16u;
    var emul = 0u;
    for (var l = 0u; l < u.num_layers; l = l + 1u) {
        var absorbed: array<f32, 16>;
        let ob = off_od + l * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            // Transmittance baked on the host (f64 exp rounded to f32), so the
            // GPU never calls `exp` here — CPU and GPU share the value.
            let trans = ec[ob + k];
            let pt = phi[k] * trans;
            absorbed[k] = phi[k] - pt;
            phi[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            var acc = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc = acc + (absorbed[k] + absorbed[k + 1u]) * (1.0 / 30.0);
            }
            planes[u.planes_base + emul * u.root_n + i] = acc;
            emul = emul + 1u;
        }
    }
}
"#;

/// Energy-conserving local scatter mix: Φ' = keep·Φ + f·blur(Φ).
pub const LOCAL_SCATTER_MIX: &str = r#"
struct U { n:u32, off:u32, keep:f32, f:f32, p0:u32, p1:u32, p2:u32, p3:u32 };
@group(0) @binding(0) var<storage, read_write> planes: array<f32>;
@group(0) @binding(1) var<storage, read_write> blurred: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    planes[u.off + i] = u.keep * planes[u.off + i] + u.f * blurred[i];
}
"#;

/// ROI Local scatter mix: single arena binding (WebGPU forbids overlapping writable aliases).
pub const LOCAL_SCATTER_MIX_ROI: &str = r#"
struct U { n:u32, plane_off:u32, blur_off:u32, keep:f32, f:f32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    arena[u.plane_off + i] = u.keep * arena[u.plane_off + i] + u.f * arena[u.blur_off + i];
}
"#;

/// Additive wide halation with per-layer bleed gains: Φ_e += gain_e · bounce.
pub const HALATION_ADD: &str = r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> planes: array<f32>;
@group(0) @binding(1) var<storage, read_write> bounce: array<f32>;
@group(0) @binding(2) var<storage, read_write> gains: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let bv = bounce[i];
    for (var e = 0u; e < u.num_emul; e = e + 1u) {
        planes[e * u.n + i] = planes[e * u.n + i] + gains[e] * bv;
    }
}
"#;

/// Multi-bounce halation accumulation: `acc = init ? w·b : acc + w·b`.
/// Mirrors the CPU decay-weighted bounce sum in `apply_spatial_exposure_effects`.
pub const HALATION_ACCUM: &str = r#"
struct U { n:u32, w:f32, init:u32, p0:u32 };
@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read_write> blurred: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let b = blurred[i];
    if (u.init != 0u) {
        acc[i] = u.w * b;
    } else {
        acc[i] = acc[i] + u.w * b;
    }
}
"#;

/// ROI Multi-bounce halation accumulation on a single arena binding.
pub const HALATION_ACCUM_ROI: &str = r#"
struct U { n:u32, out_off:u32, blur_off:u32, w:f32, init:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let b = arena[u.blur_off + i];
    if (u.init != 0u) {
        arena[u.out_off + i] = u.w * b;
    } else {
        arena[u.out_off + i] = arena[u.out_off + i] + u.w * b;
    }
}
"#;

/// ROI Additive wide halation: single arena binding + gains.
pub const HALATION_ADD_ROI: &str = r#"
struct U { n:u32, num_emul:u32, plane_base:u32, bounce_base:u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read> gains: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let bv = arena[u.bounce_base + i];
    for (var e = 0u; e < u.num_emul; e = e + 1u) {
        let idx = u.plane_base + e * u.n + i;
        arena[idx] = arena[idx] + gains[e] * bv;
    }
}
"#;

/// Capture LUT sample (developable fraction). Mirrors `DevelopableFractionLut::sample`.
pub const LUT: &str = r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> planes: array<f32>;
@group(0) @binding(1) var<storage, read_write> lc: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

const INV_LN10: f32 = 0.4342944819032518;

fn lut_sample(phi: f32, fbase: u32) -> f32 {
    if (!(phi > 0.0)) { return lc[fbase]; }
    let lp = log(phi) * INV_LN10;
    let lo0 = lc[0];
    let lo63 = lc[63];
    if (lp <= lo0) { return lc[fbase]; }
    if (lp >= lo63) { return lc[fbase + 63u]; }
    let step = lc[1] - lc[0];
    var lo = u32(floor((lp - lo0) / step));
    if (lo > 62u) { lo = 62u; }
    let hi = lo + 1u;
    let t = (lp - lc[lo]) / (lc[hi] - lc[lo]);
    return lc[fbase + lo] * (1.0 - t) + lc[fbase + hi] * t;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let eta_base = 64u + u.num_emul * 64u;
    for (var e = 0u; e < u.num_emul; e = e + 1u) {
        let idx = e * u.n + i;
        planes[idx] = lut_sample(planes[idx] * lc[eta_base + e], 64u + e * 64u);
    }
}
"#;

/// ROI Capture LUT: takes explicit plane_base (e.g. latent_workspace_base).
pub const LUT_ROI: &str = r#"
struct U { n:u32, num_emul:u32, plane_base:u32, p0:u32 };
@group(0) @binding(0) var<storage, read_write> planes: array<f32>;
@group(0) @binding(1) var<storage, read> lc: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

const INV_LN10: f32 = 0.4342944819032518;

fn lut_sample(phi: f32, fbase: u32) -> f32 {
    if (!(phi > 0.0)) { return lc[fbase]; }
    let lp = log(phi) * INV_LN10;
    let lo0 = lc[0];
    let lo63 = lc[63];
    if (lp <= lo0) { return lc[fbase]; }
    if (lp >= lo63) { return lc[fbase + 63u]; }
    let step = lc[1] - lc[0];
    var lo = u32(floor((lp - lo0) / step));
    if (lo > 62u) { lo = 62u; }
    let hi = lo + 1u;
    let t = (lp - lc[lo]) / (lc[hi] - lc[lo]);
    return lc[fbase + lo] * (1.0 - t) + lc[fbase + hi] * t;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let eta_base = 64u + u.num_emul * 64u;
    for (var e = 0u; e < u.num_emul; e = e + 1u) {
        let idx = u.plane_base + e * u.n + i;
        planes[idx] = lut_sample(planes[idx] * lc[eta_base + e], 64u + e * 64u);
    }
}
"#;

/// Reduce fraction → image/mask dye density. Mirrors `development::reduction::reduce`.
pub const REDUCE: &str = r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> planes: array<f32>;
@group(0) @binding(1) var<storage, read_write> dye: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> rc: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let e_count = u.num_emul;
    for (var e = 0u; e < e_count; e = e + 1u) {
        let f = clamp(planes[e * u.n + i], 0.0, 1.0);
        let rev = rc[3u * e_count + e];
        var eff = f;
        if (rev != 0.0) { eff = 1.0 - f; }
        // Chemical fog floor: f_eff = 1 - (1-f) * (1-f_fog). f_fog baked host-side
        // per layer as (FOG_OFFSET / d_max).clamp(0,1) into rc[5 * e_count + e].
        let fog = rc[5u * e_count + e];
        eff = 1.0 - (1.0 - eff) * (1.0 - fog);
        let dmax = rc[e];
        let ig = rc[e_count + e];
        var d = 0.0;
        if (eff > 0.0) { d = dmax * pow(eff, ig); }
        dye[e * u.n + i] = d;
        var m = 0.0;
        if (rc[4u * e_count + e] != 0.0) {
            m = rc[2u * e_count + e];
        }
        mask[e * u.n + i] = m;
    }
}
"#;

/// ROI Reduce: takes explicit plane_base, dye_base, mask_base.
pub const REDUCE_ROI: &str = r#"
struct U { n:u32, num_emul:u32, plane_base:u32, dye_base:u32, mask_base:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> planes: array<f32>;
@group(0) @binding(1) var<storage, read_write> dye: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<f32>;
@group(0) @binding(3) var<storage, read> rc: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let e_count = u.num_emul;
    for (var e = 0u; e < e_count; e = e + 1u) {
        let f = clamp(planes[u.plane_base + e * u.n + i], 0.0, 1.0);
        let rev = rc[3u * e_count + e];
        var eff = f;
        if (rev != 0.0) { eff = 1.0 - f; }
        // Chemical fog floor (mirrors `reduce`). f_fog in rc[5 * e_count + e].
        let fog = rc[5u * e_count + e];
        eff = 1.0 - (1.0 - eff) * (1.0 - fog);
        let dmax = rc[e];
        let ig = rc[e_count + e];
        var d = 0.0;
        if (eff > 0.0) { d = dmax * pow(eff, ig); }
        dye[u.dye_base + e * u.n + i] = d;
        var m = 0.0;
        if (rc[4u * e_count + e] != 0.0) {
            m = rc[2u * e_count + e];
        }
        mask[u.mask_base + e * u.n + i] = m;
    }
}
"#;

/// Fused ROI LUT and Reduce: single arena binding + lut/reduce constants.
pub const LUT_REDUCE_ROI: &str = r#"
struct U { n:u32, num_emul:u32, plane_base:u32, dye_base:u32, mask_base:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read> lc: array<f32>;
@group(0) @binding(2) var<storage, read> rc: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

const INV_LN10: f32 = 0.4342944819032518;

fn lut_sample(phi: f32, fbase: u32) -> f32 {
    if (!(phi > 0.0)) { return lc[fbase]; }
    let lp = log(phi) * INV_LN10;
    let lo0 = lc[0];
    let lo63 = lc[63];
    if (lp <= lo0) { return lc[fbase]; }
    if (lp >= lo63) { return lc[fbase + 63u]; }
    let step = lc[1] - lc[0];
    var lo = u32(floor((lp - lo0) / step));
    if (lo > 62u) { lo = 62u; }
    let hi = lo + 1u;
    let t = (lp - lc[lo]) / (lc[hi] - lc[lo]);
    return lc[fbase + lo] * (1.0 - t) + lc[fbase + hi] * t;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let e_count = u.num_emul;
    let eta_base = 64u + e_count * 64u;
    for (var e = 0u; e < e_count; e = e + 1u) {
        let idx = u.plane_base + e * u.n + i;
        let phi = arena[idx];
        let lut_val = lut_sample(phi * lc[eta_base + e], 64u + e * 64u);

        let f = clamp(lut_val, 0.0, 1.0);
        let rev = rc[3u * e_count + e];
        var eff = f;
        if (rev != 0.0) { eff = 1.0 - f; }
        // Chemical fog floor (mirrors `reduce`). f_fog in rc[5 * e_count + e].
        let fog = rc[5u * e_count + e];
        eff = 1.0 - (1.0 - eff) * (1.0 - fog);
        let dmax = rc[e];
        let ig = rc[e_count + e];
        var d = 0.0;
        if (eff > 0.0) { d = dmax * pow(eff, ig); }
        arena[u.dye_base + e * u.n + i] = d;
        var m = 0.0;
        if (rc[4u * e_count + e] != 0.0) {
            m = rc[2u * e_count + e];
        }
        arena[u.mask_base + e * u.n + i] = m;

        arena[idx] = lut_val;
    }
}
"#;

/// DIR interlayer inhibition apply. Mirrors `development::diffusion::apply_dir_inhibition`.
pub const DIR_APPLY: &str = r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> dye: array<f32>;
@group(0) @binding(1) var<storage, read_write> diffused: array<f32>;
@group(0) @binding(2) var<storage, read> mat: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let e_count = u.num_emul;
    for (var j = 0u; j < e_count; j = j + 1u) {
        var total_delta = 0.0;
        for (var s = 0u; s < e_count; s = s + 1u) {
            let weight = mat[s * e_count + j];
            let diff = diffused[s * u.n + i] - dye[s * u.n + i];
            total_delta = total_delta + weight * diff;
        }
        dye[j * u.n + i] = dye[j * u.n + i] * exp(-total_delta);
    }
}
"#;

/// ROI DIR apply: single arena binding + matrix.
pub const DIR_APPLY_ROI: &str = r#"
struct U { n:u32, num_emul:u32, dye_base:u32, work_base:u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read> mat: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let e_count = u.num_emul;
    for (var j = 0u; j < e_count; j = j + 1u) {
        var total_delta = 0.0;
        for (var s = 0u; s < e_count; s = s + 1u) {
            let weight = mat[s * e_count + j];
            let diff = arena[u.work_base + s * u.n + i] - arena[u.dye_base + s * u.n + i];
            total_delta = total_delta + weight * diff;
        }
        let idx = u.dye_base + j * u.n + i;
        arena[idx] = arena[idx] * exp(-total_delta);
    }
}
"#;

/// Adjacency (Eberhard) unsharp: D' = D + β·(D − blur(D)).
pub const ADJACENCY: &str = r#"
struct U { n:u32, off:u32, beta:f32, p0:u32 };
@group(0) @binding(0) var<storage, read_write> dye: array<f32>;
@group(0) @binding(1) var<storage, read_write> blurred: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let d = dye[u.off + i];
    dye[u.off + i] = d + u.beta * (d - blurred[i]);
}
"#;

/// ROI Adjacency: single arena binding.
pub const ADJACENCY_ROI: &str = r#"
struct U { n:u32, dye_off:u32, blur_off:u32, beta:f32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let d = arena[u.dye_off + i];
    arena[u.dye_off + i] = d + u.beta * (d - arena[u.blur_off + i]);
}
"#;

// ─── Legacy SplitMix64-based grain shaders (GRAIN_NOISE, GRAIN_NOISE_ROI,
// GRAIN_APPLY_SUB, GRAIN_APPLY_SUB_RAW_ROI, GRAIN_VAR_PARTIAL, GRAIN_SCAN_ROI,
// COPY_SCALAR_CORE_ROI) deleted in favor of the Philox4x32-10 particle-field
// grain path (`PARTICLE_FIELD`, `MICRO_MIX`, `TOE`, `SCAN_ROI`). ─────────────

/// Densitometric scan → ACEScg (Dmin-normalized). Mirrors `scan::densitometry`.
pub const SCAN: &str = r#"
struct U { n:u32, num_emul:u32, scale:f32, flags:u32 };
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dye: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

const LOG2_10: f32 = 3.3219280948873623;

fn fog2_to_exposure(d2: f32, inv_gamma: f32, fog2: f32) -> f32 {
    var d_eff = 0.0;
    if (d2 > 0.0) {
        if (d2 < fog2) {
            d_eff = (d2 * d2) / (2.0 * fog2);
        } else {
            d_eff = d2 - 0.5 * fog2;
        }
    }
    if (d_eff <= 0.0) {
        return 0.0;
    }
    return exp2(d_eff * inv_gamma) - 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let E = u.num_emul;
    let eps_base = 0u;
    let maskeps_base = E * 16u;
    let illum_base = 2u * E * 16u;
    let xbar_base = illum_base + 16u;
    let ybar_base = xbar_base + 16u;
    let zbar_base = ybar_base + 16u;
    let mat_base = zbar_base + 16u;
    let toe_base = mat_base + 9u;

    var dens: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) { dens[k] = 0.0; }
    for (var e = 0u; e < E; e = e + 1u) {
        let raw_f = dye[e * u.n + i];
        let dmax = sc[toe_base + e];
        let inv_gamma = sc[toe_base + E + e];
        let g = max(raw_f, 0.0);
        var di = raw_f;
        if (dmax > 0.0 && g > 0.0) {
            di = clamp(dmax * pow(g, inv_gamma), 0.0, 1.05 * dmax);
        } else if (dmax > 0.0) {
            di = 0.0;
        }
        let dm = mask[e * u.n + i];
        let eb = eps_base + e * 16u;
        let mb = maskeps_base + e * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            dens[k] = dens[k] + di * sc[eb + k] + dm * sc[mb + k];
        }
    }
    var t: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        t[k] = exp2(-dens[k] * LOG2_10) * sc[illum_base + k];
    }
    var X = 0.0; var Y = 0.0; var Z = 0.0;
    for (var k = 0u; k < 15u; k = k + 1u) {
        X = X + (t[k] * sc[xbar_base + k] + t[k + 1u] * sc[xbar_base + k + 1u]) * 10.0;
        Y = Y + (t[k] * sc[ybar_base + k] + t[k + 1u] * sc[ybar_base + k + 1u]) * 10.0;
        Z = Z + (t[k] * sc[zbar_base + k] + t[k + 1u] * sc[zbar_base + k + 1u]) * 10.0;
    }
    var rgb = vec3<f32>(
        (sc[mat_base + 0u] * X + sc[mat_base + 1u] * Y + sc[mat_base + 2u] * Z) * u.scale,
        (sc[mat_base + 3u] * X + sc[mat_base + 4u] * Y + sc[mat_base + 5u] * Z) * u.scale,
        (sc[mat_base + 6u] * X + sc[mat_base + 7u] * Y + sc[mat_base + 8u] * Z) * u.scale
    );

    if ((u.flags & 1u) != 0u) {
        let inv_base = toe_base + 2u * E;
        let inv_dmin = vec3<f32>(sc[inv_base + 10u], sc[inv_base + 11u], sc[inv_base + 12u]);
        let g_val = vec3<f32>(sc[inv_base + 3u], sc[inv_base + 4u], sc[inv_base + 5u]);
        let eps = sc[inv_base + 8u];
        let inv_gamma = sc[inv_base + 13u];
        let fog2 = sc[inv_base + 15u];

        let tc = vec3<f32>(
            max(rgb.x * inv_dmin.x, eps),
            max(rgb.y * inv_dmin.y, eps),
            max(rgb.z * inv_dmin.z, eps),
        );
        let d2 = vec3<f32>(
            -log2(tc.x),
            -log2(tc.y),
            -log2(tc.z),
        );
        var e_scene = vec3<f32>(
            fog2_to_exposure(d2.x, inv_gamma, fog2),
            fog2_to_exposure(d2.y, inv_gamma, fog2),
            fog2_to_exposure(d2.z, inv_gamma, fog2),
        );

        rgb = vec3<f32>(
            g_val.x * e_scene.x,
            g_val.y * e_scene.y,
            g_val.z * e_scene.z,
        );
    }

    pixels[i] = vec4<f32>(rgb.x, rgb.y, rgb.z, pixels[i].w);
}
"#;

// ─── Particle-field grain (Philox4x32-10 + per-pixel Poisson/Binomial) ───────
//
// Mirrors the CPU `apply_particle_grain_overwrite` particle overwrite:
//
//   p      = clamp(D / d_max, 0, 1)
//   prob   = p^gamma                 (developable fraction f_eff recovered)
//   sites  = sample_poisson(sites_per_cell)
//   devel  = sample_binomial(sites, prob)
//   frac   = devel / sites_per_cell
//   D      := frac                   (OVERWRITE — no base+residual)
//
// On every GPU-supported film format, `pitch ≥ 3 µm/px`, hence
// `cell_um = max(DYE_CLOUD_CORRELATION_UM, pitch) = pitch`, `cell_px = 1`,
// `cells == pixels`: the particle field is per-pixel directly and there is
// no cell-mean aggregation or bilinear upscale. The `sites_per_cell` field
// shipped per layer is `(1/κ_ref²) * pitch²` (identical to the CPU value
// when cells == pixels). Knuth product Poisson is only taken in the rare
// `λ < 16` tails; on the GPU path λ is typically very large.

/// Particle overwrite — full-frame variant. Reads D at `d_off` and writes the
/// realized fraction back to the same offset (overwrite, in-place).
pub const PARTICLE_FIELD: &str = r#"
const PF_M0: u32 = 0xD2511F53u;
const PF_M1: u32 = 0xCD9E8D57u;
const PF_W0: u32 = 0x9E3779B9u;
const PF_W1: u32 = 0xBB67AE85u;

struct PFU {
    n: u32, width: u32, height: u32, layer_idx: u32,
    d_max: f32, gamma: f32, sites_per_cell: f32, d_off: u32,
    seed_lo: u32, seed_hi: u32, sqrt_sites: f32, knuth_threshold: f32,
};
@group(0) @binding(0) var<storage, read_write> planes: array<f32>;
@group(0) @binding(1) var<uniform> u: PFU;

fn mul_full(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu; let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu; let b_hi = b >> 16u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo = (ll & 0xFFFFu) | (mid << 16u);
    let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn philox_round(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let p0 = mul_full(PF_M0, ctr.x);
    let p1 = mul_full(PF_M1, ctr.z);
    return vec4<u32>(p1.y ^ ctr.y ^ key.x, p1.x, p0.y ^ ctr.w ^ key.y, p0.x);
}

fn philox_block(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var c = ctr;
    var k = key;
    for (var i = 0u; i < 10u; i = i + 1u) {
        c = philox_round(c, k);
        k = vec2<u32>(k.x + PF_W0, k.y + PF_W1);
    }
    return c;
}

fn philox_word(x: u32, y: u32, i: u32, key: vec2<u32>) -> u32 {
    let b = i >> 2u;
    let l = i & 3u;
    let out = philox_block(vec4<u32>(x, y, 0u, b), key);
    if (l == 0u) { return out.x; }
    if (l == 1u) { return out.y; }
    if (l == 2u) { return out.z; }
    return out.w;
}

fn pf_gaussian(x: u32, y: u32, start_word: u32, key: vec2<u32>) -> f32 {
    // Bit-exact Irwin–Hall gaussian: identical to the CPU's f64 accumulation
    // of the same 12 single-word uniforms cast to f32 (`next_gaussian` in
    // grain.rs). Each term w/2^32 splits exactly into hi = (w>>12)/2^20 and
    // lo = (w&0xFFF)/2^32; the 12 hi parts (Σ ≤ 12·2^20 < 2^24) and the 12 lo
    // parts (Σ < 2^16) each accumulate exactly in f32, and the final hi+lo add
    // rounds once — the correctly rounded sum. Seeding sum_h at -6.0 makes the
    // subtraction exact too, and leaves no (a+b)-a-b compensation term that
    // Metal fast-math reassociation could collapse.
    var sum_h: f32 = -6.0;
    var sum_l: f32 = 0.0;
    for (var j = 0u; j < 12u; j = j + 1u) {
        let w = philox_word(x, y, start_word + j, key);
        sum_h = sum_h + f32(w >> 12u) / 1048576.0;
        sum_l = sum_l + f32(w & 0xFFFu) / 4294967296.0;
    }
    return sum_h + sum_l;
}

fn pf_poisson(lambda: f32, x: u32, y: u32, start_word: u32, key: vec2<u32>, sqrt_lambda: f32, threshold: f32) -> vec2<u32> {
    if (!(lambda > 0.0)) { return vec2<u32>(0u, 0u); }
    if (lambda >= 16.0) {
        let g = pf_gaussian(x, y, start_word, key);
        let draw = lambda + sqrt_lambda * g;
        // `round` matches Rust f32::round (ties away from zero). The old
        // `u32(draw + 0.5)` form double-rounds once |draw| ≥ 2^24 (half-to-even
        // instead of half-away), biasing the count by +1 on odd draws.
        let n = u32(round(max(draw, 0.0)));
        return vec2<u32>(n, 12u);
    }
    // Knuth product loop. `threshold` is exp(-lambda): baked host-side as f32
    // for the layer-constant sites draw (matching the CPU's f64 exp rounded to
    // f32); the per-pixel rare-tail calls pass WGSL exp(-lam). Each uniform
    // consumes TWO Philox words (CPU `next_unit_f64`:
    // u = ((hi<<21)|(lo>>11))/2^53), approximated in f32 as hi/2^32 while
    // advancing w by 2u so the GPU stream stays word-aligned with the CPU Philox
    // stream.
    var product: f32 = 1.0;
    var count: u32 = 0u;
    var w: u32 = start_word;
    for (var iter = 0u; iter < 100u; iter = iter + 1u) {
        if (product <= threshold) { break; }
        let hi = philox_word(x, y, w, key);
        let ut = f32(hi) / 4294967296.0;
        product = product * ut;
        count = count + 1u;
        w = w + 2u;
    }
    if (count > 0u) { count = count - 1u; }
    return vec2<u32>(count, w - start_word);
}

fn pf_binomial(trials: u32, p: f32, x: u32, y: u32, start_word: u32, key: vec2<u32>) -> vec2<u32> {
    if (trials == 0u || p <= 0.0) { return vec2<u32>(0u, 0u); }
    if (p >= 1.0) { return vec2<u32>(trials, 0u); }
    if (trials < 32u) {
        let thresh = u32(p * 4294967296.0);
        var c: u32 = 0u;
        var w: u32 = start_word;
        for (var i = 0u; i < trials; i = i + 1u) {
            let hi = philox_word(x, y, w, key);
            if (hi < thresh) { c = c + 1u; }
            w = w + 2u;
        }
        return vec2<u32>(c, w - start_word);
    }
    if (p < 0.05) {
        let lam = f32(trials) * p;
        let rp = pf_poisson(lam, x, y, start_word, key, sqrt(lam), exp(-lam));
        return vec2<u32>(min(rp.x, trials), rp.y);
    }
    if (p > 0.95) {
        let lam = f32(trials) * (1.0 - p);
        let rp = pf_poisson(lam, x, y, start_word, key, sqrt(lam), exp(-lam));
        return vec2<u32>(trials - min(rp.x, trials), rp.y);
    }
    let g = pf_gaussian(x, y, start_word, key);
    let mean = f32(trials) * p;
    let variance = mean * (1.0 - p);
    let draw = mean + sqrt(variance) * g;
    let dc = u32(round(clamp(draw, 0.0, f32(trials))));
    return vec2<u32>(dc, 12u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let x = i % u.width;
    let y = i / u.width;
    let d = planes[u.d_off + i];
    let key = vec2<u32>(u.seed_lo ^ u.layer_idx * PF_W0, u.seed_hi ^ u.layer_idx * PF_W1);
    let p = clamp(d / u.d_max, 0.0, 1.0);
    let prob = pow(p, u.gamma);
    let sp = pf_poisson(u.sites_per_cell, x, y, 0u, key, u.sqrt_sites, u.knuth_threshold);
    let dev = pf_binomial(sp.x, prob, x, y, sp.y, key);
    let fraction = f32(dev.x) / u.sites_per_cell;
    planes[u.d_off + i] = fraction;
}
"#;

/// Particle overwrite — ROI variant. Uses reflected global image coordinates
/// `(rx, ry)` for the Philox counter so that a root-halo-pixel realization
/// equals the realization at the physically-reflected image pixel (matching the
/// full-frame pass that only computes realizations for in-bounds pixels).
pub const PARTICLE_FIELD_ROI: &str = r#"
const PF_M0: u32 = 0xD2511F53u;
const PF_M1: u32 = 0xCD9E8D57u;
const PF_W0: u32 = 0x9E3779B9u;
const PF_W1: u32 = 0xBB67AE85u;

struct PFRU {
    root_x: i32, root_y: i32, root_w: u32, root_h: u32,
    root_n: u32, img_w: u32, img_h: u32, layer_idx: u32,
    d_max: f32, gamma: f32, sites_per_cell: f32, d_off: u32,
    seed_lo: u32, seed_hi: u32, sqrt_sites: f32, knuth_threshold: f32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: PFRU;

fn mul_full(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu; let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu; let b_hi = b >> 16u;
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    let mid = (ll >> 16u) + (lh & 0xFFFFu) + (hl & 0xFFFFu);
    let lo = (ll & 0xFFFFu) | (mid << 16u);
    let hi = hh + (lh >> 16u) + (hl >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn philox_round(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let p0 = mul_full(PF_M0, ctr.x);
    let p1 = mul_full(PF_M1, ctr.z);
    return vec4<u32>(p1.y ^ ctr.y ^ key.x, p1.x, p0.y ^ ctr.w ^ key.y, p0.x);
}

fn philox_block(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var c = ctr;
    var k = key;
    for (var i = 0u; i < 10u; i = i + 1u) {
        c = philox_round(c, k);
        k = vec2<u32>(k.x + PF_W0, k.y + PF_W1);
    }
    return c;
}

fn philox_word(x: u32, y: u32, i: u32, key: vec2<u32>) -> u32 {
    let b = i >> 2u;
    let l = i & 3u;
    let out = philox_block(vec4<u32>(x, y, 0u, b), key);
    if (l == 0u) { return out.x; }
    if (l == 1u) { return out.y; }
    if (l == 2u) { return out.z; }
    return out.w;
}

fn pf_gaussian(x: u32, y: u32, start_word: u32, key: vec2<u32>) -> f32 {
    // Bit-exact Irwin–Hall gaussian: identical to the CPU's f64 accumulation
    // of the same 12 single-word uniforms cast to f32 (`next_gaussian` in
    // grain.rs). Each term w/2^32 splits exactly into hi = (w>>12)/2^20 and
    // lo = (w&0xFFF)/2^32; the 12 hi parts (Σ ≤ 12·2^20 < 2^24) and the 12 lo
    // parts (Σ < 2^16) each accumulate exactly in f32, and the final hi+lo add
    // rounds once — the correctly rounded sum. Seeding sum_h at -6.0 makes the
    // subtraction exact too, and leaves no (a+b)-a-b compensation term that
    // Metal fast-math reassociation could collapse.
    var sum_h: f32 = -6.0;
    var sum_l: f32 = 0.0;
    for (var j = 0u; j < 12u; j = j + 1u) {
        let w = philox_word(x, y, start_word + j, key);
        sum_h = sum_h + f32(w >> 12u) / 1048576.0;
        sum_l = sum_l + f32(w & 0xFFFu) / 4294967296.0;
    }
    return sum_h + sum_l;
}

fn pf_poisson(lambda: f32, x: u32, y: u32, start_word: u32, key: vec2<u32>, sqrt_lambda: f32, threshold: f32) -> vec2<u32> {
    if (!(lambda > 0.0)) { return vec2<u32>(0u, 0u); }
    if (lambda >= 16.0) {
        let g = pf_gaussian(x, y, start_word, key);
        let draw = lambda + sqrt_lambda * g;
        // `round` matches Rust f32::round (ties away from zero). The old
        // `u32(draw + 0.5)` form double-rounds once |draw| ≥ 2^24 (half-to-even
        // instead of half-away), biasing the count by +1 on odd draws.
        let n = u32(round(max(draw, 0.0)));
        return vec2<u32>(n, 12u);
    }
    // Knuth product loop. `threshold` is exp(-lambda): baked host-side as f32
    // for the layer-constant sites draw (matching the CPU's f64 exp rounded to
    // f32); the per-pixel rare-tail calls pass WGSL exp(-lam). Each uniform
    // consumes TWO Philox words (CPU `next_unit_f64`:
    // u = ((hi<<21)|(lo>>11))/2^53), approximated in f32 as hi/2^32 while
    // advancing w by 2u so the GPU stream stays word-aligned with the CPU Philox
    // stream.
    var product: f32 = 1.0;
    var count: u32 = 0u;
    var w: u32 = start_word;
    for (var iter = 0u; iter < 100u; iter = iter + 1u) {
        if (product <= threshold) { break; }
        let hi = philox_word(x, y, w, key);
        let ut = f32(hi) / 4294967296.0;
        product = product * ut;
        count = count + 1u;
        w = w + 2u;
    }
    if (count > 0u) { count = count - 1u; }
    return vec2<u32>(count, w - start_word);
}

fn pf_binomial(trials: u32, p: f32, x: u32, y: u32, start_word: u32, key: vec2<u32>) -> vec2<u32> {
    if (trials == 0u || p <= 0.0) { return vec2<u32>(0u, 0u); }
    if (p >= 1.0) { return vec2<u32>(trials, 0u); }
    if (trials < 32u) {
        let thresh = u32(p * 4294967296.0);
        var c: u32 = 0u;
        var w: u32 = start_word;
        for (var i = 0u; i < trials; i = i + 1u) {
            let hi = philox_word(x, y, w, key);
            if (hi < thresh) { c = c + 1u; }
            w = w + 2u;
        }
        return vec2<u32>(c, w - start_word);
    }
    if (p < 0.05) {
        let lam = f32(trials) * p;
        let rp = pf_poisson(lam, x, y, start_word, key, sqrt(lam), exp(-lam));
        return vec2<u32>(min(rp.x, trials), rp.y);
    }
    if (p > 0.95) {
        let lam = f32(trials) * (1.0 - p);
        let rp = pf_poisson(lam, x, y, start_word, key, sqrt(lam), exp(-lam));
        return vec2<u32>(trials - min(rp.x, trials), rp.y);
    }
    let g = pf_gaussian(x, y, start_word, key);
    let mean = f32(trials) * p;
    let variance = mean * (1.0 - p);
    let draw = mean + sqrt(variance) * g;
    let dc = u32(round(clamp(draw, 0.0, f32(trials))));
    return vec2<u32>(dc, 12u);
}

fn reflect_index_signed(i: i32, len: u32) -> u32 {
    let n = i32(len);
    if (n <= 1) { return 0u; }
    var x = i;
    while (x < 0 || x >= n) {
        if (x < 0) { x = -x; }
        else { x = 2 * n - 2 - x; }
    }
    return u32(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.root_n) { return; }
    let lx = i % u.root_w;
    let ly = i / u.root_w;
    let gx = u.root_x + i32(lx);
    let gy = u.root_y + i32(ly);
    let rx = reflect_index_signed(gx, u.img_w);
    let ry = reflect_index_signed(gy, u.img_h);
    let d = arena[u.d_off + i];
    let key = vec2<u32>(u.seed_lo ^ u.layer_idx * PF_W0, u.seed_hi ^ u.layer_idx * PF_W1);
    let p = clamp(d / u.d_max, 0.0, 1.0);
    let prob = pow(p, u.gamma);
    let sp = pf_poisson(u.sites_per_cell, rx, ry, 0u, key, u.sqrt_sites, u.knuth_threshold);
    let dev = pf_binomial(sp.x, prob, rx, ry, sp.y, key);
    let fraction = f32(dev.x) / u.sites_per_cell;
    arena[u.d_off + i] = fraction;
}
"#;

/// Micro-structure composition: combines three blurred realized-fraction
/// planes (cloud, crystal, micro_cloud) into the realized fraction plane.
/// Mirrors CPU `particle_field`:
///   out = clamp(clouds + micro_weight * (crystal - micro_cloud), 0.0, 1.05)
/// Full-frame variant — separate scratch buffers per role.
pub const MICRO_MIX: &str = r#"
struct MMU { n: u32, off: u32, cloud_off: u32, crystal_off: u32, micro_off: u32, micro_weight: f32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> out_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> cloud: array<f32>;
@group(0) @binding(2) var<storage, read_write> crystal: array<f32>;
@group(0) @binding(3) var<storage, read_write> micro: array<f32>;
@group(0) @binding(4) var<uniform> u: MMU;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let cl = cloud[i];
    let cr = crystal[i];
    let mc = micro[i];
    let val = cl + u.micro_weight * (cr - mc);
    out_plane[u.off + i] = clamp(val, 0.0, 1.05);
}
"#;

/// Micro-mix — ROI variant: single arena binding + per-role offsets.
pub const MICRO_MIX_ROI: &str = r#"
struct MMU { n: u32, off: u32, cloud_off: u32, crystal_off: u32, micro_off: u32, micro_weight: f32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: MMU;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let cl = arena[u.cloud_off + i];
    let cr = arena[u.crystal_off + i];
    let mc = arena[u.micro_off + i];
    let val = cl + u.micro_weight * (cr - mc);
    arena[u.off + i] = clamp(val, 0.0, 1.05);
}
"#;

/// ROI Scan: densitometric scan → ACEScg (Dmin-normalized) with fused invert.
/// Same body as `SCAN` but addressed against the ROI arena at `dye_base`/`mask_base`
/// and writing only the core rectangle to `pixels`. Fused toe density remap.
pub const SCAN_ROI: &str = r#"
struct U {
    core_x: u32, core_y: u32, core_w: u32, core_h: u32,
    core_n: u32, root_w: u32, root_off_x: u32, root_off_y: u32,
    root_n: u32, img_w: u32, dye_base: u32, mask_base: u32,
    num_emul: u32, scale: f32, flags: u32, _p0: u32,
};
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> arena: array<f32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

const LOG2_10: f32 = 3.3219280948873623;

fn fog2_to_exposure(d2: f32, inv_gamma: f32, fog2: f32) -> f32 {
    var d_eff = 0.0;
    if (d2 > 0.0) {
        if (d2 < fog2) {
            d_eff = (d2 * d2) / (2.0 * fog2);
        } else {
            d_eff = d2 - 0.5 * fog2;
        }
    }
    if (d_eff <= 0.0) {
        return 0.0;
    }
    return exp2(d_eff * inv_gamma) - 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.core_n) { return; }

    let cx = i % u.core_w;
    let cy = i / u.core_w;

    let gx = u.core_x + cx;
    let gy = u.core_y + cy;
    let out_idx = gy * u.img_w + gx;

    let lx = u.root_off_x + cx;
    let ly = u.root_off_y + cy;
    let root_idx = ly * u.root_w + lx;

    let E = u.num_emul;
    let eps_base = 0u;
    let maskeps_base = E * 16u;
    let illum_base = 2u * E * 16u;
    let xbar_base = illum_base + 16u;
    let ybar_base = xbar_base + 16u;
    let zbar_base = ybar_base + 16u;
    let mat_base = zbar_base + 16u;
    let toe_base = mat_base + 9u;

    var dens: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) { dens[k] = 0.0; }
    for (var e = 0u; e < E; e = e + 1u) {
        let raw_f = arena[u.dye_base + e * u.root_n + root_idx];
        let dmax = sc[toe_base + e];
        let inv_gamma = sc[toe_base + E + e];
        let g = max(raw_f, 0.0);
        var di = raw_f;
        if (dmax > 0.0 && g > 0.0) {
            di = clamp(dmax * pow(g, inv_gamma), 0.0, 1.05 * dmax);
        } else if (dmax > 0.0) {
            di = 0.0;
        }
        let dm = arena[u.mask_base + e * u.root_n + root_idx];
        let eb = eps_base + e * 16u;
        let mb = maskeps_base + e * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            dens[k] = dens[k] + di * sc[eb + k] + dm * sc[mb + k];
        }
    }
    var t: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        t[k] = exp2(-dens[k] * LOG2_10) * sc[illum_base + k];
    }
    var X = 0.0; var Y = 0.0; var Z = 0.0;
    for (var k = 0u; k < 15u; k = k + 1u) {
        X = X + (t[k] * sc[xbar_base + k] + t[k + 1u] * sc[xbar_base + k + 1u]) * 10.0;
        Y = Y + (t[k] * sc[ybar_base + k] + t[k + 1u] * sc[ybar_base + k + 1u]) * 10.0;
        Z = Z + (t[k] * sc[zbar_base + k] + t[k + 1u] * sc[zbar_base + k + 1u]) * 10.0;
    }
    var rgb = vec3<f32>(
        (sc[mat_base + 0u] * X + sc[mat_base + 1u] * Y + sc[mat_base + 2u] * Z) * u.scale,
        (sc[mat_base + 3u] * X + sc[mat_base + 4u] * Y + sc[mat_base + 5u] * Z) * u.scale,
        (sc[mat_base + 6u] * X + sc[mat_base + 7u] * Y + sc[mat_base + 8u] * Z) * u.scale
    );

    if ((u.flags & 1u) != 0u) {
        let inv_base = toe_base + 2u * E;
        let inv_dmin = vec3<f32>(sc[inv_base + 10u], sc[inv_base + 11u], sc[inv_base + 12u]);
        let g_val = vec3<f32>(sc[inv_base + 3u], sc[inv_base + 4u], sc[inv_base + 5u]);
        let eps = sc[inv_base + 8u];
        let inv_gamma = sc[inv_base + 13u];
        let fog2 = sc[inv_base + 15u];

        let tc = vec3<f32>(
            max(rgb.x * inv_dmin.x, eps),
            max(rgb.y * inv_dmin.y, eps),
            max(rgb.z * inv_dmin.z, eps),
        );
        let d2 = vec3<f32>(
            -log2(tc.x),
            -log2(tc.y),
            -log2(tc.z),
        );
        let e_scene = vec3<f32>(
            fog2_to_exposure(d2.x, inv_gamma, fog2),
            fog2_to_exposure(d2.y, inv_gamma, fog2),
            fog2_to_exposure(d2.z, inv_gamma, fog2),
        );

        rgb = vec3<f32>(
            g_val.x * e_scene.x,
            g_val.y * e_scene.y,
            g_val.z * e_scene.z,
        );
    }

    pixels[out_idx] = vec4<f32>(rgb.x, rgb.y, rgb.z, 1.0);
}
"#;
