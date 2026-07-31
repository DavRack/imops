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
            let od = ec[ob + k];
            let trans = exp(-od);
            let pt = phi[k] * trans;
            absorbed[k] = phi[k] - pt;
            phi[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            // integrated_absorbed = trapz(400..700, Δ20) / 300.
            var acc = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc = acc + 0.5 * (absorbed[k] + absorbed[k + 1u]) * 20.0;
            }
            planes[emul * u.n + i] = acc / 300.0;
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
            let od = ec[ob + k];
            let trans = exp(-od);
            let pt = phi[k] * trans;
            absorbed[k] = phi[k] - pt;
            phi[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            var acc = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc = acc + 0.5 * (absorbed[k] + absorbed[k + 1u]) * 20.0;
            }
            planes[u.planes_base + emul * u.root_n + i] = acc / 300.0;
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
    for (var e = 0u; e < u.num_emul; e = e + 1u) {
        let idx = e * u.n + i;
        planes[idx] = lut_sample(planes[idx], 64u + e * 64u);
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
    for (var e = 0u; e < u.num_emul; e = e + 1u) {
        let idx = u.plane_base + e * u.n + i;
        planes[idx] = lut_sample(planes[idx], 64u + e * 64u);
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
    for (var e = 0u; e < e_count; e = e + 1u) {
        let idx = u.plane_base + e * u.n + i;
        let phi = arena[idx];
        let lut_val = lut_sample(phi, 64u + e * 64u);

        let f = clamp(lut_val, 0.0, 1.0);
        let rev = rc[3u * e_count + e];
        var eff = f;
        if (rev != 0.0) { eff = 1.0 - f; }
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
@group(0) @binding(2) var<storage, read_write> mat: array<f32>;
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

/// White Gaussian noise via SplitMix64 + Irwin–Hall (12 uniforms).
/// Bit-exact 64-bit integer stream matching `development::grain::SplitMix64`.
pub const GRAIN_NOISE: &str = r#"
struct U { width:u32, height:u32, n:u32, base_lo:u32, base_hi:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> noise: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

const GOLD: vec2<u32> = vec2<u32>(0x7F4A7C15u, 0x9E3779B9u);
const SM_C1: vec2<u32> = vec2<u32>(0x1CE4E5B9u, 0xBF58476Du);
const SM_C2: vec2<u32> = vec2<u32>(0x133111EBu, 0x94D049BBu);

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var hi = a.y + b.y;
    if (lo < a.x) { hi = hi + 1u; }
    return vec2<u32>(lo, hi);
}

fn shr64(a: vec2<u32>, s: u32) -> vec2<u32> {
    // s in {27,30,31} < 32.
    let lo = (a.x >> s) | (a.y << (32u - s));
    let hi = a.y >> s;
    return vec2<u32>(lo, hi);
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

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

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul_full(a.x, b.x);
    let lo = ll.x;
    let hi = ll.y + a.x * b.y + a.y * b.x;
    return vec2<u32>(lo, hi);
}

fn splitmix(state: vec2<u32>) -> vec2<u32> {
    var z = state;
    z = mul64(xor64(z, shr64(z, 30u)), SM_C1);
    z = mul64(xor64(z, shr64(z, 27u)), SM_C2);
    z = xor64(z, shr64(z, 31u));
    return z;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let w = u.width;
    let x = i % w;
    let y = i / w;
    let row_seed = add64(vec2<u32>(u.base_lo, u.base_hi), vec2<u32>(y, 0u));
    let base = 12u * x;
    var acc = 0.0;
    for (var j = 1u; j <= 12u; j = j + 1u) {
        let idx = base + j;
        let state = add64(row_seed, mul64(vec2<u32>(idx, 0u), GOLD));
        let z = splitmix(state);
        let zf = f32(z.y) * 4294967296.0 + f32(z.x);
        acc = acc + zf / 18446744073709551616.0;
    }
    noise[i] = acc - 6.0;
}
"#;

/// Partial sum-of-squares for grain variance. Each invocation sums `stride`
/// consecutive samples starting at `gid * stride` (+ `src_off`), writing one
/// partial to `out[out_off + i]`. CPU finishes the reduction in f64.
pub const GRAIN_VAR_PARTIAL: &str = r#"
struct U { n:u32, stride:u32, out_n:u32, src_off:u32, out_off:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.out_n) { return; }
    let start = i * u.stride;
    var acc = 0.0;
    let end = min(start + u.stride, u.n);
    for (var j = start; j < end; j = j + 1u) {
        let v = src[u.src_off + j];
        acc = acc + v * v;
    }
    out[u.out_off + i] = acc;
}
"#;

/// Grain apply on image dye only. Mirrors `development::grain::apply_grain` body.
pub const GRAIN_APPLY: &str = r#"
struct U { n:u32, off:u32, kappa:f32, dmax:f32, norm:f32, noise_off:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> dye: array<f32>;
@group(0) @binding(1) var<storage, read_write> noise: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let d0 = dye[u.off + i];
    let dens = clamp(d0, 0.0, u.dmax);
    let eps_toe = 0.05 * u.dmax;
    let taper = min(dens / (dens + eps_toe), 1.0);
    let sd = taper * sqrt(max(dens * (u.dmax - dens), 0.0));
    let noisy = dens + u.kappa * sd * noise[u.noise_off + i] * u.norm;
    let knee = 0.005 * u.dmax;
    var dd = noisy;
    if (noisy < knee) {
        dd = (knee * knee) / (2.0 * knee - noisy);
    }
    dd = min(dd, u.dmax * 1.05);
    dye[u.off + i] = dd;
}
"#;

/// Densitometric scan → ACEScg (Dmin-normalized). Mirrors `scan::densitometry`.
pub const SCAN: &str = r#"
struct U { n:u32, num_emul:u32, scale:f32, flags:u32 };
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dye: array<f32>;
@group(0) @binding(2) var<storage, read_write> mask: array<f32>;
@group(0) @binding(3) var<storage, read_write> sc: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

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

    var dens: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) { dens[k] = 0.0; }
    for (var e = 0u; e < E; e = e + 1u) {
        let di = dye[e * u.n + i];
        let dm = mask[e * u.n + i];
        let eb = eps_base + e * 16u;
        let mb = maskeps_base + e * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            dens[k] = dens[k] + di * sc[eb + k] + dm * sc[mb + k];
        }
    }
    var t: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        t[k] = pow(10.0, -dens[k]) * sc[illum_base + k];
    }
    var X = 0.0; var Y = 0.0; var Z = 0.0;
    for (var k = 0u; k < 15u; k = k + 1u) {
        X = X + 0.5 * (t[k] * sc[xbar_base + k] + t[k + 1u] * sc[xbar_base + k + 1u]) * 20.0;
        Y = Y + 0.5 * (t[k] * sc[ybar_base + k] + t[k + 1u] * sc[ybar_base + k + 1u]) * 20.0;
        Z = Z + 0.5 * (t[k] * sc[zbar_base + k] + t[k + 1u] * sc[zbar_base + k + 1u]) * 20.0;
    }
    var rgb = vec3<f32>(
        (sc[mat_base + 0u] * X + sc[mat_base + 1u] * Y + sc[mat_base + 2u] * Z) * u.scale,
        (sc[mat_base + 3u] * X + sc[mat_base + 4u] * Y + sc[mat_base + 5u] * Z) * u.scale,
        (sc[mat_base + 6u] * X + sc[mat_base + 7u] * Y + sc[mat_base + 8u] * Z) * u.scale
    );

    if ((u.flags & 1u) != 0u) {
        let inv_base = mat_base + 9u;
        let dmin = vec3<f32>(sc[inv_base], sc[inv_base + 1u], sc[inv_base + 2u]);
        let g_val = vec3<f32>(sc[inv_base + 3u], sc[inv_base + 4u], sc[inv_base + 5u]);
        let slope = sc[inv_base + 6u];
        let gamma_eff = sc[inv_base + 7u];
        let eps = sc[inv_base + 8u];
        let fog_offset = sc[inv_base + 9u];

        let tc = vec3<f32>(
            max(rgb.x / dmin.x, eps),
            max(rgb.y / dmin.y, eps),
            max(rgb.z / dmin.z, eps),
        );
        let d_img = vec3<f32>(-log(tc.x) * 0.43429448, -log(tc.y) * 0.43429448, -log(tc.z) * 0.43429448);
        let d_clamped = max(d_img - vec3<f32>(fog_offset), vec3<f32>(0.0));

        var e_scene = vec3<f32>(0.0);
        if (d_clamped.x > 0.0) { e_scene.x = pow(10.0, d_clamped.x / gamma_eff) - 1.0; } else { e_scene.x = slope * d_clamped.x; }
        if (d_clamped.y > 0.0) { e_scene.y = pow(10.0, d_clamped.y / gamma_eff) - 1.0; } else { e_scene.y = slope * d_clamped.y; }
        if (d_clamped.z > 0.0) { e_scene.z = pow(10.0, d_clamped.z / gamma_eff) - 1.0; } else { e_scene.z = slope * d_clamped.z; }

        rgb = vec3<f32>(
            g_val.x * e_scene.x,
            g_val.y * e_scene.y,
            g_val.z * e_scene.z,
        );
    }

    pixels[i] = vec4<f32>(rgb.x, rgb.y, rgb.z, pixels[i].w);
}
"#;

/// Fused Grain Apply and ROI Scan: single arena for dye/mask/noise (WebGPU-safe).
pub const GRAIN_SCAN_ROI: &str = r#"
struct U {
    core_x: u32, core_y: u32, core_w: u32, core_h: u32,
    core_n: u32, root_w: u32, root_off_x: u32, root_off_y: u32,
    root_n: u32, img_w: u32, dye_base: u32, mask_base: u32,
    num_emul: u32, scale: f32, noise_base: u32, flags: u32,
    emul: array<vec4<f32>, 16>,
};
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> arena: array<f32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn get_kappa(e: u32) -> f32 { return u.emul[e].x; }
fn get_dmax(e: u32) -> f32 { return u.emul[e].y; }
fn get_norm(e: u32) -> f32 { return u.emul[e].z; }

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

    var dens: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) { dens[k] = 0.0; }
    for (var e = 0u; e < E; e = e + 1u) {
        let d0 = arena[u.dye_base + e * u.root_n + root_idx];
        let d_max = get_dmax(e);
        let kappa = get_kappa(e);
        var di = d0;

        if (kappa > 0.0 && d_max > 0.0) {
            let dens_clamped = clamp(d0, 0.0, d_max);
            let eps_toe = 0.05 * d_max;
            let taper = min(dens_clamped / (dens_clamped + eps_toe), 1.0);
            let sd = taper * sqrt(max(dens_clamped * (d_max - dens_clamped), 0.0));
            let n_val = arena[u.noise_base + e * u.root_n + root_idx];
            let noisy = dens_clamped + kappa * sd * n_val * get_norm(e);
            let knee = 0.005 * d_max;
            if (noisy < knee) {
                di = (knee * knee) / (2.0 * knee - noisy);
            } else {
                di = noisy;
            }
            di = min(di, d_max * 1.05);
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
        t[k] = pow(10.0, -dens[k]) * sc[illum_base + k];
    }
    var X = 0.0; var Y = 0.0; var Z = 0.0;
    for (var k = 0u; k < 15u; k = k + 1u) {
        X = X + 0.5 * (t[k] * sc[xbar_base + k] + t[k + 1u] * sc[xbar_base + k + 1u]) * 20.0;
        Y = Y + 0.5 * (t[k] * sc[ybar_base + k] + t[k + 1u] * sc[ybar_base + k + 1u]) * 20.0;
        Z = Z + 0.5 * (t[k] * sc[zbar_base + k] + t[k + 1u] * sc[zbar_base + k + 1u]) * 20.0;
    }
    var rgb = vec3<f32>(
        (sc[mat_base + 0u] * X + sc[mat_base + 1u] * Y + sc[mat_base + 2u] * Z) * u.scale,
        (sc[mat_base + 3u] * X + sc[mat_base + 4u] * Y + sc[mat_base + 5u] * Z) * u.scale,
        (sc[mat_base + 6u] * X + sc[mat_base + 7u] * Y + sc[mat_base + 8u] * Z) * u.scale
    );

    if ((u.flags & 1u) != 0u) {
        let inv_base = mat_base + 9u;
        let dmin = vec3<f32>(sc[inv_base], sc[inv_base + 1u], sc[inv_base + 2u]);
        let g_val = vec3<f32>(sc[inv_base + 3u], sc[inv_base + 4u], sc[inv_base + 5u]);
        let slope = sc[inv_base + 6u];
        let gamma_eff = sc[inv_base + 7u];
        let eps = sc[inv_base + 8u];
        let fog_offset = sc[inv_base + 9u];

        let tc = vec3<f32>(
            max(rgb.x / dmin.x, eps),
            max(rgb.y / dmin.y, eps),
            max(rgb.z / dmin.z, eps),
        );
        let d_img = vec3<f32>(-log(tc.x) * 0.43429448, -log(tc.y) * 0.43429448, -log(tc.z) * 0.43429448);
        let d_clamped = max(d_img - vec3<f32>(fog_offset), vec3<f32>(0.0));

        var e_scene = vec3<f32>(0.0);
        if (d_clamped.x > 0.0) { e_scene.x = pow(10.0, d_clamped.x / gamma_eff) - 1.0; } else { e_scene.x = slope * d_clamped.x; }
        if (d_clamped.y > 0.0) { e_scene.y = pow(10.0, d_clamped.y / gamma_eff) - 1.0; } else { e_scene.y = slope * d_clamped.y; }
        if (d_clamped.z > 0.0) { e_scene.z = pow(10.0, d_clamped.z / gamma_eff) - 1.0; } else { e_scene.z = slope * d_clamped.z; }

        rgb = vec3<f32>(
            g_val.x * e_scene.x,
            g_val.y * e_scene.y,
            g_val.z * e_scene.z,
        );
    }

    pixels[out_idx] = vec4<f32>(rgb.x, rgb.y, rgb.z, 1.0);
}
"#;

/// ROI White Gaussian noise via SplitMix64 + Irwin–Hall (12 uniforms).
/// Adapts GRAIN_NOISE with ROI region parameters and global coordinate reflection.
pub const GRAIN_NOISE_ROI: &str = r#"
struct U {
    root_x: i32, root_y: i32, root_w: u32, root_h: u32,
    root_n: u32, img_w: u32, img_h: u32, base_lo: u32,
    base_hi: u32, dst_off: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

const GOLD: vec2<u32> = vec2<u32>(0x7F4A7C15u, 0x9E3779B9u);
const SM_C1: vec2<u32> = vec2<u32>(0x1CE4E5B9u, 0xBF58476Du);
const SM_C2: vec2<u32> = vec2<u32>(0x133111EBu, 0x94D049BBu);

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    while (x < 0 || x >= len) {
        if (x < 0) { x = -x; }
        else { x = 2 * len - 2 - x; }
    }
    return x;
}

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    var hi = a.y + b.y;
    if (lo < a.x) { hi = hi + 1u; }
    return vec2<u32>(lo, hi);
}

fn shr64(a: vec2<u32>, s: u32) -> vec2<u32> {
    let lo = (a.x >> s) | (a.y << (32u - s));
    let hi = a.y >> s;
    return vec2<u32>(lo, hi);
}

fn xor64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x ^ b.x, a.y ^ b.y);
}

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

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let ll = mul_full(a.x, b.x);
    let lo = ll.x;
    let hi = ll.y + a.x * b.y + a.y * b.x;
    return vec2<u32>(lo, hi);
}

fn splitmix(state: vec2<u32>) -> vec2<u32> {
    var z = state;
    z = mul64(xor64(z, shr64(z, 30u)), SM_C1);
    z = mul64(xor64(z, shr64(z, 27u)), SM_C2);
    z = xor64(z, shr64(z, 31u));
    return z;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.root_n) { return; }
    let lx = i % u.root_w;
    let ly = i / u.root_w;
    let gx = u.root_x + i32(lx);
    let gy = u.root_y + i32(ly);
    let rx = reflect_index(gx, i32(u.img_w));
    let ry = reflect_index(gy, i32(u.img_h));
    let row_seed = add64(vec2<u32>(u.base_lo, u.base_hi), vec2<u32>(u32(ry), 0u));
    let base = 12u * u32(rx);
    var acc = 0.0;
    for (var j = 1u; j <= 12u; j = j + 1u) {
        let idx = base + j;
        let state = add64(row_seed, mul64(vec2<u32>(idx, 0u), GOLD));
        let z = splitmix(state);
        let zf = f32(z.y) * 4294967296.0 + f32(z.x);
        acc = acc + zf / 18446744073709551616.0;
    }
    arena[u.dst_off + i] = acc - 6.0;
}
"#;

/// Copy a core rectangle from root-local scalar arena to full-frame scalar spill.
pub const COPY_SCALAR_CORE_ROI: &str = r#"
struct U {
    core_x: u32, core_y: u32, core_w: u32, core_h: u32,
    core_n: u32, root_w: u32, root_off_x: u32, root_off_y: u32,
    dst_off: u32, src_off: u32, p0: u32, p1: u32,
};
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.core_n) { return; }

    let cx = i % u.core_w;
    let cy = i / u.core_w;

    let lx = u.root_off_x + cx;
    let ly = u.root_off_y + cy;
    let src_idx = u.src_off + ly * u.root_w + lx;

    dst[u.dst_off + i] = src[src_idx];
}
"#;
