//! WGSL compute shaders for the GPU film path. Each mirrors a CPU stage.
//!
//! All storage bindings are declared `read_write` or `read` to match the bind-group layout
//! produced by [`crate::gpu::GpuContext`].

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
    let r = min(radius, 128);
    let x0 = i32(wid.x * 256u);
    let lx = i32(lid.x);
    let base = u.src_off + y * u.width;

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

/// Arena blur H: single storage buffer.
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

/// Arena blur V: single storage buffer.
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

/// Arena tiled blur H.
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

/// Arena tiled blur V.
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

/// Expose: ACEScg pixel -> forward absorbed fluence planes + upward bounce absorbed fluence planes.
/// Mirrors `exposure::expose_with_pitch_shutter_and_scale` (Beer-Lambert forward + upward walk).
pub const EXPOSE: &str = r#"
struct U { width:u32, height:u32, n:u32, num_layers:u32, num_emul:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> planes: array<f32>;
@group(0) @binding(2) var<storage, read_write> bounce: array<f32>;
@group(0) @binding(3) var<storage, read> ec: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let px = pixels[i];
    let r = px.x; let g = px.y; let b = px.z;

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
    let off_refl = off_pl + u.num_layers;

    var emul = 0u;
    for (var l = 0u; l < u.num_layers; l = l + 1u) {
        var absorbed: array<f32, 16>;
        let ob = off_od + l * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let trans = ec[ob + k];
            let pt = phi[k] * trans;
            absorbed[k] = phi[k] - pt;
            phi[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            var acc = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc = acc + (absorbed[k] + absorbed[k + 1u]) * 10.0;
            }
            planes[emul * u.n + i] = acc / 300.0;
            emul = emul + 1u;
        }
    }

    var phi_up: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        phi_up[k] = phi[k] * ec[off_refl + k];
    }

    var emul_up = u.num_emul;
    for (var step = 0u; step < u.num_layers; step = step + 1u) {
        let l = u.num_layers - 1u - step;
        var absorbed_up: array<f32, 16>;
        let ob = off_od + l * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let trans = ec[ob + k];
            let pt = phi_up[k] * trans;
            absorbed_up[k] = phi_up[k] - pt;
            phi_up[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            emul_up = emul_up - 1u;
            var acc_up = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc_up = acc_up + (absorbed_up[k] + absorbed_up[k + 1u]) * 10.0;
            }
            bounce[emul_up * u.n + i] = acc_up / 300.0;
        }
    }
}
"#;

/// ROI Expose: reads full-frame image with border reflection, writes forward planes + upward bounce planes into arena.
pub const EXPOSE_ROI: &str = r#"
struct U {
    root_x: i32, root_y: i32, root_w: u32, root_h: u32,
    img_w: u32, img_h: u32, root_n: u32, planes_base: u32,
    bounce_base: u32, num_layers: u32, num_emul: u32, p0: u32,
};
@group(0) @binding(0) var<storage, read> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> arena: array<f32>;
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
    let off_refl = off_pl + u.num_layers;

    var emul = 0u;
    for (var l = 0u; l < u.num_layers; l = l + 1u) {
        var absorbed: array<f32, 16>;
        let ob = off_od + l * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let trans = ec[ob + k];
            let pt = phi[k] * trans;
            absorbed[k] = phi[k] - pt;
            phi[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            var acc = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc = acc + (absorbed[k] + absorbed[k + 1u]) * 10.0;
            }
            arena[u.planes_base + emul * u.root_n + i] = acc / 300.0;
            emul = emul + 1u;
        }
    }

    var phi_up: array<f32, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) {
        phi_up[k] = phi[k] * ec[off_refl + k];
    }

    var emul_up = u.num_emul;
    for (var step = 0u; step < u.num_layers; step = step + 1u) {
        let l = u.num_layers - 1u - step;
        var absorbed_up: array<f32, 16>;
        let ob = off_od + l * 16u;
        for (var k = 0u; k < 16u; k = k + 1u) {
            let trans = ec[ob + k];
            let pt = phi_up[k] * trans;
            absorbed_up[k] = phi_up[k] - pt;
            phi_up[k] = pt;
        }
        if (ec[off_pl + l] != 0.0) {
            emul_up = emul_up - 1u;
            var acc_up = 0.0;
            for (var k = 0u; k < 15u; k = k + 1u) {
                acc_up = acc_up + (absorbed_up[k] + absorbed_up[k + 1u]) * 10.0;
            }
            arena[u.bounce_base + emul_up * u.root_n + i] = acc_up / 300.0;
        }
    }
}
"#;

/// Irradiation mixture: (1 - w) * core + w * (0.6235 * tail1 + 0.3765 * tail2).
pub const IRRADIATION_MIX: &str = r#"
struct U {
    n: u32,
    plane_off: u32,
    weight: f32,
    p0: u32,
};
@group(0) @binding(0) var<storage, read_write> plane: array<f32>;
@group(0) @binding(1) var<storage, read> core: array<f32>;
@group(0) @binding(2) var<storage, read> tail1: array<f32>;
@group(0) @binding(3) var<storage, read> tail2: array<f32>;
@group(0) @binding(4) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let c = core[i];
    let t1 = tail1[i];
    let t2 = tail2[i];
    let tail = 0.6235 * t1 + 0.3765 * t2;
    plane[u.plane_off + i] = (1.0 - u.weight) * c + u.weight * tail;
}
"#;

/// Irradiation mixture for ROI arena.
pub const IRRADIATION_MIX_ROI: &str = r#"
struct U {
    n: u32,
    plane_off: u32,
    core_off: u32,
    tail1_off: u32,
    tail2_off: u32,
    weight: f32,
    p0: u32,
    p1: u32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let c = arena[u.core_off + i];
    let t1 = arena[u.tail1_off + i];
    let t2 = arena[u.tail2_off + i];
    let tail = 0.6235 * t1 + 0.3765 * t2;
    arena[u.plane_off + i] = (1.0 - u.weight) * c + u.weight * tail;
}
"#;

/// Multi-bounce halation accumulation: `acc = init ? w*b : acc + w*b`.
pub const HALATION_ACCUM: &str = r#"
struct U { n:u32, w:f32, init:u32, p0:u32 };
@group(0) @binding(0) var<storage, read_write> acc: array<f32>;
@group(0) @binding(1) var<storage, read> blurred: array<f32>;
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

/// Multi-bounce halation accumulation for ROI arena.
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

/// Add halation accumulation to emulsion plane: `plane += acc`.
pub const HALATION_ADD_EMUL: &str = r#"
struct U { n:u32, plane_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> plane: array<f32>;
@group(0) @binding(1) var<storage, read> acc: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    plane[u.plane_off + i] = plane[u.plane_off + i] + acc[i];
}
"#;

/// Add halation accumulation to emulsion plane for ROI arena.
pub const HALATION_ADD_EMUL_ROI: &str = r#"
struct U { n:u32, plane_off:u32, acc_off:u32, p0:u32 };
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    arena[u.plane_off + i] = arena[u.plane_off + i] + arena[u.acc_off + i];
}
"#;

/// Capture LUT sample (absorbed fluence -> developable fraction).
pub const LUT: &str = r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
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
        let idx = e * u.n + i;
        planes[idx] = lut_sample(planes[idx] * lc[eta_base + e], 64u + e * 64u);
    }
}
"#;

/// ROI Capture LUT.
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

/// Reduce developable fraction -> image/mask optical density.
/// D = min(d_max, d_max * f^(1/gamma) + FOG_OFFSET).
pub const REDUCE: &str = r#"
struct U { n:u32, num_emul:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read> planes: array<f32>;
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
        let f = clamp(planes[e * u.n + i], 0.0, 1.0);
        let rev = rc[3u * e_count + e];
        var f_clamped = f;
        if (rev != 0.0) { f_clamped = 1.0 - f; }
        let dmax = rc[e];
        let ig = rc[e_count + e];
        let fog = rc[5u * e_count + e];
        var d = min(dmax, dmax * pow(f_clamped, ig) + fog);
        dye[e * u.n + i] = d;
        var m = 0.0;
        if (rc[4u * e_count + e] != 0.0) {
            m = rc[2u * e_count + e];
        }
        mask[e * u.n + i] = m;
    }
}
"#;

/// Fused ROI LUT and Reduce.
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
        var f_clamped = f;
        if (rev != 0.0) { f_clamped = 1.0 - f; }
        let dmax = rc[e];
        let ig = rc[e_count + e];
        let fog = rc[5u * e_count + e];
        var d = min(dmax, dmax * pow(f_clamped, ig) + fog);
        arena[u.dye_base + e * u.n + i] = d;
        var m = 0.0;
        if (rc[4u * e_count + e] != 0.0) {
            m = rc[2u * e_count + e];
        }
        arena[u.mask_base + e * u.n + i] = m;
    }
}
"#;

// ─── Particle-field grain (Philox4x32-10 + per-pixel Poisson/Binomial) ───────

pub const PARTICLE_FIELD: &str = r#"
const PF_M0: u32 = 0xD2511F53u;
const PF_M1: u32 = 0xCD9E8D57u;
const PF_W0: u32 = 0x9E3779B9u;
const PF_W1: u32 = 0xBB67AE85u;

struct PFU {
    n: u32, width: u32, height: u32, in_off: u32, out_off: u32,
    d_max: f32, sites_per_cell: f32, key_lo: u32, key_hi: u32,
    sqrt_sites: f32, knuth_threshold: f32, p0: u32,
};
@group(0) @binding(0) var<storage, read> in_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_plane: array<f32>;
@group(0) @binding(2) var<uniform> u: PFU;

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
    var sum_hi: u32 = 0u;
    var sum_lo: u32 = 0u;
    for (var j = 0u; j < 12u; j = j + 1u) {
        let w = philox_word(x, y, start_word + j, key);
        sum_hi = sum_hi + (w >> 12u);
        sum_lo = sum_lo + (w & 0xFFFu);
    }
    let total_hi = sum_hi + (sum_lo >> 12u);
    let total_lo = sum_lo & 0xFFFu;
    return (f32(total_hi) - 6291456.0 + f32(total_lo) / 4096.0) / 1048576.0;
}

fn pf_poisson(lambda: f32, x: u32, y: u32, start_word: u32, key: vec2<u32>, sqrt_lambda: f32, threshold: f32) -> vec2<u32> {
    if (!(lambda > 0.0)) { return vec2<u32>(0u, 0u); }
    if (lambda >= 16.0) {
        let g = pf_gaussian(x, y, start_word, key);
        let draw = lambda + sqrt_lambda * g;
        let n = u32(round(max(draw, 0.0)));
        return vec2<u32>(n, 12u);
    }
    var product: f32 = 1.0;
    var count: u32 = 0u;
    var w: u32 = start_word;
    for (var iter = 0u; iter < 100u; iter = iter + 1u) {
        if (product <= threshold) { break; }
        let hi = philox_word(x, y, w, key);
        let lo = philox_word(x, y, w + 1u, key);
        let ut = (f32(hi) + f32(lo) / 4294967296.0) / 4294967296.0;
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
        let thresh_hi = u32(p * 4294967296.0);
        let rem = p * 4294967296.0 - f32(thresh_hi);
        let thresh_lo = u32(rem * 4294967296.0);
        var c: u32 = 0u;
        var w: u32 = start_word;
        for (var i = 0u; i < trials; i = i + 1u) {
            let hi = philox_word(x, y, w, key);
            let lo = philox_word(x, y, w + 1u, key);
            if (hi < thresh_hi || (hi == thresh_hi && lo < thresh_lo)) { c = c + 1u; }
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
    let d = in_plane[u.in_off + i];
    let key = vec2<u32>(u.key_lo, u.key_hi);
    let p = clamp(d / u.d_max, 0.0, 1.0);
    let sp = pf_poisson(u.sites_per_cell, x, y, 0u, key, u.sqrt_sites, u.knuth_threshold);
    let dev = pf_binomial(sp.x, p, x, y, sp.y, key);
    let fraction = f32(dev.x) / u.sites_per_cell;
    out_plane[u.out_off + i] = fraction;
}
"#;

pub const PARTICLE_FIELD_ROI: &str = r#"
const PF_M0: u32 = 0xD2511F53u;
const PF_M1: u32 = 0xCD9E8D57u;
const PF_W0: u32 = 0x9E3779B9u;
const PF_W1: u32 = 0xBB67AE85u;

struct PFRU {
    root_x: i32, root_y: i32, root_w: u32, root_h: u32,
    root_n: u32, img_w: u32, img_h: u32, in_off: u32, out_off: u32,
    d_max: f32, sites_per_cell: f32, key_lo: u32, key_hi: u32,
    sqrt_sites: f32, knuth_threshold: f32, p0: u32,
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
    var sum_hi: u32 = 0u;
    var sum_lo: u32 = 0u;
    for (var j = 0u; j < 12u; j = j + 1u) {
        let w = philox_word(x, y, start_word + j, key);
        sum_hi = sum_hi + (w >> 12u);
        sum_lo = sum_lo + (w & 0xFFFu);
    }
    let total_hi = sum_hi + (sum_lo >> 12u);
    let total_lo = sum_lo & 0xFFFu;
    return (f32(total_hi) - 6291456.0 + f32(total_lo) / 4096.0) / 1048576.0;
}

fn pf_poisson(lambda: f32, x: u32, y: u32, start_word: u32, key: vec2<u32>, sqrt_lambda: f32, threshold: f32) -> vec2<u32> {
    if (!(lambda > 0.0)) { return vec2<u32>(0u, 0u); }
    if (lambda >= 16.0) {
        let g = pf_gaussian(x, y, start_word, key);
        let draw = lambda + sqrt_lambda * g;
        let n = u32(round(max(draw, 0.0)));
        return vec2<u32>(n, 12u);
    }
    var product: f32 = 1.0;
    var count: u32 = 0u;
    var w: u32 = start_word;
    for (var iter = 0u; iter < 100u; iter = iter + 1u) {
        if (product <= threshold) { break; }
        let hi = philox_word(x, y, w, key);
        let lo = philox_word(x, y, w + 1u, key);
        let ut = (f32(hi) + f32(lo) / 4294967296.0) / 4294967296.0;
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
        let thresh_hi = u32(p * 4294967296.0);
        let rem = p * 4294967296.0 - f32(thresh_hi);
        let thresh_lo = u32(rem * 4294967296.0);
        var c: u32 = 0u;
        var w: u32 = start_word;
        for (var i = 0u; i < trials; i = i + 1u) {
            let hi = philox_word(x, y, w, key);
            let lo = philox_word(x, y, w + 1u, key);
            if (hi < thresh_hi || (hi == thresh_hi && lo < thresh_lo)) { c = c + 1u; }
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
    let d = arena[u.in_off + i];
    let key = vec2<u32>(u.key_lo, u.key_hi);
    let p = clamp(d / u.d_max, 0.0, 1.0);
    let sp = pf_poisson(u.sites_per_cell, rx, ry, 0u, key, u.sqrt_sites, u.knuth_threshold);
    let dev = pf_binomial(sp.x, p, rx, ry, sp.y, key);
    let fraction = f32(dev.x) / u.sites_per_cell;
    arena[u.out_off + i] = fraction;
}
"#;

/// Scale blurred developable fraction by d_max back to optical density.
pub const SCALE_DMAX: &str = r#"
struct U {
    n: u32,
    in_off: u32,
    out_off: u32,
    d_max: f32,
};
@group(0) @binding(0) var<storage, read> in_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_plane: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    out_plane[u.out_off + i] = in_plane[u.in_off + i] * u.d_max;
}
"#;

/// Scale blurred developable fraction for ROI arena.
pub const SCALE_DMAX_ROI: &str = r#"
struct U {
    n: u32,
    in_off: u32,
    out_off: u32,
    d_max: f32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    arena[u.out_off + i] = arena[u.in_off + i] * u.d_max;
}
"#;

/// Two-component physical adjacency: D_i' = max(0, D_i + sum_j M_ex_ij*(D_j - blur_ex(D_j)) + sum_j M_dir_ij*(D_j - blur_dir(D_j))).
pub const ADJACENCY_TWO_COMPONENT: &str = r#"
struct U {
    n: u32,
    num_emul: u32,
    ex_active: u32,
    dir_active: u32,
};
@group(0) @binding(0) var<storage, read_write> dye: array<f32>;
@group(0) @binding(1) var<storage, read> diffused_ex: array<f32>;
@group(0) @binding(2) var<storage, read> diffused_dir: array<f32>;
@group(0) @binding(3) var<storage, read> mat_ex: array<f32>;
@group(0) @binding(4) var<storage, read> mat_dir: array<f32>;
@group(0) @binding(5) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let E = u.num_emul;
    var orig_dye: array<f32, 8>;
    for (var s = 0u; s < E; s = s + 1u) {
        orig_dye[s] = dye[s * u.n + i];
    }
    for (var j = 0u; j < E; j = j + 1u) {
        var sum_ex = 0.0;
        if (u.ex_active != 0u) {
            for (var s = 0u; s < E; s = s + 1u) {
                let m = mat_ex[j * E + s];
                if (abs(m) > 1e-8) {
                    let diff = orig_dye[s] - diffused_ex[s * u.n + i];
                    sum_ex = sum_ex + m * diff;
                }
            }
        }
        var sum_dir = 0.0;
        if (u.dir_active != 0u) {
            for (var s = 0u; s < E; s = s + 1u) {
                let m = mat_dir[j * E + s];
                if (abs(m) > 1e-8) {
                    let diff = orig_dye[s] - diffused_dir[s * u.n + i];
                    sum_dir = sum_dir + m * diff;
                }
            }
        }
        let updated = orig_dye[j] + sum_ex + sum_dir;
        dye[j * u.n + i] = max(updated, 0.0);
    }
}
"#;

/// Two-component physical adjacency for ROI arena.
pub const ADJACENCY_TWO_COMPONENT_ROI: &str = r#"
struct U {
    n: u32,
    num_emul: u32,
    dye_base: u32,
    diff_ex_base: u32,
    diff_dir_base: u32,
    ex_active: u32,
    dir_active: u32,
    p0: u32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read> mat_ex: array<f32>;
@group(0) @binding(2) var<storage, read> mat_dir: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let E = u.num_emul;
    var orig_dye: array<f32, 8>;
    for (var s = 0u; s < E; s = s + 1u) {
        orig_dye[s] = arena[u.dye_base + s * u.n + i];
    }
    for (var j = 0u; j < E; j = j + 1u) {
        var sum_ex = 0.0;
        if (u.ex_active != 0u) {
            for (var s = 0u; s < E; s = s + 1u) {
                let m = mat_ex[j * E + s];
                if (abs(m) > 1e-8) {
                    let diff = orig_dye[s] - arena[u.diff_ex_base + s * u.n + i];
                    sum_ex = sum_ex + m * diff;
                }
            }
        }
        var sum_dir = 0.0;
        if (u.dir_active != 0u) {
            for (var s = 0u; s < E; s = s + 1u) {
                let m = mat_dir[j * E + s];
                if (abs(m) > 1e-8) {
                    let diff = orig_dye[s] - arena[u.diff_dir_base + s * u.n + i];
                    sum_dir = sum_dir + m * diff;
                }
            }
        }
        let updated = orig_dye[j] + sum_ex + sum_dir;
        arena[u.dye_base + j * u.n + i] = max(updated, 0.0);
    }
}
"#;

/// Densitometric scan -> ACEScg linear RGB planes (Dmin normalized).
pub const SCAN_TO_ACESCG: &str = r#"
struct U {
    n: u32,
    num_emul: u32,
    scale: f32,
    p0: u32,
};
@group(0) @binding(0) var<storage, read> dye: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> scan_r: array<f32>;
@group(0) @binding(3) var<storage, read_write> scan_g: array<f32>;
@group(0) @binding(4) var<storage, read_write> scan_b: array<f32>;
@group(0) @binding(5) var<storage, read> sc: array<f32>;
@group(0) @binding(6) var<uniform> u: U;

const LOG2_10: f32 = 3.3219280948873623;

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
        t[k] = exp2(-dens[k] * LOG2_10) * sc[illum_base + k];
    }
    var X = 0.0; var Y = 0.0; var Z = 0.0;
    for (var k = 0u; k < 15u; k = k + 1u) {
        X = X + (t[k] * sc[xbar_base + k] + t[k + 1u] * sc[xbar_base + k + 1u]) * 10.0;
        Y = Y + (t[k] * sc[ybar_base + k] + t[k + 1u] * sc[ybar_base + k + 1u]) * 10.0;
        Z = Z + (t[k] * sc[zbar_base + k] + t[k + 1u] * sc[zbar_base + k + 1u]) * 10.0;
    }
    scan_r[i] = (sc[mat_base + 0u] * X + sc[mat_base + 1u] * Y + sc[mat_base + 2u] * Z) * u.scale;
    scan_g[i] = (sc[mat_base + 3u] * X + sc[mat_base + 4u] * Y + sc[mat_base + 5u] * Z) * u.scale;
    scan_b[i] = (sc[mat_base + 6u] * X + sc[mat_base + 7u] * Y + sc[mat_base + 8u] * Z) * u.scale;
}
"#;

/// Densitometric scan for ROI arena -> root ACEScg linear RGB planes.
pub const SCAN_TO_ACESCG_ROI: &str = r#"
struct U {
    root_n: u32,
    num_emul: u32,
    dye_base: u32,
    mask_base: u32,
    r_base: u32,
    g_base: u32,
    b_base: u32,
    scale: f32,
};
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<storage, read> sc: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

const LOG2_10: f32 = 3.3219280948873623;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.root_n) { return; }
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
        let di = arena[u.dye_base + e * u.root_n + i];
        let dm = arena[u.mask_base + e * u.root_n + i];
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
    arena[u.r_base + i] = (sc[mat_base + 0u] * X + sc[mat_base + 1u] * Y + sc[mat_base + 2u] * Z) * u.scale;
    arena[u.g_base + i] = (sc[mat_base + 3u] * X + sc[mat_base + 4u] * Y + sc[mat_base + 5u] * Z) * u.scale;
    arena[u.b_base + i] = (sc[mat_base + 6u] * X + sc[mat_base + 7u] * Y + sc[mat_base + 8u] * Z) * u.scale;
}
"#;

/// Final technical invert: converts linear scanned ACEScg to output format.
pub const INVERT: &str = r#"
struct InvertU {
    n: u32,
    mode: u32,
    inv_gamma: f32,
    scanner_s_curve: f32,
    eps: f32,
    p0: u32,
    p1: u32,
    p2: u32,
    inv_dmin: vec4<f32>,
    exponent: vec4<f32>,
    gain: vec4<f32>,
};
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> scan_r: array<f32>;
@group(0) @binding(2) var<storage, read> scan_g: array<f32>;
@group(0) @binding(3) var<storage, read> scan_b: array<f32>;
@group(0) @binding(4) var<uniform> u: InvertU;

fn apply_scanner_scurve(x: f32, s: f32) -> f32 {
    if (s <= 0.0) { return x; }
    let BASE_GAMMA: f32 = 2.38;
    let K: f32 = 0.218;
    let Y_MAX: f32 = 0.940;
    var gamma = BASE_GAMMA;
    if (s > 1.0) { gamma = BASE_GAMMA * s; }
    let k_gamma = pow(K, gamma);
    let x_pos = max(x, 0.0);
    let x_gamma = pow(x_pos, gamma);
    let f_x = Y_MAX * x_gamma / (x_gamma + k_gamma);
    if (s <= 1.0) {
        return (1.0 - s) * x + s * f_x;
    } else {
        return f_x;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    var r = scan_r[i];
    var g = scan_g[i];
    var b = scan_b[i];

    if (u.mode == 1u) {
        let tr = clamp(r * u.inv_dmin.x, u.eps, 1.0);
        let tg = clamp(g * u.inv_dmin.y, u.eps, 1.0);
        let tb = clamp(b * u.inv_dmin.z, u.eps, 1.0);
        r = 1.0 - pow(tr, u.exponent.x);
        g = 1.0 - pow(tg, u.exponent.y);
        b = 1.0 - pow(tb, u.exponent.z);
    } else if (u.mode == 2u) {
        let tr = max(r * u.inv_dmin.x, u.eps);
        let tg = max(g * u.inv_dmin.y, u.eps);
        let tb = max(b * u.inv_dmin.z, u.eps);
        var er = max(pow(tr, -u.inv_gamma) - 1.0, 0.0) * u.gain.x;
        var eg = max(pow(tg, -u.inv_gamma) - 1.0, 0.0) * u.gain.y;
        var eb = max(pow(tb, -u.inv_gamma) - 1.0, 0.0) * u.gain.z;
        if (u.scanner_s_curve > 0.0) {
            er = apply_scanner_scurve(er, u.scanner_s_curve);
            eg = apply_scanner_scurve(eg, u.scanner_s_curve);
            eb = apply_scanner_scurve(eb, u.scanner_s_curve);
        }
        r = er; g = eg; b = eb;
    }

    pixels[i] = vec4<f32>(r, g, b, 1.0);
}
"#;

/// Invert for ROI core extraction into output buffer.
pub const INVERT_ROI: &str = r#"
struct InvertRoiU {
    core_x: u32, core_y: u32, core_w: u32, core_h: u32,
    core_n: u32, root_w: u32, root_off_x: u32, root_off_y: u32,
    root_n: u32, img_w: u32, r_base: u32, g_base: u32,
    b_base: u32, mode: u32, inv_gamma: f32, scanner_s_curve: f32,
    eps: f32, p0: u32, p1: u32, p2: u32,
    inv_dmin: vec4<f32>,
    exponent: vec4<f32>,
    gain: vec4<f32>,
};
@group(0) @binding(0) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> arena: array<f32>;
@group(0) @binding(2) var<uniform> u: InvertRoiU;

fn apply_scanner_scurve(x: f32, s: f32) -> f32 {
    if (s <= 0.0) { return x; }
    let BASE_GAMMA: f32 = 2.38;
    let K: f32 = 0.218;
    let Y_MAX: f32 = 0.940;
    var gamma = BASE_GAMMA;
    if (s > 1.0) { gamma = BASE_GAMMA * s; }
    let k_gamma = pow(K, gamma);
    let x_pos = max(x, 0.0);
    let x_gamma = pow(x_pos, gamma);
    let f_x = Y_MAX * x_gamma / (x_gamma + k_gamma);
    if (s <= 1.0) {
        return (1.0 - s) * x + s * f_x;
    } else {
        return f_x;
    }
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

    var r = arena[u.r_base + root_idx];
    var g = arena[u.g_base + root_idx];
    var b = arena[u.b_base + root_idx];

    if (u.mode == 1u) {
        let tr = clamp(r * u.inv_dmin.x, u.eps, 1.0);
        let tg = clamp(g * u.inv_dmin.y, u.eps, 1.0);
        let tb = clamp(b * u.inv_dmin.z, u.eps, 1.0);
        r = 1.0 - pow(tr, u.exponent.x);
        g = 1.0 - pow(tg, u.exponent.y);
        b = 1.0 - pow(tb, u.exponent.z);
    } else if (u.mode == 2u) {
        let tr = max(r * u.inv_dmin.x, u.eps);
        let tg = max(g * u.inv_dmin.y, u.eps);
        let tb = max(b * u.inv_dmin.z, u.eps);
        var er = max(pow(tr, -u.inv_gamma) - 1.0, 0.0) * u.gain.x;
        var eg = max(pow(tg, -u.inv_gamma) - 1.0, 0.0) * u.gain.y;
        var eb = max(pow(tb, -u.inv_gamma) - 1.0, 0.0) * u.gain.z;
        if (u.scanner_s_curve > 0.0) {
            er = apply_scanner_scurve(er, u.scanner_s_curve);
            eg = apply_scanner_scurve(eg, u.scanner_s_curve);
            eb = apply_scanner_scurve(eb, u.scanner_s_curve);
        }
        r = er; g = eg; b = eb;
    }

    output[out_idx] = vec4<f32>(r, g, b, 1.0);
}
"#;
