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
@group(0) @binding(2) var<storage, read_write> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    loop {
        if (x < 0) { x = -x; }
        else if (x >= len) { x = 2 * len - 2 - x; }
        else { return x; }
    }
    return 0;
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
    tmp[y * w + x] = acc;
}
"#;

/// Vertical separable-blur pass.
/// Used when radius > [`super::BLUR_TILED_MAX_RADIUS`] (tiled path cannot cover full kernel).
pub const BLUR_V: &str = r#"
struct U { width:u32, height:u32, n:u32, radius:u32, src_off:u32, dst_off:u32, p0:u32, p1:u32 };
@group(0) @binding(0) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    loop {
        if (x < 0) { x = -x; }
        else if (x >= len) { x = 2 * len - 2 - x; }
        else { return x; }
    }
    return 0;
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
        acc = acc + tmp[u32(yy) * w + x] * ker[k];
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
@group(0) @binding(2) var<storage, read_write> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> tile: array<f32, 512>;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    loop {
        if (x < 0) { x = -x; }
        else if (x >= len) { x = 2 * len - 2 - x; }
        else { return x; }
    }
    return 0;
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
    tmp[y * u.width + u32(x)] = acc;
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
@group(0) @binding(2) var<storage, read_write> ker: array<f32>;
@group(0) @binding(3) var<uniform> u: U;

var<workgroup> tile: array<f32, 512>;

fn reflect_index(i: i32, len: i32) -> i32 {
    if (len == 1) { return 0; }
    var x = i;
    loop {
        if (x < 0) { x = -x; }
        else if (x >= len) { x = 2 * len - 2 - x; }
        else { return x; }
    }
    return 0;
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
        tile[t] = tmp[u32(yy) * w + x];
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
        phi[k] = s * ec[57u + k];
    }

    let off_od = 73u;
    let off_pl = 73u + u.num_layers * 16u;
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
            m = rc[2u * e_count + e] * (1.0 - clamp(d / dmax, 0.0, 1.0));
        }
        mask[e * u.n + i] = m;
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
        var total = 0.0;
        for (var s = 0u; s < e_count; s = s + 1u) {
            total = total + mat[s * e_count + j] * diffused[s * u.n + i];
        }
        dye[j * u.n + i] = dye[j * u.n + i] * exp(-total);
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
    let eps_toe = 0.02 * u.dmax;
    let taper = min(dens / (dens + eps_toe), 1.0);
    let sd = taper * sqrt(max(dens * (u.dmax - dens), 0.0));
    var dd = dens + u.kappa * sd * noise[u.noise_off + i] * u.norm;
    dd = clamp(dd, 0.0, u.dmax * 1.05);
    dye[u.off + i] = dd;
}
"#;

/// Densitometric scan → ACEScg (Dmin-normalized). Mirrors `scan::densitometry`.
pub const SCAN: &str = r#"
struct U { n:u32, num_emul:u32, scale:f32, p0:u32 };
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
    let r = sc[mat_base + 0u] * X + sc[mat_base + 1u] * Y + sc[mat_base + 2u] * Z;
    let g = sc[mat_base + 3u] * X + sc[mat_base + 4u] * Y + sc[mat_base + 5u] * Z;
    let b = sc[mat_base + 6u] * X + sc[mat_base + 7u] * Y + sc[mat_base + 8u] * Z;
    pixels[i] = vec4<f32>(r * u.scale, g * u.scale, b * u.scale, pixels[i].w);
}
"#;

/// PositiveLinear invert. Mirrors `scan::invert::invert_negative`.
pub const INVERT: &str = r#"
struct U { n:u32, p0:u32, p1:u32, p2:u32 };
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> ic: array<f32>;
@group(0) @binding(2) var<uniform> u: U;

fn soft_inv(inv: f32, shoulder: f32) -> f32 {
    let v = max(inv, 0.0);
    let s = max(shoulder, 1e-6);
    return v / (1.0 + v / s);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 16776960u;
    if (i >= u.n) { return; }
    let dmin = vec3<f32>(ic[0], ic[1], ic[2]);
    let g = vec3<f32>(ic[3], ic[4], ic[5]);
    let shoulder = vec3<f32>(ic[6], ic[7], ic[8]);
    let wr = ic[9]; let wg = ic[10]; let wb = ic[11];
    let chroma_keep = ic[12];
    let fade_y = ic[13];
    let eps = ic[14];

    let px = pixels[i];
    let t = vec3<f32>(
        clamp(px.x / dmin.x, eps, 1.0),
        clamp(px.y / dmin.y, eps, 1.0),
        clamp(px.z / dmin.z, eps, 1.0),
    );
    let inv = vec3<f32>(
        max(1.0 / t.x - 1.0, 0.0),
        max(1.0 / t.y - 1.0, 0.0),
        max(1.0 / t.z - 1.0, 0.0),
    );
    let raw = vec3<f32>(
        max(g.x * soft_inv(inv.x, shoulder.x), 0.0),
        max(g.y * soft_inv(inv.y, shoulder.y), 0.0),
        max(g.z * soft_inv(inv.z, shoulder.z), 0.0),
    );
    let y = max(wr * raw.x + wg * raw.y + wb * raw.z, 0.0);
    var k = 0.0;
    if (y <= 0.0) {
        k = 0.0;
    } else if (y >= fade_y) {
        k = chroma_keep;
    } else {
        k = chroma_keep * (y / fade_y);
    }
    pixels[i] = vec4<f32>(
        max(y + k * (raw.x - y), 0.0),
        max(y + k * (raw.y - y), 0.0),
        max(y + k * (raw.z - y), 0.0),
        px.w,
    );
}
"#;
