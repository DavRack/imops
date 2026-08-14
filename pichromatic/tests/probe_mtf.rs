//! TEMPORARY diagnostic probe (delete after review): physical MTF + edge
//! profile at several render pitches for the SAME physical scene.
//! v3: luminance 50 (dark) / 4000 (bright), 96 rows, smooth edge, adj/DIR
//! toggle, downsample-fine-to-35mm invariance comparison.
use pichromatic::film::blur::gaussian_blur_separable;
use pichromatic::film::development::develop;
use pichromatic::film::exposure::expose_with_pitch_shutter_and_scale;
use pichromatic::film::StockId;

const L_DARK: f32 = 50.0;
const L_BRIGHT: f32 = 4000.0;

fn mean_rows(planes: &[Vec<f32>], width: usize, height: usize) -> Vec<f32> {
    let mut acc = vec![0.0f64; width];
    for plane in planes {
        for y in 0..height {
            for x in 0..width {
                acc[x] += plane[y * width + x] as f64;
            }
        }
    }
    let n = (planes.len() * height) as f64;
    acc.iter().map(|&v| (v / n) as f32).collect()
}

fn grating_contrast(profile: &[f32], pitch_um: f32, x0_um: f32, x1_um: f32, f_cperum: f32) -> f32 {
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    let mut syy = 0.0f64;
    let mut sy = 0.0f64;
    let mut n = 0usize;
    let p0 = (x0_um / pitch_um) as usize;
    let p1 = ((x1_um / pitch_um) as usize).min(profile.len());
    for p in p0..p1 {
        let x = p as f32 * pitch_um;
        let s = (2.0 * std::f32::consts::PI * f_cperum * x).sin() as f64;
        let c = (2.0 * std::f32::consts::PI * f_cperum * x).cos() as f64;
        let v = profile[p] as f64;
        sxx += s * s;
        sxy += s * v;
        syy += c * v;
        sy += v;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    let mean = sy / n as f64;
    let a = sxy / sxx;
    let b = syy / sxx;
    let amp = (a * a + b * b).sqrt();
    (amp / mean) as f32
}

fn edge_rise_um(profile: &[f32], pitch_um: f32, x0_um: f32, x1_um: f32) -> f32 {
    let mut smooth = profile.to_vec();
    gaussian_blur_separable(&mut smooth, profile.len(), 1, 3.0);
    let p0 = (x0_um / pitch_um) as usize;
    let p1 = ((x1_um / pitch_um) as usize).min(smooth.len());
    let seg = &smooth[p0..p1];
    let lo = seg.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = seg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let t10 = lo + 0.1 * (hi - lo);
    let t90 = lo + 0.9 * (hi - lo);
    let mut x10 = None;
    let mut x90 = None;
    for (i, &v) in seg.iter().enumerate() {
        if x10.is_none() && v >= t10 {
            x10 = Some(p0 + i);
        }
        if x90.is_none() && v >= t90 {
            x90 = Some(p0 + i);
        }
    }
    match (x10, x90) {
        (Some(a), Some(b)) => (b as f32 - a as f32) * pitch_um,
        _ => f32::NAN,
    }
}

fn render_scene(
    scene_um: &[f32],
    width_um: usize,
    height: usize,
    pitch_um: f32,
    seed: u64,
    disable_adj_dir: bool,
) -> (Vec<f32>, usize) {
    let mut stock = StockId::Portra400.load().unwrap();
    stock.antihalation.reflectance = pichromatic::film::spectrum::SpectralCurve::constant(0.0);
    if disable_adj_dir {
        stock.adjacency_beta = 0.0;
        stock.dir_inhibition_matrix = vec![vec![0.0; 6]; 6];
    }
    let width = (width_um as f32 / pitch_um).ceil() as usize;
    let n = width * height;
    let mut rgb: Vec<[f32; 3]> = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            let um = x as f32 * pitch_um;
            let idx = (um.floor() as usize).min(scene_um.len() - 1);
            let v = scene_um[idx];
            rgb.push([v, v, v]);
        }
    }
    let latent =
        expose_with_pitch_shutter_and_scale(&rgb, width, height, &stock, pitch_um, 1.0 / 400.0, 1.0);
    let dyes = develop(&stock, &latent, seed, pitch_um);
    let profile = mean_rows(&dyes.image_dye, width, height);
    (profile, width)
}

#[test]
#[ignore]
fn probe_mtf_vs_pitch() {
    let width_um = 2000usize;
    let mut scene = vec![L_DARK; width_um];
    scene[400] = L_BRIGHT; // step edge at x=400..401 µm
    for x in 401..800 {
        scene[x] = L_BRIGHT;
    }
    let freqs = [0.004f32, 0.008, 0.016, 0.032];
    let bands: Vec<(f32, f32)> = freqs
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let w = 300.0;
            let x0 = 800.0 + i as f32 * w;
            (x0, x0 + w)
        })
        .collect();

    let pitches = [11.905f32, 2.646, 2.5, 2.4, 1.0, 0.5, 0.165];
    let labels = ["35mm", "8mm", "thr2.5", "2.4um", "1mm", "0.5mm", "0.5mm2"];

    for &no_adj in &[false, true] {
        println!("\n=== adjacency/DIR disabled={no_adj} ===");
        let mut fine_profile: Option<(Vec<f32>, usize)> = None;
        for (pi, &pitch) in pitches.iter().enumerate() {
            let (profile, width) = render_scene(&scene, width_um, 96, pitch, 42, no_adj);
            let edge = edge_rise_um(&profile, pitch, 380.0, 720.0);
            let mut contrasts = Vec::new();
            for (i, (x0, x1)) in bands.iter().enumerate() {
                contrasts.push(grating_contrast(&profile, pitch, *x0, *x1, freqs[i]));
            }
            println!(
                "{:>8} {:>8.3} | edge={:>8.2}um | MTF: {:>6.3} {:>6.3} {:>6.3} {:>6.3}",
                labels[pi],
                pitch,
                edge,
                contrasts[0],
                contrasts[1],
                contrasts[2],
                contrasts[3]
            );
            if pi == 6 {
                fine_profile = Some((profile, width));
            }
        }
        if let Some((fine, fine_w)) = fine_profile {
            let ratio = 11.905 / 0.1653;
            let coarse_w = (width_um as f32 / 11.905) as usize;
            let mut ds = vec![0.0f32; coarse_w];
            for x in 0..coarse_w {
                let x0 = (x as f32 * ratio) as usize;
                let x1 = ((x as f32 + 1.0) * ratio) as usize;
                let mut s = 0.0f64;
                for p in x0..x1.min(fine_w) {
                    s += fine[p] as f64;
                }
                ds[x] = (s / (x1 - x0).max(1) as f64) as f32;
            }
            let (coarse35, _) = render_scene(&scene, width_um, 96, 11.905, 42, no_adj);
            let mut diff = 0.0f64;
            let mut norm = 0.0f64;
            for x in 0..coarse_w {
                diff += ((ds[x] - coarse35[x]) as f64).abs();
                norm += coarse35[x] as f64;
            }
            println!(
                "downsample 0.5mm->35mm vs direct 35mm: mean|diff|/mean35 = {:.4}",
                diff / norm.max(1e-9)
            );
        }
    }
}
