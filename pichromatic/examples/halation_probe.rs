//! Halation diagnostic probe (original investigation).
//!
//! Section A: per-layer forward vs upward-bounce absorbed fluence for a few
//!            representative pixels (physics level, no spatial blur).
//! Section B: end-to-end film stage (expose→develop→scan, PositiveLinear) on a
//!            synthetic specular-patch scene, halation ON vs OFF, ring metrics.
//!
//! Run: cargo run --release -p pichromatic --example halation_probe

use pichromatic::color::ColorSpaceTag;
use pichromatic::film::constants::{ABSORPTION_SIGMA_SCALE_PER_UM, RADIOMETRIC_SCALE};
use pichromatic::film::exposure::absorption::{
    absorb_stack, absorb_stack_upward, mean_absorbed_fluence,
};
use pichromatic::film::exposure::radiance::{absolute_luminance_gain, sunny16_exposure};
use pichromatic::film::exposure::upsample::upsample_acescg_f32;
use pichromatic::film::stock::{LayerKind, StockId};
use pichromatic::film::units::IsoSpeed;
use pichromatic::film::{process, FilmFormat, FilmOutput, FilmParams};
use pichromatic::image::ImageMetadata;
use pichromatic::pixel::Image;

fn fluence_spectrum(px: [f32; 3]) -> [f32; 16] {
    // Mirrors exposure::pixel_fluence_spectrum with capture_scale = 1.
    let spectrum = upsample_acescg_f32(px);
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        let lambda = 400.0 + 20.0 * i as f64;
        let lambda_factor = ((lambda / 550.0) * RADIOMETRIC_SCALE) as f32;
        out[i] = spectrum[i] * lambda_factor;
    }
    out
}

fn stack_analysis(stock_name: &str, id: StockId, px: [f32; 3]) {
    let stock = id.load().expect("stock loads");
    let spectrum = fluence_spectrum(px);
    let (forward, phi_trans) =
        absorb_stack(&stock.layers, &spectrum, ABSORPTION_SIGMA_SCALE_PER_UM);

    let mut phi_refl = [0.0f32; 16];
    for i in 0..16 {
        phi_refl[i] = phi_trans[i] * stock.antihalation.reflectance.samples[i] as f32;
    }
    let upward = absorb_stack_upward(&stock.layers, &phi_refl, ABSORPTION_SIGMA_SCALE_PER_UM);

    println!("--- {stock_name}  pixel={px:?} ---");
    let idx_of = |lam: f64| ((lam - 400.0) / 20.0).round() as usize;
    for lam in [450.0, 550.0, 650.0, 680.0] {
        let i = idx_of(lam);
        println!(
            "  λ={lam:.0}nm  Φ_in={:9.4}  Φ_trans={:9.4}  R={:.4}  Φ_refl={:9.4}",
            spectrum[i], phi_trans[i], stock.antihalation.reflectance.samples[i], phi_refl[i]
        );
    }
    println!("  layer          forward_Φ    bounce_Φ   bounce/forward");
    for (f, u) in forward.iter().zip(upward.iter()) {
        if !f.produces_latent {
            continue;
        }
        let fw = mean_absorbed_fluence(&f.absorbed);
        let bc = mean_absorbed_fluence(&u.absorbed);
        println!(
            "  {:<12} {:>10.5} {:>10.5}   {:>8.4}",
            "",
            fw,
            bc,
            bc / fw.max(1e-12)
        );
    }
    // Names in stack order for emulsions.
    let names: Vec<&str> = stock
        .layers
        .iter()
        .filter(|l| l.kind == LayerKind::Emulsion)
        .map(|l| l.name)
        .collect();
    println!("  (emulsion order: {names:?})");
}

fn film_stage_run(
    stock: StockId,
    halation: bool,
    width: usize,
    height: usize,
    patch: usize,
    scene_patch: f32,
    scene_bg: f32,
) -> (Image, (usize, usize)) {
    let stock_meta = stock.load().expect("stock loads");
    let e = sunny16_exposure(IsoSpeed(stock_meta.box_iso.0));
    let gain =
        absolute_luminance_gain(e.shutter_seconds as f64, e.f_number as f64, e.iso as f64) as f32;

    let mut rgb = vec![[scene_bg, scene_bg, scene_bg] as [f32; 3]; width * height];
    let cx = width / 2;
    let cy = height / 2;
    for y in (cy - patch / 2)..(cy + patch / 2) {
        for x in (cx - patch / 2)..(cx + patch / 2) {
            rgb[y * width + x] = [scene_patch, scene_patch, scene_patch];
        }
    }
    for px in rgb.iter_mut() {
        *px = [px[0] * gain, px[1] * gain, px[2] * gain];
    }
    let mut img = Image {
        rgb_data: rgb,
        raw_data: std::sync::Arc::from([]),
        metadata: ImageMetadata {
            width,
            height,
            color_space: Some(ColorSpaceTag::AcesCg),
            shutter_seconds: Some(e.shutter_seconds),
            f_number: Some(e.f_number),
            iso: Some(e.iso),
            ..Default::default()
        },
    };
    let params = FilmParams {
        stock,
        film_format: FilmFormat::Film35mm,
        render_width_mm: Some(35.0),
        seed: 1,
        output: FilmOutput::PositiveLinear,
        enable_halation: halation,
        compensate_box_speed: true,
    };
    process(&mut img, &params).expect("film process");
    (img, (cx, cy))
}

fn ring_stats(
    label: &str,
    on: &Image,
    off: &Image,
    cx: usize,
    cy: usize,
    patch: usize,
    r_in: usize,
    r_out: usize,
) {
    let (w, _) = (on.metadata.width, on.metadata.height);
    let half = patch / 2;
    let mut sum_on = [0.0f64; 3];
    let mut sum_off = [0.0f64; 3];
    let mut n = 0usize;
    let mut max_dr = 0.0f64;
    for y in 0..on.metadata.height {
        for x in 0..w {
            let dx = (x as isize - cx as isize).unsigned_abs() as usize;
            let dy = (y as isize - cy as isize).unsigned_abs() as usize;
            let cheb = dx.max(dy);
            if cheb >= half + r_in && cheb < half + r_out {
                let p_on = on.rgb_data[y * w + x];
                let p_off = off.rgb_data[y * w + x];
                for c in 0..3 {
                    sum_on[c] += p_on[c] as f64;
                    sum_off[c] += p_off[c] as f64;
                }
                max_dr = max_dr.max((p_on[0] - p_off[0]) as f64);
                n += 1;
            }
        }
    }
    let m_on: Vec<f64> = sum_on.iter().map(|s| s / n as f64).collect();
    let m_off: Vec<f64> = sum_off.iter().map(|s| s / n as f64).collect();
    let d: Vec<f64> = m_on.iter().zip(&m_off).map(|(a, b)| a - b).collect();
    println!("{label}: ring n={n}");
    println!(
        "  mean ON  R={:.5} G={:.5} B={:.5}",
        m_on[0], m_on[1], m_on[2]
    );
    println!(
        "  mean OFF R={:.5} G={:.5} B={:.5}",
        m_off[0], m_off[1], m_off[2]
    );
    println!(
        "  Δ (ON-OFF) R={:+.5} G={:+.5} B={:+.5}   ΔR/ΔG={:.2} ΔR/ΔB={:.2}  maxΔR={:.5}",
        d[0],
        d[1],
        d[2],
        d[0] / d[1].abs().max(1e-9) * d[0].signum() * d[1].signum(),
        d[0] / d[2].abs().max(1e-9) * d[0].signum() * d[2].signum(),
        max_dr
    );
}

fn patch_stats(label: &str, on: &Image, off: &Image, cx: usize, cy: usize, patch: usize) {
    let w = on.metadata.width;
    let half = patch / 2;
    let mut sum_on = [0.0f64; 3];
    let mut sum_off = [0.0f64; 3];
    let mut n = 0usize;
    for y in (cy - half + 5)..(cy + half - 5) {
        for x in (cx - half + 5)..(cx + half - 5) {
            let p_on = on.rgb_data[y * w + x];
            let p_off = off.rgb_data[y * w + x];
            for c in 0..3 {
                sum_on[c] += p_on[c] as f64;
                sum_off[c] += p_off[c] as f64;
            }
            n += 1;
        }
    }
    println!(
        "{label}: patch mean ON  R={:.4} G={:.4} B={:.4} / OFF R={:.4} G={:.4} B={:.4}",
        sum_on[0] / n as f64,
        sum_on[1] / n as f64,
        sum_on[2] / n as f64,
        sum_off[0] / n as f64,
        sum_off[1] / n as f64,
        sum_off[2] / n as f64,
    );
}

fn main() {
    println!("================ SECTION A: stack physics ================");
    for (name, id) in [
        ("Portra400", StockId::Portra400),
        ("CineStill50D", StockId::CineStill50D),
    ] {
        stack_analysis(name, id, [9.0, 9.0, 9.0]); // white specular
        stack_analysis(name, id, [9.0, 0.0, 0.0]); // red specular
        stack_analysis(name, id, [0.185, 0.185, 0.185]); // mid gray
    }

    println!("================ SECTION B: film stage (PositiveLinear) ================");
    let width = 1500usize;
    let height = 1000usize;
    let patch = 120usize;
    for (name, id) in [
        ("Portra400", StockId::Portra400),
        ("CineStill50D", StockId::CineStill50D),
    ] {
        let (on, (cx, cy)) = film_stage_run(id, true, width, height, patch, 9.0, 0.02);
        let (off, _) = film_stage_run(id, false, width, height, patch, 9.0, 0.02);
        println!(
            "--- {name} (bg=0.02, patch=9.0, pitch={:.2}µm) ---",
            35000.0 / width as f32
        );
        patch_stats(&format!("  {name}"), &on, &off, cx, cy, patch);
        ring_stats(&format!("  {name}"), &on, &off, cx, cy, patch, 10, 60);
        ring_stats(&format!("  {name}"), &on, &off, cx, cy, patch, 60, 200);
    }
}
