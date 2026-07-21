//! Diagnose ColorChecker round-trip ΔE contributors.
use pichromatic::film::colorimetry::{acescg_to_lab, ciede2000};
use pichromatic::film::exposure::radiance::{absolute_luminance_gain, sunny16_exposure};
use pichromatic::film::exposure::upsample::{spectrum_to_acescg_rgb, upsample_acescg};
use pichromatic::film::fixtures::{colorchecker_acescg, colorchecker_image, sample_patch_means};
use pichromatic::film::units::IsoSpeed;
use pichromatic::film::{FilmFormat, FilmOutput, FilmParams, StockId};
use pichromatic::pixel::PixelOps;

fn gain() -> f32 {
    let e = sunny16_exposure(IsoSpeed(200.0));
    absolute_luminance_gain(e.shutter_seconds as f64, e.f_number as f64, e.iso as f64) as f32
}

fn de(a: [f32; 3], b: [f32; 3]) -> f64 {
    ciede2000(acescg_to_lab(a), acescg_to_lab(b))
}

fn summarize(label: &str, deltas: &[f64]) {
    let mut d = deltas.to_vec();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = d[d.len() / 2];
    let max = *d.last().unwrap();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    println!("{label}: median={med:.2} mean={mean:.2} max={max:.2}");
    let mut worst: Vec<(usize, f64)> = deltas.iter().copied().enumerate().collect();
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    print!("  worst: ");
    for (i, v) in worst.iter().take(6) {
        print!("#{i}={v:.1} ");
    }
    println!();
}

fn main() {
    let refs = colorchecker_acescg();
    let g = gain();

    // 1) Spectral upsample → reintegrate only (metamerism / basis rank)
    let mut d_up = Vec::new();
    for (i, &r) in refs.iter().enumerate() {
        let spec = upsample_acescg(r);
        let back = spectrum_to_acescg_rgb(&spec);
        let b = [back[0] as f32, back[1] as f32, back[2] as f32];
        // upsample scales with luminance; normalize by matching Y for fair color check?
        let y0 = r.luminance().max(1e-8);
        let y1 = b.luminance().max(1e-8);
        let bn = [b[0] * y0 / y1, b[1] * y0 / y1, b[2] * y0 / y1];
        d_up.push(de(r, bn));
        let _ = i;
    }
    summarize("upsample→ACEScg (Y-matched)", &d_up);

    // 2) Full PositiveLinear
    let patch = 16;
    let (mut img, _) = colorchecker_image(patch);
    for px in &mut img.rgb_data {
        *px = [px[0] * g, px[1] * g, px[2] * g];
    }
    img.film(&FilmParams {
        stock: StockId::ColorNeg200,
        film_format: FilmFormat::Film35mm,
        seed: 1,
        output: FilmOutput::PositiveLinear,
    })
    .unwrap();
    let means = sample_patch_means(&img, patch);
    let mut d_pos = Vec::new();
    let mut d_pos_chroma = Vec::new();
    for i in 0..24 {
        d_pos.push(de(refs[i], means[i]));
        let mut lo = acescg_to_lab(means[i]);
        let lr = acescg_to_lab(refs[i]);
        lo[0] = lr[0];
        d_pos_chroma.push(ciede2000(lr, lo));
    }
    summarize("PositiveLinear full ΔE00", &d_pos);
    summarize("PositiveLinear chroma-only (L* forced)", &d_pos_chroma);

    // 3) NegativeLinear — not comparable in RGB but show L* drift of positives' gray ramp
    let (mut img_n, _) = colorchecker_image(patch);
    for px in &mut img_n.rgb_data {
        *px = [px[0] * g, px[1] * g, px[2] * g];
    }
    img_n.film(&FilmParams {
        stock: StockId::ColorNeg200,
        film_format: FilmFormat::Film35mm,
        seed: 1,
        output: FilmOutput::NegativeLinear,
    })
    .unwrap();
    let means_n = sample_patch_means(&img_n, patch);
    println!("NegativeLinear gray means (patches 18..24):");
    for i in 18..24 {
        println!("  #{i} neg={:?}", means_n[i]);
    }
    println!("PositiveLinear gray means:");
    for i in 18..24 {
        let y = means[i].luminance();
        println!(
            "  #{i} pos={:?} Y={y:.4} refY={:.4}",
            means[i],
            refs[i].luminance()
        );
    }
}
