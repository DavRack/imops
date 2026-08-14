//! TEMPORARY: why does downsampled 0.5mm differ 35% from direct 35mm?
use pichromatic::film::development::develop;
use pichromatic::film::exposure::expose_with_pitch_shutter_and_scale;
use pichromatic::film::StockId;

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

fn render(scene_um: &[f32], width_um: usize, height: usize, pitch: f32, seed: u64) -> (Vec<f32>, usize) {
    let mut stock = StockId::Portra400.load().unwrap();
    stock.antihalation.reflectance = pichromatic::film::spectrum::SpectralCurve::constant(0.0);
    let width = (width_um as f32 / pitch).ceil() as usize;
    let n = width * height;
    let mut rgb = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            let um = x as f32 * pitch;
            let idx = (um.floor() as usize).min(scene_um.len() - 1);
            let v = scene_um[idx];
            rgb.push([v, v, v]);
        }
    }
    let latent = expose_with_pitch_shutter_and_scale(&rgb, width, height, &stock, pitch, 1.0 / 400.0, 1.0);
    let dyes = develop(&stock, &latent, seed, pitch);
    (mean_rows(&dyes.image_dye, width, height), width)
}

#[test]
#[ignore]
fn probe_path_means() {
    let stock = StockId::Portra400.load().unwrap();
    for (i, (_, l)) in stock.emulsion_layers().enumerate() {
        let m = l.crystal_size.as_ref().map(|d| (d.mu_ln + 0.5 * d.sigma_ln * d.sigma_ln).exp() as f32);
        println!("layer {i} mean_crystal={m:?}um 2x={:?}", m.map(|v| v * 2.0));
    }
    println!(
        "particle_resolution_limit_um = {}",
        stock
            .emulsion_layers()
            .map(|(_, l)| {
                l.crystal_size
                    .as_ref()
                    .map(|d| (d.mu_ln + 0.5 * d.sigma_ln * d.sigma_ln).exp() as f32 * 2.0)
                    .unwrap_or(0.0)
            })
            .fold(0.0f32, f32::max)
    );
    let width_um = 2000usize;
    let mut scene = vec![50.0f32; width_um];
    scene[400] = 4000.0;
    for x in 401..800 {
        scene[x] = 4000.0;
    }
    // Flat dark region far from the edge: compare mean density at each pitch.
    for &pitch in &[11.905f32, 2.646, 2.5, 2.4, 1.0, 0.5, 0.165] {
        let (prof, w) = render(&scene, width_um, 96, pitch, 42);
        let x0 = (1300.0 / pitch) as usize;
        let x1 = (1900.0 / pitch) as usize;
        let flat: Vec<f32> = prof[x0..x1.min(w)].to_vec();
        let mean = flat.iter().map(|&v| v as f64).sum::<f64>() / flat.len() as f64;
        println!("pitch={pitch:.3} flat-dark mean density={mean:.4}");
    }
    // Isolate the reduce() stage: pitch must not matter.
    for &pitch in &[11.905f32, 2.646, 0.165] {
        let mut stock = StockId::Portra400.load().unwrap();
        stock.antihalation.reflectance = pichromatic::film::spectrum::SpectralCurve::constant(0.0);
        let width = (width_um as f32 / pitch).ceil() as usize;
        let height = 96usize;
        let n = width * height;
        let mut rgb = Vec::with_capacity(n);
        for y in 0..height {
            for x in 0..width {
                let um = x as f32 * pitch;
                let idx = (um.floor() as usize).min(scene.len() - 1);
                let v = scene[idx];
                rgb.push([v, v, v]);
            }
        }
        let latent = expose_with_pitch_shutter_and_scale(&rgb, width, height, &stock, pitch, 1.0 / 400.0, 1.0);
        let reduced = pichromatic::film::development::reduction::reduce(&stock, &latent);
        let prof = mean_rows(&reduced.image_dye, width, height);
        let x0 = (1300.0 / pitch) as usize;
        let x1 = (1900.0 / pitch) as usize;
        let flat: Vec<f32> = prof[x0..x1.min(width)].to_vec();
        let mean = flat.iter().map(|&v| v as f64).sum::<f64>() / flat.len() as f64;
        println!("pitch={pitch:.3} REDUCED flat-dark mean density={mean:.4}");
    }
    // Also: reduced density (no grain) should be pitch-invariant by construction.
    // Check grain contribution: std of flat region.
    for &pitch in &[11.905f32, 2.646, 2.4, 1.0, 0.165] {
        let (prof, w) = render(&scene, width_um, 96, pitch, 42);
        let x0 = (1300.0 / pitch) as usize;
        let x1 = (1900.0 / pitch) as usize;
        let flat: Vec<f64> = prof[x0..x1.min(w)].iter().map(|&v| v as f64).collect();
        let mean = flat.iter().sum::<f64>() / flat.len() as f64;
        let var = flat.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / flat.len() as f64;
        println!("pitch={pitch:.3} flat-dark std={:.4} (grain level)", var.sqrt());
    }
}
