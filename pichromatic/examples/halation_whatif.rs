//! Halation what-if probe: midtone-destination scene + remjet-less CineStill.
//!
//! Section C: current stocks as-is, mid-gray background (steep H&D slope) with a
//!            specular patch; ring bins measured right at the patch edge.
//! Section D: CineStill50D mutated to "remjet removed": AH absorbing layer
//!            neutralized + backing peak reflectance 0.06 -> 0.40.
//!
//! Run: cargo run --release -p pichromatic --example halation_whatif

use pichromatic::film::development::develop;
use pichromatic::film::exposure::expose_with_pitch_and_shutter;
use pichromatic::film::scan::{scan, scanner_calibration_acescg, ScanMode};
use pichromatic::film::spectrum::{SpectralCurve, WavelengthGrid};
use pichromatic::film::stock::{LayerKind, StockId};
use pichromatic::film::types::LatentPlanes;

const WIDTH: usize = 1500;
const HEIGHT: usize = 1000;
const PATCH: usize = 120;
const GAIN_K64: f32 = 12.5 * 64.0; // sunny-16 absolute gain, scene-relative -> absolute

fn make_scene(bg: f32, patch_v: f32) -> Vec<[f32; 3]> {
    let mut rgb = vec![[bg, bg, bg]; WIDTH * HEIGHT];
    let (cx, cy) = (WIDTH / 2, HEIGHT / 2);
    for y in (cy - PATCH / 2)..(cy + PATCH / 2) {
        for x in (cx - PATCH / 2)..(cx + PATCH / 2) {
            rgb[y * WIDTH + x] = [patch_v, patch_v, patch_v];
        }
    }
    rgb.iter()
        .map(|p| [p[0] * GAIN_K64, p[1] * GAIN_K64, p[2] * GAIN_K64])
        .collect()
}

fn run_stock(
    stock: &pichromatic::film::stock::FilmStock,
    shutter: f32,
    scene: &Vec<[f32; 3]>,
) -> Result<Vec<[f32; 3]>, pichromatic::film::FilmError> {
    let pitch = 35000.0 / WIDTH as f32;
    let latent: LatentPlanes =
        expose_with_pitch_and_shutter(scene, WIDTH, HEIGHT, stock, pitch, shutter);
    let dyes = develop(stock, &latent, 1, pitch);
    let calibration = scanner_calibration_acescg(stock, pitch, shutter)?;
    Ok(scan(
        stock,
        &dyes,
        ScanMode::PositiveLinear {
            dmin: calibration.dmin,
            mid: calibration.mid,
        },
        pitch,
    ))
}

fn gaussian_curve_680(peak: f64) -> SpectralCurve {
    let grid = WavelengthGrid::mvp();
    let samples: Vec<f64> = grid
        .wavelengths_nm
        .iter()
        .map(|&l| peak * (-0.5 * ((l - 680.0) / 60.0).powi(2)).exp())
        .collect();
    SpectralCurve::new(grid, samples)
}

fn ring_bins(label: &str, buf: &[[f32; 3]]) {
    let (cx, cy) = (WIDTH / 2, HEIGHT / 2);
    let half = PATCH / 2;
    for (r_in, r_out) in [(0usize, 4usize), (4, 12), (12, 30)] {
        let mut sum = [0.0f64; 3];
        let mut n = 0usize;
        let mut max_dr = 0.0f64;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let dx = (x as isize - cx as isize).unsigned_abs() as usize;
                let dy = (y as isize - cy as isize).unsigned_abs() as usize;
                let cheb = dx.max(dy);
                if cheb >= half + r_in && cheb < half + r_out {
                    for c in 0..3 {
                        sum[c] += buf[y * WIDTH + x][c] as f64;
                    }
                    max_dr = max_dr.max(buf[y * WIDTH + x][0] as f64);
                    n += 1;
                }
            }
        }
        println!(
            "  {label} ring[{r_in:>2},{r_out:>2}): mean=({:.5},{:.5},{:.5}) n={n}",
            sum[0] / n as f64,
            sum[1] / n as f64,
            sum[2] / n as f64,
        );
        let _ = max_dr;
    }
}

fn main() -> Result<(), pichromatic::film::FilmError> {
    println!("=========== SECTION C: midtone bg (0.185), specular 9.0, stocks as-is ===========");
    let scene = make_scene(0.185, 9.0);
    for id in [StockId::Portra400, StockId::CineStill50D] {
        let stock = id.load().unwrap();
        let shutter = 1.0 / stock.box_iso.0;
        let buf = run_stock(&stock, shutter, &scene)?;
        ring_bins(&stock.name, &buf);
    }

    println!("=========== SECTION D: CineStill remjet-removed what-if ===========");
    let mut stock = StockId::CineStill50D.load().unwrap();
    for layer in stock.layers.iter_mut() {
        if layer.kind == LayerKind::Antihalation {
            // remjet gone: backing absorber neutralized
            layer.spectral_sensitivity = Some(SpectralCurve::constant(0.0));
        }
    }
    stock.antihalation.reflectance = gaussian_curve_680(0.40);
    let shutter = 1.0 / stock.box_iso.0;
    let buf = run_stock(&stock, shutter, &scene)?;
    ring_bins("CineStill-remjetless(R=0.40)", &buf);

    // And a milder variant closer to Fresnel-only back reflection (~0.10 net
    // red reflectance without the double AH-layer pass):
    let mut stock2 = StockId::CineStill50D.load().unwrap();
    for layer in stock2.layers.iter_mut() {
        if layer.kind == LayerKind::Antihalation {
            layer.spectral_sensitivity = Some(SpectralCurve::constant(0.0));
        }
    }
    stock2.antihalation.reflectance = gaussian_curve_680(0.10);
    let buf = run_stock(&stock2, shutter, &scene)?;
    ring_bins("CineStill-remjetless(R=0.10)", &buf);
    Ok(())
}
