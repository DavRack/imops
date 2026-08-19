//! Ignored end-to-end Portra 400 spatial-response diagnostic.
//!
//! Run with:
//! `cargo test --release -p pichromatic --test probe_mtf -- --ignored --nocapture`
//!
//! Kodak E-4050 page 4 reports processed-film B/G/R MTF for daylight exposure
//! and C-41 processing. Source calibration here uses ISO 5-3:2009 Table 4
//! spectral-product weights and same-seed paired uniform fields for DC gain.
//! The model grid truncates the ISO products outside 400–700 nm, so this is a
//! Status-M-like diagnostic, not exact full-band ISO conformance. Raw record
//! density and scanner-linear RGB remain secondary diagnostics.

use pichromatic::film::development::develop;
use pichromatic::film::exposure::expose_with_pitch_shutter_and_scale;
use pichromatic::film::scan::densitometry::scan_to_acescg;
use pichromatic::film::spectrum::SpectralCurve;
use pichromatic::film::stock::{FilmStock, LayerKind};
use pichromatic::film::types::DyePlanes;
use pichromatic::film::StockId;

const PITCH_UM: f32 = 1.0;
const WIDTH: usize = 1000; // 1.000 mm: integer cycles for every frequency below.
const HEIGHT: usize = 96;
const SEEDS: [u64; 4] = [17, 29, 43, 71];
const FREQUENCIES_CYCLES_PER_MM: [usize; 5] = [5, 10, 20, 40, 80];
// Direct vector extraction from Kodak E-4050 page 4, Exposure: Daylight,
// Process: C-41. Rows are 5, 10, 20, 40, 80 cy/mm; columns are B, G, R.
// PDF SHA256: e83ac6775d37832a4cb466892a3e1cf4c88917a6ee93384e59d6924b1cd97e3a.
const E4050_RESPONSE_PERCENT: [[f64; 3]; 5] = [
    [107.3, 108.9, 107.8],
    [111.5, 116.7, 105.8],
    [105.0, 111.3, 90.9],
    [74.5, 74.5, 49.5],
    [46.4, 38.7, 21.3],
];
// ISO 5-3:2009 Table 4 log10 spectral products, normalized to 5.000 peak,
// sampled on the model's truncated 400:20:700 nm grid. Columns are B/G/R.
const ISO5_3_LOG_PRODUCTS: [[f64; 3]; 16] = [
    [-0.397, -6.268, -55.091],
    [4.111, -4.148, -49.891],
    [4.871, -2.028, -44.691],
    [4.955, 0.092, -39.491],
    [4.343, 2.207, -34.291],
    [2.990, 3.804, -29.091],
    [-0.348, 4.626, -23.891],
    [-4.748, 5.000, -18.691],
    [-9.148, 4.818, -13.491],
    [-13.548, 3.915, -8.291],
    [-17.948, 2.239, -3.091],
    [-22.348, -0.130, 2.109],
    [-26.748, -2.530, 5.000],
    [-31.148, -4.930, 4.578],
    [-35.548, -7.330, 3.875],
    [-39.948, -9.730, 3.099],
];
const SOURCE_EXTRACTION_UNCERTAINTY_PP: f64 = 1.61;
// A source-backed calibration tolerance, not a drift pin. The remaining 2.4 pp
// allows the deliberately minimal joint PSF/adjacency family to approximate
// Kodak's processed-film curves without adding an arbitrary output filter.
const MODEL_ALLOWANCE_PP: f64 = 2.4;
const BASE_RADIANCE: f32 = 500.0;
const MODULATION: f32 = 0.05;
const SHUTTER_SECONDS: f32 = 1.0 / 400.0;

fn portra(adjacency: bool) -> FilmStock {
    let mut stock = StockId::Portra400.load().unwrap();
    stock.antihalation.reflectance = SpectralCurve::constant(0.0);
    if !adjacency {
        stock.adjacency_beta = 0.0;
        stock.adjacency_beta_record = 0.0;
        stock.adjacency_beta_cross = 0.0;
    }
    stock
}

fn expose_and_develop(
    stock: &FilmStock,
    frequency: Option<usize>,
    modulation_sign: f32,
    seed: u64,
) -> DyePlanes {
    let mut input = Vec::with_capacity(WIDTH * HEIGHT);
    for _y in 0..HEIGHT {
        for x in 0..WIDTH {
            let wave = frequency.map_or(1.0, |cycles_per_mm| {
                let x_mm = (x as f32 + 0.5) * PITCH_UM / 1000.0;
                (2.0 * std::f32::consts::PI * cycles_per_mm as f32 * x_mm).sin()
            });
            let value = BASE_RADIANCE * (1.0 + modulation_sign * MODULATION * wave);
            input.push([value; 3]);
        }
    }
    let latent = expose_with_pitch_shutter_and_scale(
        &input,
        WIDTH,
        HEIGHT,
        stock,
        PITCH_UM,
        SHUTTER_SECONDS,
        1.0,
    );
    develop(stock, &latent, seed, PITCH_UM)
}

fn record_density(dyes: &DyePlanes, pixel: usize) -> [f32; 3] {
    [
        dyes.image_dye[0][pixel] + dyes.image_dye[1][pixel],
        dyes.image_dye[2][pixel] + dyes.image_dye[3][pixel],
        dyes.image_dye[4][pixel] + dyes.image_dye[5][pixel],
    ]
}

fn status_m_like_density(stock: &FilmStock, dyes: &DyePlanes, pixel: usize) -> [f32; 3] {
    let mut spectral_density = [0.0f64; 16];
    let mut emulsion = 0usize;
    for layer in &stock.layers {
        if layer.kind != LayerKind::Emulsion {
            continue;
        }
        let coupler = layer.coupler.as_ref().unwrap();
        let image_density = dyes.image_dye[emulsion][pixel] as f64;
        let mask_density = dyes.mask_dye[emulsion][pixel] as f64;
        for wavelength in 0..16 {
            spectral_density[wavelength] += image_density * coupler.epsilon.samples[wavelength];
            if let Some(mask) = &coupler.mask_epsilon {
                spectral_density[wavelength] += mask_density * mask.samples[wavelength];
            }
        }
        emulsion += 1;
    }

    std::array::from_fn(|channel| {
        let mut weighted_transmittance = 0.0f64;
        let mut weight_sum = 0.0f64;
        for wavelength in 0..16 {
            let weight = 10.0f64.powf(ISO5_3_LOG_PRODUCTS[wavelength][channel] - 5.0);
            weighted_transmittance += weight * 10.0f64.powf(-spectral_density[wavelength]);
            weight_sum += weight;
        }
        -(weighted_transmittance / weight_sum).log10() as f32
    })
}

fn square_pixel_aperture_mtf(frequency_cycles_per_mm: usize) -> f64 {
    let x = std::f64::consts::PI * frequency_cycles_per_mm as f64 * PITCH_UM as f64 / 1000.0;
    if x == 0.0 {
        1.0
    } else {
        x.sin() / x
    }
}

fn coherent_amplitude<F>(frequency: usize, mut sample: F) -> [f64; 3]
where
    F: FnMut(usize) -> [f32; 3],
{
    let mut sin_sum = [0.0f64; 3];
    let mut cos_sum = [0.0f64; 3];
    let n = (WIDTH * HEIGHT * SEEDS.len()) as f64;
    for seed_i in 0..SEEDS.len() {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let x_mm = (x as f64 + 0.5) * PITCH_UM as f64 / 1000.0;
                let phase = 2.0 * std::f64::consts::PI * frequency as f64 * x_mm;
                let value = sample(seed_i * WIDTH * HEIGHT + y * WIDTH + x);
                for channel in 0..3 {
                    sin_sum[channel] += value[channel] as f64 * phase.sin();
                    cos_sum[channel] += value[channel] as f64 * phase.cos();
                }
            }
        }
    }
    std::array::from_fn(|channel| 2.0 * sin_sum[channel].hypot(cos_sum[channel]) / n)
}

fn mtf(
    adjacency: bool,
) -> (
    [[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()],
    [[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()],
    [[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()],
) {
    let stock = portra(adjacency);
    let mut record = [[0.0; 3]; FREQUENCIES_CYCLES_PER_MM.len()];
    let mut scanner = [[0.0; 3]; FREQUENCIES_CYCLES_PER_MM.len()];
    let mut status_m = [[0.0; 3]; FREQUENCIES_CYCLES_PER_MM.len()];
    let dc_renders: Vec<_> = SEEDS
        .iter()
        .map(|&seed| {
            (
                expose_and_develop(&stock, None, 1.0, seed),
                expose_and_develop(&stock, None, -1.0, seed),
            )
        })
        .collect();
    let mut status_m_dc = [0.0f64; 3];
    for (positive, negative) in &dc_renders {
        for pixel in 0..WIDTH * HEIGHT {
            let positive = status_m_like_density(&stock, positive, pixel);
            let negative = status_m_like_density(&stock, negative, pixel);
            for channel in 0..3 {
                status_m_dc[channel] += 0.5 * (positive[channel] - negative[channel]) as f64;
            }
        }
    }
    for value in &mut status_m_dc {
        *value = value.abs() / (SEEDS.len() * WIDTH * HEIGHT) as f64;
        assert!(*value > 0.0 && value.is_finite());
    }

    for (frequency_i, &frequency) in FREQUENCIES_CYCLES_PER_MM.iter().enumerate() {
        let renders: Vec<_> = SEEDS
            .iter()
            .map(|&seed| {
                let positive = expose_and_develop(&stock, Some(frequency), 1.0, seed);
                let negative = expose_and_develop(&stock, Some(frequency), -1.0, seed);
                let scan_positive = scan_to_acescg(&stock, &positive, PITCH_UM);
                let scan_negative = scan_to_acescg(&stock, &negative, PITCH_UM);
                (positive, negative, scan_positive, scan_negative)
            })
            .collect();
        record[frequency_i] = coherent_amplitude(frequency, |index| {
            let seed_i = index / (WIDTH * HEIGHT);
            let pixel = index % (WIDTH * HEIGHT);
            let positive = record_density(&renders[seed_i].0, pixel);
            let negative = record_density(&renders[seed_i].1, pixel);
            std::array::from_fn(|channel| 0.5 * (positive[channel] - negative[channel]))
        });
        scanner[frequency_i] = coherent_amplitude(frequency, |index| {
            let seed_i = index / (WIDTH * HEIGHT);
            let pixel = index % (WIDTH * HEIGHT);
            std::array::from_fn(|channel| {
                0.5 * (renders[seed_i].2[pixel][channel] - renders[seed_i].3[pixel][channel])
            })
        });
        let status_amplitude = coherent_amplitude(frequency, |index| {
            let seed_i = index / (WIDTH * HEIGHT);
            let pixel = index % (WIDTH * HEIGHT);
            let positive = status_m_like_density(&stock, &renders[seed_i].0, pixel);
            let negative = status_m_like_density(&stock, &renders[seed_i].1, pixel);
            std::array::from_fn(|channel| 0.5 * (positive[channel] - negative[channel]))
        });
        // Center-binned Poisson counts omit subpixel positions. Include the
        // known square pixel-aperture sinc in the diagnostic, not production.
        let aperture = square_pixel_aperture_mtf(frequency);
        status_m[frequency_i] = std::array::from_fn(|channel| {
            status_amplitude[channel] / status_m_dc[channel] * aperture
        });
    }
    (record, scanner, status_m)
}

fn covariance_and_nps() -> ([[f64; 3]; 3], [[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()]) {
    let stock = portra(true);
    let scans: Vec<_> = SEEDS
        .iter()
        .map(|&seed| {
            let dyes = expose_and_develop(&stock, None, 0.0, seed);
            scan_to_acescg(&stock, &dyes, PITCH_UM)
        })
        .collect();

    let mut means = vec![[0.0f64; 3]; SEEDS.len()];
    for (seed_i, scan) in scans.iter().enumerate() {
        for pixel in scan {
            for channel in 0..3 {
                means[seed_i][channel] += pixel[channel] as f64 / scan.len() as f64;
            }
        }
    }

    let mut covariance = [[0.0f64; 3]; 3];
    let sample_count = (SEEDS.len() * WIDTH * HEIGHT) as f64;
    for (seed_i, scan) in scans.iter().enumerate() {
        for pixel in scan {
            let residual: [f64; 3] =
                std::array::from_fn(|channel| pixel[channel] as f64 - means[seed_i][channel]);
            for a in 0..3 {
                for b in 0..3 {
                    covariance[a][b] += residual[a] * residual[b] / sample_count;
                }
            }
        }
    }

    let mut nps = [[0.0f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()];
    for (bin_i, &frequency) in FREQUENCIES_CYCLES_PER_MM.iter().enumerate() {
        for (seed_i, scan) in scans.iter().enumerate() {
            for y in 0..HEIGHT {
                let mut re = [0.0f64; 3];
                let mut im = [0.0f64; 3];
                for x in 0..WIDTH {
                    let x_mm = x as f64 * PITCH_UM as f64 / 1000.0;
                    let phase = 2.0 * std::f64::consts::PI * frequency as f64 * x_mm;
                    let pixel = scan[y * WIDTH + x];
                    for channel in 0..3 {
                        let residual = pixel[channel] as f64 - means[seed_i][channel];
                        re[channel] += residual * phase.cos();
                        im[channel] -= residual * phase.sin();
                    }
                }
                // One-sided row periodogram, units scanner-response²·mm.
                let scale = 2.0 * (PITCH_UM as f64 / 1000.0) / WIDTH as f64;
                for channel in 0..3 {
                    nps[bin_i][channel] += scale * (re[channel].powi(2) + im[channel].powi(2))
                        / (SEEDS.len() * HEIGHT) as f64;
                }
            }
        }
    }
    (covariance, nps)
}

fn print_mtf(
    label: &str,
    record: &[[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()],
    scanner: &[[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()],
    status_m: &[[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()],
) {
    println!("\n{label}");
    println!(
        "cy/mm | record B/G/R (norm {}) | scanner R/G/B (norm {}) | Status-M-like B/G/R (DC norm) | aperture",
        FREQUENCIES_CYCLES_PER_MM[0], FREQUENCIES_CYCLES_PER_MM[0]
    );
    for i in 0..FREQUENCIES_CYCLES_PER_MM.len() {
        let record_norm: [f64; 3] = std::array::from_fn(|c| record[i][c] / record[0][c]);
        let scanner_norm: [f64; 3] = std::array::from_fn(|c| scanner[i][c] / scanner[0][c]);
        println!(
            "{:>5} | {:>7.4} {:>7.4} {:>7.4} | {:>7.4} {:>7.4} {:>7.4} | {:>7.4} {:>7.4} {:>7.4} | {:>7.4}",
            FREQUENCIES_CYCLES_PER_MM[i],
            record_norm[0],
            record_norm[1],
            record_norm[2],
            scanner_norm[0],
            scanner_norm[1],
            scanner_norm[2],
            status_m[i][0],
            status_m[i][1],
            status_m[i][2],
            square_pixel_aperture_mtf(FREQUENCIES_CYCLES_PER_MM[i]),
        );
    }
}

fn report_and_check_e4050(status_m: &[[f64; 3]; FREQUENCIES_CYCLES_PER_MM.len()]) {
    let tolerance_pp = SOURCE_EXTRACTION_UNCERTAINTY_PP + MODEL_ALLOWANCE_PP;
    let mut worst = (0.0f64, 0usize, 0usize);
    println!(
        "\nKodak E-4050 Status-M-like/DC response (measured / source / error, percentage points):"
    );
    for (frequency_i, &frequency) in FREQUENCIES_CYCLES_PER_MM.iter().enumerate() {
        let measured = status_m[frequency_i].map(|response| response * 100.0);
        let error: [f64; 3] = std::array::from_fn(|channel| {
            measured[channel] - E4050_RESPONSE_PERCENT[frequency_i][channel]
        });
        println!(
            "{frequency:>5} | B {:>5.1}/{:>5.1}/{:+5.1}  G {:>5.1}/{:>5.1}/{:+5.1}  R {:>5.1}/{:>5.1}/{:+5.1}",
            measured[0],
            E4050_RESPONSE_PERCENT[frequency_i][0],
            error[0],
            measured[1],
            E4050_RESPONSE_PERCENT[frequency_i][1],
            error[1],
            measured[2],
            E4050_RESPONSE_PERCENT[frequency_i][2],
            error[2],
        );
        for channel in 0..3 {
            if error[channel].abs() > worst.0 {
                worst = (error[channel].abs(), frequency_i, channel);
            }
        }
    }
    let (error, frequency_i, channel) = worst;
    assert!(
        error <= tolerance_pp,
        "E-4050 Status-M-like/DC max error {error:.3} pp at {} cy/mm channel {channel} exceeds {tolerance_pp:.2} pp calibration tolerance",
        FREQUENCIES_CYCLES_PER_MM[frequency_i],
    );
}

#[test]
#[ignore = "release diagnostic; prints measured MTF/NPS"]
fn probe_portra_end_to_end_mtf_and_nps() {
    let (record, scanner, status_m) = mtf(true);
    let (record_no_adj, scanner_no_adj, status_m_no_adj) = mtf(false);
    print_mtf(
        "production realized population + adjacency",
        &record,
        &scanner,
        &status_m,
    );
    print_mtf(
        "diagnostic adjacency disabled",
        &record_no_adj,
        &scanner_no_adj,
        &status_m_no_adj,
    );
    report_and_check_e4050(&status_m);

    for values in record.iter().chain(scanner.iter()) {
        assert!(values.iter().all(|value| value.is_finite()));
    }
    assert!(record[0].iter().all(|&value| value > 0.0));
    assert!(scanner[0].iter().all(|&value| value > 0.0));

    let (covariance, nps) = covariance_and_nps();
    println!("\nflat scanner-linear covariance:");
    for row in covariance {
        println!("{:>12.6e} {:>12.6e} {:>12.6e}", row[0], row[1], row[2]);
    }
    println!("flat scanner-linear correlation:");
    for a in 0..3 {
        println!(
            "{:>8.4} {:>8.4} {:>8.4}",
            covariance[a][0] / (covariance[a][a] * covariance[0][0]).sqrt(),
            covariance[a][1] / (covariance[a][a] * covariance[1][1]).sqrt(),
            covariance[a][2] / (covariance[a][a] * covariance[2][2]).sqrt(),
        );
    }
    println!("flat scanner-linear one-sided row NPS [response²·mm]:");
    for (i, row) in nps.iter().enumerate() {
        println!(
            "{:>5} cy/mm | {:>12.6e} {:>12.6e} {:>12.6e}",
            FREQUENCIES_CYCLES_PER_MM[i], row[0], row[1], row[2]
        );
    }
    assert!(nps
        .iter()
        .flatten()
        .all(|value| value.is_finite() && *value >= 0.0));
}
