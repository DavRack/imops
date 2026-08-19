//! Stock-specific irradiation spread and halation from wide support bounce.
//!
//! A wide multi-bounce PSF from `psf_halation_um` is applied
//!    to each emulsion layer's physical upward-reflected bounce field B_e,
//!    derived via bidirectional Beer-Lambert absorption through the film stack.
//!    Gated by backing reflectance (`enable_halation` zeroes R).

use crate::film::blur::{exponential_blur_separable, gaussian_blur_separable};
use crate::film::spectrum::{SpectralCurve, WavelengthGrid};
use crate::film::stock::{AntihalationModel, FilmStock};

/// Representative bands for spectral reflectance sampling (nm).
const HALATION_BANDS_NM: [f64; 4] = [450.0, 550.0, 650.0, 700.0];

/// Apply stock-specific irradiation spread, then wide backing bounce in place.
///
/// `absorbed_planes` are forward absorbed emulsion planes top→bottom (e.g. blue, green, red).
/// `bounce_planes` are unblurred upward absorbed bounce fields $B_e$ in the same layer order.
pub fn apply_spatial_exposure_effects(
    absorbed_planes: &mut [Vec<f32>],
    bounce_planes: &[Vec<f32>],
    width: usize,
    height: usize,
    stock: &FilmStock,
    pixel_pitch_um: f32,
) {
    if absorbed_planes.is_empty() {
        return;
    }
    let n = width * height;
    for plane in absorbed_planes.iter() {
        assert_eq!(plane.len(), n);
    }
    for plane in bounce_planes.iter() {
        assert_eq!(plane.len(), n);
    }

    // This is one component of an effective joint record-response fit with the
    // D/4 realized-cloud footprint and realization-only adjacency. The split is
    // not independently identified irradiation physics or exact Status-M
    // calibration, and is not Kodak's final-film MTF applied wholesale here.
    if let Some(response) = stock.irradiation_response {
        let core_sigma_px = response.core_sigma_um / pixel_pitch_um.max(1e-6);
        let tail_decay_px = response.tail_decay_um / pixel_pitch_um.max(1e-6);
        for (emulsion, plane) in absorbed_planes.iter_mut().enumerate() {
            let weight = response.tail_weight_bgr[emulsion / 2];
            let mut core = plane.clone();
            let mut tail = plane.clone();
            gaussian_blur_separable(&mut core, width, height, core_sigma_px);
            exponential_blur_separable(&mut tail, width, height, tail_decay_px);
            for ((value, core), tail) in plane.iter_mut().zip(core).zip(tail) {
                *value = (1.0 - weight) * core + weight * tail;
            }
        }
    }

    // Wide support bounce with multi-bounce geometry.
    let sigma_wide = sigma_px_from_um(stock.antihalation.psf_halation_um, pixel_pitch_um);
    if sigma_wide < 1e-3 {
        return;
    }

    // Check if backing reflectance is non-zero
    let max_r = stock
        .antihalation
        .reflectance
        .samples
        .iter()
        .copied()
        .fold(0.0f64, f64::max) as f32;
    if max_r <= 0.0 {
        return;
    }

    // Multi-bounce back reflection geometry (N=3 bounces with decay rho=0.5)
    const N_BOUNCES: usize = 3;
    const RHO: f32 = 0.5;
    let mut decay_weights = [0.0f32; N_BOUNCES];
    let mut sum_decay = 0.0f32;
    for k in 0..N_BOUNCES {
        decay_weights[k] = RHO.powi(k as i32);
        sum_decay += decay_weights[k];
    }
    for k in 0..N_BOUNCES {
        decay_weights[k] /= sum_decay;
    }

    for (e, plane) in absorbed_planes.iter_mut().enumerate() {
        if e >= bounce_planes.len() {
            continue;
        }
        let mut halation_plane = vec![0.0f32; n];
        for (k, &wk) in decay_weights.iter().enumerate() {
            let bounce_k_sigma = sigma_wide * ((k + 1) as f32).sqrt();
            let mut b_k = bounce_planes[e].clone();
            gaussian_blur_separable(&mut b_k, width, height, bounce_k_sigma);
            for p in 0..n {
                halation_plane[p] += wk * b_k[p];
            }
        }
        for (p, &h) in plane.iter_mut().zip(halation_plane.iter()) {
            *p += h;
        }
    }
}

/// Effective backing reflectance integrated over an emulsion layer's spectral sensitivity curve.
pub fn effective_reflectance(reflectance: &SpectralCurve, sensitivity: &SpectralCurve) -> f32 {
    let norm = sensitivity.integrate();
    if norm <= 1e-9 {
        return reflectance.evaluate(650.0) as f32;
    }
    (reflectance.integrate_against(sensitivity) / norm) as f32
}

/// Apply additive wide-halation to a planar absorbed-fluence field.
pub fn apply_halation_plane(
    plane: &mut [f32],
    width: usize,
    height: usize,
    weight: f32,
    sigma_px: f32,
) {
    if weight <= 0.0 || sigma_px < 1e-3 {
        return;
    }
    let mut scattered = plane.to_vec();
    gaussian_blur_separable(&mut scattered, width, height, sigma_px);
    for (p, s) in plane.iter_mut().zip(scattered.iter()) {
        *p += weight * s;
    }
}

/// Weight from antihalation backing reflectance at λ.
pub fn reflectance_at(model: &AntihalationModel, wavelength_nm: f64) -> f32 {
    model.reflectance.evaluate(wavelength_nm) as f32
}

/// Sample antihalation reflectance at the 4 representative spectral bands.
pub fn band_reflectances(model: &AntihalationModel) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    for (i, &lambda) in HALATION_BANDS_NM.iter().enumerate() {
        out[i] = reflectance_at(model, lambda);
    }
    out
}

/// Convert 4-band antihalation reflectance into a 16-sample spectral curve via linear interpolation.
pub fn spectral_reflectance_curve(model: &AntihalationModel) -> [f32; 16] {
    let band_w = band_reflectances(model);
    interpolate_band_weights(band_w)
}

/// Map µm PSF to pixels via pitch.
pub fn sigma_px_from_um(sigma_um: f32, pixel_pitch_um: f32) -> f32 {
    (sigma_um / pixel_pitch_um.max(1e-6)).max(0.0)
}

/// Interpolate a 4-band weight to the MVP 16-λ grid (linear in λ).
pub fn interpolate_band_weights(band_weights: [f32; 4]) -> [f32; 16] {
    let grid = WavelengthGrid::mvp();
    let mut out = [0.0f32; 16];
    for (i, &lambda) in grid.wavelengths_nm.iter().enumerate() {
        if lambda <= HALATION_BANDS_NM[0] {
            out[i] = band_weights[0];
            continue;
        }
        if lambda >= HALATION_BANDS_NM[3] {
            out[i] = band_weights[3];
            continue;
        }
        for b in 0..3 {
            if lambda >= HALATION_BANDS_NM[b] && lambda <= HALATION_BANDS_NM[b + 1] {
                let t = ((lambda - HALATION_BANDS_NM[b])
                    / (HALATION_BANDS_NM[b + 1] - HALATION_BANDS_NM[b]))
                    as f32;
                out[i] = band_weights[b] * (1.0 - t) + band_weights[b + 1] * t;
                break;
            }
        }
    }
    out
}

pub fn band_wavelengths() -> [f64; 4] {
    HALATION_BANDS_NM
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::spectrum::SpectralCurve;

    fn elevated_ah() -> AntihalationModel {
        AntihalationModel {
            reflectance: SpectralCurve::constant(0.5),
            psf_halation_um: 20.0,
        }
    }

    #[test]
    fn halation_impulse_halo() {
        let width = 65;
        let height = 65;
        let mut plane = vec![0.0f32; width * height];
        let cx = width / 2;
        let cy = height / 2;
        plane[cy * width + cx] = 1.0;
        let ah = elevated_ah();
        let w = reflectance_at(&ah, 650.0);
        let sigma = sigma_px_from_um(ah.psf_halation_um, 5.0);
        apply_halation_plane(&mut plane, width, height, w, sigma);
        let mut prev = f32::MAX;
        for r in [2usize, 5, 10, 15] {
            let mut sum = 0.0f32;
            let mut n = 0usize;
            for dy in -(r as isize)..=(r as isize) {
                let dx = (r as isize * r as isize - dy * dy) as f32;
                if dx < 0.0 {
                    continue;
                }
                let dx = dx.sqrt().round() as isize;
                for &sx in &[-dx, dx] {
                    let x = cx as isize + sx;
                    let y = cy as isize + dy;
                    if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                        sum += plane[y as usize * width + x as usize];
                        n += 1;
                    }
                }
            }
            let mean = sum / n.max(1) as f32;
            assert!(
                mean <= prev + 1e-5,
                "radial mean should fall: r={r} mean={mean} prev={prev}"
            );
            prev = mean;
        }
        assert!(plane[cy * width + cx + 3] > 1e-6);
    }

    #[test]
    fn halation_multi_bounce_adds_to_absorbed_planes() {
        use crate::film::stock::{EmulsionLayer, FilmStock, LayerKind};
        use crate::film::units::{IsoSpeed, Microns};

        let width = 41;
        let height = 41;
        let n = width * height;
        let mut planes = vec![vec![0.0f32; n]; 3];
        let mut bounce_planes = vec![vec![0.0f32; n]; 3];
        let cx = width / 2;
        let cy = height / 2;
        // Red bounce layer has an impulse
        bounce_planes[2][cy * width + cx] = 1.0;
        // Blue bounce layer has smaller impulse
        bounce_planes[0][cy * width + cx] = 0.1;

        fn make_emulsion(name: &'static str, peak_nm: f64) -> EmulsionLayer {
            let grid = WavelengthGrid::mvp();
            let samples: Vec<f64> = grid
                .wavelengths_nm
                .iter()
                .map(|&l| {
                    let d = (l - peak_nm) / 30.0;
                    (-0.5 * d * d).exp()
                })
                .collect();
            EmulsionLayer {
                name,
                depth_from_surface: Microns(0.0),
                thickness: Microns(1.0),
                kind: LayerKind::Emulsion,
                spectral_sensitivity: Some(SpectralCurve::new(grid, samples)),
                crystal_size: None,
                silver_halide_fraction: 0.2,
                coupler: None,
                gamma_contrast: 1.0,
                capture_k: 1.0,
                reciprocity_p: 1.0,
                is_reversal: false,
            }
        }

        let stock = FilmStock {
            name: "test",
            box_iso: IsoSpeed(200.0),
            layers: vec![
                make_emulsion("blue", 450.0),
                make_emulsion("green", 550.0),
                make_emulsion("red", 650.0),
            ],
            antihalation: elevated_ah(),
            irradiation_response: None,
            developer_diffusion_length: Microns(1.0),
            adjacency_beta: 0.0,
            adjacency_beta_record: 0.0,
            adjacency_beta_cross: 0.0,
            scanner_light: SpectralCurve::constant(1.0),
            capture_luts: vec![],
            grain_kappa: vec![],
            tabular_grain_thickness_um: None,
        };

        apply_spatial_exposure_effects(&mut planes, &bounce_planes, width, height, &stock, 5.0);

        let blue_halo = planes[0][cy * width + cx + 4];
        let green_halo = planes[1][cy * width + cx + 4];
        let red_halo = planes[2][cy * width + cx + 4];
        assert!(
            blue_halo > 1e-6,
            "blue should receive halation blur from blue bounce"
        );
        assert_eq!(
            green_halo, 0.0,
            "green had zero bounce, receives 0 halation"
        );
        assert!(
            red_halo > blue_halo,
            "red halo > blue halo due to stronger red bounce"
        );
    }

    #[test]
    fn portra_irradiation_response_preserves_uniform_fluence() {
        use crate::film::stock::StockId;

        let mut stock = StockId::Portra400.load().unwrap();
        stock.antihalation.reflectance = SpectralCurve::constant(0.0);
        let width = 64;
        let height = 64;
        let mut planes = vec![vec![1.0; width * height]; 6];
        let bounce = vec![vec![0.0; width * height]; 6];

        apply_spatial_exposure_effects(&mut planes, &bounce, width, height, &stock, 1.0);

        assert!(planes
            .iter()
            .flatten()
            .all(|value| (*value - 1.0).abs() < 1e-5));
    }

    #[test]
    fn halation_flag_off_identity() {
        let width = 32;
        let height = 32;
        let mut a = vec![0.0f32; width * height];
        a[16 * width + 16] = 1.0;
        let b = a.clone();
        apply_halation_plane(&mut a, width, height, 0.0, 5.0);
        assert_eq!(a, b);
    }

    #[test]
    fn halation_red_weighted() {
        let ah = AntihalationModel {
            reflectance: {
                let grid = WavelengthGrid::mvp();
                let samples: Vec<f64> = grid
                    .wavelengths_nm
                    .iter()
                    .map(|&l| if l >= 600.0 { 0.4 } else { 0.05 })
                    .collect();
                SpectralCurve::new(grid, samples)
            },
            psf_halation_um: 30.0,
        };
        let w_blue = reflectance_at(&ah, 450.0);
        let w_red = reflectance_at(&ah, 650.0);
        assert!(w_red >= w_blue);
    }

    #[test]
    fn cinestill50d_halation_physics_red_signature() {
        use crate::film::exposure::expose_with_pitch;
        use crate::film::stock::StockId;

        let stock = StockId::CineStill50D.load().expect("CineStill 50D loads");
        let width = 45;
        let height = 45;
        let cx = width / 2;
        let cy = height / 2;

        // Bright red specular highlight patch (7x7 pixels at center, 5000 nits)
        let mut red_img = vec![[0.0f32, 0.0, 0.0]; width * height];
        for dy in -3..=3 {
            for dx in -3..=3 {
                red_img[(cy as isize + dy) as usize * width + (cx as isize + dx) as usize] =
                    [5000.0, 0.0, 0.0];
            }
        }

        let latent_red = expose_with_pitch(&red_img, width, height, &stock, 5.0);
        // Scatter is always on; isolate bounce by repeating with R = 0.
        let mut stock_no_bounce = stock.clone();
        stock_no_bounce.antihalation.reflectance = SpectralCurve::constant(0.0);
        let latent_scatter = expose_with_pitch(&red_img, width, height, &stock_no_bounce, 5.0);
        // Emulsion order: [BF, BS, GF, GS, RF, RS]
        // Outside the specular core (offset 7 px), red fast receives backing bounce;
        // blue fast must not receive bounce from a red highlight (scatter may still leak).
        let pos = cy * width + cx + 7;
        let red_bounce = latent_red.layers[4][pos] - latent_scatter.layers[4][pos];
        let blue_bounce = latent_red.layers[0][pos] - latent_scatter.layers[0][pos];
        assert!(
            red_bounce > 1e-4,
            "CineStill 50D must produce strong red backing bounce in RF layer: {red_bounce}"
        );
        assert!(
            blue_bounce.abs() < 1e-6,
            "Blue emulsion must receive no backing bounce from a red highlight: BF_bounce={blue_bounce}"
        );
    }
}
