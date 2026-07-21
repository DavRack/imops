//! Halation and local gelatin scatter on absorbed-fluence planes.
//!
//! Two stages (film-implementation.md §5.6):
//! 1. **Local scatter** — always-on in-gelatin blur via `psf_local_um`
//!    (`Φ' = (1−f)·Φ + f·(Φ ⊛ PSF_local)`). Softens even with zero backing R.
//! 2. **Support bounce** — wide PSF from `psf_halation_um`, weighted by backing
//!    reflectance. Bounce is formed from the deepest emulsion's *absorbed*
//!    fluence (not transmitted-to-backing). Absorbed ≈ complement of transmitted
//!    only in a thin-layer sense; for MVP both broadly track incident red, so
//!    absorbed is used as a cheap bounce-source proxy. Shallower layers receive
//!    a red-biased bleed so the halo is colored, not gray.

use crate::film::blur::gaussian_blur_separable;
use crate::film::constants::{HALATION_BLEED_WEIGHTS_BGR, LOCAL_SCATTER_MIX};
use crate::film::spectrum::WavelengthGrid;
use crate::film::stock::{AntihalationModel, FilmStock};

/// Representative bands for spectral reflectance sampling (nm).
const HALATION_BANDS_NM: [f64; 4] = [450.0, 550.0, 650.0, 700.0];

/// Apply local scatter + wide backing halation with cross-layer bleed in place.
///
/// `absorbed_planes` are emulsion planes top→bottom (e.g. blue, green, red).
pub fn apply_spatial_exposure_effects(
    absorbed_planes: &mut [Vec<f32>],
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

    // --- 1. Local gelatin scatter (always on if psf_local > 0) ---
    let sigma_local = sigma_px_from_um(stock.antihalation.psf_local_um, pixel_pitch_um);
    if sigma_local >= 1e-3 && LOCAL_SCATTER_MIX > 0.0 {
        for plane in absorbed_planes.iter_mut() {
            apply_local_scatter(plane, width, height, LOCAL_SCATTER_MIX, sigma_local);
        }
    }

    // --- 2. Wide support bounce with cross-layer bleed ---
    let sigma_wide = sigma_px_from_um(stock.antihalation.psf_halation_um, pixel_pitch_um);
    let w_red = reflectance_at(&stock.antihalation, 650.0);
    if w_red <= 0.0 || sigma_wide < 1e-3 {
        return;
    }

    // Bounce source: deepest emulsion (last plane), which sits nearest the support.
    let deep = absorbed_planes.len() - 1;
    let mut bounce = absorbed_planes[deep].clone();
    gaussian_blur_separable(&mut bounce, width, height, sigma_wide);

    // Normalize bleed weights to the number of emulsion planes present.
    let bleed = bleed_weights(absorbed_planes.len());
    for (e, plane) in absorbed_planes.iter_mut().enumerate() {
        let gain = w_red * bleed[e];
        if gain <= 0.0 {
            continue;
        }
        for (p, &b) in plane.iter_mut().zip(bounce.iter()) {
            *p += gain * b;
        }
    }
}

fn bleed_weights(emulsion_count: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; emulsion_count];
    for i in 0..emulsion_count {
        // Map onto BGR template: first→blue-ish, last→red-ish.
        let t = if emulsion_count <= 1 {
            1.0
        } else {
            i as f32 / (emulsion_count - 1) as f32
        };
        // Interpolate from blue weight toward red weight.
        let b = HALATION_BLEED_WEIGHTS_BGR[0];
        let r = HALATION_BLEED_WEIGHTS_BGR[2];
        let g = HALATION_BLEED_WEIGHTS_BGR[1];
        w[i] = if t <= 0.5 {
            b + (g - b) * (t * 2.0)
        } else {
            g + (r - g) * ((t - 0.5) * 2.0)
        };
    }
    w
}

/// Energy-conserving local scatter: `Φ' = (1−f)·Φ + f·blur(Φ)`.
pub fn apply_local_scatter(
    plane: &mut [f32],
    width: usize,
    height: usize,
    mix: f32,
    sigma_px: f32,
) {
    let f = mix.clamp(0.0, 1.0);
    if f <= 0.0 || sigma_px < 1e-3 {
        return;
    }
    let mut scattered = plane.to_vec();
    gaussian_blur_separable(&mut scattered, width, height, sigma_px);
    let keep = 1.0 - f;
    for (p, s) in plane.iter_mut().zip(scattered.iter()) {
        *p = keep * *p + f * s;
    }
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
                    / (HALATION_BANDS_NM[b + 1] - HALATION_BANDS_NM[b])) as f32;
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
            psf_local_um: 2.0,
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
    fn local_scatter_softens_impulse() {
        let width = 33;
        let height = 33;
        let mut plane = vec![0.0f32; width * height];
        let cx = width / 2;
        let cy = height / 2;
        plane[cy * width + cx] = 1.0;
        apply_local_scatter(&mut plane, width, height, 0.5, 2.0);
        assert!(plane[cy * width + cx] < 1.0);
        assert!(plane[cy * width + cx + 2] > 1e-4);
        let sum: f32 = plane.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "energy should conserve, sum={sum}");
    }

    #[test]
    fn cross_layer_bleed_hits_shallower_more_from_deep_source() {
        use crate::film::stock::{EmulsionLayer, FilmStock, LayerKind, LogNormalDist};
        use crate::film::units::{IsoSpeed, Microns};

        let width = 41;
        let height = 41;
        let n = width * height;
        let mut planes = vec![vec![0.0f32; n]; 3];
        // Only deep (red) layer has an impulse.
        let cx = width / 2;
        let cy = height / 2;
        planes[2][cy * width + cx] = 1.0;

        let stock = FilmStock {
            name: "test",
            box_iso: IsoSpeed(200.0),
            layers: vec![EmulsionLayer {
                name: "dummy",
                depth_from_surface: Microns(0.0),
                thickness: Microns(1.0),
                kind: LayerKind::Overcoat,
                spectral_sensitivity: None,
                crystal_size: None::<LogNormalDist>,
                silver_halide_fraction: 0.0,
                coupler: None,
                gamma_contrast: 1.0,
                capture_k: 1.0,
                reciprocity_p: 1.0,
                is_reversal: false,
            }],
            antihalation: elevated_ah(),
            developer_diffusion_length: Microns(1.0),
            adjacency_beta: 0.0,
            dir_diffusion_length: Microns(1.0),
            dir_inhibition_matrix: vec![],
            scanner_light: SpectralCurve::constant(1.0),
            capture_luts: vec![],
            grain_kappa: vec![],
        };

        apply_spatial_exposure_effects(&mut planes, width, height, &stock, 5.0);

        // Shallower planes must pick up some of the deep bounce (colored halo path).
        let blue_halo = planes[0][cy * width + cx + 4];
        let green_halo = planes[1][cy * width + cx + 4];
        let red_halo = planes[2][cy * width + cx + 4];
        assert!(blue_halo > 1e-6, "blue should receive bleed");
        assert!(green_halo > blue_halo, "green bleed ≥ blue");
        assert!(red_halo > green_halo, "red (source layer) gets most bounce");
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
            psf_local_um: 2.0,
            psf_halation_um: 30.0,
        };
        let w_blue = reflectance_at(&ah, 450.0);
        let w_red = reflectance_at(&ah, 650.0);
        assert!(w_red >= w_blue);
    }
}
