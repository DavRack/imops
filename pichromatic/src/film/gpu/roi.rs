//! Bounded Region of Interest (ROI) geometry planner and ROI-aware reflection primitives.

use crate::film::blur::{gaussian_radius, reflect_index};
use crate::film::constants::DYE_CLOUD_CORRELATION_UM;
use crate::film::exposure::halation::sigma_px_from_um;
use crate::film::stock::FilmStock;
use crate::film::types::FilmFormat;

/// Signed 2D integer rectangle in global coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RectI {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl RectI {
    #[inline]
    pub(super) const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    #[inline]
    pub(super) const fn right(&self) -> i32 {
        self.x + self.width as i32
    }

    #[inline]
    pub(super) const fn bottom(&self) -> i32 {
        self.y + self.height as i32
    }

    /// Check if a global point `(gx, gy)` lies inside this rectangle.
    #[inline]
    pub(super) const fn contains(&self, gx: i32, gy: i32) -> bool {
        gx >= self.x && gx < self.right() && gy >= self.y && gy < self.bottom()
    }

    /// Expand the rectangle symmetrically by `dx` horizontally and `dy` vertically.
    #[inline]
    pub(super) fn expand(&self, dx: u32, dy: u32) -> Self {
        Self {
            x: self.x - dx as i32,
            y: self.y - dy as i32,
            width: self.width + 2 * dx,
            height: self.height + 2 * dy,
        }
    }
}

/// ROI Execution Plan containing core destination bounds, root expanded bounds,
/// and exact stage radii computed via the repository's `ceil(3σ)` policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RoiPlan {
    pub(super) core: RectI,
    pub(super) root: RectI,
    pub(super) local_radius: u32,
    pub(super) wide_radius: u32,
    pub(super) dir_radius: u32,
    pub(super) adjacency_radius: u32,
    pub(super) grain_radius: u32,
}

impl RoiPlan {
    /// Calculate exact Gaussian radius (pixels) covering 99.7% of mass using shared `gaussian_radius`.
    #[inline]
    pub(super) fn sigma_to_radius(sigma: f32) -> u32 {
        gaussian_radius(sigma) as u32
    }

    /// Build ROI execution plan for a target core region and stock / format parameters.
    pub(super) fn build(
        core: RectI,
        stock: &FilmStock,
        film_format: FilmFormat,
        image_width: u32,
    ) -> Self {
        let pitch = film_format.pixel_pitch_um(image_width as usize);

        let sigma_local = sigma_px_from_um(stock.antihalation.psf_local_um, pitch);
        let sigma_wide = sigma_px_from_um(stock.antihalation.psf_halation_um, pitch);
        // CPU multi-bounce halation blurs at σ√(k+1); the widest bounce (k=2) is σ√3.
        let sigma_wide = if sigma_wide >= 1e-3 {
            sigma_wide * 3.0f32.sqrt()
        } else {
            sigma_wide
        };
        let sigma_dir = stock.dir_diffusion_length.0 / pitch.max(1e-6);
        let sigma_adj = stock.developer_diffusion_length.0 / pitch.max(1e-6);

        let grain_sigma = (DYE_CLOUD_CORRELATION_UM / pitch.max(1e-6)).max(1.0);

        let local_radius = Self::sigma_to_radius(sigma_local);
        let wide_radius = Self::sigma_to_radius(sigma_wide);
        let dir_radius =
            if !stock.dir_inhibition_matrix.is_empty() { Self::sigma_to_radius(sigma_dir) } else { 0 };
        let adjacency_radius = if stock.adjacency_beta.abs() >= 1e-8 {
            Self::sigma_to_radius(sigma_adj)
        } else {
            0
        };
        let grain_radius = Self::sigma_to_radius(grain_sigma);

        // Content dependency halo: max(local + wide halation + DIR + adjacency, grain_radius).
        // Back-propagated through blurs and pointwise stages.
        let blur_halo = local_radius + wide_radius + dir_radius + adjacency_radius;
        let halo_radius = blur_halo.max(grain_radius);
        let root = core.expand(halo_radius, halo_radius);

        Self {
            core,
            root,
            local_radius,
            wide_radius,
            dir_radius,
            adjacency_radius,
            grain_radius,
        }
    }
}

/// Mirror-reflect a 1D signed global coordinate `gx` relative to physical image length `len`.
/// Tile/core edges MUST NOT be used here — `len` MUST be physical `image_width` or `image_height`.
///
/// Delegates directly to `blur::reflect_index` with safe type conversion to maintain a single reflection rule.
#[inline]
pub(super) fn reflect_global_index(gx: i32, len: u32) -> u32 {
    reflect_index(gx as isize, len as usize) as u32
}

/// Map a signed global coordinate `(gx, gy)` into physical image coordinates `(px, py)`
/// via physical border reflection, and return the relative offset `(lx, ly)` inside the
/// root ROI rectangle `root`.
///
/// Panics if `(gx, gy)` lies outside `root` to prevent silent u32 wrapping/truncation.
#[inline]
pub(super) fn global_to_root_local(gx: i32, gy: i32, root: RectI, img_w: u32, img_h: u32) -> (u32, u32, u32, u32) {
    assert!(
        root.contains(gx, gy),
        "Global coordinate ({gx}, {gy}) outside root bounds {:?}",
        root
    );
    let px = reflect_global_index(gx, img_w);
    let py = reflect_global_index(gy, img_h);
    let lx = (gx - root.x) as u32;
    let ly = (gy - root.y) as u32;
    (px, py, lx, ly)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::film::blur::gaussian_blur_separable;
    use crate::film::stock::StockId;

    #[test]
    fn reflect_global_index_matches_blur_reflect_index() {
        for len in [1, 2, 3, 5, 10, 100] {
            for gx in -500..500 {
                let direct_roi = reflect_global_index(gx, len);
                let blur_ref = reflect_index(gx as isize, len as usize) as u32;
                assert_eq!(
                    direct_roi, blur_ref,
                    "Mismatch for gx={gx}, len={len}: direct={direct_roi}, blur_ref={blur_ref}"
                );
            }
        }
    }

    #[test]
    fn reflect_global_index_degenerate_dimensions() {
        assert_eq!(reflect_global_index(-10, 1), 0);
        assert_eq!(reflect_global_index(0, 1), 0);
        assert_eq!(reflect_global_index(10, 1), 0);

        assert_eq!(reflect_global_index(0, 0), 0);
    }

    #[test]
    fn reflect_global_index_physical_edges() {
        // len = 4 -> indices 0, 1, 2, 3
        assert_eq!(reflect_global_index(0, 4), 0);
        assert_eq!(reflect_global_index(3, 4), 3);
        assert_eq!(reflect_global_index(-1, 4), 1);
        assert_eq!(reflect_global_index(-2, 4), 2);
        assert_eq!(reflect_global_index(4, 4), 2);
        assert_eq!(reflect_global_index(5, 4), 1);
    }

    #[test]
    fn roi_plan_backward_propagation() {
        let core = RectI::new(100, 100, 1024, 1024);
        let stock = StockId::ColorNeg200.load().unwrap(); // Stock has non-zero halo radii
        let plan = RoiPlan::build(core, &stock, FilmFormat::Film35mm, 4032);

        let blur_halo = plan.local_radius + plan.wide_radius + plan.dir_radius + plan.adjacency_radius;
        let total_halo = blur_halo.max(plan.grain_radius);
        assert_eq!(plan.root.x, core.x - total_halo as i32);
        assert_eq!(plan.root.y, core.y - total_halo as i32);
        assert_eq!(plan.root.width, core.width + 2 * total_halo);
        assert_eq!(plan.root.height, core.height + 2 * total_halo);
    }

    #[test]
    fn roi_plan_radii_range_0_to_128_and_thresholds() {
        assert_eq!(RoiPlan::sigma_to_radius(0.0), 0);
        assert_eq!(RoiPlan::sigma_to_radius(0.00099), 0);
        assert_eq!(RoiPlan::sigma_to_radius(0.001), 1);
        assert_eq!(RoiPlan::sigma_to_radius(0.1), 1);
        assert_eq!(RoiPlan::sigma_to_radius(0.33), 1);
        assert_eq!(RoiPlan::sigma_to_radius(0.34), 2);

        for r in 0..=128 {
            let sigma = if r == 0 { 0.0 } else { r as f32 / 3.0 };
            let computed_r = RoiPlan::sigma_to_radius(sigma);
            let shared_r = gaussian_radius(sigma) as u32;
            assert_eq!(computed_r, shared_r);
            assert_eq!(computed_r, r as u32);
        }
    }

    #[test]
    fn global_to_root_local_mapping_and_out_of_bounds_assertion() {
        let root = RectI::new(-10, -5, 100, 100);
        assert_eq!(root.right(), 90);
        assert_eq!(root.bottom(), 95);

        let (px, py, lx, ly) = global_to_root_local(-2, 3, root, 4000, 3000);
        assert_eq!(px, 2);
        assert_eq!(py, 3);
        assert_eq!(lx, 8);
        assert_eq!(ly, 8);

        let result = std::panic::catch_unwind(|| {
            global_to_root_local(-11, 3, root, 4000, 3000);
        });
        assert!(result.is_err(), "Expected panic on global_to_root_local out-of-bounds x");
    }

    #[test]
    fn cpu_roi_blur_equivalence_test() {
        // CPU-only equivalence test between full-frame blur and ROI root-expanded blur.
        let cases = [
            // (img_w, img_h, core_x, core_y, core_w, core_h, sigma)
            (64, 64, 16, 16, 16, 16, 2.0), // interior core
            (64, 64, 0, 0, 16, 16, 2.5),   // physical edge core (top-left)
            (64, 64, 48, 48, 16, 16, 3.0), // physical edge core (bottom-right)
            (60, 50, 48, 32, 12, 18, 2.0), // partial final core (ends at image right/bottom)
            (1, 1, 0, 0, 1, 1, 1.5),       // 1x1 degenerate
            (1, 32, 0, 8, 1, 8, 2.0),      // 1xN degenerate
            (32, 1, 8, 0, 8, 1, 2.0),      // Nx1 degenerate
            (33, 33, 0, 0, 32, 32, 2.0),   // boundary edge cases (core right=32 near img_w=33)
            (33, 33, 32, 32, 1, 1, 2.0),   // partial 1x1 core at bottom-right corner
        ];

        for &(img_w, img_h, cx, cy, cw, ch, sigma) in &cases {
            let radius = gaussian_radius(sigma) as u32;

            // 1. Create synthetic test pattern full-frame image with impulses and ramps.
            let mut full_image = vec![0.0f32; img_w * img_h];
            for y in 0..img_h {
                for x in 0..img_w {
                    let idx = (y * img_w + x) as usize;
                    full_image[idx] = (x * 17 + y * 31 % 101) as f32 / 100.0;
                }
            }
            // Add sharp impulses at borders and corners
            if img_w > 0 && img_h > 0 {
                full_image[0] += 10.0;
                full_image[(img_h - 1) * img_w + (img_w - 1)] += 5.0;
            }

            // 2. Run full-frame blur reference.
            let mut full_blurred = full_image.clone();
            gaussian_blur_separable(&mut full_blurred, img_w as usize, img_h as usize, sigma);

            // 3. Construct ROI plan for the given core & radius.
            let core = RectI::new(cx as i32, cy as i32, cw, ch);
            let root = core.expand(radius, radius);

            // Populate root buffer by reading physical reflected global coordinates.
            let mut root_buf = vec![0.0f32; (root.width * root.height) as usize];
            for ry in 0..root.height {
                let gy = root.y + ry as i32;
                for rx in 0..root.width {
                    let gx = root.x + rx as i32;
                    let (px, py, lx, ly) = global_to_root_local(gx, gy, root, img_w as u32, img_h as u32);
                    let root_idx = (ly * root.width + lx) as usize;
                    let img_idx = (py * img_w as u32 + px) as usize;
                    root_buf[root_idx] = full_image[img_idx];
                }
            }

            // 4. Run separable blur on root buffer.
            gaussian_blur_separable(
                &mut root_buf,
                root.width as usize,
                root.height as usize,
                sigma,
            );

            // 5. Extract core from blurred root and assert EXACT float bit equality against full_blurred.
            for cy_offset in 0..core.height {
                let gy = core.y + cy_offset as i32;
                let py = reflect_global_index(gy, img_h as u32);
                for cx_offset in 0..core.width {
                    let gx = core.x + cx_offset as i32;
                    let px = reflect_global_index(gx, img_w as u32);

                    let lx = (gx - root.x) as u32;
                    let ly = (gy - root.y) as u32;

                    let root_idx = (ly * root.width + lx) as usize;
                    let full_idx = (py * img_w as u32 + px) as usize;

                    let val_roi = root_buf[root_idx];
                    let val_full = full_blurred[full_idx];

                    assert_eq!(
                        val_roi.to_bits(),
                        val_full.to_bits(),
                        "Bitwise blur mismatch for image {img_w}x{img_h}, core ({cx},{cy},{cw},{ch}), sigma {sigma} at local ({cx_offset},{cy_offset}): roi={val_roi} ({:#x}), full={val_full} ({:#x})",
                        val_roi.to_bits(),
                        val_full.to_bits()
                    );
                }
            }
        }
    }
}
