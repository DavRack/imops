//! Wavelength grid and spectral curve operations.
//!
//! Optical density convention: base-10, T = 10^(−D).

use crate::film::constants::{
    WAVELENGTH_MAX_NM, WAVELENGTH_MIN_NM, WAVELENGTH_SAMPLE_COUNT, WAVELENGTH_STEP_NM,
};

/// Fixed wavelength sampling grid (nm).
#[derive(Clone, Debug, PartialEq)]
pub struct WavelengthGrid {
    pub wavelengths_nm: Vec<f64>,
}

impl WavelengthGrid {
    /// MVP visible grid: 400, 420, …, 700 nm (16 samples).
    pub fn mvp() -> Self {
        let mut wavelengths_nm = Vec::with_capacity(WAVELENGTH_SAMPLE_COUNT);
        let mut lambda = WAVELENGTH_MIN_NM;
        while lambda <= WAVELENGTH_MAX_NM + 1e-9 {
            wavelengths_nm.push(lambda);
            lambda += WAVELENGTH_STEP_NM;
        }
        debug_assert_eq!(wavelengths_nm.len(), WAVELENGTH_SAMPLE_COUNT);
        Self { wavelengths_nm }
    }

    pub fn len(&self) -> usize {
        self.wavelengths_nm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.wavelengths_nm.is_empty()
    }
}

/// Spectral samples aligned to a [`WavelengthGrid`].
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralCurve {
    pub grid: WavelengthGrid,
    pub samples: Vec<f64>,
}

impl SpectralCurve {
    pub fn new(grid: WavelengthGrid, samples: Vec<f64>) -> Self {
        assert_eq!(
            grid.len(),
            samples.len(),
            "SpectralCurve samples must match grid length"
        );
        Self { grid, samples }
    }

    /// Constant value on the MVP grid.
    pub fn constant(value: f64) -> Self {
        let grid = WavelengthGrid::mvp();
        let samples = vec![value; grid.len()];
        Self { grid, samples }
    }

    /// Linear interpolation of the curve at `wavelength_nm`.
    pub fn evaluate(&self, wavelength_nm: f64) -> f64 {
        let w = &self.grid.wavelengths_nm;
        if wavelength_nm <= w[0] {
            return self.samples[0];
        }
        let last = w.len() - 1;
        if wavelength_nm >= w[last] {
            return self.samples[last];
        }
        for i in 0..last {
            if wavelength_nm >= w[i] && wavelength_nm <= w[i + 1] {
                let t = (wavelength_nm - w[i]) / (w[i + 1] - w[i]);
                return self.samples[i] * (1.0 - t) + self.samples[i + 1] * t;
            }
        }
        self.samples[last]
    }

    /// Pointwise product of two curves on the same grid.
    pub fn multiply(&self, other: &SpectralCurve) -> SpectralCurve {
        assert_eq!(self.grid.wavelengths_nm, other.grid.wavelengths_nm);
        let samples = self
            .samples
            .iter()
            .zip(other.samples.iter())
            .map(|(a, b)| a * b)
            .collect();
        SpectralCurve {
            grid: self.grid.clone(),
            samples,
        }
    }

    /// Beer–Lambert transmittance from optical density samples: T = 10^(−D).
    ///
    /// `density` is base-10 OD per sample (already including any ε scaling).
    pub fn beer_lambert_transmittance(density: &SpectralCurve) -> SpectralCurve {
        let samples = density.samples.iter().map(|&d| 10f64.powf(-d)).collect();
        SpectralCurve {
            grid: density.grid.clone(),
            samples,
        }
    }

    /// Trapezoidal integral of the curve over the wavelength span (nm units).
    pub fn integrate(&self) -> f64 {
        let w = &self.grid.wavelengths_nm;
        let s = &self.samples;
        if w.len() < 2 {
            return 0.0;
        }
        let mut acc = 0.0;
        for i in 0..w.len() - 1 {
            let dlambda = w[i + 1] - w[i];
            acc += 0.5 * (s[i] + s[i + 1]) * dlambda;
        }
        acc
    }

    /// Integrate `self * other` over wavelength (trapezoidal).
    pub fn integrate_against(&self, other: &SpectralCurve) -> f64 {
        self.multiply(other).integrate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavelength_grid_mvp_len() {
        let g = WavelengthGrid::mvp();
        assert_eq!(g.len(), 16);
        assert_eq!(g.wavelengths_nm[0], 400.0);
        assert_eq!(*g.wavelengths_nm.last().unwrap(), 700.0);
    }

    #[test]
    fn beer_lambert_zero_density() {
        let dens = SpectralCurve::constant(0.0);
        let t = SpectralCurve::beer_lambert_transmittance(&dens);
        for s in &t.samples {
            assert!((s - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn beer_lambert_unit_density() {
        // dens=1, ε folded in → T = 10^(−1) = 0.1
        let dens = SpectralCurve::constant(1.0);
        let t = SpectralCurve::beer_lambert_transmittance(&dens);
        for s in &t.samples {
            assert!((s - 0.1).abs() < 1e-12);
        }
    }

    #[test]
    fn curve_integrate_uniform() {
        let value = 2.0;
        let curve = SpectralCurve::constant(value);
        let span = 700.0 - 400.0; // nm
        let expected = value * span;
        let got = curve.integrate();
        let rel = ((got - expected) / expected).abs();
        assert!(
            rel < 0.01,
            "integrate uniform: got {got}, expected {expected}, rel {rel}"
        );
    }
}
