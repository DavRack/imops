/// Configuration parameters for the Reference Gamut Compression.
/// Defaults match the ACES 1.3 LMT implementation.
#[derive(Debug, Clone)]
pub struct RgcParams {
    pub threshold: [f32; 3], // [Cyan, Magenta, Yellow] direction thresholds
    pub limits: [f32; 3],    // [Cyan, Magenta, Yellow] direction limits
    pub power: f32,
    pub invert: bool,
}

impl Default for RgcParams {
    fn default() -> Self {
        // Defaults from the Python 'main' function arguments
        // limit = value + 1.0 (as per the python code: cyan+1, etc.)
        Self {
            threshold: [0.815, 0.803, 0.88],
            limits: [1.147, 1.264, 1.312], // 0.147+1, 0.264+1, 0.312+1
            power: 1.2,
            invert: false,
        }
    }
}

/// The core compression math function.
/// Maps input distance 'x' through the power curve.
///
/// Port of Python: `compress(dist, lim, thr, invert, power)`
#[inline]
fn compress_value(dist: f32, lim: f32, thr: f32, power: f32, invert: bool) -> f32 {
    // 1. If distance is below threshold, no compression needed.
    // The Python code does this via masking: `cdist[dist < thr] = dist[dist < thr]`
    if dist < thr {
        return dist;
    }

    // 2. Calculate scale factor 's'
    // Python: s = (lim-thr)/np.power(np.power((1-thr)/(lim-thr),-power)-1,1/power)
    // This calculates the y=1 intersect to ensure smooth continuity.
    let base_inner = (1.0 - thr) / (lim - thr);
    
    // Safety check for div by zero or invalid pow inputs
    if base_inner <= 0.0 || (lim - thr).abs() < 1e-6 {
        return dist; 
    }

    let denom_inner = base_inner.powf(-power) - 1.0;
    
    // Safety check for complex numbers result
    if denom_inner <= 0.0 {
         return dist;
    }

    let s = (lim - thr) / denom_inner.powf(1.0 / power);

    // 3. Apply Curve
    let dist_norm = (dist - thr) / s; // (dist-thr)/s

    if !invert {
        // Forward Compression
        // y = thr + s * ( ((dist-thr)/s) / (1 + ((dist-thr)/s)^p)^(1/p) )
        let denominator = (1.0 + dist_norm.powf(power)).powf(1.0 / power);
        thr + s * (dist_norm / denominator)
    } else {
        // Inverse (Un-compression)
        // y = thr + s * ( - ( x^p / (x^p - 1) ) )^(1/p)
        // Note: The python math for inverse is simpler to write as:
        // inner = dist_norm^p / (dist_norm^p - 1)
        // result = thr + s * (-inner)^(1/p)
        
        let dn_pow = dist_norm.powf(power);
        let inner = dn_pow / (dn_pow - 1.0);
        
        // Safety for inverse: if inner is positive (which implies -inner is neg),
        // we can't root it without complex numbers.
        if inner >= 0.0 {
             dist 
        } else {
            thr + s * (-inner).powf(1.0 / power)
        }
    }
}

/// Applies ACES Reference Gamut Compression to a single pixel.
/// 
/// Input: ACEScg RGB [f32; 3]
/// Output: Compressed RGB [f32; 3]
pub fn gamut_compress_pixel(rgb: [f32; 3], params: &RgcParams) -> [f32; 3] {
    // 1. Calculate Achromatic axis (Max(R, G, B))
    // Python: ach = np.max(rgb, axis=-1)
    let ach = rgb[0].max(rgb[1]).max(rgb[2]);

    // Handle pure black to avoid division by zero
    if ach == 0.0 {
        return [0.0, 0.0, 0.0];
    }

    // 2. Calculate Distance
    // Python: dist = np.where(ach == 0.0, 0.0, (ach-rgb)/np.abs(ach))
    // Since we handled ach=0 above, we just divide.
    let abs_ach = ach.abs();
    
    // We compute this per channel
    let dist = [
        (ach - rgb[0]) / abs_ach,
        (ach - rgb[1]) / abs_ach,
        (ach - rgb[2]) / abs_ach,
    ];

    // 3. Compress Distance
    // The Python code maps lim/thr arrays to the R, G, B channels respectively.
    let cdist = [
        compress_value(dist[0], params.limits[0], params.threshold[0], params.power, params.invert),
        compress_value(dist[1], params.limits[1], params.threshold[1], params.power, params.invert),
        compress_value(dist[2], params.limits[2], params.threshold[2], params.power, params.invert),
    ];

    // 4. Reconstruct RGB
    // Python: crgb = ach - cdist * np.abs(ach)
    [
        ach - cdist[0] * abs_ach,
        ach - cdist[1] * abs_ach,
        ach - cdist[2] * abs_ach,
    ]
}
