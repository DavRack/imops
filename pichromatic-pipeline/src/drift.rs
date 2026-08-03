//! Guard-Banded Canonicalizer with Exceptions.
//!
//! Drift-proof quantization for validating float32 render pipelines (CPU vs
//! GPU, cross-architecture).
//!
//! ## Representation space
//! Every channel is mapped into a monotonically increasing 32-bit Ordered
//! Integer Space ([`ordered_ulp`]); `-0.0` is normalized to `+0.0` and all
//! distances are absolute integer (ULP) gaps.
//!
//! ## Guard banding
//! Samples are bucketed with a global bucket width `W` strictly larger than
//! the max allowed drift radius `D` (`W = 8 × D`). A sample's bucket index
//! is `sample ÷ W`. Drift of at most `D` ULPs can only move a sample across
//! a bucket edge when it is within `D` of that edge (the guard band), so:
//!
//! - deep inside a bucket (offset in `[D, W − D)`): canonical range `(b, b)`
//! - inside the lower guard band (offset `< D`): canonical range `(b−1, b)`
//! - inside the upper guard band (offset `≥ W − D`): canonical range `(b, b+1)`
//!
//! Two samples agree iff their canonical bucket ranges overlap, which gives:
//!
//! - **Zero false positives**: drift `≤ D` ULPs always agrees.
//! - **Zero false negatives**: deviation `≥ W + 2·D` ULPs (one full bucket
//!   plus both guard bands) never agrees, so a rogue pixel that far off
//!   fails the check.

/// Monotonically increasing 32-bit Ordered Integer Space of an `f32`
/// (`-0.0` normalized to `+0.0`), so absolute integer gaps are ULP distances.
pub fn ordered_ulp(value: f32) -> u32 {
    let value = if value == 0.0 { 0.0 } else { value };
    let bits = value.to_bits();
    if bits >> 31 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

/// Guard-banded canonicalizer: buckets of width `W = 8 × D` with exception
/// handling for samples inside the `D`-wide guard bands at bucket edges.
#[derive(Clone, Copy, Debug)]
pub struct Canonicalizer {
    /// Global bucket width `W`, strictly larger than the drift radius `D`.
    pub width: u32,
    /// Max allowed drift radius `D`, in ULPs.
    pub radius: u32,
}

impl Canonicalizer {
    /// `W = 8 × D` as recommended by the guard-banding strategy.
    pub fn new(max_drift_ulps: u32) -> Self {
        let radius = max_drift_ulps.max(1);
        Self {
            width: radius.checked_mul(8).unwrap_or(u32::MAX),
            radius,
        }
    }

    /// Bucket index of a sample: `sample ÷ W`.
    pub fn bucket(&self, sample: u32) -> u32 {
        sample / self.width
    }

    /// Canonical value of a sample: the inclusive bucket-ID range it can
    /// legitimately occupy under at most `radius` ULPs of drift.
    pub fn canonical(&self, sample: u32) -> (u32, u32) {
        let bucket = self.bucket(sample);
        let offset = sample % self.width;
        if offset < self.radius {
            (bucket.saturating_sub(1), bucket)
        } else if offset >= self.width - self.radius {
            (bucket, bucket + 1)
        } else {
            (bucket, bucket)
        }
    }

    /// Two samples agree iff their canonical bucket ranges overlap.
    ///
    /// Guarantees: `|a − b| ≤ radius` always agrees (zero false positives);
    /// `|a − b| ≥ width + 2·radius` never agrees (zero false negatives, a
    /// deviation of one full bucket plus both guard bands).
    pub fn agree(&self, a: u32, b: u32) -> bool {
        let (a_lo, a_hi) = self.canonical(a);
        let (b_lo, b_hi) = self.canonical(b);
        a_lo <= b_hi && b_lo <= a_hi
    }

    /// Single-value digest of a sample's canonical range, for whole-image
    /// hashing: `(lo, hi)` packed into one `u64`.
    pub fn digest(&self, sample: u32) -> u64 {
        let (lo, hi) = self.canonical(sample);
        ((lo as u64) << 32) | hi as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_ulp_is_monotonic_across_the_full_range() {
        let samples = [
            f32::NEG_INFINITY,
            -f32::MAX,
            -1.0e30,
            -1.0,
            -1.0e-10,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0e-10,
            1.0,
            1.0e30,
            f32::MAX,
            f32::INFINITY,
        ];
        for pair in samples.windows(2) {
            if pair[0] == -0.0 {
                continue;
            }
            assert!(
                ordered_ulp(pair[0]) < ordered_ulp(pair[1]),
                "ordered({:?}) must be < ordered({:?})",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn negative_zero_normalizes_to_positive_zero() {
        assert_eq!(ordered_ulp(-0.0), ordered_ulp(0.0));
    }

    #[test]
    fn ulp_gap_matches_float_gap() {
        let a = 1.0f32;
        let b = a.next_up();
        assert_eq!(ordered_ulp(b) - ordered_ulp(a), 1);
    }

    /// Zero false positives: drift of at most `D` ULPs in either direction
    /// must always agree, across every offset of interest (guard bands,
    /// interiors, bucket crossings, large magnitudes).
    #[test]
    fn drift_within_radius_always_agrees() {
        let canon = Canonicalizer::new(100);
        let width = canon.width;
        for u in [
            0,
            1,
            canon.radius - 1,
            canon.radius,
            width - canon.radius - 1,
            width - canon.radius,
            width - 1,
            width,
            width + 3,
            1 << 24,
        ] {
            for delta in 0..=canon.radius {
                assert!(canon.agree(u, u + delta), "u={u} delta={delta}");
                if u >= delta {
                    assert!(canon.agree(u - delta, u), "u={u} delta={delta}");
                }
            }
        }
    }

    /// Zero false negatives: a deviation of one full bucket plus both guard
    /// bands (`W + 2·D` ULPs) must never agree, so a single rogue pixel that
    /// far off fails the check.
    #[test]
    fn deviation_beyond_bucket_width_never_agrees() {
        let canon = Canonicalizer::new(100);
        let fail_at = canon.width + 2 * canon.radius;
        for u in [0, canon.radius, canon.width, 1 << 20, u32::MAX / 2] {
            for delta in (fail_at..fail_at + canon.width).step_by(13) {
                assert!(
                    !canon.agree(u, u + delta),
                    "u={u} delta={delta} must not agree"
                );
            }
        }
    }

    /// The digest is deterministic and reflects the canonical range.
    #[test]
    fn digest_packs_canonical_range() {
        let canon = Canonicalizer::new(100);
        let (lo, hi) = canon.canonical(1 << 20);
        assert_eq!(canon.digest(1 << 20), ((lo as u64) << 32) | hi as u64);
    }

    #[test]
    fn width_is_strictly_larger_than_radius() {
        let canon = Canonicalizer::new(100);
        assert_eq!(canon.width, 8 * canon.radius);
        assert!(canon.width > canon.radius);
    }
}
