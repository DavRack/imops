# Portra 400 calibration comparison

This is a development comparison only. Set `SPEKTRAFILM_PORTRA_PROFILE` to the
temporary local reference profile when running the ignored comparison test.
The profile is intentionally supplied at test time rather than stored here.

The profile is not copied into imops, and no profile values are used as runtime
constants or drift pins.

## Grid and resampling

The reference has 81 wavelength samples from 380–780 nm at 5 nm spacing.
imops currently uses 16 samples from 400–700 nm at 20 nm spacing. Comparison
was made by linear interpolation of the reference onto the imops grid. To make
the loss visible, the 16-point result was linearly reconstructed at every
5 nm reference point in the shared 400–700 nm interval:

| reference field | channel order | max reconstruction error | RMS reconstruction error |
| --- | --- | ---: | ---: |
| log sensitivity | R, G, B | 0.242278, 0.116468, 0.165251 | 0.054374, 0.032445, 0.035117 |
| channel density | R, G, B | 0.022951, 0.052008, 0.047690 | 0.008120, 0.017831, 0.015616 |

The 380–400 nm and 700–780 nm tails are outside the imops runtime grid and
are therefore not silently extrapolated into the model.

## Differences

| quantity | local reference summary | committed imops model | classification / consequence |
| --- | --- | --- | --- |
| spectral sensitivity | R/G/B peaks near 645/545/400 nm; processed reference curves | B/G/R grid peaks 400/540/640 nm, shared by fast/slow pairs | reference data versus fitted Gaussian curves; grid quantization and shape differences remain visible |
| dye-density spectra | R/G/B channel-density peaks near 695/540/450 nm | B/G/R Gaussian dye peaks 440/540/680 nm | reference processed density versus fitted spectral absorptivity; no replacement was made |
| maximum dye density | channel density is a wavelength-dependent reference field | B/G/R totals are 2.82/2.38/1.76 OD, split across fast/slow layers | imops stock constants are fitted/empirical model parameters, not copied profile data |
| base density | reference base-density curve spans approximately 0.188–0.818 over its wavelength grid | orange-mask residual is modeled as `0.4 × d_max`, then scan-normalized; Portra runtime D-min is `[1.000, 0.495, 0.119]` in normalized ACEScg | different conventions; base density is not silently substituted |
| mid-scale neutral | reference provides a wavelength-dependent midscale-neutral density field, approximately 0.736–1.887 | no separate stored neutral-density field; `capture_k`, layer gamma, dye density, and the positive-inversion mid-gray probe establish the anchor | fitted runtime calibration versus reference measurement |
| H&D curves | 256 samples over log exposure −3 to 4, including layered curves | 64-sample log-fluence developable LUT (−4 to 8), followed by `D_max · f^(1/gamma)` per layer | different physical parameterization; monotonicity and finite-value tests cover the runtime curves |
| illuminants | D55 reference illuminant and D50 viewing illuminant | scanner/viewing light is explicit D50; capture has no separate D55 field | D50 handling is tested; D55 reference-light separation remains a calibration gap |

## Test coverage

The runtime-invariant tests in `stock/portra_400.rs` independently check finite
and grid-resolved sensitivity/dye curves, layer ordering and peaks, density
totals, normalized base density, D50 scanner handling, monotonic H&D and
layered density curves, and finite mid-gray negative/inversion reference
generation. They do not claim that the runtime curves are the local reference.

The independent local-profile comparison is the ignored integration test
`portra_reference_comparison` in `tests/portra_calibration.rs`. Run it only
when the local profile is available:

```text
SPEKTRAFILM_PORTRA_PROFILE=/path/to/kodak_portra_400.json \
  cargo test --release --test portra_calibration -- --ignored --nocapture
```

It reads the profile at test time, checks the reference grid, illuminant tags,
curve shape and finiteness, compares the runtime spectral/density summaries
against the reference fields, and reports the explicit 20 nm reconstruction
and normalized-curve errors. Raw measured curves are not required to be
monotonic; the runtime model's monotonicity is tested separately. It does not
copy or embed the data.

Measured/reference data remains a comparison input; fitted and empirical imops
parameters remain explicitly identified above rather than being silently
re-pinned.
