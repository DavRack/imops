# Pichromatic — Scene-Referred RAW Pipeline Implementation Plan (Phase 1)

**Audience:** Agy (implementing agent), working inside the existing `pichromatic` repo.
**Goal of this document:** turn the "world-class RAW pipeline" design conversation into an incremental, testable build plan that fits real constraints: pure Rust, minimal dependencies, CPU-only, and one shared core library consumed by CLI, WebAssembly, iOS, and Android.

This is not a from-scratch spec. Pichromatic already exists. Every step below is a unit of work that either adds a new module, replaces a naive placeholder inside an existing module, or adds tests. Nothing here should require a rewrite of the app shell — only the image-processing core and its bindings.

---

## 1. Ground rules (do not deviate without flagging it back)

1. **Pure Rust.** No C/C++ bindings, no FFI to system image libraries (no LibRaw, no OpenColorIO, no libjpeg-turbo). If a pure-Rust crate doesn't exist for something, we write it ourselves.
2. **Minimum dependencies**, and a hard split between two tiers:
   - **Core algorithm crate** (raw decode → pixels → color science → grading → output pixels): target **zero non-std production dependencies** unless a step explicitly says otherwise. Every dependency added here has to justify itself against "could we write 200 lines of Rust instead."
   - **Platform adapter crates** (CLI arg parsing, `wasm-bindgen` glue, `uniffi`/`cbindgen` glue): these may take on normal per-platform dependencies. They must not leak into the core crate.
   - `dev-dependencies` (test-only) are cheap and fine to use liberally — they never ship in a binary or a `.wasm`/`.a`/`.aar` artifact.
3. **CPU only.** No GPU compute paths in this phase (no CUDA/Metal/Vulkan/WebGPU). No SIMD intrinsics yet either — write clean scalar Rust first, vectorize later once correctness is locked in by tests. Multi-threading is also deferred (see §6) because thread pools behave differently on wasm32/iOS/Android and we don't want that variable in play while we're still proving correctness.
4. **One core library, four consumers.** Every public type and function in the core crate must be `Send`-agnostic-safe to compile on `wasm32-unknown-unknown`, `aarch64-apple-ios`, and `aarch64-linux-android`, in addition to native. That means: no `std::thread`, no filesystem access inside the core algorithms (I/O happens at the edges), no assumptions about available memory beyond "a few full-resolution float buffers."
5. **Pipeline, not DAG.** Build this as an ordered sequence of stages with a config struct, not a node graph / dependency graph engine. A `PipelineConfig` in, an `Image` out. Non-destructive editing, caching graphs, and re-orderable nodes are explicitly future work — do not build the abstraction for it now, it will just slow this phase down and guess wrong about the eventual API.
6. **No masking / local adjustments in this phase.** Every operation below is global (applies to the whole frame). Mask infrastructure (brushes, gradients, luminosity ranges, subject/sky detection) is out of scope until this global pipeline is correct and fast.
7. **We cannot do this in one PR.** The steps below are deliberately small. Each one should be its own commit/PR with its own tests. Do not batch multiple steps into one change — if a step feels too big to review in one sitting, split it further rather than merging steps together.

---

## 2. Step 0 — Audit before writing any pipeline code

Before Step 1, spend real time here. This determines whether the rest of the plan slots in cleanly or needs renaming.

**Do:**
- Map the current `pichromatic` repo: crate/module boundaries, what image types already exist, what (if anything) currently does raw decoding or demosaicing, what the CLI/wasm/iOS/Android entry points currently call.
- Identify what's already a dependency (are we already pulling in an `image`-crate-style dependency? Any existing raw decode path, even a placeholder?).
- Identify the current output path (how does a processed image currently get from Rust to each of the 4 consumers?).
- Write a short `ARCHITECTURE_AUDIT.md` (or add to existing docs) mapping old module names → new module names used in this plan, so nothing below gets implemented twice under two names.

**Test:** N/A (this is a documentation/discovery step) — but its "done" criterion is concrete.

**Done when:** a written mapping exists and is referenced at the top of the actual tracking issue/board, so every later step says "goes in `<existing path>`" instead of guessing.

---

## 3. Architecture decisions locked for this phase

- **Core image buffer:** planar `f32`, one contiguous `Vec<f32>` per channel (not interleaved), wrapped in a small `Image` struct carrying width/height/channel count/color space tag. Planar layout makes per-channel raw-domain operations (which dominate the early pipeline) simpler and more cache-friendly than interleaved, and costs nothing extra at the interleave boundary (only needed once, at final export).
- **Color space is a typed tag, not a convention.** Every `Image` carries an enum (`RawMosaic`, `CameraLinear`, `Aces2065_1`, `AcesCg`, `AcesCct`, `OkLab`, `DisplaySrgb`, `DisplayP3`, `DisplayRec709`, …). Functions that convert between spaces consume one tag and produce another; this makes an entire class of bugs ("did this already get gamma-encoded?") a compile-time-adjacent, at-least-runtime-assertable thing instead of a silent mistake.
- **RAW format support starts with DNG only.** DNG is openly documented (it's TIFF-EP based) and can be fully implemented in pure Rust without reverse-engineering a vendor's proprietary compression. CR3/NEF/ARW/RAF support is real future work but is a separate, large effort per format (each needs its own often-undocumented decompression). Until then, users of proprietary raw formats convert via Adobe DNG Converter or camera-native DNG mode. This is called out explicitly so nobody is surprised later.
- **ACES colorimetry, implemented directly, no OCIO.** ACES2065-1 (AP0), ACEScg (AP1), and ACEScct are just published primaries/whitepoint/matrices/transfer curves (SMPTE ST 2065-1, Academy S-2014-001, Academy S-2016-001). We hard-code the matrices as `const` 3×3s and implement the ACEScct log curve as a pure function. This gets us real ACES-compatible scene-referred colorimetry with zero runtime dependency on OCIO/CTL.
- **Output transform is "ACES-inspired," not bit-exact ACES 2, in this phase.** The full ACES 2 output transform uses a Hellwig-derived JMh appearance model with its own tone-scale and gamut-compression machinery — that's a substantial project on its own. Phase 1 ships a simplified but principled equivalent: Oklab/OkLCh for the perceptual space, a hue-preserving chroma compressor, and a filmic tone curve on lightness. This is flagged as a deliberate simplification with a named follow-up (§8, Future Work).
- **No EXR in this phase.** Archival/interchange output as OpenEXR is real future work (a pure-Rust EXR crate exists and can be adopted later). Phase 1's "master" format is just the in-memory float `Image`, and the only file export is PNG (8/16-bit).

---

## 4. Dependency policy (be explicit about this in every PR)

**Core crate — allowed today:** nothing beyond `std`. Every step below that needs new capability (bit unpacking, matrix math, a LUT curve, an OETF) writes it directly.

**Core crate — pre-approved narrow exceptions**, because writing them ourselves buys nothing and re-implementing them badly is a real risk:
- `png` (pure Rust, only pulls in `miniz_oxide`/`crc32fast`, both pure Rust, both compile cleanly to wasm32/iOS/Android) — for the export step only, isolated behind an `export` module so it's easy to strip or swap later.

**Anything else** (a raw-decoder crate, an image crate, a linear algebra crate, a color crate) requires a one-paragraph justification in the PR description: what it saves us, what it costs us on each of the 4 targets, and why hand-rolling it is worse here. Default answer should be "write it ourselves" — this pipeline's whole value proposition is owning the color science, so outsourcing the math undermines the point.

**Platform adapter crates — normal rules apply:** `clap` for the CLI, `wasm-bindgen`/`js-sys` for wasm, `uniffi` or hand-written `cbindgen` headers for iOS/Android. These never appear in the core crate's `Cargo.toml`.

**Dev-dependencies — liberal use is fine:** `approx` (float comparisons), `proptest` or hand-rolled property tests, `criterion` for benchmarking later. These don't ship.

---

## 5. Testing philosophy (applies to every step below — read this once, not 50 times)

Every step must satisfy all of the following before it's "done," in addition to whatever step-specific test is listed:

1. **Compiles everywhere.** `cargo check` (native), `cargo check --target wasm32-unknown-unknown`, `cargo check --target aarch64-apple-ios`, `cargo check --target aarch64-linux-android` all pass for the core crate. Set this up as a CI matrix in Step 6 and never let it regress.
2. **Has an automated test that fails if the step is wrong or is later broken by a refactor.** "I ran it and the picture looked fine" is not a test.
3. **Uses one of these verification strategies**, chosen per step below:
   - **Known-value unit test:** hand-computed input → exact (or epsilon-bounded) expected output. Used for pure math: matrices, curves, OETFs, encode/decode round trips.
   - **Synthetic image test:** generate a test image with a *known* ground truth (solid patches, ramps, checkerboards, a synthetic forward-mosaiced Bayer pattern from a known RGB source, a synthetic noisy frame with known σ, a synthetic clipped-highlight scene), run the stage, compare against ground truth with a numeric metric (PSNR/SSIM for image similarity, mean/median error for scalar corrections, ΔE2000 for color accuracy).
   - **Reference-file regression test:** a small corpus of real DNG files (aim for 3–5 diverse cameras/sensors, checked in as test fixtures or pulled by a fixture script) with **approved** outputs locked as regression baselines. A step that touches decode/demosaic/color reruns this corpus and fails if output drifts beyond a tolerance band without a human re-approving the new baseline.
   - **Determinism test:** run the same input through the same stage twice (and, where feasible, on two targets), assert bit-identical (or ULP-bounded) output. Critical because we have 4 build targets and no GPU/thread nondeterminism to hide behind — if results diverge, it's a real bug.
   - **Round-trip / invertibility test:** for operations that should be (near-)invertible (space conversions, encode/decode pairs), forward then inverse and check we land back within float epsilon.
   - **Robustness/fuzz test:** feed malformed/truncated/randomized bytes at parsers; assert "returns an `Err`, never panics, never reads out of bounds." This matters a lot more than usual because a panic inside the core crate takes down an iOS/Android host app.
4. **Ships its fixtures with it.** Any test image, reference chart value, or sample DNG needed to reproduce the test is committed alongside the code (or fetched by a documented, pinned script) — not "trust me, it worked on my machine."

Two small test-support utilities are needed early and are called out as their own step (Step 6):
- A **synthetic test-image generator** (solid patches, ramps, checkerboards, a synthetic ColorChecker-like chart with known Lab values, frequency-sweep charts for moiré, forward-mosaicing of a known RGB image into a fake Bayer/X-Trans pattern for demosaic ground truth).
- A **CIEDE2000 (ΔE00) function**, used only by tests/QC to score color accuracy against reference chart values. (Reference ColorChecker Lab/xyY values are public numeric data — safe to embed as test constants.)

---

## 6. Explicitly out of scope for this phase

Naming these so nobody accidentally scope-creeps into them, and so it's clear they're deferred, not forgotten:

- Local adjustments / masking (brushes, gradients, range masks, subject/sky detection)
- Non-destructive node-graph / DAG editing engine
- GPU compute (CUDA/Metal/Vulkan/WebGPU)
- Multi-threading / SIMD (correctness first; performance work comes after, once we have a regression suite that would catch a parallelism bug)
- Non-DNG raw formats (CR3, NEF, ARW, RAF, RW2, ORF, 3FR, …)
- Full ACES 2 output transform (Hellwig/JMh appearance model, real RGC)
- OpenEXR archival export
- 3D LUT engine / film emulation / density-domain color mixing
- Print / ICC / CMYK workflows (already excluded per the original design conversation)
- Multi-frame/burst fusion, pixel-shift, spectral camera profiling (needs a lab, not just code)
- Row/column dark-current temperature modeling beyond a basic per-channel offset (needs calibration data we don't have yet)

---

## 7. Milestone overview

| Milestone | Theme | Steps |
|---|---|---|
| M0 | Foundations & recon | 0–6 |
| M1 | RAW decoding (DNG) + walking skeleton | 7–14 |
| M2 | Raw-domain corrections | 15–21 |
| M3 | White balance, clipping, demosaic | 22–28 |
| M4 | Geometric & chromatic-aberration correction | 29–31 |
| M5 | Camera color science (IDT) | 32–37 |
| M6 | Scene-referred working spaces & grading | 38–43 |
| M7 | Gamut mapping & output transform | 44–48 |
| M8 | Export & pipeline orchestration | 49–52 |
| M9 | Cross-platform hardening & QC | 53–56 |

**Delivery strategy:** don't wait until M9 to prove the whole chain works. M1 ends with a deliberately crude but *complete* end-to-end path (Step 14, "walking skeleton") — DNG in, PNG out, wired into the CLI, buildable on all 4 targets. That de-risks the architecture (file I/O, buffer layout, FFI boundary shape) before we've sunk weeks into demosaic quality or color science. Every later milestone plugs a better stage into that same skeleton rather than bolting a new pipeline together at the end.

---

## 8. The steps

### Milestone 0 — Foundations & recon

#### Step 1 — (see §2 above: repo audit)
Covered above; listed here so numbering lines up with tracking issues.

#### Step 2 — Workspace/crate layout for core + platform shims
**Do:** Establish (or confirm/adjust, per Step 1's audit) a workspace with a dependency-free core crate (e.g. `pichromatic-core`) and thin adapter crates (`pichromatic-cli`, `pichromatic-wasm`, `pichromatic-ios`, `pichromatic-android`) that depend on it. No algorithm code lives outside the core crate.
**Test:** `cargo build --workspace` succeeds; an empty smoke test in each adapter crate calls one trivial core function (e.g. a version string) to prove the dependency wiring works.
**Done when:** all 5 crates exist, build, and the core crate's `Cargo.toml` has zero non-std dependencies.

#### Step 3 — Core image buffer & numeric types
**Do:** Define `Image { width, height, channels: SmallVecOrArray<Vec<f32>>, space: ColorSpace }` (planar float storage) and the `ColorSpace` enum from §3. Provide basic constructors, a pixel accessor, and NaN/Inf-safe helpers (`is_finite_everywhere`, `clamp_finite`).
**Test:** known-value unit tests for constructors and accessors; a property test that random-fills a buffer, round-trips through get/set, and matches.
**Done when:** the type compiles on all 4 targets and has >90% line coverage on its own module.

#### Step 4 — Metadata & pipeline config types
**Do:** Define `RawMetadata` (mirrors the fields the raw decoder will need to fill: CFA pattern, black/white levels, camera matrices, orientation, ISO, exposure time, lens info placeholder) and `PipelineConfig` (the single struct that will drive the whole linear pipeline — one field per stage's parameters, all with sane defaults).
**Test:** known-value test that `PipelineConfig::default()` produces a config that later stages accept without erroring (a "does it at least parse" test, expanded as stages land).
**Done when:** types exist, are `Clone + Debug`, and default construction is covered by a test.

#### Step 5 — Error handling strategy
**Do:** One core error enum (e.g. `PichromaticError`) with variants for decode failure, unsupported format, malformed metadata, numeric failure, etc. No `panic!`/`unwrap()`/`expect()` on untrusted input anywhere in the core crate (allowed only on truly-invariant internal logic, and even then prefer `debug_assert!`).
**Test:** a `#[deny(clippy::unwrap_used)]`-style lint (or a grep-based CI check if that lint is impractical) scoped to the core crate's non-test code; a fuzz-style test in Step 13 exercises this directly.
**Done when:** the lint/check is in CI and passing.

#### Step 6 — Test harness, fixtures & CI skeleton
**Do:** Set up the CI matrix from §5 (native + 3 cross-compile checks). Build the synthetic test-image generator and the ΔE00 test utility described in §5. Create a `test-fixtures/` (or confirm existing) directory convention for sample DNGs and reference outputs.
**Test:** the generator itself gets unit-tested (e.g. "a synthetic solid-red patch has the RGB values we asked for"; "forward-mosaicing then naively de-mosaicing a flat field returns the flat field").
**Done when:** CI runs on every PR and includes the 4-target compile matrix; the generator and ΔE00 utility are available to every later test.

---

### Milestone 1 — RAW decoding (DNG) + walking skeleton

#### Step 7 — TIFF/IFD reader primitives
**Do:** Implement a minimal, pure-Rust TIFF container reader: byte-order detection, IFD (tag directory) walking, tag value reading for the numeric/string/array types DNG actually uses. No compression support yet — just structure.
**Test:** known-value tests against a hand-built minimal TIFF byte buffer (constructed in the test itself, not a real file) asserting specific tags parse to expected values.
**Done when:** the reader can enumerate all top-level and sub-IFD tags of a real DNG file without erroring (even though we're not yet interpreting most of them).

#### Step 8 — DNG tag extraction
**Do:** Extract the tags the pipeline actually needs: `CFAPattern`, `BlackLevel`/`BlackLevelDeltaH`/`BlackLevelDeltaV`, `WhiteLevel`, `ColorMatrix1/2`, `ForwardMatrix1/2`, `CalibrationIlluminant1/2`, `AsShotNeutral`, `Orientation`, `ActiveArea`, `DefaultCropOrigin/Size`, bits-per-sample, compression tag.
**Test:** reference-file regression test against 2–3 real DNGs, asserting each extracted field matches values independently confirmed via `exiftool` (or the file's known camera spec sheet) — check these expected values into the test as constants.
**Done when:** all listed fields extract correctly for the fixture corpus.

#### Step 9 — Raw sample unpacking
**Do:** Implement unpacking for the bit depths DNG commonly uses (8/12/14/16-bit, both byte-aligned and bit-packed) into a flat `u16` (or `f32`, pre-normalization) mosaic buffer.
**Test:** known-value unit test: hand-construct a small packed byte buffer for each bit depth, assert unpacked values match by-hand-computed expectations.
**Done when:** round-trip test (pack a synthetic pattern with a tiny reference packer written in the test, unpack it, compare) passes for every supported bit depth.

#### Step 10 — Linearization table application
**Do:** Apply the DNG `LinearizationTable` (if present) to map stored codes to linear-ish codes before further processing.
**Test:** known-value test with a synthetic small LUT; identity-LUT case must be a no-op (tested explicitly).
**Done when:** presence/absence of the tag is both handled without branching bugs, covered by two separate tests.

#### Step 11 — Baseline lossless-JPEG decompression for compressed DNG
**Do:** Implement the lossless-JPEG (predictor-based) decompression scheme DNG uses for compressed tiles/strips. This is the single most algorithmically involved decode step — budget real time for it, and consider shipping uncompressed-DNG support first (Steps 7–10 alone already unblock Step 14's skeleton) and landing this as its own follow-up PR.
**Test:** reference-file regression test against real compressed DNGs from the fixture corpus, comparing decoded pixel values against a trusted independent decode (e.g., values cross-checked once by hand/via a known-good external tool, then locked as the baseline — no runtime dependency on that tool afterward).
**Done when:** the fixture corpus's compressed DNGs decode to within the locked baseline tolerance.

#### Step 12 — RawFrame assembly & decode validation tests
**Do:** Wire Steps 7–11 into a single `decode_dng(bytes: &[u8]) -> Result<RawFrame, PichromaticError>` where `RawFrame` bundles the mosaic buffer + `RawMetadata`.
**Test:** integration test decoding every fixture DNG end-to-end, asserting dimensions, CFA pattern, and black/white levels all match expectations in one pass.
**Done when:** this function is the single public entry point for DNG decode and nothing downstream touches TIFF tags directly.

#### Step 13 — Malformed-input robustness tests
**Do:** Nothing new to implement beyond defensive checks already required by Step 5 — this step is about proving it. Add bounds checks anywhere Step 7–11 indexed into a buffer using a value read from the file.
**Test:** fuzz-style test (can be a simple loop over randomized/truncated/bit-flipped copies of a fixture file, doesn't need full `cargo-fuzz` infrastructure yet, though that's a good follow-up) asserting `decode_dng` always returns `Err` or a valid `RawFrame`, never panics.
**Done when:** thousands of randomized inputs run in CI without a panic.

#### Step 14 — Walking skeleton: DNG → naive RGB → PNG
**Do:** Wire the crudest possible complete path: `decode_dng` → naive nearest-neighbor or bilinear "demosaic" (a throwaway implementation, fine to replace in Step 24) → naive black/white normalization → a hardcoded gamma curve (not real color science yet) → 8-bit PNG export (Step 50's encoder can be built early just enough for this) → callable from the CLI adapter crate.
**Test:** end-to-end test: given a fixture DNG, the CLI command produces a valid PNG file (parses back with the `png` crate, correct dimensions, not all-black/all-white). This is a smoke test, not a quality test.
**Done when:** a human can run the CLI on a real photo and see a recognizable (if ugly) picture. This is the milestone gate — don't start M2 until this works, because it validates the whole plumbing (decode → buffer → export → CLI) that every later stage builds on.

---

### Milestone 2 — Raw-domain corrections

#### Step 15 — Black level subtraction & white-level normalization
**Do:** Implement the per-channel (and, where metadata provides it, per-row/per-column) black model from the design doc, normalizing into `[~0, 1]`-ish float range without early clamping (small negatives preserved).
**Test:** known-value test on a synthetic flat-field frame with an injected known black level — output mean should match the expected normalized value within epsilon; a "no premature clamping" test asserts small negative inputs survive the stage unmodified in sign.
**Done when:** replacing the walking skeleton's naive normalization with this stage doesn't break the Step 14 smoke test and the new unit tests pass.

#### Step 16 — Hot/dead pixel detection & correction
**Do:** Implement the local-median/MAD detector and same-color-neighbor replacement described in the design doc, operating in raw mosaic space.
**Test:** synthetic image test: take a clean synthetic mosaic, inject N random hot/dead pixels at known locations, run correction, assert >95% of injected defects are corrected within a small error bound and that non-defective pixels are unchanged.
**Done when:** the synthetic-defect test passes and a reference DNG with known real sensor defects (if available in the fixture corpus) shows no visible defect artifacts in a rendered preview.

#### Step 17 — Row/column banding correction
**Do:** Implement the robust row/column offset estimation (median or trimmed-mean based; full masked-optical-black-pixel usage only if the fixture cameras expose that data, otherwise operate on the active area statistics) and subtraction.
**Test:** synthetic image test: inject a known synthetic row/column banding pattern onto a flat field, run correction, assert residual banding amplitude drops below a threshold (measured as the std-dev of row/column means after correction vs. before).
**Done when:** the synthetic banding test passes and running this stage on a clean (no banding) synthetic frame doesn't introduce artifacts (regression-tested as a no-op case).

#### Step 18 — Flat-field / vignetting correction
**Do:** Implement the radial polynomial vignetting model (`V_c(r) = 1 + a·r² + b·r⁴ + c·r⁶`) with per-channel coefficients read from `PipelineConfig` (a simple built-in generic profile is fine for now — real per-lens calibration data is future work).
**Test:** synthetic image test: apply a known synthetic vignette to a flat field, run the inverse correction with matching coefficients, assert the result is flat within a small tolerance (max deviation from mean below threshold).
**Done when:** the synthetic round-trip test passes.

#### Step 19 — Green channel equilibration
**Do:** For Bayer CFAs, compute and apply the G1/G2 gain correction described in the design doc.
**Test:** synthetic image test: inject a known G1/G2 imbalance into a synthetic Bayer mosaic, run equilibration, assert the two green planes converge (mean difference below threshold) in flat regions.
**Done when:** test passes; also add a regression check that a downstream demosaic (once Step 24 lands) shows reduced zippering on a fixture image with visible green imbalance — this second check can be added retroactively once Step 24 exists.

#### Step 20 — Noise model estimation (Poisson-Gaussian)
**Do:** Implement `σ²_c(s) = α_c·s + β_c` parameter estimation from flat-field/dark-frame-style regions of an image (or from `PipelineConfig`-supplied per-ISO constants as a starting point if we don't yet have per-camera calibration data), producing a `NoiseMap`.
**Test:** synthetic image test: generate a synthetic image with pixel noise drawn from a *known* α/β, run estimation, assert recovered α/β are within a tolerance band of ground truth.
**Done when:** estimation recovers known synthetic parameters within tolerance across at least 3 different α/β combinations (simulating different ISOs).

#### Step 21 — Raw-domain denoise
**Do:** Implement one solid, simple, CPU-cheap edge-aware denoiser first (e.g. a bilateral or guided filter operating per-channel on the mosaic, using the `NoiseMap` from Step 20 to set strength) — not the full BM3D/wavelet menu from the design doc; those are later upgrades once this baseline is proven and measured.
**Test:** synthetic image test: take a clean synthetic image, add known-σ synthetic noise, run denoise, measure PSNR/SSIM improvement vs. the noisy input and absolute PSNR vs. ground truth; assert both clear a defined threshold. Also assert a "detail preservation" check: a known sharp edge in the synthetic image doesn't get blurred beyond a tolerance (edge-width measurement before/after).
**Done when:** thresholds pass and the stage is off by default at very low estimated noise (a no-op regression test on a clean synthetic frame).

---

### Milestone 3 — White balance, clipping, demosaic

#### Step 22 — White balance application
**Do:** Implement the diagonal gain application (`RGB_wb = D_wb · RGB_cam`), normalized so green gain is 1.0, sourced from `AsShotNeutral`/camera metadata first, with a manual-gains override path in `PipelineConfig`.
**Test:** known-value test: apply known gains to a known input, assert exact output; synthetic-chart test: a synthetic neutral-gray patch, after WB, has R≈G≈B within epsilon.
**Done when:** both tests pass.

#### Step 23 — Clipping / highlight mask generation
**Do:** Implement per-channel clipped-pixel detection against `WhiteLevel` (with the near-clip threshold from the design doc), producing single/two/all-channel-clipped masks.
**Test:** synthetic image test: construct a synthetic frame with known clipped regions (single-channel, two-channel, all-channel), assert the generated masks exactly match the known ground-truth regions.
**Done when:** mask accuracy is 100% on the synthetic test (this is a deterministic classification, not a fuzzy metric).

#### Step 24 — Bilinear demosaic (reference baseline)
**Do:** Implement standard bilinear Bayer demosaic. This *replaces* the walking skeleton's throwaway demosaic and also becomes the "known-simple" baseline other algorithms are compared against in tests.
**Test:** synthetic image test: forward-mosaic a known smooth synthetic RGB image (using the generator from Step 6) into a fake Bayer pattern, demosaic it, compare against the known original via PSNR/SSIM; also a known-value test on a tiny hand-computed 4×4 mosaic patch.
**Done when:** PSNR on the smooth synthetic test clears a baseline threshold (this threshold becomes the floor that Step 25's better algorithm must beat).

#### Step 25 — Malvar-He-Cutler demosaic
**Do:** Implement the gradient-corrected bilinear (Malvar-He-Cutler) algorithm as the new default.
**Test:** same synthetic PSNR/SSIM harness as Step 24, asserting this algorithm measurably beats the Step 24 bilinear baseline on both the smooth-gradient test and a higher-frequency synthetic test (fine stripes/edges) where bilinear is known to be weak.
**Done when:** both comparative thresholds pass; the bilinear implementation is kept (not deleted) as the test baseline and as a fast-preview option in `PipelineConfig`.

#### Step 26 — False-color / zipper suppression
**Do:** Implement a local chroma-consistency pass (median-of-neighborhood chroma difference smoothing near detected high-frequency/edge regions) to reduce zippering/maze artifacts from Step 25's output.
**Test:** synthetic image test using a moiré/frequency-sweep chart from the Step 6 generator; measure a "false color" metric (chroma variance in regions that are known-neutral/known-single-hue in ground truth) before/after, assert reduction.
**Done when:** the false-color metric improves on the synthetic torture chart without regressing the Step 25 PSNR thresholds on the smooth/edge tests (regression-tested together).

#### Step 27 — Highlight reconstruction
**Do:** Implement single-channel and two-channel clipped-highlight reconstruction using the ratio-based methods from the design doc, driven by the Step 23 masks, running on the demosaiced (or partially-demosaiced) data as appropriate.
**Test:** synthetic image test: construct a synthetic scene with known pre-clip color and artificially clip 1 or 2 channels, run reconstruction, compare reconstructed color against the known pre-clip ground truth (ΔE00 threshold).
**Done when:** reconstructed hue/chroma stays within a defined ΔE00 tolerance of ground truth for both the single- and two-channel-clipped synthetic cases.

#### Step 28 — Demosaic quality regression suite
**Do:** Consolidate Steps 24–27's synthetic tests plus the real fixture corpus into one CI-tracked regression suite with locked baseline scores (PSNR/SSIM/false-color metric/ΔE00), so any future change to this area shows its impact immediately.
**Test:** is the test — this step is about assembling and locking the harness itself.
**Done when:** the suite runs in CI on every PR touching the demosaic module and fails on regression beyond a defined tolerance.

---

### Milestone 4 — Geometric & chromatic-aberration correction

#### Step 29 — Lateral chromatic aberration correction
**Do:** Implement the per-channel radial scaling model (`r_R' = r·(1 + k_R1·r² + k_R2·r⁴)`, etc.) with coefficients from `PipelineConfig` (generic defaults for now; real per-lens profiles are future work, same caveat as Step 18).
**Test:** synthetic image test: apply a known synthetic lateral-CA displacement to a synthetic high-contrast edge image, run correction with matching coefficients, assert channel misalignment (measured as the spatial offset between channel edge positions) drops below a pixel-fraction threshold.
**Done when:** the synthetic round-trip test passes.

#### Step 30 — Radial lens distortion correction
**Do:** Implement the radial (`k1,k2,k3`) + tangential (`p1,p2`) distortion model from the design doc.
**Test:** synthetic image test: render a synthetic grid pattern, apply a known synthetic distortion, run correction with matching coefficients, measure grid-line straightness (deviation from a fitted line) before/after.
**Done when:** post-correction grid deviation drops below a defined pixel threshold.

#### Step 31 — Unified geometric warp + Lanczos resampler
**Do:** Compose Steps 29–30 (plus crop/rotation) into a single resampling pass using Lanczos interpolation, so the image is only resampled once rather than once per correction (per the design doc's "avoid repeated resampling" rule).
**Test:** determinism/quality test: verify that composing-then-single-resampling produces measurably less blur/ringing than naively chaining two separate nearest/bilinear resample passes on the same synthetic distorted+CA-shifted grid (compare edge sharpness metric between the two approaches). Also a known-value test on the Lanczos kernel itself against hand-computed coefficients.
**Done when:** the composed single-pass path is the only one wired into the pipeline, and its quality advantage over naive chaining is captured as a regression-tested number.

---

### Milestone 5 — Camera color science (IDT)

#### Step 32 — 3×3 matrix / linear algebra primitives
**Do:** Implement a small, dependency-free `Mat3` type (multiply, invert, matrix-vector apply) — just enough linear algebra for color matrices, not a general linear algebra crate.
**Test:** known-value tests: multiply/invert against hand-computed examples; `M * M⁻¹ ≈ I` round-trip test with `approx`-based epsilon comparison (dev-dependency).
**Done when:** all tests pass and this is the only matrix type used anywhere in the color pipeline (no ad-hoc matrix math scattered elsewhere).

#### Step 33 — Camera-to-XYZ via DNG ColorMatrix/ForwardMatrix
**Do:** Implement the camera-to-XYZ transform path using the DNG `ColorMatrix`/`ForwardMatrix`/`CameraCalibration` tags extracted in Step 8, being careful about which direction each matrix maps (per the design doc's warning about `ColorMatrix` being XYZ→camera and `ForwardMatrix` being camera→PCS).
**Test:** known-value test using a real fixture DNG's embedded matrices: convert its `AsShotNeutral` white-balanced-neutral through this path and assert the result lands near the expected neutral point in XYZ (i.e., a near-gray input should map to a near-gray-axis XYZ, not an arbitrarily tinted one).
**Done when:** the neutral-axis test passes for every fixture camera profile.

#### Step 34 — Dual-illuminant interpolation (mired space)
**Do:** Implement mired-space linear interpolation between `ColorMatrix1`/`ColorMatrix2` (and their `ForwardMatrix` counterparts) based on estimated/selected white balance color temperature.
**Test:** known-value tests at the two endpoint illuminants (interpolated matrix must exactly equal the corresponding stored matrix at `t=0` and `t=1`) and a monotonicity test (matrix coefficients vary smoothly, no discontinuity, across the interpolation range).
**Done when:** endpoint and monotonicity tests pass.

#### Step 35 — Bradford chromatic adaptation transform
**Do:** Implement the Bradford CAT as a `Mat3`-based function (fixed cone-response matrix constants from the published spec) for adapting between illuminant white points.
**Test:** known-value test: adapting a known white point to itself is the identity transform; adapting D65→D50 (or another well-documented pair) matches published reference values within epsilon.
**Done when:** both tests pass.

#### Step 36 — XYZ → ACES AP0/AP1 fixed-matrix conversion
**Do:** Hard-code the XYZ→AP0 (ACES2065-1) and AP0↔AP1 (ACEScg) matrices from SMPTE ST 2065-1 / Academy S-2014-001, plus their inverses.
**Test:** known-value tests against the published reference matrix constants (transcribed once, carefully, into test assertions distinct from the production constants — i.e., don't just assert the code equals itself); round-trip test (`XYZ → AP0 → XYZ ≈ identity`).
**Done when:** both tests pass.

#### Step 37 — Camera profile validation against reference chart
**Do:** Wire Steps 33–36 into one `camera_to_aces(RawFrame_metadata, white_balance) -> Mat3` profile function. Validate it against a physically photographed ColorChecker-style chart (one of the fixture DNGs should be a chart shot, ideally under a known illuminant).
**Test:** using the Step 6 ΔE00 utility and the public reference chart Lab values, measure ΔE00 between the pipeline's rendered chart-patch colors and the reference values (after also applying a matching neutral output transform from M7, or a simple placeholder linear-to-sRGB for this test if M7 isn't done yet). Track the number as a regression baseline, don't just threshold it — chart ΔE alone isn't the whole story (per the design doc's warning), but it's a necessary sanity check.
**Done when:** the fixture chart shot produces a ΔE00 report that's reviewed by a human before being locked as the regression baseline.

---

### Milestone 6 — Scene-referred working spaces & grading

#### Step 38 — Linear working-space operator: exposure
**Do:** Implement `RGB' = RGB · 2^EV` on ACEScg-space images, no clamping.
**Test:** known-value test for exact scaling; a "values above 1.0 survive" test (explicit regression against accidental clamping).
**Done when:** both pass.

#### Step 39 — Log grading space (ACEScct-equivalent) encode/decode
**Do:** Implement the ACEScct piecewise log curve (per Academy S-2016-001) as `acescg_to_acescct`/`acescct_to_acescg`.
**Test:** known-value tests at documented reference points (e.g. the curve's linear-segment breakpoint) matching the published spec values; round-trip test (`encode(decode(x)) ≈ x` and vice versa) across a sweep of values including negatives and values above 1.0.
**Done when:** both pass across the full tested value sweep.

#### Step 40 — Log-domain contrast operator
**Do:** Implement the pivot-based log-domain contrast operator from the design doc (`L' = pivot + contrast·(L - pivot)`, reapplied as a luminance-ratio scale on RGB).
**Test:** known-value test: contrast=1.0 is an exact no-op; a specific contrast value against a hand-computed expected output for a known input.
**Done when:** both pass.

#### Step 41 — Lift/Gamma/Gain (ASC CDL-style) operator
**Do:** Implement `out = clamp(in·slope + offset, 0, ∞)^power` plus saturation-around-luma, operating in ACEScct.
**Test:** known-value tests for each of slope/offset/power independently (isolate one parameter at a time against hand-computed expected values); identity-parameters no-op test.
**Done when:** all pass.

#### Step 42 — Luminance-preserving saturation operator
**Do:** Implement `RGB' = Y + s·(RGB - Y)` with the configured luminance weights.
**Test:** known-value test; a "saturation=1.0 is identity, saturation=0.0 produces exact luma-matched gray" pair of tests.
**Done when:** both pass.

#### Step 43 — Space-conversion round-trip test suite
**Do:** Consolidate Steps 33–39's round-trip properties into one CI-tracked property-test suite (random-sampled inputs, not just fixed known-values) covering camera→XYZ→ACES→ACEScg→ACEScct and back.
**Test:** is the test — property-based (via `proptest` dev-dependency or a hand-rolled random sweep) round-trip epsilon assertions across the whole chain.
**Done when:** the suite runs in CI and passes across at least 10,000 random samples per property.

---

### Milestone 7 — Gamut mapping & output transform

#### Step 44 — Oklab/OkLCh perceptual space conversion
**Do:** Implement the published Oklab matrices/cube-root nonlinearity and the Cartesian↔polar (OkLCh) conversion.
**Test:** known-value tests against the published Oklab reference conversions (e.g. converting known sRGB primaries and checking against publicly documented Oklab values); round-trip test (`RGB → OkLab → RGB ≈ identity`).
**Done when:** both pass.

#### Step 45 — Hue-preserving gamut compression
**Do:** Implement the soft chroma-compression function from the design doc (`M' = threshold + (M-threshold)/(1+(M-threshold)/shoulder)` beyond a threshold) operating in OkLCh against a target display gamut boundary function.
**Test:** synthetic test: feed colors known to be outside the target gamut (constructed directly in OkLCh at chroma values that exceed the gamut boundary), assert (a) hue (`h`) is unchanged before/after within epsilon, (b) resulting chroma is inside the boundary, (c) in-gamut colors below threshold are untouched (no-op regression).
**Done when:** all three assertions pass across a sweep of hues and lightness levels.

#### Step 46 — Filmic tone-mapping operator
**Do:** Implement a simple, well-understood filmic/shoulder tone curve (e.g. a Reinhard-style or a parametric shoulder curve) applied to lightness in the perceptual space, explicitly documented in code comments as the Phase-1 stand-in for a full ACES 2 tone-scale.
**Test:** known-value tests at key points (curve passes through defined black/mid/white points; curve is monotonic — tested via a dense sweep asserting non-decreasing output); a "no negative or >1 output for any finite input" test.
**Done when:** all pass.

#### Step 47 — Display OETFs (sRGB / Display P3 / Rec.709)
**Do:** Implement the sRGB piecewise OETF (IEC 61966-2-1), Rec.709 OETF (ITU-R BT.709), and Display P3's OETF (same curve as sRGB, different primaries — Apple's published Display P3 spec), plus the corresponding primaries matrices for AP1→each display gamut.
**Test:** known-value tests against each standard's published reference points (e.g. sRGB's exact breakpoint at 0.0031308, known input/output pairs from the spec); round-trip test (`OETF(EOTF(x)) ≈ x`) for each curve.
**Done when:** all three curves pass their known-value and round-trip tests.

#### Step 48 — Output-transform regression suite
**Do:** Wire Steps 44–47 into one `render_to_display(AcesCg_image, target: DisplaySpace) -> Image` function and build a CI-tracked regression suite: known ACES input colors (including deliberately out-of-gamut and >1.0 "highlight" values) mapped to expected display output ranges/values.
**Test:** is the test — assert ACES 100%-reflectance white maps to display white within tolerance; assert no output pixel is outside `[0,1]` for the target encoding; assert hue-preservation holds end-to-end (not just within Step 45 in isolation) for a sweep of saturated synthetic inputs.
**Done when:** the suite runs in CI and is the locked baseline for this module.

---

### Milestone 8 — Export & pipeline orchestration

#### Step 49 — Quantization & dithering (8/16-bit)
**Do:** Implement float→integer quantization with ordered or blue-noise dithering to avoid banding in smooth gradients, for both 8-bit and 16-bit output.
**Test:** synthetic image test: quantize a smooth synthetic gradient with and without dithering, measure banding via a histogram-based metric (e.g. count of "flat run" lengths), assert dithered output has meaningfully shorter flat runs; known-value test that quantization without dithering rounds correctly at exact boundary values.
**Done when:** both pass.

#### Step 50 — PNG export encoder integration
**Do:** Wire the `png` crate (the one pre-approved exception from §4) behind an `export::write_png` function taking an already-quantized buffer; replace the walking skeleton's placeholder export with this.
**Test:** integration test: export a known buffer, re-read it with the same `png` crate's decoder, assert byte-exact match; a metadata test asserting correct bit-depth/color-type flags are written for both 8-bit and 16-bit paths.
**Done when:** both pass and Step 14's CLI path uses this instead of its placeholder.

#### Step 51 — Full linear Pipeline struct
**Do:** Assemble every stage from M1–M7 into one ordered `Pipeline::process(raw_bytes: &[u8], config: &PipelineConfig) -> Result<Image, PichromaticError>` function — literally an ordered sequence of function calls per §1's "pipeline, not DAG" rule, each stage's output space type-checked into the next stage's expected input space.
**Test:** a compile-time-flavored test: the function signature itself proves every stage's output/input color-space tags line up (if they don't, it won't compile); a runtime smoke test that the full function runs without error on every fixture DNG.
**Done when:** the full chain runs end-to-end on all fixtures without error, and the walking skeleton's ad-hoc wiring from Step 14 is deleted in favor of this.

#### Step 52 — End-to-end golden-image integration test
**Do:** For each fixture DNG, run the full `Pipeline::process` with a fixed default `PipelineConfig`, and lock the output PNG as a golden/reference image (human-reviewed once).
**Test:** is the test — byte-level or near-byte-level (small tolerance for legitimate float rounding differences across a refactor) comparison against the golden image on every CI run.
**Done when:** golden images exist for every fixture and the suite is green in CI; this becomes the single highest-value regression test in the whole repo, since it exercises the entire chain at once.

---

### Milestone 9 — Cross-platform hardening & QC

#### Step 53 — Determinism tests
**Do:** Nothing new to implement — add tests that run `Pipeline::process` twice on the same input (same process, and, where CI infrastructure allows, compare native output against a `wasmtime`-executed wasm32 build's output) and assert results match.
**Test:** is the test — bit-identical (or documented ULP-bounded, if any float operation ordering legitimately differs across targets — investigate and fix rather than accept if so) comparison.
**Done when:** native-vs-native repeat runs are bit-identical, and native-vs-wasm32 (via `wasmtime`) is identical within a documented, justified tolerance.

#### Step 54 — Cross-compilation smoke tests
**Do:** Confirm (this should already be true from Step 2/6's CI matrix, but now exercise it against the *complete* pipeline, not just empty crates) that `cargo build --target` succeeds for wasm32-unknown-unknown, aarch64-apple-ios, and aarch64-linux-android with the full pipeline code included.
**Test:** CI job per target; for wasm32, additionally load the built `.wasm` in a minimal `wasmtime` harness and call `Pipeline::process` on a small fixture, asserting it returns successfully.
**Done when:** all 3 cross-compile targets build the full pipeline and the wasm32 target's runtime smoke test passes.

#### Step 55 — Minimal FFI boundary per platform + smoke tests
**Do:** In the platform adapter crates (not core): wire minimal bindings — `wasm-bindgen` export for wasm, a `uniffi` (or hand-written `cbindgen`) header for iOS/Android — exposing just enough of `Pipeline::process` (byte buffer in, byte buffer out, config as a simple struct/JSON) to prove the boundary shape works. Full native UI integration is a separate app-level effort, not part of this doc.
**Test:** per-platform smoke test: wasm — a small JS test harness (e.g. Node + the built wasm module) calls the exported function on a fixture and checks it returns a valid PNG buffer; iOS/Android — at minimum, confirm the generated headers/bindings compile against a trivial host-language snippet (a full simulator/emulator run is a nice-to-have, not a blocker for this phase).
**Done when:** the wasm smoke test passes in CI; iOS/Android binding compilation is verified (simulator/emulator execution tracked as a fast-follow if infrastructure isn't ready yet).

#### Step 56 — Regression corpus + CI wiring
**Do:** Consolidate every regression suite built along the way (Step 28 demosaic, Step 37 chart ΔE, Step 43 round-trips, Step 48 output-transform, Step 52 golden images, Step 53 determinism) into one clearly documented `make test-full` (or equivalent) target, with a short README explaining what each suite catches and how to re-approve a baseline after an intentional change.
**Test:** is the test — this is the meta-suite. Its "done" bar is that a new contributor can run one command and understand, from failures alone, roughly which stage broke.
**Done when:** the consolidated suite runs in CI on every PR, and the README exists.

---

## 9. Future work (explicitly deferred, not forgotten)

Once the above is solid and shipping:

- Local adjustments / masking engine
- Non-destructive DAG-based editing (the node-graph architecture from the original design conversation)
- Full ACES 2 output transform (Hellwig/JMh appearance model, true RGC)
- Additional raw formats (CR3, NEF, ARW, RAF, RW2, ORF, …) — likely one milestone-sized effort *each*
- Multi-threading and SIMD, once a stable regression suite exists to catch parallelism bugs
- GPU compute paths
- OpenEXR archival export, 3D LUT engine, film emulation / density-domain color mixing
- Real per-camera and per-lens calibration data (noise models, vignetting, CA, distortion) to replace today's generic/config-driven placeholders
- Multi-frame/burst fusion, pixel-shift support
- `cargo-fuzz`-grade continuous fuzzing (Step 13 ships a lighter-weight version now)

---

## 10. Summary

~56 steps across 10 milestones. Each step is small enough to be one reviewable PR, ships with the specific test that proves it works (known-value math tests, synthetic-ground-truth image tests, real-fixture regression tests, determinism tests, or round-trip tests — never "looked fine to me"), and every step keeps the core crate dependency-free and buildable across native/wasm32/iOS/Android from day one rather than as a late integration surprise. The walking skeleton at Step 14 is the key risk-reduction move: it proves the whole architecture end-to-end early, cheaply, before any single stage's quality work begins.
