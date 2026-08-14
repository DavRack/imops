# Changes Made
1. **Grain-Coupled Fine-Pitch DIR / Adjacency**:
   - Modified `apply_particle_grain_overwrite` in `grain.rs` to operate on $f_{dev}$ (developable fraction) directly.
   - DIR inhibition and Eberhard adjacency are now correctly coupled to the stochastically resolved particle fraction $f_{dev}$ instead of the expected / smooth scene baseline, maintaining physics constraints.
   - Skipped layers keep their raw expected $f_{dev}$ without destroying the data in the planes loop, ensuring exact matches when elements are skipped without crashing tests.
   - Updated `fine_pitch_reference` and `develop` in `mod.rs` to provide `sigma_dir_px`, `dir_inhibition_matrix`, `sigma_px`, and `adjacency_beta`.
   - Adjusted `grain.rs` unit tests to include empty DIR matrix arguments for backward compatibility.
2. **Deep Shadow Latitude Fine-Tuning**:
   - Increased the `sigma_ln` on the `blue_fast`, `green_fast`, and `red_fast` emulsion layers of `Portra 400` from `0.65` to `0.88`. This widens the crystal distribution and correctly deepens the toe curve to support ~3.5 to 4 stops of shadow latitude in `fast_layer_capture_toe_has_shadow_latitude` logic.
   - Checked no pins (e.g. `H_PIN` / `capture_k` logic or test pin invariants) were improperly touched. All tests run locally pass without any baseline pin failure.

Tests were successfully compiled and run under `cargo test --release -p pichromatic`.
