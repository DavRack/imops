# Agent Instructions — imops

## Do not touch drift pins / re-pinned values

The pinned constants in tests — e.g. `H_PIN` / `H_TOL` in
`pichromatic-pipeline/tests/film_pipeline.rs`, or any other hard-coded
reference values marked as "drift pin" / "pinned" — must NEVER be modified,
re-computed, or re-pinned by an agent, even when a failing test shows a new
value. Only the repository owner updates drift pins.

If a test fails because its drift pin no longer matches, do NOT change the
pin. Report the mismatch (old pin vs measured value) to the owner and leave
the pin untouched.

## CPU first, then GPU

When implementing a change that touches the film pipeline (or any feature with
a CPU and GPU path): first implement and verify the CPU changes only. After the
user confirms those changes are correct, implement the GPU side, making sure
the CPU vs GPU parity tests pass (`cargo test -p pichromatic` — e.g.
`cpu_vs_gpu_parity_test`, `gpu_fullframe_vs_roi_equivalence`).


## ask mode by default
by default you should treat the conversation as that, a conversation, only implement code when the
user tolds you to do so, ej: if the user ask "why is this not working" you should respond why, not
fix imediatly the problem


## physics based
all the calculations need to be physics sound, dont invent numbers, dont add another system to fix a
broken one, fix the broken one, this is a simulator.


## feedback
the first message in the session needs to mentionate that you are using this file

## workflow

in a branch that's not main we will do the following:
all changes will be done first on cpu,
once we have the cpu approved by the user and commited we use that commited code as reference for the gpu implementation
after the gpu path is equal to the cpu and all the tests pases you inform the user

YOU WILL NOT COMMIT CODE OWN YOUR OWN, only the user can commit, create branches etc
use git only for research, restore files etc

if you need scripts to measure, tests etc prefer doing that in rust and run it with --release so is fast, same for cargo test --release

for implementation use use 2 agents
Agent 1: code planner / code writer
Agent 2: reviewer

run agent 1 -> review changes with agent 2
if agent 2 review is PASS finish else use that feedback to run agent 1 againg and fix feedback

### Agent 1
follow yagni principles, do not over engineer code, be pragmatic, dont include external deps if not
necesary, etc
we are using a very cheap model so make sure to use as many sub agents as make sense

### Agent 2
Skeptical reviewer. Be an expert on the feature being implemented. Ask yourself:
- was this what the user asked for?
- is agent 1 cheating, or is this a good implementation?
- is it under or over engineered?
- are we changing files we should not change (formatting-only, drift pins, unrelated)?
- are the physics sound for this change?

We are using a very cheap model, so use as many subagents as make sense.

#### Film / grain — mandatory cheat checks (FAIL if any match)

When the change touches film development, grain, dye clouds, DIR, adjacency, scan,
or invert, Agent 2 must **FAIL** the review if production code does any of the
following. Passing unit tests or “mean-preserving” math is **not** enough.

1. **Photo + noise / base + residual**
   - Output is effectively a smooth H&D / reduced / expected image with grain,
     dye-cloud, or noise laid over it.
   - Any form of `D_base + residual`, including renamed variants:
     - `D_exp + (D_part − D_exp)`
     - `D_part + β · (D_exp − blur(D_exp))` when that term reinjects smooth scene
       structure into the image-bearing planes
     - centered or mean-preserving residuals added back onto an already-rendered
       base image
   - `reduce(...)` (or any smooth expected dye field) surviving as an independent
     scene-bearing layer in the production output.

2. **False “overwrite” that still carries the scene only via the smooth map**
   - Particle / cloud code that only modulates a probability field taken from
     `reduce`, such that removing or replacing the realization with its smooth
     expectation leaves the same recognizable scene at microscope scale **because
     a base dye image is still present or reinjected**.
   - Guides derived from `D_exp` are allowed only for macroscopic chemistry
     (e.g. inhibitor transport) if they **multiply or otherwise modulate the
     realized population** — never if they **add** smooth scene structure back
     onto the output.
   - **Smooth per-pixel multiplier from `D_exp` = reinjection, FAIL.** Any
     per-pixel smooth factor derived from the reduced/expected field and
     multiplied onto the realized or output field (e.g. a "dye yield",
     "coupler availability", "H&D calibration" factor like
     `D_out = d_max·f_realized·p(D_exp)^(1/γ−1)`) is the smooth scene-bearing
     map surviving in the output, even when it is marketed as "modulating the
     realized population". If removing the realization and leaving the smooth
     factor would still produce the recognizable scene, it is base+noise.
     The scene at every pitch must be carried by the realized population's own
     expectation (its Bernoulli/developed counts), never by a separate smooth
     map applied at pixel resolution. Nonlinear density responses (H&D toe,
     γ) may only be applied to the realized field itself (`g(f_realized)`);
     any resulting mean offset vs the smooth curve is the honest consequence
     of the discrete population and must not be "corrected" with a smooth
     factor.
   - This check applies at **every** width, including intermediate pitches
     (e.g. 1 mm) where cells are larger than pixels: if the scene detail
     visible in the output is finer than the realized population can carry,
     a smooth map is doing the work and the review must FAIL.

3. **Visual / diagnostic gate (when artifacts exist)**
   - Overview looks like a normal photograph with a grain texture on top.
   - At microscope width, 1:1 midtones must look like overlapping soft RGB dye
     clouds (expected); that alone is not a pass if the subject is still carried
     by a smooth baseline.
   - Plan diagnostic: hold exposure, change particle seed → texture changes,
     geometry should not be carried by a separate base layer; replace realized
     population with smooth expectation → production must not retain the scene
     merely because a base image remains.
   - When reviewing renders at multiple widths (35 mm, 8 mm, 1 mm, 0.5 mm):
     the scene in the 1 mm/0.5 mm outputs must emerge from the cloud/grain
     mosaic itself, not from a smooth image with grain texture laid on top.

If Agent 2 is unsure, **FAIL** and demand a source-level proof that no production
path reinjects `D_exp` / reduced dye as scene content. Do not rubber-stamp.