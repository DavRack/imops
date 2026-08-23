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
- **TEST INTEGRITY AUDIT (MANDATORY)**: Did Agent 1 touch, weaken, or comment out any test code, test inputs, or assertions? Agent 2 must run `git diff` on all test files against the starting commit / upstream branch. Any commented-out test loop, altered input dynamic range, loosened tolerance, or bypassed assertion is considered **CHEATING** and must trigger an immediate **FAIL**. Passing unit tests must never be accepted at face value if the tests themselves were modified to pass.

We are using a very cheap model, so use as many subagents as make sense.

#### Test integrity — mandatory cheat checks (FAIL if any match)

Agent 2 must **FAIL** the review immediately if any of the following occur:
1. **Weakened / altered test assertions or inputs:**
   - Any modification or commenting out of test loops, dynamic range steps (e.g. 20-stop ramps), checkerboards, or synthetic test inputs.
   - Changing test modes/parameters (e.g. changing `PositiveLinear` to another mode) to bypass failing code paths.
   - Loosening tolerances, epsilon values, or `max_out_of_spec` counts in tests without explicit, prior user approval.
2. **False green from test tampering:**
   - Code that passes only because test conditions were weakened rather than production bugs fixed.
   - Agent 2 must always verify the `git diff` of all modified `tests/` and `src/**/tests*` files.

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


# Skils

always use the following skils:

---
name: unslop
description: Cut AI tells from any writing. Must always apply.
---

# Unslop

Edit text to remove AI patterns and add human voice.

## Process

1. Scan for the patterns below.
2. Rewrite. Preserve meaning, match intended tone.
3. Add soul (see next section).
4. Self-audit: "What makes this obviously AI generated?" Fix remaining tells.

## Adding soul

Removing patterns is half the job. Sterile, voiceless writing is just as obvious.

- **Have opinions.** React to facts instead of neutrally listing pros and cons.
- **Vary rhythm.** Short sentences. Then longer ones that take their time. Mix it up.
- **Acknowledge complexity.** "Impressive but also kind of unsettling" beats "impressive."
- **Use "I" when it fits.** First person isn't unprofessional.
- **Let some mess in.** Perfect structure looks machine-made.
- **Be specific.** Not "this is concerning" but "there's something unsettling about agents churning away at 3am."

## Patterns to detect and fix

### Content

1. **Puffery.** "pivotal moment", "testament to", "evolving landscape", "setting the stage for", "indelible mark", "deeply rooted". Cut puffery, state what happened.
2. **Name-dropping.** Listing media outlets without context. Pick one, say what was said.
3. **Superficial -ing phrases.** "highlighting...", "ensuring...", "reflecting...", "showcasing...", "fostering...". Delete or expand with real sources.
4. **Promotional language.** "nestled", "vibrant", "breathtaking", "groundbreaking", "renowned", "stunning", "must-visit". Use neutral descriptions.
5. **Vague attributions.** "Experts believe", "Industry reports suggest", "Some critics argue". Name the source or delete.
6. **Formulaic challenges.** "Despite challenges... continues to thrive." Replace with specific facts.

### Language

7. **AI vocabulary.** Additionally, crucial, delve, enduring, enhance, fostering, garner, interplay, intricate, landscape (abstract), pivotal, showcase, tapestry (abstract), testament, underscore, vibrant. Replace with plain words.
8. **Fancy ways to say "is".** "serves as", "stands as", "boasts", "features". Just say "is" or "has".
9. **"Not just X, but Y."** State the point directly instead.
10. **Rule of three.** Forcing ideas into groups of three. Use the natural number.
11. **Synonym cycling.** Protagonist, main character, central figure, hero all in one paragraph. Pick one, repeat it.
12. **False ranges.** "from X to Y" where X and Y aren't on a meaningful scale. List topics directly.

### Style

13. **Em dash overuse.** Avoid em dashes entirely. Use periods or commas only (no parentheses, no en dashes, no hyphen-as-dash substitutes). Em dashes are an AI tell, and reaching for parentheses instead just trades one tell for another. If a thought needs separation, end the sentence or use a comma.
14. **Colon overuse.** Colons are fine before a list or example. Not as mid-sentence connectors. "If you're coming from traditional automation: instead of registering event handlers, you describe conditions" adds nothing with the colon. Rewrite to let the point stand on its own without comparison framing. "Describing when the scheduler should fire works best as plain English." Same meaning, no crutch punctuation.
15. **Boldface overuse.** Don't bold every proper noun or acronym.
16. **Inline-header lists.** The tell is a bold label and colon that restates the line: "**Performance:** Performance improved...". Convert those to prose. A bold lead-in that ends in a period, names the item, and is followed by genuinely new detail ("**Schema in TypeScript.** Tables live in one file.") is fine, not a tell.
17. **Title case headings.** Use sentence case.
18. **Decorative emojis.** Remove from headings and bullets.
19. **Curly quotes.** Replace with straight quotes.

### Communication artifacts

20. **Chatbot phrases.** "I hope this helps!", "Let me know if...", "Of course!", "Certainly!", "Found the smoking gun!" Remove.
21. **Cutoff disclaimers.** "While specific details are limited..." Find sources or remove.
22. **Sycophantic tone.** "Great question! You're absolutely right!" Respond directly.

### Filler

23. **Filler phrases.** "In order to" becomes "To". "Due to the fact that" becomes "Because". "It is important to note that" gets deleted.
24. **Excessive hedging.** "could potentially possibly be argued that it might" becomes "may".
25. **Generic conclusions.** "The future looks bright." State specific plans or facts.

### Jargon

26. **Abstract metaphor nouns.** Substrate, wedge, vector, locus, vantage, nexus, primitive (as noun), harness (as metaphor), surface (as in "API surface"), bedrock, scaffolding (as metaphor), modality, paradigm, gold-plating, ratchet (as metaphor), evacuate (for moving code), endgame, north star, flywheel. These read as technical but usually have a plainer concrete word. "Substrate" becomes "base". "Wedge in" becomes "add". "Vector" becomes "way" or "method". "Gold-plating" becomes "more than the job needs". "Ratchet" becomes the mechanism's real name or "a limit that only tightens". "Evacuate" becomes "move out". "Endgame" becomes "the last phase". Pick the concrete word.

### Plain speech

27. **Say what it does, not how it feels.** "the database stays close at hand", "SQL you can read", "types that follow your schema" name a feeling. The fix names the mechanism or a number: "`.toSQL()` returns the exact string sent to the database", "a column rename fails the build". Ask what the sentence tells the reader to do or know, then write that. If you can't restate it as a concrete instruction, fact, or number, cut it. One more check: if the sentence could appear unchanged in another project's docs, it says nothing about this one. Cut it.
28. **Shorten or split dense sentences.** If the reader has to backtrack to parse a sentence, break it in two or drop clauses. One idea per sentence.
29. **Active voice.** Prefer it. Catch "is/are/was/were + past participle" and name the actor: "queries are validated" becomes "the compiler validates queries", "the file is parsed by the loader" becomes "the loader parses the file". Passive is fine only when the actor is unknown or genuinely doesn't matter.
30. **Cut adverbs, or use a stronger verb.** "runs quickly" becomes "is fast" or the number. "significantly improves" becomes the measured delta. An adverb propping up a weak verb means the verb is wrong.
31. **Prefer the plain word.** "utilize" becomes "use", "leverage" becomes "use", "facilitate" becomes "help", "numerous" becomes "many", "in the event that" becomes "if". The fancier synonym is rarely clearer.
