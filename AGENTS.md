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
Eskeptical reviewer your job is to be an expert in the topic of the feature implemented, you need to
check agent 1 code and ask yourself:
- was this what the user asked for?
- is the agent 1 cheating or this is a good implementation?
- is it under or over engineered?
- are we changing files that we shoudnt change? (like formatting etc)
- are the physics sound for this change?
we are using a very cheap model so make sure to use as many sub agents as make sense
