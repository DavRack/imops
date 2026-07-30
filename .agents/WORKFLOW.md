# Multi-Agent Workflow Orchestration

## Overview
This document defines the execution pipeline, shared artifacts, and feedback loop for the `/buildc` workflow, consisting of a Planner & Coder agent and a Reviewer agent.

---

## Workflow: /buildc (Feature Development & Refactoring)

### Core Principles
- **YAGNI (You Aren't Gonna Need It)**: Both agents strictly follow YAGNI principles. Do not overbuild, do not add unrequested features or preemptive abstractions.
- **Strict Prompt Fidelity**: Implement exactly what the user asks for.

### Non functional requirements
- Performance: this is a real app, the code needs to execute in a rasonable time and memory, for prototiping max 1m max 20gb ram, for production ready max 2s max 4gb ram (not exact values, just guidelines)

### Shared Artifacts Pipeline (.scratchpad/)

```text
User Input ──────► [Agent 1: Planner & Coder]
                        │
                        ▼ (Code Files & .scratchpad/plan.md)
                   [Agent 2: Reviewer]
                        │
                  ┌─────┴─────┐
                  ▼           ▼
           NEEDS_REVISION    PASS
                  │           │
      (Loop to Agent 1)     (Terminate - Success)
```

### Workflow Steps

#### Step 1: Planning and Implementation
- **Agent**: `planner_coder` (`.agents/roles/01_planner_coder.md`)
- **Action**: Plans the implementation following YAGNI principles and writes the code. Fixes issues based on reviewer feedback if in a loop.
- **Output**: Application codebase updates and (optional) `.scratchpad/plan.md`.

#### Step 2: Code Review & Validation
- **Agent**: `reviewer` (`.agents/roles/02_reviewer.md`)
- **Action**: Reviews the code changes made by Agent 1 against a strict set of questions and YAGNI principles.
- **Output**: `.scratchpad/review_report.md` (Contains Verdict: PASS / NEEDS_REVISION).

#### Step 3: Self-Correction Loop / Termination
- **Check Condition**:
  - If Agent 2's review report contains `NEEDS_REVISION`:
    - **Trigger**: Loop back to **Step 1** (Agent 1 addresses feedback).
  - Else if it contains `PASS`:
    - **Trigger**: Complete pipeline with SUCCESS status.

---

## Workflow: /investigate (Deep Investigation & Skeptical Review)

### Core Principles
- **2-Agent Pipeline**: Agent 1 conducts the initial creative investigation; Agent 2 critically evaluates and challenges the findings.
- **Skeptical Approval**: Agent 2 must remain highly skeptical and only approve (`PASS`) when provided with sufficient physical, factual, and mathematical evidence.
- **User Delivery**: Only after Agent 2 issues a `PASS` is the response finalized and presented to the user.

### Shared Artifacts Pipeline (.scratchpad/)

```text
User Query ──────► [Agent 1: Lead Investigator]
                        │
                        ▼ (.scratchpad/investigation_report.md)
                   [Agent 2: Skeptical Reviewer]
                        │
                  ┌─────┴─────┐
                  ▼           ▼
           NEEDS_REVISION    PASS
                  │           │
      (Loop to Agent 1)     (Deliver Response to User)
```

### Workflow Steps

#### Step 1: Investigation & Research
- **Agent**: `investigator` (`.agents/roles/03_investigator.md`)
- **Action**: Performs tight research, explores the problem space, proposes creative/novel solutions, and compiles theoretical and empirical evidence.
- **Output**: `.scratchpad/investigation_report.md`

#### Step 2: Skeptical Review & Accuracy Audit
- **Agent**: `skeptical_reviewer` (`.agents/roles/04_skeptical_reviewer.md`)
- **Action**: Evaluates the report strictly focused on correctness, physical accuracy, factual accuracy, and mathematical rigor. Remains skeptical and requires strong proof.
- **Output**: `.scratchpad/investigation_review.md` (Contains Verdict: `PASS` or `NEEDS_REVISION`).

#### Step 3: Feedback Loop / User Delivery
- **Check Condition**:
  - If `investigation_review.md` contains `NEEDS_REVISION`:
    - **Trigger**: Loop back to **Step 1** (Agent 1 addresses doubts and provides additional evidence).
  - Else if it contains `PASS`:
    - **Trigger**: Present the verified investigation response to the user.

