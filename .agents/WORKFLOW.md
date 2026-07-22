# Assembly Line Multi-Agent Workflow Orchestration

## Overview
This document defines the strict execution pipeline, shared artifact handoffs, and feedback loop conditions for feature development using an Assembly Line architecture.

---

## Core Workflow Principles
1. **Strict Scope & Prompt Fidelity (No Invented Goals or Features)**:
   - Build strictly what the user requested. Agents **MUST NOT** invent arbitrary goals, unrequested feature additions, unstated scope extensions, or artificial constraints/tolerances that were not specified in the user's prompt.
2. **Clarification Protocol for Ambiguity**:
   - If a requirement, parameter, design decision, or assumption is genuinely necessary to proceed but was **NOT** specified in the user's request, agents **MUST** explicitly ask the user for clarification rather than inventing arbitrary goals or assumptions on their own.

---

## Shared Artifacts Pipeline (.scratchpad/)

```text
User Input ──────► [Step 1: Prompt Engineer]
                        │
                        ▼ (.scratchpad/final_spec.md)
                   [Step 2: Researcher]
                        │
                        ▼ (.scratchpad/docs_context.md)
                   [Step 3: Coder]
                        │
                        ▼ (Code Files)
                ┌───────┴───────┐
                ▼ (Parallel)    ▼ (Parallel)
         [Step 4a: QA]    [Step 4b: SWE]
                │               │
                ▼               ▼ (.scratchpad/swe_report.md)
        (.scratchpad/qa_report.md)
                └───────┬───────┘
                        ▼
               [Step 5: Integrator]
                        │
                        ▼
             [Step 6: Loop Check]
           /                      \
   NEEDS_REVISION               PASS
         /                          \
(Loop to Step 4)                (Terminate - Success)
```

---

## Workflow Steps

### Step 1: Prompt Refinement
- **Agent**: `prompt_engineer` (`.agents/roles/01_prompt_engineer.md`)
- **Action**: Transform trigger prompt into structured spec.
- **Output**: `.scratchpad/final_spec.md`

### Step 2: Architecture & Context Research
- **Agent**: `researcher` (`.agents/roles/02_researcher.md`)
- **Action**: Inspect codebase, find target files, identify dependencies/APIs.
- **Output**: `.scratchpad/docs_context.md`

### Step 3: Initial Code Implementation
- **Agent**: `coder` (`.agents/roles/03_coder.md`)
- **Action**: Implement feature according to spec and research context.
- **Output**: Application codebase updates.

### Step 4: Parallel Code Review (Concurrent Step)
- **Step 4a**:
  - **Agent**: `qa_reviewer` (`.agents/roles/04_qa_reviewer.md`)
  - **Action**: Audit correctness, boundary conditions, edge cases.
  - **Output**: `.scratchpad/qa_report.md` (Contains Verdict: PASS / NEEDS_REVISION)
- **Step 4b**:
  - **Agent**: `swe_reviewer` (`.agents/roles/05_swe_reviewer.md`)
  - **Action**: Audit architecture, design patterns, maintainability.
  - **Output**: `.scratchpad/swe_report.md` (Contains Verdict: PASS / NEEDS_REVISION)

### Step 5: Code Integration & Refactoring
- **Agent**: `integrator` (`.agents/roles/06_integrator.md`)
- **Action**: Apply reviewer suggestions from QA and SWE reports.
- **Output**: Codebase updates.

### Step 6: Self-Correction Loop / Termination Condition
- **Check Condition**:
  - If `.scratchpad/qa_report.md` or `.scratchpad/swe_report.md` contains `NEEDS_REVISION` AND loop count < max_iterations:
    - **Trigger**: Loop back to **Step 4** (Parallel Reviewers re-audit code).
  - Else if both contain `PASS`:
    - **Trigger**: Complete pipeline with SUCCESS status.
  - Else (max_iterations reached):
    - **Trigger**: Terminate with WARNING (max revision attempts reached).
