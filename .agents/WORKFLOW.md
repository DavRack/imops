# Assembly Line Multi-Agent Workflow Orchestration

## Overview
This document defines the execution pipelines, shared artifact handoffs, and feedback loop conditions for multi-agent workflows:
1. `/build-feature`: Feature Development Pipeline (Specification → Research → Coding → Reviews → Integration).
2. `/ask`: Codebase Q&A & Deep Research Pipeline (**Strictly Read-Only App Code** → Query Spec → Investigation/Probing → Audit → Final Answer).

---

## Core Workflow Principles
1. **Strict Scope & Prompt Fidelity (No Invented Goals or Features)**:
   - Build strictly what the user requested. Agents **MUST NOT** invent arbitrary goals, unrequested feature additions, unstated scope extensions, or artificial constraints/tolerances that were not specified in the user's prompt.
2. **Clarification Protocol for Ambiguity**:
   - If a requirement, parameter, design decision, or assumption is genuinely necessary to proceed but was **NOT** specified in the user's request, agents **MUST** explicitly ask the user for clarification rather than inventing arbitrary goals or assumptions on their own.
3. **No Superficial Symptom Patches or Dummy Fallbacks**:
   - Agents **MUST NOT** insert dummy fallbacks, fake default numbers, or artificial patches (e.g. magic default gains or silent fallback values) just to avoid failing tests or to "not be wrong".
   - Trace and fix the true upstream data parser/provider or handle missing data cleanly rather than inventing silent fallback values.

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

---

## Codebase Q&A & Research Pipeline (/ask-codebase)

### Trigger
`"/ask-codebase <question>"`

### Constraint Protocol
- **STRICT_READ_ONLY_APP_CODE**: Zero modifications to application source files (`pichromatic/`, `pichromatic-pipeline/`, `src/`, etc.).
- **PROBE_SCRIPTS_ALLOWED**: Agents may create temporary diagnostic scripts in `.scratchpad/scratch/` or run workspace test/benchmark commands to empirically verify behavior.

### Flow Diagram
```text
User Question ──► [Step 1: Question Clarifier]
                         │
                         ▼ (.scratchpad/qa_query_spec.md)
                  [Step 2: Codebase Investigator] (Searches, probes, runs diagnostic scripts)
                         │
                         ▼ (.scratchpad/qa_investigation.md)
                  [Step 3: QA Analyst] (Audits completeness & zero app code edits)
                         │
                         ▼ (.scratchpad/qa_audit_report.md)
                  [Step 4: Response Synthesizer]
                         │
                         ▼ (.scratchpad/final_answer.md)
```

