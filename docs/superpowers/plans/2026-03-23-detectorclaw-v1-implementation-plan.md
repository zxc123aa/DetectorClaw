# DetectorClaw V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the DetectorClaw research/spec baseline and define the minimum V1 architecture for online diagnosis-first development.

**Architecture:** Start with documentation and interface contracts, not premature software. V1 is organized around a shared shot data model, an online diagnosis workflow, and human-in-the-loop decision outputs; joint inversion is reserved for the next implementation phase.

**Tech Stack:** Markdown-first repository, future codebase unspecified, physics-led detector pipelines, auditable schema contracts.

---

### Task 1: Formalize the system spec

**Files:**
- Create: `docs/superpowers/specs/2026-03-23-detectorclaw-design.md`

- [x] **Step 1: Convert the exported discussion into a formal spec**
- [x] **Step 2: Lock V1 as online-diagnosis-first**
- [x] **Step 3: Document four layers, public contracts, and validation criteria**
- [x] **Step 4: Review for scope control**

Expected result: a spec that is strong enough for proposal writing and concrete enough for future engineering decisions.

### Task 2: Define the shared data contracts

**Files:**
- Create: `docs/architecture/detectorclaw-data-contracts.md`

- [x] **Step 1: Define the shared entities**
Document `ShotContext`, `DetectorPayload`, `OnlineAssessment`, and `DecisionRecord`.

- [x] **Step 2: Add field-level semantics**
Mark required fields, optional fields, provenance fields, and version references.

- [x] **Step 3: Add an example lifecycle**
Show how one shot moves from raw detector inputs to an operator-facing recommendation.

- [x] **Step 4: Review for extensibility**
Confirm the contracts can later absorb joint inversion outputs without breaking V1.

### Task 3: Define the V1 online diagnosis workflow

**Files:**
- Create: `docs/architecture/detectorclaw-online-diagnosis-v1.md`

- [x] **Step 1: Define the workflow stages**
Cover pre-shot setup, in-shot ingestion, online checks, recommendation synthesis, and operator confirmation.

- [x] **Step 2: Define failure and downgrade paths**
Specify what happens when a detector is missing, noisy, saturated, or inconsistent.

- [x] **Step 3: Define the confidence model**
Describe how confidence is reported in V1 without pretending to have a full probabilistic inference engine.

- [x] **Step 4: Review for human-in-the-loop safety**
Ensure the workflow produces advice, not autonomous control.

### Task 4: Create the near-term execution map

**Files:**
- Modify: `docs/superpowers/plans/2026-03-23-detectorclaw-v1-implementation-plan.md`

- [x] **Step 1: Break the roadmap into work packages**
Add WP1 through WP4 alignment for research and prototype work.

- [x] **Step 2: Add acceptance scenarios**
Capture the minimum scenarios that V1 must satisfy before code implementation begins.

- [x] **Step 3: Add open issues**
List the unresolved items that depend on future detector-specific codebases and data availability.

- [x] **Step 4: Review for handoff readiness**
Make sure another engineer or agent can begin implementation from this plan alone.

### Task 5: Verify repository deliverables

**Files:**
- Review: `docs/superpowers/specs/2026-03-23-detectorclaw-design.md`
- Review: `docs/superpowers/plans/2026-03-23-detectorclaw-v1-implementation-plan.md`
- Review: `docs/architecture/detectorclaw-data-contracts.md`
- Review: `docs/architecture/detectorclaw-online-diagnosis-v1.md`

- [x] **Step 1: Confirm all referenced files exist**
Run: `find docs -maxdepth 4 -type f | sort`

- [x] **Step 2: Confirm headings render and naming is consistent**
Run: `sed -n '1,240p' <file>`

- [x] **Step 3: Confirm the repo still reflects document-first scope**
Run: `find . -maxdepth 4 -type f | sort`

- [x] **Step 4: Record remaining implementation prerequisites**
Expected: a short list of missing future inputs such as detector-specific repositories, sample data, and calibration inventories.

## Acceptance Scenarios
- One shot can be described in a normalized structure without tying the design to one detector.
- V1 can represent detector failure, missing input, and degraded confidence explicitly.
- The decision output includes rationale and provenance.
- The plan leaves Phase 2 space for multi-detector joint inversion without redesigning V1 contracts.

## Open Issues
- The future software language and runtime are not selected yet.
- Detector-specific raw data formats are not yet documented in this repository.
- No sample shot dataset is present for workflow validation.
- This workspace is not a Git repository, so branch/worktree execution is not available here.
