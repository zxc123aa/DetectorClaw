# DetectorClaw Design Spec

**Date:** 2026-03-23  
**Status:** Drafted from exported discussion and refined into an executable repository spec  
**Project Type:** Dual-track doctoral research spec + system prototype spec

## 1. Goal
DetectorClaw is a physics-constrained agent system for laser-driven proton or ion-beam diagnosis. Its purpose is not to replace core inversion with an LLM, but to unify multi-detector observations, forward models, experimental knowledge, and decision support into one closed-loop workflow.

The project runs on two tracks:

1. **Research track:** define the scientific questions, methodology, and publication roadmap.
2. **System track:** define a layered prototype architecture that can support experiment design, online diagnosis, later joint inversion, and knowledge capture.

## 2. Problem Statement
Current diagnosis is fragmented across detectors and analysis chains. RCF, scintillating fiber, TOF, and activation methods each solve part of the problem, but the experiment still depends heavily on manual judgment for detector layout, data quality control, result comparison, and next-shot decisions.

The core gap is an intelligent coordination layer that can:

- represent heterogeneous detector inputs in a common structure,
- preserve physics constraints in analysis,
- provide fast online assessment during repeated shots,
- support later multi-detector joint inference,
- leave an auditable record of decisions and confidence.

## 3. Scope and Phase Boundary
### V1 priority
V1 is **online diagnosis first**. It must support experiment-time quick-look and human-in-the-loop recommendations before the repository expands into a full analysis platform.

### V1 in scope
- experiment context capture,
- detector payload normalization,
- online quality control,
- rough cutoff-energy estimation,
- anomaly labeling,
- next-shot suggestion generation,
- confidence and explanation output,
- versioned result logging.

### Explicitly out of scope for V1
- full multi-detector Bayesian joint inversion,
- autonomous experiment control,
- replacing detector-specific physics pipelines with generic LLM inference,
- publication automation as a primary deliverable.

## 4. Research Questions
1. How can heterogeneous detector responses be represented in one diagnosable and extensible framework?
2. How can physics priors, response matrices, and data-driven approximations be combined without degrading interpretability?
3. How can an experiment-time agent produce useful, auditable recommendations while keeping humans in the loop?

## 5. System Architecture
DetectorClaw is defined as a four-layer system:

1. **Data foundation layer**  
   Normalizes shot context, detector payloads, calibration references, quality flags, and result versions.
2. **Online diagnosis layer**  
   Performs background checks, saturation checks, EMP contamination checks, rough spectrum indicators, anomaly detection, and recommendation generation.
3. **Physics and inversion layer**  
   Starts in V1 as detector-specific quick-look hooks and expands in Phase 2 into multi-detector joint inference with uncertainty propagation.
4. **Knowledge and reporting layer**  
   Stores decisions, versions, operator notes, and reusable summaries for papers, logs, and training.

## 6. Public Contracts
The first shared interfaces are documentation contracts, not code contracts.

### ShotContext
- shot identifier,
- target parameters,
- laser parameters,
- detector configuration,
- timestamp,
- operator note,
- calibration and algorithm version references.

### DetectorPayload
- detector type,
- raw data reference,
- preprocessing state,
- calibration reference,
- acquisition status,
- optional detector-specific metadata.

### OnlineAssessment
- quality flags,
- anomaly labels,
- rough cutoff-energy estimate,
- detector-specific quick-look metrics,
- confidence score,
- explanation summary.

### DecisionRecord
- recommendation for next shot,
- reason set,
- confidence,
- operator confirmation status,
- links to source payloads and assessments.

## 7. Validation Criteria
V1 is successful when it can process one shot into a reproducible assessment package that includes:

- a normalized experiment context,
- per-detector data status,
- at least one rough beam-quality indicator,
- explicit anomaly reporting when data quality is poor,
- a next-shot recommendation with explanation,
- traceable version references for calibration and logic.

## 8. Roadmap
### Near-term, 6 to 12 months
- formalize the research framing and architecture,
- lock shared data contracts,
- define the V1 online diagnosis workflow,
- pilot manual or semi-manual quick-look evaluation against existing detector practice.

### Doctoral roadmap
- **WP1:** unified modeling and system framework,
- **WP2:** online diagnosis and experiment-time decision support,
- **WP3:** multi-detector joint inversion and uncertainty propagation,
- **WP4:** platform validation, reporting, and thesis integration.

## 9. Non-Negotiable Constraints
- Core inversion stays physics-led.
- LLM use is limited to orchestration, explanation, retrieval, and reporting.
- V1 remains human-in-the-loop.
- Recommendations must be traceable to data, calibration versions, and documented rules.
