# DetectorClaw Online Diagnosis V1

## Objective
Define the minimum closed-loop workflow for an online-diagnosis-first DetectorClaw prototype. V1 is designed for repeated experiments where operators need quick quality judgments and next-shot suggestions without waiting for full offline inversion.

## Workflow
### 1. Pre-shot setup
- Register the shot context.
- Confirm active detector configuration.
- Load the calibration and logic versions that will be used for this shot.

### 2. Data ingestion
- Accept detector payloads as they arrive.
- Mark each payload as `ok`, `missing`, `partial`, `invalid`, or `delayed`.
- Preserve raw references instead of copying or rewriting the source data.

### 3. Online checks
- Run background and baseline checks.
- Detect saturation or clipping.
- Detect obvious EMP contamination or timing corruption.
- Generate detector-specific quick-look metrics.
- Produce a coarse rough cutoff-energy estimate when enough signal exists.

### 4. Recommendation synthesis
- Combine the quality flags, anomalies, and quick-look metrics.
- Generate one operator-facing recommendation such as:
  - repeat current parameters,
  - adjust shielding,
  - adjust target thickness,
  - recheck detector placement,
  - ignore this shot for spectrum comparison.
- Attach a concise explanation and a confidence score.

### 5. Human confirmation
- Require an operator response: accept, reject, or defer.
- Store the final recommendation and operator decision in the shot record.

## Downgrade and Failure Paths
### Missing detector
The workflow continues with reduced confidence and explicitly reports which detector was absent.

### Noisy or saturated detector
The workflow suppresses strong beam claims from that detector and raises a quality flag first.

### Conflicting detector indications
V1 does not force a unified physics result. It reports conflict and advises caution or offline review.

### Insufficient signal
The workflow reports that no reliable quick-look conclusion is available and avoids fabricated recommendations.

## Confidence Model
V1 confidence is operational, not Bayesian. It should be treated as a structured judgment based on:

- detector availability,
- data integrity,
- consistency of quick-look indicators,
- calibration availability,
- rule coverage for the current scenario.

Suggested human-readable bands:

- `high`: sufficient signal and no major quality issue,
- `medium`: usable data with one known limitation,
- `low`: degraded or conflicting evidence,
- `blocked`: recommendation cannot be trusted for decision-making.

## Acceptance Conditions
V1 is acceptable when it can:

- ingest a shot context and detector payload set,
- emit at least one online assessment,
- downgrade explicitly under poor data conditions,
- keep a provenance trail for the recommendation,
- stop short of autonomous experiment control.

## Follow-on Work
Phase 2 should connect this workflow to multi-detector joint inversion, uncertainty propagation, and detector credibility scoring. Those additions should enrich the same shot record rather than replace the V1 structure.
