# DetectorClaw Data Contracts

## Purpose
This document defines the first shared entities for DetectorClaw V1. They are written as repository contracts so that future software, analysis notebooks, or experiment tooling can use the same concepts even before a codebase exists.

## 1. ShotContext
Represents one experimental shot and the context needed to interpret detector outputs.

| Field | Required | Meaning |
| --- | --- | --- |
| `shot_id` | yes | Unique identifier for the shot |
| `timestamp` | yes | Acquisition time |
| `target` | yes | Target material, thickness, geometry, or relevant run descriptor |
| `laser` | yes | Energy, pulse, pointing, and other laser settings needed for diagnosis |
| `detector_config` | yes | List of detectors and their placement/configuration for this shot |
| `operator_note` | no | Free-text note from operator or experiment log |
| `calibration_versions` | yes | References to the calibration set used for this shot |
| `analysis_version` | yes | Version of logic or workflow that produced derived outputs |

## 2. DetectorPayload
Represents one detector contribution for one shot.

| Field | Required | Meaning |
| --- | --- | --- |
| `detector_type` | yes | `rcf`, `scint_fiber`, `tof`, `activation`, or future type |
| `payload_ref` | yes | Path, URI, or identifier of raw input |
| `status` | yes | `ok`, `missing`, `partial`, `invalid`, or `delayed` |
| `preprocess_state` | yes | Whether background subtraction, denoising, or normalization has run |
| `calibration_ref` | yes | Exact calibration used by this payload |
| `metadata` | no | Detector-specific fields such as gain, trigger, layer, distance |

## 3. OnlineAssessment
Represents the V1 quick-look interpretation of one shot.

| Field | Required | Meaning |
| --- | --- | --- |
| `quality_flags` | yes | List of data-quality judgments such as saturation or low signal |
| `anomaly_labels` | yes | Labels such as EMP contamination, mismatch, or abnormal background |
| `rough_cutoff_energy` | no | Fast approximate cutoff-energy estimate |
| `quicklook_metrics` | no | Detector-specific V1 summary metrics |
| `confidence_score` | yes | Coarse confidence for operator use, not a posterior probability |
| `explanation` | yes | Short explanation of why the system reached this assessment |

## 4. DecisionRecord
Represents what DetectorClaw recommends after online assessment.

| Field | Required | Meaning |
| --- | --- | --- |
| `recommendation` | yes | Suggested next action |
| `reason_set` | yes | Structured reasons behind the recommendation |
| `confidence` | yes | Confidence attached to the recommendation |
| `operator_confirmation` | yes | Whether a human accepted, rejected, or deferred the advice |
| `source_refs` | yes | Links back to `ShotContext`, `DetectorPayload`, and `OnlineAssessment` |

## 5. Relationship Model
One `ShotContext` may have many `DetectorPayload` records.  
One online pass generates one `OnlineAssessment`.  
One `OnlineAssessment` may produce one `DecisionRecord`.  
Later phases may attach joint inversion outputs to the same shot without changing these base entities.

## 6. Example Lifecycle
1. A new shot is created in `ShotContext`.
2. Raw inputs from TOF and scintillating fiber arrive as `DetectorPayload` records.
3. The V1 online logic evaluates quality, anomalies, and rough energy indicators into `OnlineAssessment`.
4. A human-readable next-shot recommendation is stored as `DecisionRecord`.
5. In Phase 2, a joint inversion module may append richer reconstructed beam outputs to the same shot lineage.

## 7. Contract Rules
- Every derived result must reference calibration and analysis versions.
- Missing data must be represented explicitly; silent omission is not allowed.
- Confidence must be reported even when the system downgrades itself.
- V1 explanations must stay traceable to detector observations or documented rules.
