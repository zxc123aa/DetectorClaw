# RCF GUI Auto-Rectification Design

## Summary

This spec defines the next GUI and segmentation upgrade for RCF stack review in DetectorClaw.
The current patch detector is driven by dark dose regions, which fails on low-contrast films such as the first HD layer and leaves obviously tilted films in axis-aligned boxes.

The new default mode is `auto-initial-rectification`:

1. detect full RCF sheets from rectangular outer edges, not dose darkness,
2. estimate an initial per-patch rotation angle,
3. present rotated previews in the GUI,
4. let the operator assign a patch directly to stack position `N`.

The goal is not final autonomous analysis. The goal is stable human-in-the-loop review on real multi-scan TIFF inputs.

## Problems To Fix

### 1. Low-contrast sheets are under-detected

The first HD sheet may appear much lighter than EBT layers.
Dark-region thresholding captures only the central exposed area instead of the whole film.

### 2. Tilt is visible but not modeled

Several sheets are visibly rotated within the scan.
The current GUI only stores an axis-aligned `bbox`, so white borders remain obvious in extracted patches.

### 3. Stack ordering is slow to edit

`Up/Down` controls are not workable for 12+ layers across multiple scans.
The operator needs direct stack assignment, such as “this patch is layer 7”.

## Design

### Detection Pipeline

Replace dark-mask-first segmentation with a rectangle-first detector:

1. normalize the scan background and denoise lightly,
2. detect outer edges of sheet-like regions,
3. fit candidate rectangles or near-rectangles,
4. filter by expected size, aspect ratio, rectangularity, and scan layout,
5. keep low-confidence candidates instead of silently dropping them.

Dose darkness remains a secondary cue only.

### Auto Initial Rectification

For each detected sheet:

1. estimate `angle_deg` from the candidate rectangle orientation,
2. generate a rotated preview patch,
3. derive a tighter crop in rotated coordinates,
4. mark confidence state.

New patch state fields:

- `bbox`
- `angle_deg`
- `rotated_bbox`
- `crop_bbox`
- `confidence`
- `status_flags`

### GUI Interaction

After `Load Session`, the GUI should show:

- detected outer sheet boxes on each scan,
- one selected patch with raw and rotated previews,
- direct stack assignment by entering a target position `N`.

Manual actions remain:

- move/resize box,
- re-estimate angle from two edge points,
- fine-tune angle numerically,
- set refined crop,
- assign patch to stack position `N`.

Reordering should shift other patches automatically rather than requiring repeated `Up/Down` clicks.

## Scope

### In Scope

- rectangle-edge-driven film detection,
- automatic initial angle estimation,
- rotated preview generation,
- direct stack-position assignment,
- confidence flags for low-quality detections,
- real-data validation on current two-scan inputs.

### Out of Scope

- final dose-pipeline refactor,
- background ROI GUI,
- perspective correction beyond rotated rectangles,
- autonomous final stack ordering without operator confirmation.

## Acceptance Criteria

The current real dataset is the acceptance baseline.

1. The first HD film is boxed as a full sheet, not just its central dark region.
2. Tilted films such as `5`, `6`, and `8` load with a meaningful initial angle and visibly reduced white-border waste.
3. The operator can assign any patch directly to stack position `N`.
4. Two scans can be reviewed as one merged stack session.
5. Weak detections are labeled as low confidence instead of being silently omitted.

## Implementation Notes

Implementation should proceed in this order:

1. detector and data-model update,
2. auto-angle estimation and rotated preview plumbing,
3. GUI stack-position assignment,
4. real-data regression using the current TIFF pair and stack JSON.

This spec intentionally keeps the boundary narrow: make review reliable first, then reconnect refined patch geometry to downstream physics processing.
