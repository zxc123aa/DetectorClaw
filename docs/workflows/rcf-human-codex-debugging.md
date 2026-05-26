# RCF Human-Codex Debugging Workflow

## Goal
Use one real RCF scan, one film background image, one scanner background image, and a film type to iteratively stabilize the Python CLI on real data.

## What You Provide
- One input TIFF containing multiple expanded RCF pieces
- Film type, such as `HDV2`, `MDV3`, or `EBT3`
- Film background TIFF
- Scanner background TIFF
- Expected patch count
- Optional note about known bad pieces or strange scan artifacts

## What Codex Produces
For each run, the CLI should produce:

- `overlay_raw.png`
- `overlay_final.png`
- `mask.png`
- `components.json`
- `review.json`
- `summary.json`
- `debug.log`

## Debug Loop
### Round 1: Segmentation
You check only:
- patch count
- wrong bounding boxes
- wrong order

Codex then updates either:
- the segmentation logic, or
- the `review.json` override

### Round 2: Dose sanity
You check only:
- obviously wrong dose magnitude
- missing exposed region
- bad background compensation

Codex then updates:
- calibration configuration
- QC reporting
- background handling logic

### Round 3: Repeatability
Run the same real input twice and confirm the same patch order, bounding boxes, and dose summary fields.

## Review Rule
Keep human edits limited to `review.json` patch order and bounding boxes. Do not introduce ad-hoc manual crop steps into the normal workflow.
