# 2026-03-26 Dose View Debugging And Repository Reorganization

## Summary

The dose pseudocolor failure was traced to invalid background paths in the example config, not to Rust rendering. In parallel, the repository layout was normalized so reference assets, run artifacts, and code are no longer mixed at the root.

## Main changes

- Confirmed the MATLAB reference script `RCF_RECT2.mlx` uses these external background paths:
  - `C:\Songtan\怀柔\RCF\dosimetry\background\HD-new-background.tif`
  - `C:\Songtan\怀柔\RCF\dosimetry\background\EBT4-new-background.tif`
  - `C:\Songtan\怀柔\RCF\dosimetry\background\ScannerBackground.tif`
- Added explicit GUI dose-availability state and stopped treating missing background files as fake `404` patch errors.
- Made shot resolution prefer `reference/shots/shot_XXX/` and fall back to the old root layout.
- Moved tracked reference assets into `reference/` and runtime/debug assets into `runs/`.

## Verification

- Dose preview tests now distinguish missing calibration assets from missing session resources.
- Shot loading works with the new `reference/shots/shot_001/` layout and still supports the old root layout as fallback.
- Browser console no longer fills with repeated `dose-image` `404` requests when dose calculation is unavailable.

## Notes

- The current repository still references external background files by path only; the actual background TIFFs remain outside version control.
