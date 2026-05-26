# MATLAB Reference

`RCF_RECT2.mlx` is the MATLAB Live Script used as the main external reference for the current RCF dose workflow.

Key extracted facts:

- Shot geometry references `calculate_rcf_positions('RCF1.json')`
- Background TIFF paths:
  - `C:\Songtan\怀柔\RCF\dosimetry\background\HD-new-background.tif`
  - `C:\Songtan\怀柔\RCF\dosimetry\background\EBT4-new-background.tif`
  - `C:\Songtan\怀柔\RCF\dosimetry\background\ScannerBackground.tif`
- Dose logic:
  - `OD = log10((original_Rmean - BG0_Rmean) ./ (image_R - BG0_Rmean))`
  - `OD2 = log10((original_Rmean - BG0_Rmean) ./ (BGmean - BG0_Rmean))`
  - Per-film calibration curves are then evaluated and background-subtracted

Notes:

- The MATLAB `indx == 2` branch sets `RCF_name = 'HD'` but does not visibly load an `original` background image in the extracted code.
- The current Python implementation matches the same high-level pattern: red-channel OD, calibration-curve evaluation, and background subtraction.
