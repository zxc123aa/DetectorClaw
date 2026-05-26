# 2026-03-24 Native Preview And Segmentation

## Summary

Heavy image preview and segmentation paths started moving out of pure Python into the Rust preview core.

## Main changes

- Added native preview commands for scan, bbox, and patch rendering.
- Introduced process-internal native calls instead of repeated subprocess-only execution.
- Tightened segmentation fallback behavior so large scans benefit from the native path without breaking small or low-confidence cases.

## Verification

- Rust crate built and passed targeted tests.
- Python wrappers loaded the native library successfully.
- GUI preview paths continued to render after native integration.

## Notes

- The major observed gains came from removing subprocess overhead, not from changing the physical dose logic.
