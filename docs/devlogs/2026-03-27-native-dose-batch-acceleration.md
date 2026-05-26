# 2026-03-27 Native Dose Batch Acceleration

## Summary

Dose precompute and cache warm paths now support a native batch route (`dose-batch`) in `rcf_preview_core`, with up to 8 tasks per call and parallel execution. The GUI precompute pipeline attempts this native route first and falls back to the existing Python path when native symbols/binaries are unavailable.

## Main changes

- Added native `dose-batch` request/response contracts in `native/rcf_preview_core`.
- Added parallel per-task processing in Rust (`rayon`) for 8-task batches.
- Implemented native dose math and pseudocolor rendering (`gray`/`turbo`/`jet`) for preview assets.
- Exported `rcf_dose_batch_json` from the cdylib and wired a Python wrapper `preview.run_native_dose_batch(...)`.
- Updated GUI precompute dose stages (`dose-single`, `dose-high`, `dose-overview`) to try native batch first.
- Updated on-demand `/dose-image` miss handling to try native single-variant warm before Python fallback.
- Extended precompute status payload with `interactive_queue`, `bulk_queue`, `inflight_batch`, and `backend`.

## Verification

- Targeted GUI tests:
  - `test_gui_dose_overview_prewarm_batches_assigned_patches`
  - `test_gui_load_starts_session_precompute_and_populates_caches`
  - `test_gui_assignment_requeues_session_precompute_for_current_revision`
  - `test_gui_precompute_status_and_start_endpoints`
  - `test_gui_dose_image_cache_only_skips_synchronous_compute`
  - Result: `5 passed`.
- Preview tests: `tests/test_rcf_preview.py` result `9 passed`.
- Native crate build:
  - `cargo check` passed.
  - `cargo build --release` passed.

## Measured impact (synthetic 8-patch 3000x4000 case)

- Full precompute after assigning 8 patches:
  - Python fallback: ~6893 ms
  - Native batch: ~3996 ms
  - Improvement: ~42%
- First uncached single dose preview (320):
  - Python fallback: ~249 ms
  - Native batch route: ~105 ms
  - Improvement: ~58%
