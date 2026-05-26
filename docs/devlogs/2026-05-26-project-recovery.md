# 2026-05-26 Project Recovery

## Context
The workspace was damaged during cloud-drive backup/deletion. Recovery used the exported DetectorClaw design document, residual zip archives, existing tests, recovered source directories, and Python bytecode metadata as reconstruction anchors.

## Restored Implementation
- Rebuilt missing Python modules:
  - `detectorclaw/rcf/calibration.py`
  - `detectorclaw/rcf/stack.py`
  - `detectorclaw/rcf/live_browser.py`
- Restored native Rust project metadata from `rcf_preview_core.zip`:
  - `native/rcf_preview_core/Cargo.toml`
  - `native/rcf_preview_core/Cargo.lock`
  - `native/rcf_preview_core/README.md`
- Fixed Python package discovery in `pyproject.toml` so editable installs only package `detectorclaw*`.
- Rebuilt the missing Git index from `HEAD` with `git read-tree HEAD`; no working-tree files were overwritten.

## Validation
- `python -m pip install -e . pytest`
- `pytest -q`

Result:

```text
93 passed, 4 deselected in 12.26s
```

The deselected tests are the live browser integration tests marked `live_browser`.

## Remaining Gaps
- Rust native core was not validated because `cargo` is not available in the current shell.
- `playwright-cli` is not on `PATH`, so live browser automation cannot be exercised yet.
- Most recovered project files are still untracked by Git; review and add the intended files once the recovered layout is accepted.
