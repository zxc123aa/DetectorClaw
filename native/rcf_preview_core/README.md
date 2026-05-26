# rcf_preview_core

Native image backend for DetectorClaw.

## Scope

This binary currently accelerates:

- scan preview downsampling
- raw patch bbox extraction
- patch perspective rectification
- segment-mode patch detection for file-based CLI/GUI paths
- JPEG/PNG preview encoding

Python orchestration remains in:

- [`detectorclaw/rcf/preview.py`](../../detectorclaw/rcf/preview.py)
- [`detectorclaw/rcf/segment.py`](../../detectorclaw/rcf/segment.py)

## Build

```bash
. "$HOME/.cargo/env"
cargo build --release
```

The compiled artifacts are emitted to:

```text
native/rcf_preview_core/target/release/rcf_preview_core
native/rcf_preview_core/target/release/librcf_preview_core.so
```

On macOS the shared library suffix is `.dylib`; on Windows it is `.dll`.

## Runtime

The Python preview layer auto-detects the compiled binary in `target/release` or `target/debug`.

The Python segment layer now prefers the in-process shared library and falls back to the CLI binary only when the library is unavailable.

You can also force a specific binary with:

```bash
export DETECTORCLAW_RCF_NATIVE_PREVIEW_BIN=/abs/path/to/rcf_preview_core
export DETECTORCLAW_RCF_NATIVE_PREVIEW_LIB=/abs/path/to/librcf_preview_core.so
export DETECTORCLAW_RCF_NATIVE_SEGMENT_BIN=/abs/path/to/rcf_preview_core
export DETECTORCLAW_RCF_NATIVE_SEGMENT_LIB=/abs/path/to/librcf_preview_core.so
```

## CLI contract

The binary reads JSON from stdin and writes JSON to stdout.

Supported commands:

- `scan-preview`
- `bbox-preview`
- `patch-preview`
- `segment-detect`

Response fields:

- `media_type`
- `content_hex`

`patch-preview` optionally accepts `crop_bbox`, applied after perspective rectification and before resize/encoding.

`segment-detect` returns:

- `component_count`
- `mask_png_hex`
- `components`
- `patches`
