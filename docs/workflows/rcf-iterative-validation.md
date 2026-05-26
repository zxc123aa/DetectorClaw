# RCF Iterative Validation Workflow

## Goal
Run repeatable validation rounds for shot-based RCF GUI behavior without running the entire test suite.

## Scope
- Data set: `reference/shots/shot_001`
- Validation type: smoke tests + browser key-path acceptance + cache latency gates
- Performance gates:
  - First single-patch pseudocolor request: `<= 450 ms`
  - Cached next-patch switching P95 (10 samples): `<= 120 ms`
- Probe defaults:
  - Dose preview quality: `quality=78`

## One-Round Command
From repo root:

```bash
python scripts/rcf_iterative_validation.py \
  --data-root /mnt/c/songtan/detectorclaw \
  --shot 001 \
  --round-label round1
```

Round report path:

- `runs/validation/iterative-validation-<round-label>.json`

## What The Script Executes
1. Smoke tests:
   - `pytest tests/test_rcf_preview.py`
   - `pytest tests/test_rcf_gui.py -k "<dose/precompute/cache subset>"`
   - `pytest tests/test_rcf_cli.py -k "<live-browser session subset>"`
2. GUI/session setup:
   - Start an isolated GUI server (`http://127.0.0.1:18013/rcf/gui` by default)
   - Run `live-open` and `live-shot` to keep project browser session attached
3. API validation:
   - `POST /api/rcf/session/load`
   - 自动把前 `8` 片设为已分配（用于导航和缓存热路径验收）
   - wait until `GET /precompute/status` reaches `done` (or fail on `error/timeout`)
   - `GET /assets/manifest` and verify assigned patches exist
   - latency sampling on `GET /patch/{patch_id}/dose-image`

## Manual Browser Acceptance (Each Round)
After the script returns, validate in the same browser session:

1. Switch modes in order:
   - `原始图 -> 修正图 -> 剂量计算图 -> 单片伪色`
2. In `单片伪色`, switch palette:
   - `Turbo`, `Jet`, `Gray`
3. Use `↑ 上一片 / ↓ 下一片`:
   - Only assigned patches are navigated
   - Display label matches assigned order ("第 N 片")
4. Colorbar behavior:
   - Remains visible in dose modes
   - No obvious layout jump during patch/palette switch

## Failure Handling
- If smoke tests fail: fix failing area first, rerun the same round.
- If precompute status is `error`: inspect config/background paths before UI debugging.
- If latency gate fails with no functional failures: prioritize cache warm-path and asset readiness debugging.
- Do not run full `pytest` by default; only expand test scope after repeated failures.
