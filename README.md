# DetectorClaw

DetectorClaw is a local RCF review and online-diagnosis prototype centered on shot-based workflows.

## Repository Layout

- `detectorclaw/`: Python package for the RCF CLI, GUI, preview, segmentation, and browser broker.
- `native/rcf_preview_core/`: Rust native core for preview and segmentation hot paths.
- `configs/`: YAML configuration files.
- `reference/`: curated reference assets, including MATLAB references and shot fixtures.
- `docs/devlogs/`: dated development logs for major implementation milestones.
- `runs/`: local runtime outputs and debug artifacts. Ignored by git.
- `tests/`: targeted regression tests.

## Common Commands

- `python -m detectorclaw.rcf gui --host 127.0.0.1 --port 8013`
- `python -m detectorclaw.rcf live-shot --shot 001 --data-root /mnt/c/songtan/detectorclaw`
- `python -m detectorclaw.rcf live-session --data-root /mnt/c/songtan/detectorclaw --history 5`
- `python scripts/rcf_iterative_validation.py --data-root /mnt/c/songtan/detectorclaw --shot 001 --round-label round1`

## Reference Data

- MATLAB reference: `reference/matlab/RCF_RECT2.mlx`
- Shot 001 fixture: `reference/shots/shot_001/`
- Background path notes: `reference/backgrounds/README.md`
