from __future__ import annotations

from pathlib import Path


def _normalize_shot_id(shot_id: str) -> tuple[str, int]:
    digits = "".join(ch for ch in str(shot_id) if ch.isdigit())
    if not digits:
        raise ValueError("shot_id must contain digits")
    numeric = int(digits)
    return f"{numeric:03d}", numeric


def discover_config_file(data_root: Path) -> tuple[Path, str]:
    data_root = Path(data_root)
    config_dir = data_root / "configs"
    candidates: list[Path] = []
    if config_dir.exists():
        candidates.extend(sorted(config_dir.glob("rcf*.yaml")))
        candidates.extend(sorted(config_dir.glob("rcf*.yml")))
    non_example = [path for path in candidates if ".example." not in path.name]
    if non_example:
        return non_example[0], "discovered"
    example = config_dir / "rcf.example.yaml"
    if example.exists():
        return example, "example"
    raise FileNotFoundError(f"No RCF config file found under {config_dir}")


def _shot_reference_dir(data_root: Path, padded_shot_id: str) -> Path:
    return Path(data_root) / "reference" / "shots" / f"shot_{padded_shot_id}"


def _discover_shot_assets(data_root: Path, padded: str, numeric: int) -> tuple[list[Path], Path]:
    data_root = Path(data_root)
    reference_dir = _shot_reference_dir(data_root, padded)
    reference_scan_1 = reference_dir / f"RCF{padded}.tif"
    reference_scan_2 = reference_dir / f"RCF{padded}_2.tif"
    reference_stack = reference_dir / f"RCF{numeric}.json"
    if reference_scan_1.exists() and reference_scan_2.exists() and reference_stack.exists():
        return [reference_scan_1, reference_scan_2], reference_stack

    legacy_scan_1 = data_root / f"RCF{padded}.tif"
    legacy_scan_2 = data_root / f"RCF{padded}_2.tif"
    legacy_stack = data_root / f"RCF{numeric}.json"
    return [legacy_scan_1, legacy_scan_2], legacy_stack


def resolve_shot_inputs(shot_id: str, data_root: Path) -> dict:
    padded, numeric = _normalize_shot_id(shot_id)
    data_root = Path(data_root)
    input_files, stack_config = _discover_shot_assets(data_root, padded, numeric)
    config_file, config_source = discover_config_file(data_root)
    output_dir = data_root / "runs" / "gui" / f"shot_{padded}_review"

    missing = [str(path) for path in [*input_files, stack_config, config_file] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Shot {padded} is missing required files: {', '.join(missing)}")

    return {
        "shot_id": padded,
        "data_root": data_root,
        "input_files": input_files,
        "stack_config_file": stack_config,
        "config_file": config_file,
        "config_source": config_source,
        "output_dir": output_dir,
    }
