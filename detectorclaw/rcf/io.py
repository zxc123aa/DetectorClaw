from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_rgb_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Input file not found: {image_path}")
    return np.asarray(Image.open(image_path).convert("RGB"))


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    patches_dir = output_dir / "patches"
    dose_dir = output_dir / "dose"
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    dose_dir.mkdir(parents=True, exist_ok=True)
    return patches_dir, dose_dir


def save_patch_image(path: Path, patch_rgb: np.ndarray) -> None:
    Image.fromarray(patch_rgb.astype(np.uint8), mode="RGB").save(path)


def save_dose_preview(path: Path, dose: np.ndarray) -> None:
    max_value = float(dose.max())
    if max_value <= 0:
        preview = np.zeros_like(dose, dtype=np.uint8)
    else:
        preview = np.clip(dose / max_value * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_overlay(path: Path, image_rgb: np.ndarray, patches: list[dict]) -> None:
    overlay = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay)

    for patch in patches:
        x, y, width, height = patch["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline=(255, 0, 0), width=3)
        draw.text((x + 3, y + 3), str(patch["order"]), fill=(255, 0, 0))

    overlay.save(path)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_debug_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
