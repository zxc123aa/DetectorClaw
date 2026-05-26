from __future__ import annotations

from pathlib import Path

import yaml


def _validate_film_type(config: dict, film_type: str) -> None:
    film_models = config["calibration"]["film_models"]
    if film_type not in film_models:
        raise ValueError(f"Film type {film_type!r} not found in calibration.film_models")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    for key in ("film_type", "background", "calibration"):
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    background = config["background"]
    for key in ("film_path", "scanner_path"):
        if key not in background:
            raise ValueError(f"Missing required background field: {key}")

    calibration = config["calibration"]
    film_models = calibration.get("film_models")
    if not isinstance(film_models, dict) or not film_models:
        raise ValueError("Missing calibration.film_models mapping")

    film_type = config["film_type"]
    _validate_film_type(config, film_type)

    film_paths = background.get("film_paths", {})
    if film_paths is not None and not isinstance(film_paths, dict):
        raise ValueError("background.film_paths must be a mapping when provided")

    material_film_types = config.get("material_film_types", {})
    if material_film_types is not None and not isinstance(material_film_types, dict):
        raise ValueError("material_film_types must be a mapping when provided")
    for mapped_film_type in material_film_types.values():
        _validate_film_type(config, str(mapped_film_type))

    segmentation = config.setdefault("segmentation", {})
    segmentation.setdefault("min_area", 5000)
    segmentation.setdefault("padding", 4)
    segmentation.setdefault("sort_mode", "yx")
    segmentation.setdefault("backend", "cpu")
    if segmentation["backend"] not in {"cpu", "auto", "cuda"}:
        raise ValueError("segmentation.backend must be cpu, auto, or cuda")

    calibration.setdefault("background_quantile", 95)
    calibration.setdefault("backend", "auto")
    if calibration["backend"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("calibration.backend must be auto, cpu, or cuda")

    return config


def resolve_film_type(config: dict, stack_mapping: dict | None = None) -> str:
    if stack_mapping is not None:
        material_name = stack_mapping.get("material_name")
        mapped = config.get("material_film_types", {}).get(material_name)
        if mapped:
            return str(mapped)
    return str(config["film_type"])


def resolve_background_paths(config: dict, film_type: str) -> tuple[Path, Path]:
    background = config["background"]
    film_paths = background.get("film_paths", {}) or {}
    film_path = film_paths.get(film_type, background["film_path"])
    return Path(film_path), Path(background["scanner_path"])
