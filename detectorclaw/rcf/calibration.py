from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .runtime_cache import file_signature

_BACKGROUND_MEAN_CACHE: dict[tuple[str, int, int], float] = {}


def load_background_mean(image_path: Path) -> float:
    signature = file_signature(Path(image_path))
    cached = _BACKGROUND_MEAN_CACHE.get(signature)
    if cached is not None:
        return cached

    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float64)
    mean = float(array[:, :, 0].mean())
    _BACKGROUND_MEAN_CACHE[signature] = mean
    return mean


def _nearest_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    q = max(0.0, min(100.0, float(percentile))) / 100.0
    target = int(round((values.size - 1) * q))
    return float(np.partition(values, target)[target])


def _evaluate_curve_model(film_model: dict, od: np.ndarray | float) -> np.ndarray | float:
    kind = str(film_model.get("kind", "polynomial")).lower()
    if kind == "polynomial":
        coefficients = film_model.get("coefficients")
        if not isinstance(coefficients, list) or not coefficients:
            raise ValueError("polynomial film model requires non-empty coefficients")
        result = 0.0
        for power, coefficient in enumerate(coefficients):
            result = result + float(coefficient) * np.power(od, power)
        return result
    if kind in {"power_law", "power-law", "powerlaw"}:
        scale = float(film_model["scale"])
        exponent = float(film_model["exponent"])
        return scale * np.power(od, exponent)
    raise ValueError(f"Unsupported film model kind: {kind}")


def dose_from_patch(
    patch_rgb: np.ndarray,
    film_background_mean: float,
    scanner_background_mean: float,
    film_model: dict,
    background_quantile: float,
    backend: str = "auto",
) -> tuple[np.ndarray, float]:
    backend = str(backend).lower()
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be auto, cpu, or cuda")

    if patch_rgb.size == 0:
        raise ValueError("empty patch image")

    red = np.asarray(patch_rgb[:, :, 0], dtype=np.float64)
    signal_floor = max(float(film_background_mean) - float(scanner_background_mean), 1e-6)
    patch_background_mean = _nearest_percentile(red.reshape(-1).copy(), background_quantile)
    background_signal = max(patch_background_mean - float(scanner_background_mean), 1e-6)
    od_background = np.log10(max(signal_floor / background_signal, 1.0))
    baseline = float(_evaluate_curve_model(film_model, od_background))

    patch_signal = np.maximum(red - float(scanner_background_mean), 1e-6)
    od = np.log10(np.maximum(signal_floor / patch_signal, 1.0))
    dose = np.maximum(np.asarray(_evaluate_curve_model(film_model, od), dtype=np.float64) - baseline, 0.0)
    return dose.astype(np.float32), float(patch_background_mean)
