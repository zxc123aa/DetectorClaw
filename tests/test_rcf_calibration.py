from pathlib import Path

import numpy as np
from PIL import Image


def test_load_background_mean_reuses_cached_file_signature(tmp_path: Path) -> None:
    from detectorclaw.rcf import calibration

    image_path = tmp_path / "background.tif"
    Image.fromarray(np.full((8, 8, 3), 123, dtype=np.uint8), mode="RGB").save(image_path)

    original_open = calibration.Image.open
    calls = {"count": 0}

    def wrapped_open(*args, **kwargs):
        calls["count"] += 1
        return original_open(*args, **kwargs)

    calibration._BACKGROUND_MEAN_CACHE.clear()
    calibration.Image.open = wrapped_open
    try:
        first = calibration.load_background_mean(image_path)
        second = calibration.load_background_mean(image_path)
    finally:
        calibration.Image.open = original_open

    assert first == second == 123.0
    assert calls["count"] == 1


def test_dose_from_patch_supports_backend_argument() -> None:
    from detectorclaw.rcf.calibration import dose_from_patch

    patch_rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    film_model = {"kind": "polynomial", "coefficients": [0.0, 100.0]}

    dose_cpu, background_cpu = dose_from_patch(
        patch_rgb=patch_rgb,
        film_background_mean=220.0,
        scanner_background_mean=10.0,
        film_model=film_model,
        background_quantile=95,
        backend="cpu",
    )
    dose_auto, background_auto = dose_from_patch(
        patch_rgb=patch_rgb,
        film_background_mean=220.0,
        scanner_background_mean=10.0,
        film_model=film_model,
        background_quantile=95,
        backend="auto",
    )

    assert dose_cpu.shape == patch_rgb.shape[:2]
    assert dose_auto.shape == patch_rgb.shape[:2]
    assert background_cpu == background_auto
    assert float(dose_cpu.min()) >= 0.0
    assert float(dose_auto.min()) >= 0.0
