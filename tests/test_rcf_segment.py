import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from detectorclaw.rcf.runtime_cache import LRUCache
from detectorclaw.rcf.segment import _compute_patch_film_mask
from detectorclaw.rcf.segment import _estimate_patch_background
from detectorclaw.rcf.segment import detect_patches
from detectorclaw.rcf.segment import detect_patches_path


def _save_rgb_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _low_contrast_sheet_with_dark_center() -> np.ndarray:
    canvas = Image.new("RGB", (420, 240), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((30, 30, 160, 180), fill=(210, 220, 220), outline=(120, 130, 130), width=4)
    draw.rectangle((70, 70, 120, 130), fill=(130, 150, 180))
    return np.asarray(canvas)


def _rotated_sheet() -> np.ndarray:
    canvas = Image.new("RGB", (420, 300), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(180, 50), (300, 90), (250, 240), (130, 200)], fill=(215, 218, 200), outline=(100, 110, 90))
    return np.asarray(canvas)


def _vertically_merged_sheets() -> np.ndarray:
    canvas = Image.new("RGB", (320, 420), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((50, 30, 190, 170), fill=(210, 220, 220), outline=(120, 130, 130), width=4)
    draw.rectangle((50, 185, 190, 325), fill=(208, 218, 218), outline=(120, 130, 130), width=4)
    draw.rectangle((105, 166, 135, 191), fill=(208, 218, 218), outline=(120, 130, 130), width=2)
    return np.asarray(canvas)


def _rotated_near_square_sheet_with_dark_center() -> np.ndarray:
    canvas = Image.new("RGB", (240, 240), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(60, 45), (180, 60), (165, 180), (45, 165)], fill=(238, 240, 240), outline=(210, 215, 215))
    draw.rectangle((95, 90, 130, 125), fill=(120, 150, 180))
    return np.asarray(canvas)


def test_estimate_patch_background_and_mask_keep_full_sheet() -> None:
    patch = _rotated_near_square_sheet_with_dark_center()

    background = _estimate_patch_background(patch, border_width=20)
    film_mask = _compute_patch_film_mask(patch, background)
    ys, xs = np.nonzero(film_mask)

    assert np.allclose(background, np.array([245.0, 245.0, 245.0]), atol=2.0)
    assert xs.min() <= 48
    assert ys.min() <= 48
    assert xs.max() >= 175
    assert ys.max() >= 175
    assert film_mask.sum() >= 12000


def test_detect_patches_prefers_full_low_contrast_sheet_over_dark_center() -> None:
    image = _low_contrast_sheet_with_dark_center()

    detection = detect_patches(image, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert len(detection["patches"]) == 1
    patch = detection["patches"][0]

    assert patch["bbox"][0] <= 32
    assert patch["bbox"][1] <= 32
    assert patch["bbox"][2] >= 125
    assert patch["bbox"][3] >= 145
    assert patch["angle_deg"] == 0.0
    assert patch["angle_source"] == "contour_rect"
    assert patch["angle_confidence"] > 0.0
    assert patch["status_flags"] == []


def test_detect_patches_uses_contour_rect_angle_for_rotated_near_square_sheet() -> None:
    image = _rotated_near_square_sheet_with_dark_center()

    detection = detect_patches(image, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert len(detection["patches"]) == 1
    patch = detection["patches"][0]

    assert abs(patch["angle_deg"]) >= 5.0
    assert patch["angle_source"] == "contour_rect"
    assert patch["angle_confidence"] > 0.0
    assert patch["status_flags"] == []


def test_detect_patches_emits_initial_angle_for_rotated_sheet() -> None:
    image = _rotated_sheet()

    detection = detect_patches(image, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert len(detection["patches"]) == 1
    patch = detection["patches"][0]

    assert "angle_deg" in patch
    assert abs(patch["angle_deg"]) >= 5.0
    assert patch["angle_source"] == "contour_rect"
    assert patch["angle_confidence"] > 0.0
    assert patch["status_flags"] == []


def test_detect_patches_supports_backend_argument() -> None:
    image = _rotated_sheet()

    detection_cpu = detect_patches(image, {"min_area": 1000, "padding": 4, "sort_mode": "yx", "backend": "cpu"})
    detection_auto = detect_patches(image, {"min_area": 1000, "padding": 4, "sort_mode": "yx", "backend": "auto"})

    assert len(detection_cpu["patches"]) == 1
    assert len(detection_auto["patches"]) == 1
    assert max(abs(a - b) for a, b in zip(detection_cpu["patches"][0]["bbox"], detection_auto["patches"][0]["bbox"])) <= 4
    assert abs(detection_cpu["patches"][0]["angle_deg"] - detection_auto["patches"][0]["angle_deg"]) < 1.0


def test_detect_patches_splits_tall_merged_component_into_multiple_sheets() -> None:
    image = _vertically_merged_sheets()

    detection = detect_patches(image, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert len(detection["patches"]) == 2
    assert detection["patches"][0]["bbox"][3] < 220
    assert detection["patches"][1]["bbox"][3] < 220
    assert all("angle_source" in patch for patch in detection["patches"])


def test_detect_patches_path_falls_back_to_python_without_native_binary(tmp_path: Path, monkeypatch) -> None:
    from detectorclaw.rcf import segment

    image = _rotated_sheet()
    image_path = tmp_path / "scan.tif"
    _save_rgb_image(image_path, image)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_SEGMENT_BIN", raising=False)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_SEGMENT_LIB", raising=False)
    monkeypatch.setattr(segment, "load_native_segment_library", lambda: None)
    monkeypatch.setattr(segment, "locate_native_segment_binary", lambda: None)

    detection = detect_patches_path(image_path, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert len(detection["patches"]) == 1
    assert detection["patches"][0]["angle_source"] == "contour_rect"
    assert detection["component_count"] >= 1
    assert detection["mask"].dtype == np.bool_


def test_detect_patches_path_prefers_native_binary_when_available(tmp_path: Path, monkeypatch) -> None:
    from detectorclaw.rcf import segment

    image = _rotated_sheet()
    image_path = tmp_path / "scan.tif"
    _save_rgb_image(image_path, image)
    fake_bin = tmp_path / "rcf_preview_core"
    fake_bin.write_text("", encoding="utf-8")
    monkeypatch.setenv("DETECTORCLAW_RCF_NATIVE_SEGMENT_BIN", str(fake_bin))
    monkeypatch.setattr(segment, "load_native_segment_library", lambda: None)

    seen: dict[str, object] = {}

    def fake_run(command: list[str], input: str, capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["payload"] = json.loads(input)
        mask = Image.new("L", (32, 24), 255)
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        payload = json.dumps(
            {
                "component_count": 1,
                "mask_png_hex": buffer.getvalue().hex(),
                "patches": [
                    {
                        "order": 1,
                        "bbox": [4, 5, 20, 12],
                        "angle_deg": 11.5,
                        "angle_confidence": 0.88,
                        "angle_source": "contour_rect",
                        "status_flags": [],
                    }
                ],
                "components": [
                    {
                        "component_id": 1,
                        "area": 240,
                        "bbox": [4, 5, 20, 12],
                        "angle_deg": 11.5,
                        "angle_confidence": 0.88,
                        "angle_source": "contour_rect",
                        "status_flags": [],
                        "kept": True,
                    }
                ],
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr(segment.subprocess, "run", fake_run)

    detection = detect_patches_path(image_path, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert seen["command"] == [str(fake_bin), "segment-detect"]
    assert seen["payload"] == {
        "scan_file": str(image_path),
        "min_area": 1000,
        "padding": 4,
        "sort_mode": "yx",
    }
    assert detection["component_count"] == 1
    assert detection["patches"][0]["bbox"] == [4, 5, 20, 12]
    assert detection["mask"].shape == (24, 32)


def test_detect_patches_path_prefers_native_library_when_available(tmp_path: Path, monkeypatch) -> None:
    from detectorclaw.rcf import segment

    image = _rotated_sheet()
    image_path = tmp_path / "scan.tif"
    _save_rgb_image(image_path, image)

    fake_library = object()
    monkeypatch.setattr(segment, "load_native_segment_library", lambda: fake_library)
    monkeypatch.setattr(
        segment,
        "_run_native_segment_detection_via_library",
        lambda library, image_path, config: {
            "mask": np.ones((24, 32), dtype=bool),
            "component_count": 1,
            "components": [],
            "patches": [
                {
                    "order": 1,
                    "bbox": [4, 5, 20, 12],
                    "angle_deg": 7.5,
                    "angle_confidence": 0.91,
                    "angle_source": "contour_rect",
                    "status_flags": [],
                }
            ],
        },
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess fallback should not run when native library is available")

    monkeypatch.setattr(segment.subprocess, "run", fail_run)

    detection = detect_patches_path(image_path, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert detection["component_count"] == 1
    assert detection["patches"][0]["angle_source"] == "contour_rect"


def test_detect_patches_path_caches_identical_requests_and_misses_on_config_change(
    tmp_path: Path, monkeypatch
) -> None:
    from detectorclaw.rcf import segment

    image = _rotated_sheet()
    image_path = tmp_path / "scan.tif"
    _save_rgb_image(image_path, image)
    monkeypatch.setattr(segment, "_SEGMENT_RESULT_CACHE", LRUCache(max_entries=8))
    fake_library = object()
    monkeypatch.setattr(segment, "load_native_segment_library", lambda: fake_library)

    calls: list[int] = []

    def fake_library_call(library, image_path: Path, config: dict) -> dict:
        calls.append(int(config["padding"]))
        return {
            "mask": np.ones((24, 32), dtype=bool),
            "component_count": 1,
            "components": [],
            "patches": [
                {
                    "order": 1,
                    "bbox": [4, 5, 20, 12],
                    "angle_deg": float(config["padding"]),
                    "angle_confidence": 0.88,
                    "angle_source": "contour_rect",
                    "status_flags": [],
                }
            ],
        }

    monkeypatch.setattr(segment, "_run_native_segment_detection_via_library", fake_library_call)

    first = detect_patches_path(image_path, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})
    second = detect_patches_path(image_path, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})
    third = detect_patches_path(image_path, {"min_area": 1000, "padding": 6, "sort_mode": "yx"})

    assert first["patches"][0]["angle_deg"] == 4.0
    assert second["patches"][0]["angle_deg"] == 4.0
    assert third["patches"][0]["angle_deg"] == 6.0
    assert calls == [4, 6]


def test_detect_patches_path_falls_back_to_python_for_low_confidence_native_angle(
    tmp_path: Path, monkeypatch
) -> None:
    from detectorclaw.rcf import segment

    image = _rotated_sheet()
    image_path = tmp_path / "scan.tif"
    _save_rgb_image(image_path, image)

    monkeypatch.setattr(
        segment,
        "_run_native_segment_detection",
        lambda image_path, config: {
            "mask": np.ones((24, 32), dtype=bool),
            "component_count": 1,
            "components": [
                {
                    "component_id": 1,
                    "area": 240,
                    "bbox": [4, 5, 20, 12],
                    "angle_deg": 0.0,
                    "angle_confidence": 0.0,
                    "angle_source": "low_confidence_zero",
                    "status_flags": ["low_confidence_angle"],
                    "kept": True,
                }
            ],
            "patches": [
                {
                    "order": 1,
                    "bbox": [4, 5, 20, 12],
                    "angle_deg": 0.0,
                    "angle_confidence": 0.0,
                    "angle_source": "low_confidence_zero",
                    "status_flags": ["low_confidence_angle"],
                }
            ],
        },
    )

    python_result = {
        "mask": np.ones((24, 32), dtype=bool),
        "component_count": 1,
        "components": [],
        "patches": [
            {
                "order": 1,
                "bbox": [4, 5, 20, 12],
                "angle_deg": 12.5,
                "angle_confidence": 0.82,
                "angle_source": "contour_rect",
                "status_flags": [],
            }
        ],
    }
    monkeypatch.setattr(segment, "detect_patches", lambda image_rgb, config: python_result)

    detection = detect_patches_path(image_path, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert detection["patches"] == python_result["patches"]
    assert detection["components"] == python_result["components"]
    assert np.array_equal(detection["mask"], python_result["mask"])
    assert detection["mask"] is not python_result["mask"]
    assert detection["patches"][0]["angle_source"] == "contour_rect"


def test_detect_patches_path_keeps_native_multi_patch_result_without_python_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    from detectorclaw.rcf import segment

    image = _vertically_merged_sheets()
    image_path = tmp_path / "scan.tif"
    _save_rgb_image(image_path, image)

    native_result = {
        "mask": np.ones((48, 64), dtype=bool),
        "component_count": 2,
        "components": [
            {
                "component_id": 1,
                "area": 600,
                "bbox": [8, 8, 20, 28],
                "angle_deg": 0.0,
                "angle_confidence": 0.85,
                "angle_source": "contour_rect",
                "status_flags": [],
                "kept": True,
            },
            {
                "component_id": 2,
                "area": 580,
                "bbox": [34, 10, 20, 26],
                "angle_deg": 0.0,
                "angle_confidence": 0.0,
                "angle_source": "low_confidence_zero",
                "status_flags": ["low_confidence_angle"],
                "kept": True,
            },
        ],
        "patches": [
            {
                "order": 1,
                "bbox": [8, 8, 20, 28],
                "angle_deg": 0.0,
                "angle_confidence": 0.85,
                "angle_source": "contour_rect",
                "status_flags": [],
            },
            {
                "order": 2,
                "bbox": [34, 10, 20, 26],
                "angle_deg": 0.0,
                "angle_confidence": 0.0,
                "angle_source": "low_confidence_zero",
                "status_flags": ["low_confidence_angle"],
            },
        ],
    }
    monkeypatch.setattr(segment, "_run_native_segment_detection", lambda image_path, config: native_result)

    def fail_detect_patches(image_rgb, config):
        raise AssertionError("python fallback should not run for mixed multi-patch native results")

    monkeypatch.setattr(segment, "detect_patches", fail_detect_patches)

    detection = detect_patches_path(image_path, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})

    assert detection["patches"] == native_result["patches"]
    assert detection["components"] == native_result["components"]
    assert np.array_equal(detection["mask"], native_result["mask"])
    assert detection["mask"] is not native_result["mask"]
    assert detection["patches"][1]["angle_source"] == "low_confidence_zero"
