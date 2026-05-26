import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detectorclaw.rcf.runtime_cache import LRUCache


def _save_rgb_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _build_large_scan(path: Path) -> None:
    canvas = np.full((1800, 3200, 3), 245, dtype=np.uint8)
    canvas[200:900, 200:1000, 0] = 110
    canvas[200:900, 200:1000, 1:] = 55
    _save_rgb_image(path, canvas)


def _library_preview_result() -> tuple[bytes, str]:
    image = Image.new("RGB", (96, 64), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue(), "image/jpeg"


def test_preview_scan_falls_back_to_python_without_native_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_BIN", raising=False)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_LIB", raising=False)
    monkeypatch.setattr(preview, "load_native_preview_library", lambda: None)
    monkeypatch.setattr(preview, "locate_native_preview_binary", lambda: None)

    content, media_type = preview.render_scan_preview(
        scan_file=scan_path,
        max_dim=320,
        preview_format="jpeg",
        quality=70,
    )

    assert media_type == "image/jpeg"
    decoded = Image.open(io.BytesIO(content))
    assert decoded.format == "JPEG"
    assert max(decoded.size) <= 320


def test_preview_scan_prefers_native_binary_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    fake_bin = tmp_path / "rcf_preview_core"
    fake_bin.write_text("", encoding="utf-8")
    monkeypatch.setenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_BIN", str(fake_bin))
    monkeypatch.setattr(preview, "load_native_preview_library", lambda: None)

    seen: dict[str, object] = {}

    def fake_run(command: list[str], input: str, capture_output: bool, text: bool, check: bool) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["payload"] = json.loads(input)
        image = Image.new("RGB", (120, 80), (10, 20, 30))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        payload = json.dumps({"media_type": "image/jpeg", "content_hex": buffer.getvalue().hex()})
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr="")

    monkeypatch.setattr(preview.subprocess, "run", fake_run)

    content, media_type = preview.render_scan_preview(
        scan_file=scan_path,
        max_dim=200,
        preview_format="jpeg",
        quality=65,
    )

    assert media_type == "image/jpeg"
    assert seen["command"] == [str(fake_bin), "scan-preview"]
    assert seen["payload"] == {
        "scan_file": str(scan_path),
        "max_dim": 200,
        "preview_format": "jpeg",
        "quality": 65,
    }
    decoded = Image.open(io.BytesIO(content))
    assert decoded.size == (120, 80)


def test_preview_scan_prefers_native_library_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    fake_library = object()
    monkeypatch.setattr(preview, "load_native_preview_library", lambda: fake_library)
    monkeypatch.setattr(
        preview,
        "_run_native_preview_via_library",
        lambda library, command_name, payload: (
            Image.new("RGB", (96, 64), (10, 20, 30)).tobytes(),
            "image/raw",
        )
        if False
        else _library_preview_result(),
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess fallback should not run when native preview library is available")

    monkeypatch.setattr(preview.subprocess, "run", fail_run)

    content, media_type = preview.render_scan_preview(
        scan_file=scan_path,
        max_dim=200,
        preview_format="jpeg",
        quality=65,
    )

    assert media_type == "image/jpeg"
    decoded = Image.open(io.BytesIO(content))
    assert decoded.size == (96, 64)


def test_preview_scan_caches_identical_requests_and_misses_on_param_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    monkeypatch.setattr(preview, "_PREVIEW_RESULT_CACHE", LRUCache(max_entries=8))
    fake_library = object()
    monkeypatch.setattr(preview, "load_native_preview_library", lambda: fake_library)

    calls: list[tuple[str, int]] = []

    def fake_library_call(library, command_name: str, payload: dict) -> tuple[bytes, str]:
        calls.append((command_name, int(payload["max_dim"])))
        return _library_preview_result()

    monkeypatch.setattr(preview, "_run_native_preview_via_library", fake_library_call)

    first = preview.render_scan_preview(scan_path, max_dim=320, preview_format="jpeg", quality=70)
    second = preview.render_scan_preview(scan_path, max_dim=320, preview_format="jpeg", quality=70)
    third = preview.render_scan_preview(scan_path, max_dim=240, preview_format="jpeg", quality=70)

    assert first == second
    assert third == _library_preview_result()
    assert calls == [("scan-preview", 320), ("scan-preview", 240)]


def test_preview_patch_falls_back_to_python_and_resizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_BIN", raising=False)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_LIB", raising=False)
    monkeypatch.setattr(preview, "load_native_preview_library", lambda: None)
    monkeypatch.setattr(preview, "locate_native_preview_binary", lambda: None)

    content, media_type = preview.render_patch_preview(
        scan_file=scan_path,
        quad_points=[[200.0, 200.0], [1000.0, 200.0], [1000.0, 900.0], [200.0, 900.0]],
        max_dim=240,
        preview_format="jpeg",
        quality=70,
    )

    assert media_type == "image/jpeg"
    decoded = Image.open(io.BytesIO(content))
    assert decoded.format == "JPEG"
    assert max(decoded.size) <= 240


def test_preview_bbox_falls_back_to_python_and_resizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_BIN", raising=False)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_LIB", raising=False)
    monkeypatch.setattr(preview, "load_native_preview_library", lambda: None)
    monkeypatch.setattr(preview, "locate_native_preview_binary", lambda: None)

    content, media_type = preview.render_bbox_preview(
        scan_file=scan_path,
        bbox=[200, 200, 800, 700],
        max_dim=200,
        preview_format="jpeg",
        quality=70,
    )

    assert media_type == "image/jpeg"
    decoded = Image.open(io.BytesIO(content))
    assert decoded.format == "JPEG"
    assert decoded.size == (200, 175)


def test_preview_patch_applies_crop_bbox_in_python_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_BIN", raising=False)
    monkeypatch.delenv("DETECTORCLAW_RCF_NATIVE_PREVIEW_LIB", raising=False)
    monkeypatch.setattr(preview, "load_native_preview_library", lambda: None)
    monkeypatch.setattr(preview, "locate_native_preview_binary", lambda: None)

    content, media_type = preview.render_patch_preview(
        scan_file=scan_path,
        quad_points=[[200.0, 200.0], [1000.0, 200.0], [1000.0, 900.0], [200.0, 900.0]],
        crop_bbox=[10, 20, 160, 120],
        max_dim=500,
        preview_format="jpeg",
        quality=70,
    )

    assert media_type == "image/jpeg"
    decoded = Image.open(io.BytesIO(content))
    assert decoded.format == "JPEG"
    assert decoded.size == (160, 120)


def test_extract_patch_image_supports_backend_argument(tmp_path: Path) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    quad = [[200.0, 200.0], [1000.0, 200.0], [1000.0, 900.0], [200.0, 900.0]]

    cpu_image = preview.extract_patch_image(scan_path, quad, backend="cpu")
    auto_image = preview.extract_patch_image(scan_path, quad, backend="auto")

    assert cpu_image.size == auto_image.size
    cpu_array = np.asarray(cpu_image.convert("RGB"), dtype=np.int16)
    auto_array = np.asarray(auto_image.convert("RGB"), dtype=np.int16)
    assert np.mean(np.abs(cpu_array - auto_array)) < 2.0


def test_extract_patch_image_respects_max_dim_during_warp(tmp_path: Path) -> None:
    from detectorclaw.rcf import preview

    scan_path = tmp_path / "scan.tif"
    _build_large_scan(scan_path)
    quad = [[200.0, 200.0], [1000.0, 200.0], [1000.0, 900.0], [200.0, 900.0]]

    full_image = preview.extract_patch_image(scan_path, quad, backend="cpu")
    limited_image = preview.extract_patch_image(scan_path, quad, backend="cpu", max_dim=200)

    assert max(full_image.size) > 200
    assert max(limited_image.size) <= 200

    resized_full = preview.resize_for_preview(full_image, 200)
    full_array = np.asarray(resized_full.convert("RGB"), dtype=np.int16)
    limited_array = np.asarray(limited_image.convert("RGB"), dtype=np.int16)
    assert np.mean(np.abs(full_array - limited_array)) < 3.0
