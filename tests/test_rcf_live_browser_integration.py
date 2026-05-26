import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import pytest
from PIL import Image

from detectorclaw.rcf.live_browser import load_shot_into_gui


pytestmark = pytest.mark.live_browser


def _save_rgb_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _build_synthetic_scan(path: Path) -> None:
    canvas = np.full((220, 320, 3), 245, dtype=np.uint8)
    canvas[40:120, 30:120, 0] = 110
    canvas[40:120, 30:120, 1:] = 55
    canvas[60:170, 180:280, 0] = 135
    canvas[60:170, 180:280, 1:] = 70
    _save_rgb_image(path, canvas)


def _build_three_patch_scan(path: Path) -> None:
    canvas = np.full((240, 360, 3), 245, dtype=np.uint8)
    canvas[30:110, 20:110, 0] = 110
    canvas[30:110, 20:110, 1:] = 55
    canvas[70:180, 200:300, 0] = 135
    canvas[70:180, 200:300, 1:] = 70
    canvas[140:220, 30:120, 0] = 120
    canvas[140:220, 30:120, 1:] = 60
    _save_rgb_image(path, canvas)


def _build_background(path: Path, red_level: int) -> None:
    background = np.full((80, 80, 3), red_level, dtype=np.uint8)
    _save_rgb_image(path, background)


def _write_config(path: Path, film_path: Path, scanner_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "film_type: TEST",
                "background:",
                f"  film_path: {film_path.as_posix()}",
                f"  scanner_path: {scanner_path.as_posix()}",
                "segmentation:",
                "  min_area: 1000",
                "  padding: 4",
                "calibration:",
                "  background_quantile: 95",
                "  film_models:",
                "    TEST:",
                "      kind: polynomial",
                "      coefficients: [0.0, 100.0]",
            ]
        ),
        encoding="utf-8",
    )


def _write_stack_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "material_name": "HD",
                        "thickness": "105",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 0, "table_ID": 1, "Cutoff_ene": 3.6},
                    },
                    {
                        "material_name": "EBT",
                        "thickness": "280",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 1, "table_ID": 3, "Cutoff_ene": 28.6},
                    },
                    {
                        "material_name": "EBT",
                        "thickness": "280",
                        "thickness_type": "fixed",
                        "rcf": {"rcf_id": 2, "table_ID": 5, "Cutoff_ene": 43.7},
                    },
                ],
                "custom_materials": {},
            }
        ),
        encoding="utf-8",
    )


def _prepare_repo_style_shot(root: Path) -> None:
    _build_synthetic_scan(root / "RCF001.tif")
    _build_three_patch_scan(root / "RCF001_2.tif")
    _build_background(root / "film_background.tif", red_level=220)
    _build_background(root / "scanner_background.tif", red_level=10)
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config_dir / "rcf.example.yaml", root / "film_background.tif", root / "scanner_background.tif")
    _write_stack_config(root / "RCF1.json")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, process: subprocess.Popen[str], timeout_s: float = 15.0) -> None:
    opener = build_opener(ProxyHandler({}))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"GUI server exited early: {stderr.strip()}")
        try:
            with opener.open(url, timeout=1.0) as response:  # noqa: S310
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for server: {url}")


def _run_playwright(session_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["playwright-cli", f"-s={session_name}", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "playwright-cli command failed")
    return result


def _extract_eval_value(stdout: str):
    text = stdout.strip()
    if "### Result" not in text:
        return text
    after = text.split("### Result", 1)[1].lstrip()
    if "\n\n### " in after:
        after = after.split("\n\n### ", 1)[0]
    line = after.strip().splitlines()[0].strip()
    return json.loads(line)


def _wait_for_revision(session_name: str, min_revision: int, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    last_state = None
    while time.time() < deadline:
        last_state = json.loads(
            _extract_eval_value(
                _run_playwright(session_name, "eval", 'JSON.stringify(window.__detectorclawRcfGui.getState())').stdout
            )
        )
        if last_state["revision"] >= min_revision:
            return last_state
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for revision >= {min_revision}; last state: {last_state}")


def _post_json(url: str, payload: dict) -> dict:
    opener = build_opener(ProxyHandler({}))
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=5.0) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def test_live_shot_loads_visible_gui_and_tracks_manual_revision(tmp_path: Path) -> None:
    if shutil.which("playwright-cli") is None:
        raise AssertionError("playwright-cli is required for live browser integration tests")

    _prepare_repo_style_shot(tmp_path)
    port = _find_free_port()
    gui_url = f"http://127.0.0.1:{port}/rcf/gui"
    session_name = f"rcf-live-integration-{int(time.time() * 1000)}"
    repo_root = Path(__file__).resolve().parents[1]
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "detectorclaw.rcf.gui:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_http(gui_url, server)
        result = load_shot_into_gui(
            shot_id="001",
            data_root=tmp_path,
            gui_url=gui_url,
            detection_mode="autocrop",
            session_name=session_name,
            browser="chrome",
        )

        assert result["session_name"] == session_name
        assert "已加载" in result["status_text"]

        initial_state = json.loads(
            _extract_eval_value(
                _run_playwright(session_name, "eval", 'JSON.stringify(window.__detectorclawRcfGui.getState())').stdout
            )
        )
        assert initial_state["shot_id"] == "001"
        assert initial_state["revision"] == 0
        assert len(initial_state["patches"]) == 5

        first_patch = initial_state["patches"][0]
        _post_json(
            f"http://127.0.0.1:{port}/api/rcf/session/{initial_state['session_id']}/patch/{first_patch['patch_id']}/geometry",
            {
                "rotated_rect": {
                    **first_patch["rotated_rect"],
                    "angle_deg": first_patch["rotated_rect"]["angle_deg"] + 4,
                }
            },
        )

        revised_state = _wait_for_revision(session_name, 1)
        status_text = _extract_eval_value(
            _run_playwright(
                session_name,
                "eval",
                'document.getElementById("status")?.textContent || ""',
            ).stdout
        )

        assert revised_state["revision"] >= 1
        assert revised_state["last_modified_patch_id"] == first_patch["patch_id"]
        assert revised_state["patches"][0]["rotated_rect"]["angle_deg"] == first_patch["rotated_rect"]["angle_deg"] + 4
        assert "检测到手动修改" in status_text
        assert first_patch["patch_id"] in status_text
    finally:
        subprocess.run(["playwright-cli", f"-s={session_name}", "close"], capture_output=True, text=True)
        server.terminate()
        server.wait(timeout=5)


def test_live_gui_can_move_selected_patch_and_refresh_revision(tmp_path: Path) -> None:
    if shutil.which("playwright-cli") is None:
        raise AssertionError("playwright-cli is required for live browser integration tests")

    _prepare_repo_style_shot(tmp_path)
    port = _find_free_port()
    gui_url = f"http://127.0.0.1:{port}/rcf/gui"
    session_name = f"rcf-live-drag-{int(time.time() * 1000)}"
    repo_root = Path(__file__).resolve().parents[1]
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "detectorclaw.rcf.gui:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_http(gui_url, server)
        load_shot_into_gui(
            shot_id="001",
            data_root=tmp_path,
            gui_url=gui_url,
            detection_mode="autocrop",
            session_name=session_name,
            browser="chrome",
        )

        initial_state = json.loads(
            _extract_eval_value(
                _run_playwright(session_name, "eval", 'JSON.stringify(window.__detectorclawRcfGui.getState())').stdout
            )
        )
        first_patch = initial_state["patches"][0]
        initial_cx = first_patch["rotated_rect"]["cx"]
        initial_cy = first_patch["rotated_rect"]["cy"]

        _run_playwright(
            session_name,
            "eval",
            'window.__detectorclawRcfGui.debugMoveSelectedPatch(12, 8)',
        )

        revised_state = _wait_for_revision(session_name, 1)
        moved_patch = revised_state["patches"][0]
        status_text = _extract_eval_value(
            _run_playwright(
                session_name,
                "eval",
                'document.getElementById("status")?.textContent || ""',
            ).stdout
        )

        assert moved_patch["rotated_rect"]["cx"] == initial_cx + 12
        assert moved_patch["rotated_rect"]["cy"] == initial_cy + 8
        assert revised_state["last_modified_patch_id"] == first_patch["patch_id"]
        assert "已应用拖拽" in status_text
    finally:
        subprocess.run(["playwright-cli", f"-s={session_name}", "close"], capture_output=True, text=True)
        server.terminate()
        server.wait(timeout=5)


def test_live_gui_can_rotate_and_resize_selected_patch(tmp_path: Path) -> None:
    if shutil.which("playwright-cli") is None:
        raise AssertionError("playwright-cli is required for live browser integration tests")

    _prepare_repo_style_shot(tmp_path)
    port = _find_free_port()
    gui_url = f"http://127.0.0.1:{port}/rcf/gui"
    session_name = f"rcf-live-reshape-{int(time.time() * 1000)}"
    repo_root = Path(__file__).resolve().parents[1]
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "detectorclaw.rcf.gui:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_http(gui_url, server)
        load_shot_into_gui(
            shot_id="001",
            data_root=tmp_path,
            gui_url=gui_url,
            detection_mode="autocrop",
            session_name=session_name,
            browser="chrome",
        )

        initial_state = json.loads(
            _extract_eval_value(
                _run_playwright(session_name, "eval", 'JSON.stringify(window.__detectorclawRcfGui.getState())').stdout
            )
        )
        first_patch = initial_state["patches"][0]
        initial_angle = first_patch["rotated_rect"]["angle_deg"]
        initial_width = first_patch["rotated_rect"]["width"]
        initial_height = first_patch["rotated_rect"]["height"]

        _run_playwright(
            session_name,
            "eval",
            'window.__detectorclawRcfGui.debugRotateSelectedPatch(15)',
        )
        rotated_state = _wait_for_revision(session_name, 1)
        assert rotated_state["patches"][0]["rotated_rect"]["angle_deg"] == initial_angle + 15

        _run_playwright(
            session_name,
            "eval",
            'window.__detectorclawRcfGui.debugResizeSelectedPatch(20, 10)',
        )
        resized_state = _wait_for_revision(session_name, 2)
        resized_patch = resized_state["patches"][0]

        assert resized_patch["rotated_rect"]["width"] == initial_width + 20
        assert resized_patch["rotated_rect"]["height"] == initial_height + 10
    finally:
        subprocess.run(["playwright-cli", f"-s={session_name}", "close"], capture_output=True, text=True)
        server.terminate()
        server.wait(timeout=5)


def test_live_gui_can_assign_selected_patch_as_next_order(tmp_path: Path) -> None:
    if shutil.which("playwright-cli") is None:
        raise AssertionError("playwright-cli is required for live browser integration tests")

    _prepare_repo_style_shot(tmp_path)
    port = _find_free_port()
    gui_url = f"http://127.0.0.1:{port}/rcf/gui"
    session_name = f"rcf-live-assign-{int(time.time() * 1000)}"
    repo_root = Path(__file__).resolve().parents[1]
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "detectorclaw.rcf.gui:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "error",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_http(gui_url, server)
        load_shot_into_gui(
            shot_id="001",
            data_root=tmp_path,
            gui_url=gui_url,
            detection_mode="autocrop",
            session_name=session_name,
            browser="chrome",
        )

        _run_playwright(
            session_name,
            "eval",
            'window.__detectorclawRcfGui.debugAssignSelectedPatchAsNext()',
        )
        revised_state = _wait_for_revision(session_name, 1)
        first_patch = revised_state["patches"][0]
        status_text = _extract_eval_value(
            _run_playwright(
                session_name,
                "eval",
                'document.getElementById("status")?.textContent || ""',
            ).stdout
        )

        assert first_patch["assignment_status"] == "assigned"
        assert first_patch["assigned_order"] == 1
        assert "设为下一片" in status_text
        assignment_text = _extract_eval_value(
            _run_playwright(
                session_name,
                "eval",
                'document.getElementById("patch-assignment")?.textContent || ""',
            ).stdout
        )
        assert "当前人工片序：第 1 片" == assignment_text
    finally:
        subprocess.run(["playwright-cli", f"-s={session_name}", "close"], capture_output=True, text=True)
        server.terminate()
        server.wait(timeout=5)
