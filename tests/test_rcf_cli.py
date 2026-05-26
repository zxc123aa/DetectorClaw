import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw


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


def _build_rotated_scan(path: Path) -> None:
    canvas = Image.new("RGB", (420, 300), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(180, 50), (300, 90), (250, 240), (130, 200)], fill=(215, 218, 200), outline=(100, 110, 90))
    canvas.save(path)


def _build_faint_rotated_scan(path: Path) -> None:
    canvas = Image.new("RGB", (420, 320), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(165, 55), (305, 88), (272, 248), (132, 215)], fill=(239, 241, 241), outline=(216, 220, 220))
    draw.polygon([(198, 110), (255, 123), (241, 183), (184, 170)], fill=(120, 150, 180))
    canvas.save(path)


def _build_misaligned_center_scan(path: Path) -> None:
    canvas = Image.new("RGB", (420, 320), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(165, 55), (305, 88), (272, 248), (132, 215)], fill=(239, 241, 241), outline=(216, 220, 220))
    draw.rectangle((185, 110, 255, 190), fill=(120, 150, 180))
    canvas.save(path)


def _build_register_patch(path: Path, tint: int = 220) -> None:
    canvas = Image.new("RGB", (260, 260), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(30, 20), (235, 35), (220, 240), (18, 225)], fill=(tint, tint + 2, tint - 4), outline=(180, 182, 176))
    draw.ellipse((90, 90, 170, 170), fill=(120, 90, 70))
    canvas.save(path)


def _build_registered_layer(path: Path, base_level: int) -> None:
    canvas = np.full((100, 120, 3), 245, dtype=np.uint8)
    canvas[15:70, 25:90, 2] = base_level
    canvas[15:70, 25:90, 0:2] = 90
    _save_rgb_image(path, canvas)


def _build_mask(path: Path, x0: int, y0: int, x1: int, y1: int) -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask, mode="L").save(path)


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


def test_live_browser_reuses_healthy_named_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import live_browser

    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(live_browser, "_run_playwright", fake_run)

    outcome = live_browser.ensure_session(gui_url="http://127.0.0.1:8013/rcf/gui", session_name="rcf-live")

    assert outcome == "reused"
    assert calls == [["playwright-cli", "-s=rcf-live", "snapshot"]]


def test_live_browser_reuses_running_gui_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import live_browser

    spawn_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(live_browser, "_gui_url_is_ready", lambda gui_url, timeout_s=1.0: True)
    monkeypatch.setattr(
        live_browser,
        "_spawn_gui_server",
        lambda host, port: spawn_calls.append((host, port)),
    )

    outcome = live_browser.ensure_gui_server("http://127.0.0.1:8013/rcf/gui")

    assert outcome == "reused"
    assert spawn_calls == []


def test_live_browser_starts_gui_server_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import live_browser

    spawn_calls: list[tuple[str, int]] = []
    readiness = iter([False, True])

    monkeypatch.setattr(live_browser, "_gui_url_is_ready", lambda gui_url, timeout_s=1.0: next(readiness))
    monkeypatch.setattr(
        live_browser,
        "_spawn_gui_server",
        lambda host, port: spawn_calls.append((host, port)),
    )

    outcome = live_browser.ensure_gui_server("http://127.0.0.1:8013/rcf/gui")

    assert outcome == "started"
    assert spawn_calls == [("127.0.0.1", 8013)]


def test_live_browser_restarts_broken_named_session(monkeypatch: pytest.MonkeyPatch) -> None:
    from detectorclaw.rcf import live_browser

    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-1] == "snapshot":
            raise live_browser.PlaywrightCliError("broken session")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(live_browser, "_run_playwright", fake_run)

    outcome = live_browser.ensure_session(
        gui_url="http://127.0.0.1:8013/rcf/gui",
        session_name="rcf-live",
        browser="chrome",
    )

    assert outcome == "restarted"
    assert calls == [
        ["playwright-cli", "-s=rcf-live", "snapshot"],
        ["playwright-cli", "-s=rcf-live", "close"],
        ["playwright-cli", "-s=rcf-live", "open", "http://127.0.0.1:8013/rcf/gui", "--browser=chrome", "--persistent", "--headed"],
    ]


def test_cli_live_shot_forwards_to_live_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from detectorclaw.rcf import cli

    captured: dict = {}

    def fake_load_shot_into_gui(**kwargs) -> dict:
        captured.update(kwargs)
        return {
            "session_name": kwargs["session_name"],
            "gui_server_state": "reused",
            "session_state": "reused",
            "status_text": "Loaded shot 001",
        }

    monkeypatch.setattr("detectorclaw.rcf.live_browser.load_shot_into_gui", fake_load_shot_into_gui)

    exit_code = cli.main(
        [
            "live-shot",
            "--shot",
            "001",
            "--data-root",
            str(tmp_path),
            "--gui-url",
            "http://127.0.0.1:8013/rcf/gui",
            "--detection-mode",
            "autocrop",
            "--session-name",
            "rcf-live",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "shot_id": "001",
        "data_root": tmp_path,
        "gui_url": "http://127.0.0.1:8013/rcf/gui",
        "detection_mode": "autocrop",
        "session_name": "rcf-live",
        "browser": "chrome",
        "config_file": None,
        "output_dir": None,
        "stack_config_file": None,
    }


def test_live_browser_persists_session_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from detectorclaw.rcf import live_browser

    shot_dir = tmp_path / "reference" / "shots" / "shot_001"
    shot_dir.mkdir(parents=True)
    scan_1_path = shot_dir / "RCF001.tif"
    scan_2_path = shot_dir / "RCF001_2.tif"
    stack_config_path = shot_dir / "RCF1.json"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "rcf.example.yaml"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    def fake_run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        if len(command) >= 2 and command[-2] == "eval":
            return subprocess.CompletedProcess(command, 0, stdout='### Result\n"已加载 发次 001"\n', stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(live_browser, "ensure_gui_server", lambda gui_url=live_browser.DEFAULT_GUI_URL: "reused")
    monkeypatch.setattr(live_browser, "ensure_session", lambda gui_url=live_browser.DEFAULT_GUI_URL, session_name=live_browser.DEFAULT_SESSION_NAME, browser=live_browser.DEFAULT_BROWSER: "reused")
    monkeypatch.setattr(live_browser, "_run_playwright", fake_run)

    result = live_browser.load_shot_into_gui(
        shot_id="001",
        data_root=tmp_path,
        gui_url="http://127.0.0.1:8013/rcf/gui",
        detection_mode="autocrop",
        session_name="rcf-live",
        browser="chrome",
    )

    record_path = live_browser.session_record_path(tmp_path)
    assert record_path.exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["session_name"] == "rcf-live"
    assert record["shot_id"] == "001"
    assert record["gui_url"] == "http://127.0.0.1:8013/rcf/gui"
    assert record["data_root"] == str(tmp_path.resolve())
    assert record["config_file"] == str(config_path)
    assert record["output_dir"] == str(tmp_path / "runs" / "gui" / "shot_001_review")
    assert record["stack_config_file"] == str(stack_config_path)
    assert result["record_path"] == str(record_path)


def test_live_browser_open_project_session_writes_active_record_and_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from detectorclaw.rcf import live_browser

    monkeypatch.setattr(live_browser, "ensure_gui_server", lambda gui_url=live_browser.DEFAULT_GUI_URL: "reused")
    monkeypatch.setattr(
        live_browser,
        "ensure_session",
        lambda gui_url=live_browser.DEFAULT_GUI_URL, session_name=live_browser.DEFAULT_SESSION_NAME, browser=live_browser.DEFAULT_BROWSER: "reused",
    )

    result = live_browser.open_project_session(
        data_root=tmp_path,
        gui_url="http://127.0.0.1:8013/rcf/gui",
        session_name="rcf-live",
        browser="chrome",
    )

    record = live_browser.load_session_record(tmp_path)
    history = live_browser.list_session_history(tmp_path)
    assert result["session_name"] == "rcf-live"
    assert record is not None
    assert record["active_session_name"] == "rcf-live"
    assert record["backend"] == "playwright-cli"
    assert record["last_command"] == "open_project_session"
    assert len(history) == 1
    assert history[0]["event"] == "open_session"


def test_live_browser_open_project_session_falls_back_to_system_chromium(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from detectorclaw.rcf import live_browser

    monkeypatch.setattr(live_browser, "ensure_gui_server", lambda gui_url=live_browser.DEFAULT_GUI_URL: "reused")

    def fake_ensure_session(**kwargs) -> str:
        raise live_browser.PlaywrightSessionError("Chromium distribution 'chrome' is not found")

    monkeypatch.setattr(live_browser, "ensure_session", fake_ensure_session)
    monkeypatch.setattr(
        live_browser,
        "_open_system_chromium_session",
        lambda **kwargs: {"status_text": "Session ready"},
    )

    result = live_browser.open_project_session(
        data_root=tmp_path,
        gui_url="http://127.0.0.1:8013/rcf/gui",
        session_name="rcf-live",
        browser="chrome",
    )

    record = live_browser.load_session_record(tmp_path)
    assert result["session_state"] == "system-chromium"
    assert result["backend"] == "python-playwright-system-chromium"
    assert record is not None
    assert record["backend"] == "python-playwright-system-chromium"


def test_live_browser_attach_project_session_recovers_default_session_without_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from detectorclaw.rcf import live_browser

    monkeypatch.setattr(live_browser, "session_is_healthy", lambda session_name=live_browser.DEFAULT_SESSION_NAME: session_name == "rcf-live")

    result = live_browser.attach_project_session(
        data_root=tmp_path,
        gui_url="http://127.0.0.1:8013/rcf/gui",
    )

    record = live_browser.load_session_record(tmp_path)
    assert result["session_name"] == "rcf-live"
    assert result["session_state"] == "attached"
    assert record is not None
    assert record["session_name"] == "rcf-live"
    assert record["session_state"] == "attached"


def test_cli_live_session_prints_saved_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from detectorclaw.rcf import cli

    record_path = tmp_path / ".detectorclaw" / "browser_session.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "session_name": "rcf-live",
                "gui_url": "http://127.0.0.1:8013/rcf/gui",
                "shot_id": "001",
            }
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["live-session", "--data-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["session_name"] == "rcf-live"
    assert payload["shot_id"] == "001"


def test_cli_live_open_forwards_to_live_browser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from detectorclaw.rcf import cli

    captured: dict = {}

    def fake_open_project_session(**kwargs) -> dict:
        captured.update(kwargs)
        return {
            "session_name": kwargs["session_name"],
            "session_state": "reused",
            "status_text": "Session ready",
        }

    monkeypatch.setattr("detectorclaw.rcf.live_browser.open_project_session", fake_open_project_session)

    exit_code = cli.main(
        [
            "live-open",
            "--data-root",
            str(tmp_path),
            "--gui-url",
            "http://127.0.0.1:8013/rcf/gui",
            "--session-name",
            "rcf-live",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "data_root": tmp_path,
        "gui_url": "http://127.0.0.1:8013/rcf/gui",
        "session_name": "rcf-live",
        "browser": "chrome",
    }


def test_cli_live_attach_forwards_reopen_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from detectorclaw.rcf import cli
    from detectorclaw.rcf import live_browser

    captured: dict = {}

    def fake_attach_project_session(**kwargs) -> dict:
        captured.update(kwargs)
        return {
            "session_name": kwargs.get("session_name") or "rcf-live",
            "session_state": "restarted",
            "status_text": "Session attached",
        }

    monkeypatch.setattr("detectorclaw.rcf.live_browser.attach_project_session", fake_attach_project_session)

    exit_code = cli.main(
        [
            "live-attach",
            "--data-root",
            str(tmp_path),
            "--reopen",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "data_root": tmp_path,
        "gui_url": live_browser.DEFAULT_GUI_URL,
        "session_name": None,
        "browser": "chrome",
        "reopen": True,
    }


def test_cli_live_session_prints_history_when_requested(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from detectorclaw.rcf import cli
    from detectorclaw.rcf import live_browser

    live_browser.save_session_record(
        tmp_path,
        {
            "schema_version": 1,
            "session_name": "rcf-live",
            "active_session_name": "rcf-live",
            "gui_url": "http://127.0.0.1:8013/rcf/gui",
            "shot_id": "001",
        },
    )
    live_browser.append_history_event(
        tmp_path,
        {
            "event": "open_session",
            "session_name": "rcf-live",
            "result": "ok",
        },
    )

    exit_code = cli.main(["live-session", "--data-root", str(tmp_path), "--history", "5"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["active"]["session_name"] == "rcf-live"
    assert payload["history"][0]["event"] == "open_session"


def test_cli_live_doctor_prints_versions(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from detectorclaw.rcf import cli

    monkeypatch.setattr(
        "detectorclaw.rcf.live_browser.doctor_playwright_env",
        lambda data_root=None: {
            "playwright_cli": {"available": True, "version": "0.1.0"},
            "node_playwright": {"available": True, "version": "1.56.1"},
            "python_playwright": {"available": True, "version": "1.57.0"},
            "warnings": ["version mismatch"],
        },
    )

    exit_code = cli.main(["live-doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["playwright_cli"]["version"] == "0.1.0"
    assert payload["warnings"] == ["version mismatch"]


def test_cli_processes_single_scan_and_emits_outputs(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "out"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "process",
            "--input",
            str(scan_path),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr

    overlay_path = output_dir / "overlay.png"
    overlay_raw_path = output_dir / "overlay_raw.png"
    overlay_final_path = output_dir / "overlay_final.png"
    mask_path = output_dir / "mask.png"
    components_path = output_dir / "components.json"
    patches_dir = output_dir / "patches"
    dose_dir = output_dir / "dose"
    summary_path = output_dir / "summary.json"
    review_path = output_dir / "review.json"
    debug_log_path = output_dir / "debug.log"

    assert overlay_path.exists()
    assert overlay_raw_path.exists()
    assert overlay_final_path.exists()
    assert mask_path.exists()
    assert components_path.exists()
    assert patches_dir.exists()
    assert dose_dir.exists()
    assert summary_path.exists()
    assert review_path.exists()
    assert debug_log_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["input_file"] == str(scan_path)
    assert summary["film_type"] == "TEST"
    assert summary["segmentation_status"] == "ok"
    assert summary["review_applied"] is False
    assert summary["calibration_status"] == "dose"
    assert summary["qc_flags"] == []
    assert len(summary["patches"]) == 2
    assert all(patch["dose_mean"] >= 0 for patch in summary["patches"])
    assert all(patch["status"] == "ok" for patch in summary["patches"])
    assert all("angle_source" in patch for patch in summary["patches"])
    assert all("angle_confidence" in patch for patch in summary["patches"])
    assert all("status_flags" in patch for patch in summary["patches"])

    components = json.loads(components_path.read_text(encoding="utf-8"))
    assert components["component_count"] >= 2
    assert len(components["components"]) >= 2

    patch_files = sorted(patches_dir.glob("*.png"))
    dose_files = sorted(dose_dir.glob("*.png"))
    assert len(patch_files) == 2
    assert len(dose_files) == 2


def test_cli_autocrop_processes_single_scan_and_writes_summary(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.tif"
    output_dir = tmp_path / "autocrop_out"

    _build_three_patch_scan(scan_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "autocrop",
            "--input",
            str(scan_path),
            "--output",
            str(output_dir),
            "--expected-count",
            "3",
            "--min-side-px",
            "70",
            "--save-debug",
            "--save-montage",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "patches").exists()
    assert (output_dir / "overlays" / "scan_overlay.png").exists()
    assert (output_dir / "metadata" / "scan.json").exists()
    assert (output_dir / "debug" / "scan_mask.png").exists()
    assert (output_dir / "debug" / "scan_white_distance.png").exists()
    assert (output_dir / "montage" / "scan_montage.png").exists()

    patch_files = sorted((output_dir / "patches").glob("scan_rcf_*.png"))
    assert len(patch_files) == 3

    metadata = json.loads((output_dir / "metadata" / "scan.json").read_text(encoding="utf-8"))
    assert metadata["detected_count"] == 3
    assert len(metadata["candidates"]) == 3
    assert all(Path(candidate["patch_path"]).exists() for candidate in metadata["candidates"])


def test_cli_uses_review_file_to_override_detection_order(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "out"
    review_override_path = tmp_path / "review_override.json"

    _build_synthetic_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    review_override = {
        "patches": [
            {"order": 1, "bbox": [176, 56, 108, 118]},
            {"order": 2, "bbox": [26, 36, 98, 88]},
        ]
    }
    review_override_path.write_text(
        json.dumps(review_override),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "process",
            "--input",
            str(scan_path),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--review",
            str(review_override_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["review_applied"] is True
    assert summary["segmentation_status"] == "review_override"
    assert [patch["order"] for patch in summary["patches"]] == [1, 2]
    assert summary["patches"][0]["bbox"] == [176, 56, 108, 118]


def test_cli_debug_patch_emits_single_patch_rectification_artifacts(tmp_path: Path) -> None:
    scan_path = tmp_path / "rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "patch_debug"

    _build_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "debug-patch",
            "--input",
            str(scan_path),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--patch-order",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "patch_raw.png").exists()
    assert (output_dir / "patch_lab_distance.png").exists()
    assert (output_dir / "patch_mask.png").exists()
    assert (output_dir / "patch_threshold.png").exists()
    assert (output_dir / "patch_component.png").exists()
    assert (output_dir / "patch_contour.png").exists()
    assert (output_dir / "patch_boxpoints.png").exists()
    assert (output_dir / "patch_rotated.png").exists()
    assert (output_dir / "patch_warped.png").exists()
    assert (output_dir / "patch_refined_crop.png").exists()
    assert (output_dir / "patch_debug.json").exists()

    debug_payload = json.loads((output_dir / "patch_debug.json").read_text(encoding="utf-8"))
    assert debug_payload["patch_order"] == 1
    assert debug_payload["rectification_source"] == "opencv_min_area_rect"
    assert debug_payload["rectification_confidence"] > 0.5
    assert debug_payload["angle_source"] in {"contour_rect", "hough", "regionprops_fallback", "low_confidence_zero"}
    assert len(debug_payload["box_points"]) == 4
    assert "refined_crop_bbox" in debug_payload
    assert len(debug_payload["refined_crop_bbox"]) == 4


def test_cli_debug_patch_projection_expands_to_sheet_boundary(tmp_path: Path) -> None:
    scan_path = tmp_path / "faint_rotated_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "patch_debug"

    _build_faint_rotated_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "debug-patch",
            "--input",
            str(scan_path),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--patch-order",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr

    debug_payload = json.loads((output_dir / "patch_debug.json").read_text(encoding="utf-8"))
    assert debug_payload["sheet_coverage_ratio"] > debug_payload["component_coverage_ratio"]
    assert debug_payload["projection_threshold"] > 0.0


def test_cli_debug_patch_edge_residual_corrects_misaligned_center(tmp_path: Path) -> None:
    scan_path = tmp_path / "misaligned_center_scan.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "patch_debug"

    _build_misaligned_center_scan(scan_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "debug-patch",
            "--input",
            str(scan_path),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
            "--patch-order",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr

    debug_payload = json.loads((output_dir / "patch_debug.json").read_text(encoding="utf-8"))
    assert abs(debug_payload["edge_residual_angle_deg"]) > 5.0
    assert abs(debug_payload["rotation_angle_deg"]) > 5.0
    assert debug_payload["edge_line_confidence"] > 0.2


def test_cli_fails_when_background_file_is_missing(tmp_path: Path) -> None:
    scan_path = tmp_path / "scan.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    output_dir = tmp_path / "out"

    _build_synthetic_scan(scan_path)
    _build_background(scanner_background_path, red_level=10)

    config_path.write_text(
        "\n".join(
            [
                "film_type: TEST",
                "background:",
                "  film_path: missing_background.tif",
                f"  scanner_path: {scanner_background_path.as_posix()}",
                "calibration:",
                "  film_models:",
                "    TEST:",
                "      kind: polynomial",
                "      coefficients: [0.0, 100.0]",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "process",
            "--input",
            str(scan_path),
            "--config",
            str(config_path),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode != 0
    assert "background" in result.stderr.lower()


def test_cli_processes_multiple_scans_and_maps_stack_metadata(tmp_path: Path) -> None:
    scan_1_path = tmp_path / "scan_1.tif"
    scan_2_path = tmp_path / "scan_2.tif"
    film_background_path = tmp_path / "film_background.tif"
    scanner_background_path = tmp_path / "scanner_background.tif"
    config_path = tmp_path / "rcf.yaml"
    stack_config_path = tmp_path / "stack.json"
    output_dir = tmp_path / "out"

    _build_synthetic_scan(scan_1_path)
    _build_three_patch_scan(scan_2_path)
    _build_background(film_background_path, red_level=220)
    _build_background(scanner_background_path, red_level=10)
    _write_config(config_path, film_background_path, scanner_background_path)
    _write_stack_config(stack_config_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "process",
            "--input",
            str(scan_1_path),
            str(scan_2_path),
            "--config",
            str(config_path),
            "--stack-config",
            str(stack_config_path),
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["scan_count"] == 2
    assert summary["stack_config_file"] == str(stack_config_path)
    assert len(summary["patches"]) == 5
    assert summary["qc_flags"] == ["stack_config_patch_count_mismatch"]
    assert summary["patches"][0]["scan_index"] == 1
    assert summary["patches"][0]["global_order"] == 1
    assert summary["patches"][0]["stack"]["rcf_id"] == 0
    assert summary["patches"][1]["stack"]["rcf_id"] == 1
    assert summary["patches"][2]["stack"]["rcf_id"] == 2

    assert (output_dir / "scans" / "scan_01" / "summary.json").exists()
    assert (output_dir / "scans" / "scan_02" / "summary.json").exists()


def test_cli_register_outputs_registered_masks_and_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "register_out"
    input_dir.mkdir()
    _build_register_patch(input_dir / "layer_01.png", tint=220)
    _build_register_patch(input_dir / "layer_02.png", tint=228)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "register",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--out-size",
            "256x256",
            "--save-debug",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "registered").exists()
    assert (output_dir / "registered_rgba").exists()
    assert (output_dir / "masks").exists()
    assert (output_dir / "overlays").exists()
    assert (output_dir / "metadata").exists()
    assert (output_dir / "debug").exists()
    assert (output_dir / "summary.csv").exists()

    metadata = json.loads((output_dir / "metadata" / "layer_01.json").read_text(encoding="utf-8"))
    assert metadata["registered_size_before_crop"] == {"width": 256, "height": 256}
    assert metadata["registered_size_after_crop"] == {"width": 256, "height": 256}
    assert metadata["crop_mode"] == "fixed"
    assert Path(metadata["outputs"]["registered"]).exists()
    assert Path(metadata["outputs"]["mask"]).exists()


def test_cli_od_stack_uses_intersection_bbox_auto_roi(tmp_path: Path) -> None:
    registered_dir = tmp_path / "registered"
    masks_dir = tmp_path / "masks"
    output_dir = tmp_path / "od_out"
    registered_dir.mkdir()
    masks_dir.mkdir()

    _build_registered_layer(registered_dir / "layer_01_registered.png", base_level=140)
    _build_registered_layer(registered_dir / "layer_02_registered.png", base_level=170)
    _build_mask(masks_dir / "layer_01_registered_mask.png", 25, 15, 90, 70)
    _build_mask(masks_dir / "layer_02_registered_mask.png", 30, 20, 85, 65)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "od-stack",
            "--registered-dir",
            str(registered_dir),
            "--masks-dir",
            str(masks_dir),
            "--output-dir",
            str(output_dir),
            "--channel",
            "red",
            "--reference-mode",
            "per-image-percentile",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "od_npy").exists()
    assert (output_dir / "od_png").exists()
    assert (output_dir / "overlays").exists()
    assert (output_dir / "metadata").exists()
    assert (output_dir / "summary.csv").exists()

    metadata = json.loads((output_dir / "metadata" / "layer_01_registered.json").read_text(encoding="utf-8"))
    assert metadata["roi"] == [30, 20, 85, 65]
    assert metadata["roi_source"] == "auto:intersection-bbox"
    assert Path(output_dir / "od_npy" / "layer_01_registered_od.npy").exists()
    assert Path(output_dir / "od_png" / "layer_02_registered_od.png").exists()


def test_cli_workflow_runs_register_then_od_stack(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "workflow_out"
    input_dir.mkdir()
    _build_register_patch(input_dir / "layer_01.png", tint=220)
    _build_register_patch(input_dir / "layer_02.png", tint=228)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "workflow",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--out-size",
            "256x256",
            "--channel",
            "red",
            "--reference-mode",
            "per-image-percentile",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "register" / "registered").exists()
    assert (output_dir / "register" / "summary.csv").exists()
    assert (output_dir / "od" / "summary.csv").exists()
    assert (output_dir / "od" / "od_png").exists()
