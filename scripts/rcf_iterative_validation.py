#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

GUI_K_EXPRESSION = (
    "dose_overview_prewarm or "
    "load_starts_session_precompute or "
    "assignment_requeues_session_precompute or "
    "dose_image_cache_only_skips_synchronous_compute or "
    "assets_manifest_reads_disk_cache_readiness or "
    "serves_gray_and_pseudocolor_dose_previews"
)
CLI_K_EXPRESSION = (
    "live_browser_open_project_session_writes_active_record_and_history or "
    "live_browser_attach_project_session_recovers_default_session_without_record"
)
SMOKE_COMMANDS = [
    ["pytest", "tests/test_rcf_preview.py"],
    ["pytest", "tests/test_rcf_gui.py", "-k", GUI_K_EXPRESSION],
    ["pytest", "tests/test_rcf_cli.py", "-k", CLI_K_EXPRESSION],
]


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "duration_ms": round(self.duration_ms, 3),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _api_base_from_gui_url(gui_url: str) -> str:
    parsed = urlparse(gui_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Unsupported GUI URL: {gui_url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _parse_gui_binding(gui_url: str) -> tuple[str, int]:
    parsed = urlparse(gui_url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
        raise ValueError(f"GUI URL must be http://host:port/rcf/gui, got: {gui_url}")
    return parsed.hostname, int(parsed.port)


def _run_command(command: list[str], cwd: Path) -> CommandResult:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_ms=duration_ms,
    )


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout_s: float = 20.0) -> tuple[int, dict[str, Any], float]:
    opener = build_opener(ProxyHandler({}))
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout_s) as response:  # noqa: S310
            body_bytes = response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            content_type = (response.headers.get("Content-Type") or "").lower()
            decoded = _decode_payload(body_bytes, content_type=content_type)
            return int(response.status), decoded, elapsed_ms
    except HTTPError as exc:
        body_bytes = exc.read() if exc.fp is not None else b""
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content_type = (exc.headers.get("Content-Type") if exc.headers is not None else "") or ""
        decoded = _decode_payload(body_bytes, content_type=str(content_type).lower())
        if not decoded:
            decoded = {"detail": str(exc)}
        return int(exc.code), decoded, elapsed_ms
    except URLError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return 0, {"detail": str(exc)}, elapsed_ms


def _decode_payload(body_bytes: bytes, content_type: str) -> dict[str, Any]:
    if not body_bytes:
        return {}
    if "application/json" in content_type:
        try:
            return json.loads(body_bytes.decode("utf-8"))
        except Exception:
            return {"detail": body_bytes.decode("utf-8", errors="replace")}
    return {"bytes": len(body_bytes)}


def _http_status(url: str, timeout_s: float = 2.0) -> int:
    opener = build_opener(ProxyHandler({}))
    request = Request(url, method="GET")
    try:
        with opener.open(request, timeout=timeout_s) as response:  # noqa: S310
            return int(response.status)
    except HTTPError as exc:
        return int(exc.code)
    except URLError:
        return 0


def _wait_for_gui_ready(gui_url: str, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _http_status(gui_url, timeout_s=1.0) == 200:
            return
        time.sleep(0.25)
    raise TimeoutError(f"GUI did not become ready at {gui_url}")


def _spawn_gui_server(gui_url: str, cwd: Path) -> subprocess.Popen[str]:
    host, port = _parse_gui_binding(gui_url)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "gui",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _compute_p95(values: list[float]) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _wait_precompute_done(api_base: str, session_id: str, timeout_s: float = 180.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status_code, payload, _elapsed = _http_json(
            "GET",
            f"{api_base}/api/rcf/session/{session_id}/precompute/status",
            timeout_s=20.0,
        )
        if status_code != 200:
            raise RuntimeError(f"Failed to fetch precompute status: HTTP {status_code}, payload={payload}")
        state = str(payload.get("state", ""))
        last_payload = payload
        if state in {"done", "error"}:
            return payload
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for precompute done. Last payload: {last_payload}")


def _sorted_assigned_patches(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    patches = list(manifest.get("patches") or [])
    patches.sort(key=lambda item: (int(item.get("assigned_order") or 1_000_000), str(item.get("patch_id"))))
    return patches


def _measure_dose_image_latency(
    api_base: str,
    session_id: str,
    patches: list[dict[str, Any]],
    switch_count: int,
    cache_only: bool,
    quality: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    if not patches:
        return latencies, failures
    for index in range(switch_count):
        patch = patches[index % len(patches)]
        patch_id = str(patch["patch_id"])
        url = (
            f"{api_base}/api/rcf/session/{session_id}/patch/{patch_id}/dose-image"
            f"?palette=turbo&max_dim=320&format=jpeg&quality={int(quality)}"
            f"&cache_only={'true' if cache_only else 'false'}"
        )
        status_code, payload, elapsed_ms = _http_json("GET", url, timeout_s=20.0)
        if status_code != 200:
            failures.append(
                {
                    "patch_id": patch_id,
                    "status_code": status_code,
                    "payload": payload,
                    "elapsed_ms": round(elapsed_ms, 3),
                }
            )
            continue
        latencies.append(elapsed_ms)
    return latencies, failures


def _auto_assign_patches(
    api_base: str,
    session_id: str,
    count: int,
) -> list[dict[str, Any]]:
    status_code, state_payload, _elapsed = _http_json(
        "GET",
        f"{api_base}/api/rcf/session/{session_id}/state",
        timeout_s=20.0,
    )
    if status_code != 200:
        raise RuntimeError(f"Failed to fetch session state: HTTP {status_code}, payload={state_payload}")
    patches = list(state_payload.get("patches") or [])
    patches.sort(key=lambda item: str(item.get("patch_id")))
    selected = patches[: max(0, count)]
    assignments: list[dict[str, Any]] = []
    for index, patch in enumerate(selected, start=1):
        patch_id = str(patch["patch_id"])
        payload = {"assignment_status": "assigned", "assigned_order": index}
        patch_status, patch_payload, _patch_elapsed = _http_json(
            "POST",
            f"{api_base}/api/rcf/session/{session_id}/patch/{patch_id}/assignment",
            payload=payload,
            timeout_s=20.0,
        )
        assignments.append(
            {
                "patch_id": patch_id,
                "status_code": patch_status,
                "payload": patch_payload,
            }
        )
        if patch_status != 200:
            raise RuntimeError(f"Assignment failed for {patch_id}: HTTP {patch_status}, payload={patch_payload}")
    return assignments


def _open_browser_session(data_root: Path, gui_url: str, shot_id: str) -> list[CommandResult]:
    commands = [
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "live-open",
            "--data-root",
            str(data_root),
            "--gui-url",
            gui_url,
        ],
        [
            sys.executable,
            "-m",
            "detectorclaw.rcf",
            "live-shot",
            "--shot",
            shot_id,
            "--data-root",
            str(data_root),
            "--gui-url",
            gui_url,
            "--detection-mode",
            "autocrop",
        ],
    ]
    results: list[CommandResult] = []
    for command in commands:
        results.append(_run_command(command, cwd=data_root))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one iterative validation round for DetectorClaw RCF GUI.")
    parser.add_argument("--data-root", type=Path, default=Path.cwd(), help="Project/data root containing reference/shots.")
    parser.add_argument("--shot", type=str, default="001", help="Shot identifier to load, default 001.")
    parser.add_argument("--gui-url", type=str, default="http://127.0.0.1:18013/rcf/gui", help="GUI URL.")
    parser.add_argument("--detection-mode", type=str, default="autocrop", choices=["autocrop", "segment"])
    parser.add_argument("--switch-count", type=int, default=10, help="Dose next-switch sample count.")
    parser.add_argument("--jpeg-quality", type=int, default=78, help="Dose preview JPEG quality for latency probe URLs.")
    parser.add_argument("--first-threshold-ms", type=float, default=450.0, help="First single dose latency P95 threshold.")
    parser.add_argument("--switch-threshold-ms", type=float, default=120.0, help="Cached switch latency P95 threshold.")
    parser.add_argument("--precompute-timeout-s", type=float, default=180.0, help="Precompute timeout.")
    parser.add_argument("--auto-assign-count", type=int, default=8, help="Assign first N patches before manifest/latency checks.")
    parser.add_argument("--round-label", type=str, default="", help="Optional round label for report file name.")
    parser.add_argument("--skip-smoke-tests", action="store_true", help="Skip smoke test commands.")
    parser.add_argument("--skip-browser-open", action="store_true", help="Skip live-open/live-shot browser steps.")
    parser.add_argument(
        "--reuse-existing-gui",
        action="store_true",
        help="Reuse existing GUI process on --gui-url instead of starting an isolated one.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    api_base = _api_base_from_gui_url(args.gui_url)
    report: dict[str, Any] = {
        "timestamp": int(time.time()),
        "data_root": str(data_root),
        "shot": args.shot,
        "gui_url": args.gui_url,
        "smoke_tests": [],
        "browser_commands": [],
        "api": {},
        "metrics": {},
        "thresholds_ms": {
            "first_p95_max": args.first_threshold_ms,
            "switch_p95_max": args.switch_threshold_ms,
        },
        "probe_quality": int(args.jpeg_quality),
        "pass": False,
    }
    gui_process: subprocess.Popen[str] | None = None
    exit_code = 1
    try:
        if not args.skip_smoke_tests:
            for command in SMOKE_COMMANDS:
                result = _run_command(command, cwd=data_root)
                report["smoke_tests"].append(result.to_json())

        if args.reuse_existing_gui:
            status = _http_status(args.gui_url, timeout_s=2.0)
            if status != 200:
                raise RuntimeError(f"GUI is not ready at {args.gui_url}; got status {status}")
        else:
            gui_process = _spawn_gui_server(args.gui_url, cwd=data_root)
            _wait_for_gui_ready(args.gui_url, timeout_s=20.0)

        if not args.skip_browser_open:
            browser_results = _open_browser_session(data_root, args.gui_url, args.shot)
            report["browser_commands"] = [item.to_json() for item in browser_results]

        load_payload = {
            "shot_id": args.shot,
            "data_root": str(data_root),
            "detection_mode": args.detection_mode,
        }
        load_status, load_response, load_elapsed_ms = _http_json(
            "POST",
            f"{api_base}/api/rcf/session/load",
            payload=load_payload,
            timeout_s=30.0,
        )
        report["api"]["load"] = {
            "status_code": load_status,
            "elapsed_ms": round(load_elapsed_ms, 3),
            "payload": load_response,
        }
        if load_status != 200:
            raise RuntimeError(f"Session load failed: HTTP {load_status}, payload={load_response}")

        session_id = str(load_response["session_id"])
        precompute_status = _wait_precompute_done(api_base, session_id, timeout_s=args.precompute_timeout_s)
        report["api"]["precompute_status"] = precompute_status

        assignments = _auto_assign_patches(
            api_base=api_base,
            session_id=session_id,
            count=args.auto_assign_count,
        )
        report["api"]["auto_assign"] = {
            "count": len(assignments),
            "results": assignments,
        }
        precompute_status = _wait_precompute_done(api_base, session_id, timeout_s=args.precompute_timeout_s)
        report["api"]["precompute_status_after_assign"] = precompute_status

        manifest_status, manifest_payload, manifest_elapsed_ms = _http_json(
            "GET",
            f"{api_base}/api/rcf/session/{session_id}/assets/manifest",
            timeout_s=20.0,
        )
        report["api"]["manifest"] = {
            "status_code": manifest_status,
            "elapsed_ms": round(manifest_elapsed_ms, 3),
            "payload": manifest_payload,
        }
        if manifest_status != 200:
            raise RuntimeError(f"Assets manifest failed: HTTP {manifest_status}, payload={manifest_payload}")

        assigned_patches = _sorted_assigned_patches(manifest_payload)
        if not assigned_patches:
            raise RuntimeError("No assigned patches found in assets manifest")

        first_patch_id = str(assigned_patches[0]["patch_id"])
        first_url = (
            f"{api_base}/api/rcf/session/{session_id}/patch/{first_patch_id}/dose-image"
            f"?palette=turbo&max_dim=320&format=jpeg&quality={int(args.jpeg_quality)}&cache_only=false"
        )
        first_status, first_payload, first_elapsed_ms = _http_json("GET", first_url, timeout_s=20.0)
        report["api"]["first_dose_request"] = {
            "status_code": first_status,
            "elapsed_ms": round(first_elapsed_ms, 3),
            "payload": first_payload if first_status != 200 else {"detail": "image-bytes"},
        }

        switch_latencies, switch_failures = _measure_dose_image_latency(
            api_base=api_base,
            session_id=session_id,
            patches=assigned_patches,
            switch_count=args.switch_count,
            cache_only=True,
            quality=args.jpeg_quality,
        )
        first_p95 = _compute_p95([first_elapsed_ms]) if first_status == 200 else math.nan
        switch_p95 = _compute_p95(switch_latencies)
        report["metrics"] = {
            "assigned_patch_count": len(assigned_patches),
            "first_latency_ms": round(first_elapsed_ms, 3) if first_status == 200 else math.nan,
            "first_p95_ms": round(first_p95, 3) if not math.isnan(first_p95) else math.nan,
            "switch_samples": [round(value, 3) for value in switch_latencies],
            "switch_failures": switch_failures,
            "switch_p95_ms": round(switch_p95, 3) if not math.isnan(switch_p95) else math.nan,
        }

        smoke_ok = all(item["returncode"] == 0 for item in report["smoke_tests"]) if report["smoke_tests"] else True
        browser_ok = all(item["returncode"] == 0 for item in report["browser_commands"]) if report["browser_commands"] else True
        precompute_ok = str(precompute_status.get("state")) == "done"
        first_ok = first_status == 200 and first_elapsed_ms <= args.first_threshold_ms
        switch_ok = (not switch_failures) and (not math.isnan(switch_p95)) and switch_p95 <= args.switch_threshold_ms
        report["pass"] = all([smoke_ok, browser_ok, precompute_ok, first_ok, switch_ok])
        exit_code = 0 if report["pass"] else 1
    except Exception as exc:  # noqa: BLE001
        report["error"] = str(exc)
        exit_code = 1
    finally:
        if gui_process is not None:
            gui_process.terminate()
            try:
                gui_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                gui_process.kill()
            report["gui_server"] = {
                "mode": "isolated",
                "returncode": gui_process.returncode,
            }
        elif args.reuse_existing_gui:
            report["gui_server"] = {
                "mode": "reused",
            }

    report_path = _write_report(data_root, report, args.round_label)
    print(json.dumps({"pass": report["pass"], "report": str(report_path), "metrics": report.get("metrics", {}), "error": report.get("error")}, ensure_ascii=False, indent=2))
    return exit_code


def _write_report(data_root: Path, report: dict[str, Any], round_label: str) -> Path:
    output_dir = data_root / "runs" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = round_label.strip() or time.strftime("%Y%m%d-%H%M%S", time.localtime())
    path = output_dir / f"iterative-validation-{suffix}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
