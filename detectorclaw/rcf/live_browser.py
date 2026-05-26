from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .shot_resolver import resolve_shot_inputs

DEFAULT_SESSION_NAME = "rcf-live"
DEFAULT_BROWSER = "chrome"
DEFAULT_GUI_URL = "http://127.0.0.1:8013/rcf/gui"
SESSION_RECORD_DIRNAME = ".detectorclaw"
SESSION_RECORD_FILENAME = "browser_session.json"
SESSION_HISTORY_FILENAME = "browser_history.jsonl"
SESSION_SCHEMA_VERSION = 1
PLAYWRIGHT_BACKEND = "playwright-cli"
SYSTEM_CHROMIUM_BACKEND = "python-playwright-system-chromium"
SYSTEM_CHROMIUM_CANDIDATES = ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable")


class PlaywrightCliError(RuntimeError):
    pass


class PlaywrightSessionError(RuntimeError):
    pass


class GuiLoadError(RuntimeError):
    pass


class GuiServerError(RuntimeError):
    pass


_SYSTEM_CHROMIUM_CONTROLLER = r"""
import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

payload = json.loads(sys.argv[1])
status_path = Path(payload["status_path"])
command_path = Path(payload["command_path"])
status_path.parent.mkdir(parents=True, exist_ok=True)
command_path.parent.mkdir(parents=True, exist_ok=True)
last_command_id = None


def write_status(status):
    status["pid"] = os.getpid()
    status["heartbeat_at"] = time.time()
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def write_heartbeat():
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        status = {"ok": True, "command_id": last_command_id, "status_text": "alive"}
    write_status(status)


def read_command():
    if not command_path.exists():
        return None
    try:
        return json.loads(command_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_command(page, command):
    page.goto(command["gui_url"], wait_until="networkidle")

    if command["action"] == "load_shot":
        page.locator("#shot-id").fill(command["shot_id"])
        page.locator("#data-root").fill(command["data_root"])
        page.locator("#detection-mode").select_option(command["detection_mode"])
        page.locator("#config-file").fill(command["config_file"] or "")
        page.locator("#output-dir").fill(command["output_dir"] or "")
        page.locator("#stack-config-file").fill(command["stack_config_file"] or "")
        page.locator("#load-session").click()
        page.wait_for_function(
            "() => {"
            " const text = document.getElementById('status')?.textContent || '';"
            " return text.includes('已加载') || text.includes('Loaded');"
            " }",
            timeout=15000,
        )

    try:
        status_text = page.locator("#status").text_content(timeout=3000) or "Session ready"
    except Exception:
        status_text = "Session ready"
    write_status(
        {
            "ok": True,
            "command_id": command["command_id"],
            "action": command["action"],
            "status_text": status_text,
            "url": page.url,
        }
    )


try:
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=payload["profile_dir"],
            executable_path=payload["executable_path"],
            headless=False,
            args=["--new-window"],
        )
        page = context.pages[0] if context.pages else context.new_page()
        initial_command = {
            key: payload[key]
            for key in (
                "command_id",
                "action",
                "gui_url",
                "shot_id",
                "data_root",
                "detection_mode",
                "config_file",
                "output_dir",
                "stack_config_file",
            )
        }
        run_command(page, initial_command)
        last_command_id = initial_command["command_id"]

        while True:
            current = read_command()
            if current and current.get("command_id") != last_command_id:
                run_command(page, current)
                last_command_id = current["command_id"]
            if not context.pages:
                write_status({"ok": False, "error": "browser window was closed", "command_id": last_command_id})
                raise SystemExit(0)
            write_heartbeat()
            time.sleep(1)
except Exception as exc:
    write_status({"ok": False, "error": str(exc), "command_id": last_command_id})
    raise
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _run_playwright(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "playwright-cli command failed"
        raise PlaywrightCliError(detail)
    return result


def _extract_result_text(stdout: str) -> str:
    text = stdout.strip()
    marker = "### Result"
    if marker not in text:
        return text
    after = text.split(marker, 1)[1].lstrip()
    if "\n\n### " in after:
        after = after.split("\n\n### ", 1)[0]
    line = after.strip().splitlines()[0].strip()
    try:
        return json.loads(line)
    except Exception:
        return line


def _session_command(session_name: str, *args: str) -> list[str]:
    return ["playwright-cli", f"-s={session_name}", *args]


def _system_chromium_path() -> str | None:
    configured = os.environ.get("DETECTORCLAW_CHROMIUM_BIN")
    if configured and Path(configured).exists():
        return configured
    for candidate in SYSTEM_CHROMIUM_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _playwright_cli_browser_unavailable(error: Exception) -> bool:
    message = str(error)
    return (
        "chrome-for-testing" in message
        or "is not installed" in message
        or "is not found" in message
        or "does not support chromium" in message
        or "Executable doesn't exist" in message
    )


def _resolve_root(data_root: Path) -> Path:
    return Path(data_root).expanduser().resolve()


def session_record_path(data_root: Path) -> Path:
    return _resolve_root(data_root) / SESSION_RECORD_DIRNAME / SESSION_RECORD_FILENAME


def session_history_path(data_root: Path) -> Path:
    return _resolve_root(data_root) / SESSION_RECORD_DIRNAME / SESSION_HISTORY_FILENAME


def _normalize_record(data_root: Path, payload: dict) -> dict:
    root = _resolve_root(data_root)
    record = dict(payload)
    record.setdefault("schema_version", SESSION_SCHEMA_VERSION)
    record.setdefault("backend", PLAYWRIGHT_BACKEND)
    record.setdefault("project_root", str(root))
    if record.get("active_session_name") is None and record.get("session_name") is not None:
        record["active_session_name"] = record["session_name"]
    if record.get("session_name") is None and record.get("active_session_name") is not None:
        record["session_name"] = record["active_session_name"]
    if record.get("page_url") is None and record.get("gui_url") is not None:
        record["page_url"] = record["gui_url"]
    record.setdefault("updated_at", _utc_now_iso())
    return record


def _system_chromium_status_path(data_root: Path, session_name: str) -> Path:
    return _resolve_root(data_root) / SESSION_RECORD_DIRNAME / f"{session_name}_system_chromium_status.json"


def _system_chromium_command_path(data_root: Path, session_name: str) -> Path:
    return _resolve_root(data_root) / SESSION_RECORD_DIRNAME / f"{session_name}_system_chromium_command.json"


def _system_chromium_profile_dir(data_root: Path, session_name: str) -> Path:
    timestamp = int(time.time() * 1000)
    return _resolve_root(data_root) / SESSION_RECORD_DIRNAME / "chromium-profiles" / f"{session_name}-{timestamp}"


def _process_is_running(pid: int | str | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _system_chromium_record_is_healthy(record: dict | None) -> bool:
    if not record or record.get("backend") != SYSTEM_CHROMIUM_BACKEND:
        return False
    if not _process_is_running(record.get("browser_process_pid") or record.get("browser_pid")):
        return False
    status_path = record.get("browser_status_path")
    if not status_path or not record.get("browser_command_path"):
        return False
    path = Path(status_path)
    if not path.exists():
        return False
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not status.get("ok"):
        return False
    heartbeat_at = status.get("heartbeat_at")
    if heartbeat_at is not None and time.time() - float(heartbeat_at) > 10.0:
        return False
    return True


def _write_system_chromium_command(command_path: Path, command: dict) -> None:
    temp_path = command_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(command_path)


def _wait_for_system_chromium_status(
    status_path: Path,
    command_id: str,
    process: subprocess.Popen[str] | None = None,
    timeout_s: float = 20.0,
) -> dict:
    deadline = time.monotonic() + timeout_s
    last_status: dict | None = None
    while time.monotonic() < deadline:
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            last_status = status
            if status.get("command_id") == command_id:
                if status.get("ok"):
                    return status
                raise PlaywrightSessionError(status.get("error") or "system Chromium fallback failed")
        if process is not None and process.poll() is not None:
            raise PlaywrightSessionError("system Chromium fallback exited early")
        time.sleep(0.25)
    detail = f"; last status: {last_status}" if last_status else ""
    raise PlaywrightSessionError(f"Timed out waiting for system Chromium command {command_id}{detail}")


def _open_system_chromium_session(
    data_root: Path,
    gui_url: str,
    session_name: str,
    action: str,
    shot_id: str | None = None,
    detection_mode: str = "autocrop",
    config_file: Path | None = None,
    output_dir: Path | None = None,
    stack_config_file: Path | None = None,
    timeout_s: float = 20.0,
) -> dict:
    executable_path = _system_chromium_path()
    if executable_path is None:
        raise PlaywrightSessionError("No system Chromium executable found for fallback browser launch")

    root = _resolve_root(data_root)
    status_path = _system_chromium_status_path(root, session_name)
    command_path = _system_chromium_command_path(root, session_name)
    profile_dir = _system_chromium_profile_dir(root, session_name)
    log_path = root / SESSION_RECORD_DIRNAME / f"{session_name}_system_chromium.log"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    command_id = f"{int(time.time() * 1000)}-{action}"
    command = {
        "command_id": command_id,
        "action": action,
        "gui_url": gui_url,
        "shot_id": str(shot_id or ""),
        "data_root": str(root),
        "detection_mode": detection_mode,
        "config_file": str(config_file) if config_file is not None else "",
        "output_dir": str(output_dir) if output_dir is not None else "",
        "stack_config_file": str(stack_config_file) if stack_config_file is not None else "",
    }

    existing = load_session_record(root)
    if existing and existing.get("session_name") == session_name and _system_chromium_record_is_healthy(existing):
        command_path = Path(existing.get("browser_command_path") or command_path)
        status_path = Path(existing.get("browser_status_path") or status_path)
        _write_system_chromium_command(command_path, command)
        status = _wait_for_system_chromium_status(status_path, command_id, timeout_s=timeout_s)
        status["process_pid"] = existing.get("browser_process_pid") or existing.get("browser_pid")
        status["profile_dir"] = existing.get("browser_profile_dir")
        status["log_path"] = existing.get("browser_log_path")
        status["executable_path"] = existing.get("browser_executable_path") or executable_path
        status["status_path"] = str(status_path)
        status["command_path"] = str(command_path)
        return status

    if status_path.exists():
        status_path.unlink()

    payload = {
        **command,
        "executable_path": executable_path,
        "profile_dir": str(profile_dir),
        "status_path": str(status_path),
        "command_path": str(command_path),
    }
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", _SYSTEM_CHROMIUM_CONTROLLER, json.dumps(payload)],
            stdout=log,
            stderr=log,
            start_new_session=True,
            text=True,
        )

    status = _wait_for_system_chromium_status(status_path, command_id, process=process, timeout_s=timeout_s)
    status["process_pid"] = process.pid
    status["profile_dir"] = str(profile_dir)
    status["log_path"] = str(log_path)
    status["executable_path"] = executable_path
    status["status_path"] = str(status_path)
    status["command_path"] = str(command_path)
    return status


def load_session_record(data_root: Path) -> dict | None:
    path = session_record_path(data_root)
    if not path.exists():
        return None
    return _normalize_record(data_root, json.loads(path.read_text(encoding="utf-8")))


def load_active_session(data_root: Path) -> dict | None:
    return load_session_record(data_root)


def save_session_record(data_root: Path, payload: dict) -> Path:
    path = session_record_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _normalize_record(data_root, payload)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_history_event(data_root: Path, payload: dict) -> Path:
    path = session_history_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(payload)
    event.setdefault("timestamp", _utc_now_iso())
    event.setdefault("backend", PLAYWRIGHT_BACKEND)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return path


def list_session_history(data_root: Path, limit: int | None = None) -> list[dict]:
    path = session_history_path(data_root)
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    entries.reverse()
    return entries[:limit] if limit is not None else entries


def _parse_gui_binding(gui_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(gui_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise GuiServerError(f"Unsupported GUI URL: {gui_url}")
    return parsed.hostname, parsed.port


def _gui_url_is_ready(gui_url: str, timeout_s: float = 1.0) -> bool:
    request = urllib.request.Request(gui_url, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            content_type = response.headers.get("Content-Type", "")
            return response.status < 500 and "text/html" in content_type
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _spawn_gui_server(host: str, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "detectorclaw.rcf", "gui", "--host", host, "--port", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )


def ensure_gui_server(gui_url: str = DEFAULT_GUI_URL, startup_timeout_s: float = 10.0) -> str:
    if _gui_url_is_ready(gui_url):
        return "reused"
    host, port = _parse_gui_binding(gui_url)
    _spawn_gui_server(host, port)
    deadline = time.monotonic() + startup_timeout_s
    while time.monotonic() < deadline:
        if _gui_url_is_ready(gui_url):
            return "started"
        time.sleep(0.25)
    raise GuiServerError(f"GUI server did not become ready at {gui_url}")


def session_is_healthy(session_name: str = DEFAULT_SESSION_NAME) -> bool:
    try:
        _run_playwright(_session_command(session_name, "snapshot"))
        return True
    except PlaywrightCliError:
        return False


def ensure_session(
    gui_url: str = DEFAULT_GUI_URL,
    session_name: str = DEFAULT_SESSION_NAME,
    browser: str = DEFAULT_BROWSER,
) -> str:
    if session_is_healthy(session_name):
        return "reused"
    _run_playwright(_session_command(session_name, "close"), check=False)
    try:
        _run_playwright(
            _session_command(
                session_name,
                "open",
                gui_url,
                f"--browser={browser}",
                "--persistent",
                "--headed",
            )
        )
    except PlaywrightCliError as exc:
        raise PlaywrightSessionError(str(exc)) from exc
    return "restarted"


def _session_summary(
    data_root: Path,
    session_name: str,
    gui_url: str,
    browser: str,
    session_state: str,
    gui_server_state: str,
    last_command: str,
    status_text: str,
    shot_id: str | None = None,
    config_file: Path | None = None,
    output_dir: Path | None = None,
    stack_config_file: Path | None = None,
) -> dict:
    return _normalize_record(
        data_root,
        {
            "session_name": session_name,
            "active_session_name": session_name,
            "gui_url": gui_url,
            "page_url": gui_url,
            "browser": browser,
            "session_state": session_state,
            "gui_server_state": gui_server_state,
            "last_command": last_command,
            "status_text": status_text,
            "shot_id": shot_id,
            "data_root": str(_resolve_root(data_root)),
            "config_file": str(config_file) if config_file is not None else None,
            "output_dir": str(output_dir) if output_dir is not None else None,
            "stack_config_file": str(stack_config_file) if stack_config_file is not None else None,
        },
    )


def open_project_session(
    data_root: Path,
    gui_url: str = DEFAULT_GUI_URL,
    session_name: str = DEFAULT_SESSION_NAME,
    browser: str = DEFAULT_BROWSER,
) -> dict:
    gui_server_state = ensure_gui_server(gui_url=gui_url)
    backend = PLAYWRIGHT_BACKEND
    try:
        session_state = ensure_session(gui_url=gui_url, session_name=session_name, browser=browser)
        status_text = "Session ready"
    except PlaywrightSessionError as exc:
        if not _playwright_cli_browser_unavailable(exc):
            raise
        fallback = _open_system_chromium_session(
            data_root=data_root,
            gui_url=gui_url,
            session_name=session_name,
            action="open",
        )
        backend = SYSTEM_CHROMIUM_BACKEND
        session_state = "system-chromium"
        status_text = fallback["status_text"]
    record = _session_summary(
        data_root=data_root,
        session_name=session_name,
        gui_url=gui_url,
        browser=browser,
        session_state=session_state,
        gui_server_state=gui_server_state,
        last_command="open_project_session",
        status_text=status_text,
    )
    record["backend"] = backend
    if backend == SYSTEM_CHROMIUM_BACKEND:
        record["browser_pid"] = fallback.get("pid")
        record["browser_process_pid"] = fallback.get("process_pid")
        record["browser_profile_dir"] = fallback.get("profile_dir")
        record["browser_status_path"] = fallback.get("status_path")
        record["browser_command_path"] = fallback.get("command_path")
        record["browser_log_path"] = fallback.get("log_path")
        record["browser_executable_path"] = fallback.get("executable_path")
    record_path = save_session_record(data_root, record)
    append_history_event(
        data_root,
        {
            "event": "open_session",
            "session_name": session_name,
            "gui_url": gui_url,
            "page_url": gui_url,
            "result": session_state,
            "detail": record["status_text"],
            "backend": backend,
        },
    )
    record["record_path"] = str(record_path)
    return record


def _list_session_names() -> list[str]:
    result = _run_playwright(["playwright-cli", "list"], check=False)
    names: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            names.append(line.split()[0])
    return names


def attach_project_session(
    data_root: Path,
    gui_url: str = DEFAULT_GUI_URL,
    session_name: str | None = None,
    browser: str = DEFAULT_BROWSER,
    reopen: bool = False,
) -> dict:
    existing = load_session_record(data_root)
    target_session_name = session_name or (existing.get("active_session_name") if existing else DEFAULT_SESSION_NAME)
    target_gui_url = gui_url or (existing.get("gui_url") if existing else DEFAULT_GUI_URL)
    gui_server_state = "reused"
    session_state = "attached"

    if not session_is_healthy(target_session_name):
        if reopen:
            gui_server_state = ensure_gui_server(gui_url=target_gui_url)
            session_state = ensure_session(gui_url=target_gui_url, session_name=target_session_name, browser=browser)
        elif target_session_name not in _list_session_names():
            raise PlaywrightSessionError(f"No healthy Playwright session available for {target_session_name}")

    record = _session_summary(
        data_root=data_root,
        session_name=target_session_name,
        gui_url=target_gui_url,
        browser=browser,
        session_state=session_state if session_state != "reused" else "attached",
        gui_server_state=gui_server_state,
        last_command="attach_project_session",
        status_text="Session attached",
        shot_id=existing.get("shot_id") if existing else None,
        config_file=Path(existing["config_file"]) if existing and existing.get("config_file") else None,
        output_dir=Path(existing["output_dir"]) if existing and existing.get("output_dir") else None,
        stack_config_file=Path(existing["stack_config_file"]) if existing and existing.get("stack_config_file") else None,
    )
    record_path = save_session_record(data_root, record)
    append_history_event(
        data_root,
        {
            "event": "attach_session",
            "session_name": target_session_name,
            "gui_url": target_gui_url,
            "page_url": target_gui_url,
            "result": record["session_state"],
            "detail": record["status_text"],
        },
    )
    record["record_path"] = str(record_path)
    return record


def _build_load_script(
    shot_id: str,
    data_root: Path,
    gui_url: str,
    detection_mode: str,
    config_file: Path | None,
    output_dir: Path | None,
    stack_config_file: Path | None,
) -> str:
    payload = {
        "shotId": str(shot_id),
        "dataRoot": str(data_root),
        "detectionMode": str(detection_mode),
        "configFile": str(config_file) if config_file is not None else "",
        "outputDir": str(output_dir) if output_dir is not None else "",
        "stackConfigFile": str(stack_config_file) if stack_config_file is not None else "",
    }
    payload_json = json.dumps(payload)
    return f"""
async page => {{
  const payload = {payload_json};
  await page.goto({json.dumps(gui_url)}, {{ waitUntil: "networkidle" }});
  await page.locator("#shot-id").fill(payload.shotId);
  await page.locator("#data-root").fill(payload.dataRoot);
  await page.locator("#detection-mode").selectOption(payload.detectionMode);
  await page.locator("#config-file").fill(payload.configFile);
  await page.locator("#output-dir").fill(payload.outputDir);
  await page.locator("#stack-config-file").fill(payload.stackConfigFile);
  await page.locator("#load-session").click();
  await page.waitForFunction(() => {{
    const text = document.getElementById("status")?.textContent || "";
    return text.includes("已加载") || text.includes("Loaded");
  }}, {{ timeout: 15000 }});
}}
""".strip()


def _resolve_live_shot_paths(
    shot_id: str,
    data_root: Path,
    config_file: Path | None,
    output_dir: Path | None,
    stack_config_file: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    if config_file is not None and output_dir is not None and stack_config_file is not None:
        return config_file, output_dir, stack_config_file
    resolved = resolve_shot_inputs(shot_id, data_root)
    return (
        config_file or resolved["config_file"],
        output_dir or resolved["output_dir"],
        stack_config_file or resolved["stack_config_file"],
    )


def load_shot_into_gui(
    shot_id: str,
    data_root: Path,
    gui_url: str = DEFAULT_GUI_URL,
    detection_mode: str = "autocrop",
    session_name: str = DEFAULT_SESSION_NAME,
    browser: str = DEFAULT_BROWSER,
    config_file: Path | None = None,
    output_dir: Path | None = None,
    stack_config_file: Path | None = None,
) -> dict:
    data_root = Path(data_root)
    config_file, output_dir, stack_config_file = _resolve_live_shot_paths(
        shot_id=str(shot_id),
        data_root=data_root,
        config_file=config_file,
        output_dir=output_dir,
        stack_config_file=stack_config_file,
    )
    gui_server_state = ensure_gui_server(gui_url=gui_url)
    backend = PLAYWRIGHT_BACKEND
    try:
        session_state = ensure_session(gui_url=gui_url, session_name=session_name, browser=browser)
    except PlaywrightSessionError as exc:
        if not _playwright_cli_browser_unavailable(exc):
            raise
        fallback = _open_system_chromium_session(
            data_root=data_root,
            gui_url=gui_url,
            session_name=session_name,
            action="load_shot",
            shot_id=shot_id,
            detection_mode=detection_mode,
            config_file=config_file,
            output_dir=output_dir,
            stack_config_file=stack_config_file,
        )
        status_text = fallback["status_text"]
        record = _session_summary(
            data_root=data_root,
            session_name=session_name,
            gui_url=gui_url,
            browser=browser,
            session_state="system-chromium",
            gui_server_state=gui_server_state,
            last_command="load_shot_into_gui",
            status_text=status_text,
            shot_id=str(shot_id),
            config_file=config_file,
            output_dir=output_dir,
            stack_config_file=stack_config_file,
        )
        record["backend"] = SYSTEM_CHROMIUM_BACKEND
        record["browser_pid"] = fallback.get("pid")
        record["browser_process_pid"] = fallback.get("process_pid")
        record["browser_profile_dir"] = fallback.get("profile_dir")
        record["browser_status_path"] = fallback.get("status_path")
        record["browser_command_path"] = fallback.get("command_path")
        record["browser_log_path"] = fallback.get("log_path")
        record["browser_executable_path"] = fallback.get("executable_path")
        record_path = save_session_record(data_root, record)
        append_history_event(
            data_root,
            {
                "event": "load_shot",
                "session_name": session_name,
                "gui_url": gui_url,
                "page_url": gui_url,
                "shot_id": str(shot_id),
                "result": "ok",
                "detail": status_text,
                "backend": SYSTEM_CHROMIUM_BACKEND,
            },
        )
        record["record_path"] = str(record_path)
        return record

    try:
        _run_playwright(_session_command(session_name, "goto", gui_url))
        _run_playwright(
            _session_command(
                session_name,
                "run-code",
                _build_load_script(
                    shot_id=shot_id,
                    data_root=data_root,
                    gui_url=gui_url,
                    detection_mode=detection_mode,
                    config_file=config_file,
                    output_dir=output_dir,
                    stack_config_file=stack_config_file,
                ),
            )
        )
        status = _run_playwright(
            _session_command(session_name, "eval", 'document.getElementById("status")?.textContent || ""')
        ).stdout
    except PlaywrightCliError as exc:
        append_history_event(
            data_root,
            {
                "event": "load_shot",
                "session_name": session_name,
                "gui_url": gui_url,
                "page_url": gui_url,
                "shot_id": str(shot_id),
                "result": "error",
                "detail": str(exc),
            },
        )
        raise GuiLoadError(str(exc)) from exc

    status_text = _extract_result_text(status)
    record = _session_summary(
        data_root=data_root,
        session_name=session_name,
        gui_url=gui_url,
        browser=browser,
        session_state=session_state,
        gui_server_state=gui_server_state,
        last_command="load_shot_into_gui",
        status_text=status_text,
        shot_id=str(shot_id),
        config_file=config_file,
        output_dir=output_dir,
        stack_config_file=stack_config_file,
    )
    record["backend"] = backend
    record_path = save_session_record(data_root, record)
    append_history_event(
        data_root,
        {
            "event": "load_shot",
            "session_name": session_name,
            "gui_url": gui_url,
            "page_url": gui_url,
            "shot_id": str(shot_id),
            "result": "ok",
            "detail": status_text,
            "backend": backend,
        },
    )
    record["record_path"] = str(record_path)
    return record


def _safe_command_version(command: list[str]) -> dict:
    if shutil.which(command[0]) is None:
        return {"available": False, "version": None, "error": "not found"}
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {"available": False, "version": None, "error": result.stderr.strip() or result.stdout.strip()}
    return {"available": True, "version": result.stdout.strip(), "error": None}


def _safe_python_package_version(package_name: str) -> dict:
    try:
        return {"available": True, "version": metadata.version(package_name), "error": None}
    except metadata.PackageNotFoundError:
        return {"available": False, "version": None, "error": "not installed"}


def _safe_npm_package_version(package_name: str) -> dict:
    if shutil.which("npm") is None:
        return {"available": False, "version": None, "error": "npm not found"}
    result = subprocess.run(["npm", "view", package_name, "version", "--json"], capture_output=True, text=True)
    if result.returncode != 0:
        return {"available": False, "version": None, "error": result.stderr.strip() or result.stdout.strip()}
    try:
        payload = json.loads(result.stdout)
        version = payload.get("version") if isinstance(payload, dict) else payload
    except json.JSONDecodeError:
        version = result.stdout.strip()
    return {"available": True, "version": version, "error": None}


def _detect_browser_binaries() -> dict:
    candidates = {
        "chrome": ["google-chrome", "google-chrome-stable", "chrome"],
        "chromium": ["chromium", "chromium-browser"],
        "msedge": ["msedge", "microsoft-edge"],
    }
    return {name: any(shutil.which(binary) for binary in binaries) for name, binaries in candidates.items()}


def doctor_playwright_env(data_root: Path | None = None) -> dict:
    report = {
        "playwright_cli": _safe_command_version(["playwright-cli", "--version"]),
        "node_playwright": _safe_command_version(["playwright", "--version"]),
        "python_playwright": _safe_python_package_version("playwright"),
        "playwright_mcp": _safe_npm_package_version("playwright-mcp"),
        "browser_binaries": _detect_browser_binaries(),
        "warnings": [],
    }
    if data_root is not None:
        report["session_record_path"] = str(session_record_path(data_root))
        report["history_path"] = str(session_history_path(data_root))

    node_version = report["node_playwright"]["version"]
    python_version = report["python_playwright"]["version"]
    if node_version and python_version:
        node_version = node_version.replace("Version", "").strip()
        if node_version != python_version:
            report["warnings"].append(f"Node playwright {node_version} differs from Python playwright {python_version}")
    if not report["playwright_cli"]["available"]:
        report["warnings"].append("playwright-cli is not available on PATH")
    return report
