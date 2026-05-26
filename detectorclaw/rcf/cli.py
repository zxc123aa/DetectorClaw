from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .autocrop import autocrop_inputs, parse_fixed_size
from . import live_browser
from .od_stack import od_stack_registered, parse_roi
from .pipeline import debug_patch_rectification, process_scan
from .register import parse_size, register_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detectorclaw.rcf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process one RCF scan TIFF")
    process_parser.add_argument("--input", required=True, nargs="+", type=Path, help="One or more input TIFF scans")
    process_parser.add_argument("--config", required=True, type=Path, help="YAML config")
    process_parser.add_argument("--output", required=True, type=Path, help="Output directory")
    process_parser.add_argument(
        "--stack-config",
        type=Path,
        default=None,
        help="Optional JSON stack configuration for multi-scan ordering",
    )
    process_parser.add_argument(
        "--review",
        type=Path,
        default=None,
        help="Optional JSON file overriding patch order or boxes",
    )

    gui_parser = subparsers.add_parser("gui", help="Start local RCF review GUI")
    gui_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    gui_parser.add_argument("--port", default=8000, type=int, help="Port to bind")

    live_shot_parser = subparsers.add_parser("live-shot", help="Reuse a fixed Playwright browser session and load one shot into the GUI")
    live_shot_parser.add_argument("--shot", required=True, type=str, help="Shot identifier, e.g. 001")
    live_shot_parser.add_argument("--data-root", required=True, type=Path, help="Repository or data root containing RCF files")
    live_shot_parser.add_argument("--gui-url", default=live_browser.DEFAULT_GUI_URL, type=str, help="Visible RCF GUI URL")
    live_shot_parser.add_argument("--detection-mode", default="autocrop", choices=["autocrop", "segment"])
    live_shot_parser.add_argument("--session-name", default=live_browser.DEFAULT_SESSION_NAME, type=str, help="Fixed Playwright session name")
    live_shot_parser.add_argument("--browser", default=live_browser.DEFAULT_BROWSER, type=str, help="Browser for playwright-cli open")
    live_shot_parser.add_argument("--config-file", default=None, type=Path)
    live_shot_parser.add_argument("--output-dir", default=None, type=Path)
    live_shot_parser.add_argument("--stack-config-file", default=None, type=Path)

    live_open_parser = subparsers.add_parser("live-open", help="Open or reuse the active Playwright browser session for this project")
    live_open_parser.add_argument("--data-root", required=True, type=Path, help="Data root containing the browser session record")
    live_open_parser.add_argument("--gui-url", default=live_browser.DEFAULT_GUI_URL, type=str, help="Visible RCF GUI URL")
    live_open_parser.add_argument("--session-name", default=live_browser.DEFAULT_SESSION_NAME, type=str, help="Fixed Playwright session name")
    live_open_parser.add_argument("--browser", default=live_browser.DEFAULT_BROWSER, type=str, help="Browser for playwright-cli open")

    live_attach_parser = subparsers.add_parser("live-attach", help="Attach to the active Playwright browser session for this project")
    live_attach_parser.add_argument("--data-root", required=True, type=Path, help="Data root containing the browser session record")
    live_attach_parser.add_argument("--gui-url", default=live_browser.DEFAULT_GUI_URL, type=str, help="Visible RCF GUI URL")
    live_attach_parser.add_argument("--session-name", default=None, type=str, help="Optional explicit Playwright session name")
    live_attach_parser.add_argument("--browser", default=live_browser.DEFAULT_BROWSER, type=str, help="Browser used when reopening a stale session")
    live_attach_parser.add_argument("--reopen", action="store_true", help="Reopen the session when the recorded browser is stale")

    live_session_parser = subparsers.add_parser("live-session", help="Show the last recorded Playwright browser session for one data root")
    live_session_parser.add_argument("--data-root", required=True, type=Path, help="Data root containing the browser session record")
    live_session_parser.add_argument("--history", default=0, type=int, help="Include up to N most recent history events")

    live_doctor_parser = subparsers.add_parser("live-doctor", help="Inspect local Playwright CLI, Python, and MCP environment state")
    live_doctor_parser.add_argument("--data-root", default=None, type=Path, help="Optional data root for session record path reporting")

    debug_patch_parser = subparsers.add_parser("debug-patch", help="Debug rectification for one detected patch")
    debug_patch_parser.add_argument("--input", required=True, type=Path, help="Input TIFF scan")
    debug_patch_parser.add_argument("--config", required=True, type=Path, help="YAML config")
    debug_patch_parser.add_argument("--output", required=True, type=Path, help="Output directory")
    debug_patch_parser.add_argument("--patch-order", required=True, type=int, help="Detected patch order to inspect")

    autocrop_parser = subparsers.add_parser("autocrop", help="Autocrop and rectify RCF sheets using OpenCV baseline")
    autocrop_parser.add_argument("--input", required=True, type=Path, help="Input image file or directory")
    autocrop_parser.add_argument("--output", required=True, type=Path, help="Output directory")
    autocrop_parser.add_argument("--expected-count", type=int, default=None, help="Keep the most likely first N sheets")
    autocrop_parser.add_argument("--fixed-size", type=str, default=None, help="Optional fixed patch size, e.g. 800x800")
    autocrop_parser.add_argument("--min-area-ratio", type=float, default=0.01)
    autocrop_parser.add_argument("--max-area-ratio", type=float, default=0.50)
    autocrop_parser.add_argument("--min-side-px", type=int, default=80)
    autocrop_parser.add_argument("--max-aspect-ratio", type=float, default=1.8)
    autocrop_parser.add_argument("--blur-ksize", type=int, default=5)
    autocrop_parser.add_argument("--morph-ksize", type=int, default=9)
    autocrop_parser.add_argument("--manual-threshold", type=int, default=None)
    autocrop_parser.add_argument("--save-debug", action="store_true")
    autocrop_parser.add_argument("--save-montage", action="store_true")

    register_parser = subparsers.add_parser("register", help="Register one or more already-cut RCF layers")
    register_parser.add_argument("--input", required=True, type=Path, help="Input image file or directory")
    register_parser.add_argument("--output", required=True, type=Path, help="Output directory")
    register_parser.add_argument("--out-size", default="2000x2000", type=str, help="Unified registration plane size")
    register_parser.add_argument("--crop-mode", default="fixed", choices=["fixed", "tight"])
    register_parser.add_argument("--blur-ksize", default=5, type=int)
    register_parser.add_argument("--morph-ksize", default=7, type=int)
    register_parser.add_argument("--manual-threshold", default=None, type=int)
    register_parser.add_argument("--min-area-ratio", default=0.4, type=float)
    register_parser.add_argument("--save-debug", action="store_true")

    od_stack_parser = subparsers.add_parser("od-stack", help="Compute OD maps and ROI stats from registered layers")
    od_stack_parser.add_argument("--registered-dir", required=True, type=Path)
    od_stack_parser.add_argument("--masks-dir", required=True, type=Path)
    od_stack_parser.add_argument("--output-dir", required=True, type=Path)
    od_stack_parser.add_argument("--channel", default="red", choices=["red", "green", "blue", "gray", "rgb-mean"])
    od_stack_parser.add_argument("--reference-mode", default="per-image-percentile", choices=["per-image-percentile", "reference-file"])
    od_stack_parser.add_argument("--reference-file", default=None, type=Path)
    od_stack_parser.add_argument("--reference-percentile", default=99.5, type=float)
    od_stack_parser.add_argument("--roi", default=None, type=str)
    od_stack_parser.add_argument("--auto-roi", default="intersection-bbox", choices=["intersection-bbox"])
    od_stack_parser.add_argument("--png-vmin", default=None, type=float)
    od_stack_parser.add_argument("--png-vmax", default=None, type=float)

    workflow_parser = subparsers.add_parser("workflow", help="Run register then OD stack in one command")
    workflow_parser.add_argument("--input", required=True, type=Path, help="Input image file or directory")
    workflow_parser.add_argument("--output", required=True, type=Path, help="Workflow root output directory")
    workflow_parser.add_argument("--out-size", default="2000x2000", type=str)
    workflow_parser.add_argument("--crop-mode", default="fixed", choices=["fixed", "tight"])
    workflow_parser.add_argument("--blur-ksize", default=5, type=int)
    workflow_parser.add_argument("--morph-ksize", default=7, type=int)
    workflow_parser.add_argument("--manual-threshold", default=None, type=int)
    workflow_parser.add_argument("--min-area-ratio", default=0.4, type=float)
    workflow_parser.add_argument("--save-debug", action="store_true")
    workflow_parser.add_argument("--channel", default="red", choices=["red", "green", "blue", "gray", "rgb-mean"])
    workflow_parser.add_argument("--reference-mode", default="per-image-percentile", choices=["per-image-percentile", "reference-file"])
    workflow_parser.add_argument("--reference-file", default=None, type=Path)
    workflow_parser.add_argument("--reference-percentile", default=99.5, type=float)
    workflow_parser.add_argument("--roi", default=None, type=str)
    workflow_parser.add_argument("--auto-roi", default="intersection-bbox", choices=["intersection-bbox"])
    workflow_parser.add_argument("--png-vmin", default=None, type=float)
    workflow_parser.add_argument("--png-vmax", default=None, type=float)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "process":
            if len(args.input) == 1 and args.stack_config is None:
                process_scan(
                    input_path=args.input[0],
                    config_path=args.config,
                    output_dir=args.output,
                    review_path=args.review,
                )
            else:
                from .pipeline import process_scan_series

                process_scan_series(
                    input_paths=args.input,
                    config_path=args.config,
                    output_dir=args.output,
                    stack_config_path=args.stack_config,
                    review_path=args.review,
                )
            return 0
        if args.command == "gui":
            import uvicorn

            from .gui import create_app

            uvicorn.run(create_app(), host=args.host, port=args.port)
            return 0
        if args.command == "live-shot":
            result = live_browser.load_shot_into_gui(
                shot_id=args.shot,
                data_root=args.data_root,
                gui_url=args.gui_url,
                detection_mode=args.detection_mode,
                session_name=args.session_name,
                browser=args.browser,
                config_file=args.config_file,
                output_dir=args.output_dir,
                stack_config_file=args.stack_config_file,
            )
            print(
                f"{result['session_name']} {result['session_state']}: {result['status_text']}",
                file=sys.stdout,
            )
            return 0
        if args.command == "live-open":
            result = live_browser.open_project_session(
                data_root=args.data_root,
                gui_url=args.gui_url,
                session_name=args.session_name,
                browser=args.browser,
            )
            print(
                f"{result['session_name']} {result['session_state']}: {result['status_text']}",
                file=sys.stdout,
            )
            return 0
        if args.command == "live-attach":
            result = live_browser.attach_project_session(
                data_root=args.data_root,
                gui_url=args.gui_url,
                session_name=args.session_name,
                browser=args.browser,
                reopen=args.reopen,
            )
            print(
                f"{result['session_name']} {result['session_state']}: {result['status_text']}",
                file=sys.stdout,
            )
            return 0
        if args.command == "live-session":
            record = live_browser.load_session_record(args.data_root)
            if record is None:
                print(f"No browser session record found under {args.data_root}", file=sys.stderr)
                return 1
            payload: dict = {"active": record} if args.history > 0 else record
            if args.history > 0:
                payload["history"] = live_browser.list_session_history(args.data_root, limit=args.history)
            print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stdout)
            return 0
        if args.command == "live-doctor":
            report = live_browser.doctor_playwright_env(data_root=args.data_root)
            print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stdout)
            return 0
        if args.command == "debug-patch":
            debug_patch_rectification(
                input_path=args.input,
                config_path=args.config,
                output_dir=args.output,
                patch_order=args.patch_order,
            )
            return 0
        if args.command == "autocrop":
            autocrop_inputs(
                input_path=args.input,
                output_dir=args.output,
                expected_count=args.expected_count,
                fixed_size=parse_fixed_size(args.fixed_size),
                min_area_ratio=args.min_area_ratio,
                max_area_ratio=args.max_area_ratio,
                min_side_px=args.min_side_px,
                max_aspect_ratio=args.max_aspect_ratio,
                blur_ksize=args.blur_ksize,
                morph_ksize=args.morph_ksize,
                manual_threshold=args.manual_threshold,
                save_debug=args.save_debug,
                save_montage=args.save_montage,
            )
            return 0
        if args.command == "register":
            register_inputs(
                input_path=args.input,
                output_dir=args.output,
                out_size=parse_size(args.out_size),
                crop_mode=args.crop_mode,
                blur_ksize=args.blur_ksize,
                morph_ksize=args.morph_ksize,
                manual_threshold=args.manual_threshold,
                min_area_ratio=args.min_area_ratio,
                save_debug=args.save_debug,
            )
            return 0
        if args.command == "od-stack":
            od_stack_registered(
                registered_dir=args.registered_dir,
                masks_dir=args.masks_dir,
                output_dir=args.output_dir,
                channel=args.channel,
                reference_mode=args.reference_mode,
                reference_file=args.reference_file,
                reference_percentile=args.reference_percentile,
                roi=parse_roi(args.roi),
                auto_roi_mode=args.auto_roi,
                png_vmin=args.png_vmin,
                png_vmax=args.png_vmax,
            )
            return 0
        if args.command == "workflow":
            register_output_dir = args.output / "register"
            od_output_dir = args.output / "od"
            register_inputs(
                input_path=args.input,
                output_dir=register_output_dir,
                out_size=parse_size(args.out_size),
                crop_mode=args.crop_mode,
                blur_ksize=args.blur_ksize,
                morph_ksize=args.morph_ksize,
                manual_threshold=args.manual_threshold,
                min_area_ratio=args.min_area_ratio,
                save_debug=args.save_debug,
            )
            od_stack_registered(
                registered_dir=register_output_dir / "registered",
                masks_dir=register_output_dir / "masks",
                output_dir=od_output_dir,
                channel=args.channel,
                reference_mode=args.reference_mode,
                reference_file=args.reference_file,
                reference_percentile=args.reference_percentile,
                roi=parse_roi(args.roi),
                auto_roi_mode=args.auto_roi,
                png_vmin=args.png_vmin,
                png_vmax=args.png_vmax,
            )
            return 0
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    parser.print_help(sys.stderr)
    return 2
