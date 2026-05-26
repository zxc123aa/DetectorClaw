from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .autocrop import detect_rcf_rectangles
from .config import load_config
from .io import load_rgb_image, save_json
from .segment import detect_patches, detect_patches_path
from .shot_resolver import resolve_shot_inputs
from .stack import load_stack_entries


def _rotated_rect_from_bbox(bbox: list[int], angle_deg: float) -> dict:
    x, y, width, height = bbox
    return {
        "cx": float(x + width / 2.0),
        "cy": float(y + height / 2.0),
        "width": float(width),
        "height": float(height),
        "angle_deg": float(angle_deg),
    }


def _rotated_rect_corners(rotated_rect: dict) -> list[list[float]]:
    cx = float(rotated_rect["cx"])
    cy = float(rotated_rect["cy"])
    width = float(rotated_rect["width"])
    height = float(rotated_rect["height"])
    angle_rad = math.radians(float(rotated_rect["angle_deg"]))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    half_w = width / 2.0
    half_h = height / 2.0
    local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    corners = []
    for dx, dy in local:
        x = cx + dx * cos_a - dy * sin_a
        y = cy + dx * sin_a + dy * cos_a
        corners.append([float(x), float(y)])
    return corners


def _rotated_rect_aabb(rotated_rect: dict) -> list[int]:
    corners = _rotated_rect_corners(rotated_rect)
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    x0 = math.floor(min(xs))
    y0 = math.floor(min(ys))
    x1 = math.ceil(max(xs))
    y1 = math.ceil(max(ys))
    return [int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))]


def _normalize_quad(points: list[list[float]] | None) -> list[list[float]] | None:
    if points is None:
        return None
    return [[float(point[0]), float(point[1])] for point in points]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _version_id() -> str:
    return uuid.uuid4().hex[:12]


def _display_label(patch: dict) -> str:
    if patch.get("assignment_status") == "assigned" and patch.get("assigned_order") is not None:
        return f"第 {int(patch['assigned_order'])} 片"
    if patch.get("assignment_status") == "ignored_duplicate":
        return f"候{int(patch.get('local_order') or 0)}"
    return f"候{int(patch.get('local_order') or 0)}"


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}

    def create_session(
        self,
        input_files: list[Path] | None,
        config_file: Path | None,
        output_dir: Path | None,
        stack_config_file: Path | None = None,
        detection_mode: str = "autocrop",
        shot_id: str | None = None,
        data_root: Path | None = None,
        force_redetect: bool = False,
    ) -> dict:
        resolved = self._resolve_inputs(
            input_files=input_files,
            config_file=config_file,
            output_dir=output_dir,
            stack_config_file=stack_config_file,
            detection_mode=detection_mode,
            shot_id=shot_id,
            data_root=data_root,
        )
        session_file = self._session_file(resolved["output_dir"])
        if session_file.exists():
            session = self._restore_session_file(session_file)
            session.update(
                {
                    "shot_id": resolved["shot_id"],
                    "data_root": str(resolved["data_root"]) if resolved["data_root"] is not None else None,
                    "input_files": [str(path) for path in resolved["input_files"]],
                    "config_file": str(resolved["config_file"]),
                    "config_source": resolved["config_source"],
                    "output_dir": str(resolved["output_dir"]),
                    "stack_config_file": str(resolved["stack_config_file"]) if resolved["stack_config_file"] is not None else None,
                    "identity_key": resolved["identity_key"],
                }
            )
            self._rebind_session_input_files(session, resolved["input_files"])
            self._sessions[session["session_id"]] = session
            if force_redetect:
                self.redetect_session(session["session_id"], detection_mode=resolved["detection_mode"])
                return self.get_session(session["session_id"])
            session["last_load_source"] = "restored"
            self._autosave_session(session)
            return session

        session = self._new_session_record(resolved)
        self._sessions[session["session_id"]] = session
        self._autosave_session(session)
        return session

    def redetect_session(self, session_id: str, detection_mode: str | None = None) -> dict:
        session = self.get_session(session_id)
        version = self._build_version(
            input_files=[Path(path) for path in session["input_files"]],
            config_file=Path(session["config_file"]),
            stack_config_file=Path(session["stack_config_file"]) if session["stack_config_file"] else None,
            detection_mode=detection_mode or self._active_version(session)["detection_mode"],
            session_source="new_detection",
        )
        session["versions"].append(version)
        session["active_version_id"] = version["version_id"]
        session["last_load_source"] = "new_detection"
        self._autosave_session(session)
        return session

    def activate_version(self, session_id: str, version_id: str) -> dict:
        session = self.get_session(session_id)
        self._get_version(session, version_id)
        session["active_version_id"] = version_id
        session["last_load_source"] = "restored"
        self._autosave_session(session)
        return self.serialize_session(session)

    def get_session(self, session_id: str) -> dict:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown GUI session: {session_id}") from exc

    def serialize_session(self, session: dict) -> dict:
        version = self._active_version(session)
        return {
            "session_id": session["session_id"],
            "shot_id": session["shot_id"],
            "data_root": session["data_root"],
            "input_files": session["input_files"],
            "config_file": session["config_file"],
            "config_source": session.get("config_source", "explicit"),
            "output_dir": session["output_dir"],
            "stack_config_file": session["stack_config_file"],
            "version_id": version["version_id"],
            "active_version_id": session["active_version_id"],
            "available_versions": [self._serialize_version_summary(item, index + 1) for index, item in enumerate(session["versions"])],
            "detection_mode": version["detection_mode"],
            "qc_flags": version["qc_flags"],
            "revision": version["revision"],
            "last_modified_patch_id": version["last_modified_patch_id"],
            "session_source": session.get("last_load_source", version.get("session_source", "new_detection")),
            "autosaved_at": version.get("autosaved_at"),
            "scans": version["scans"],
            "patches": self._ordered_patches(version),
        }

    def update_patch_geometry(self, session: dict, patch_id: str, rotated_rect: dict) -> dict:
        version = self._active_version(session)
        patch = self._find_patch(version, patch_id)
        patch["rotated_rect"] = {
            "cx": float(rotated_rect["cx"]),
            "cy": float(rotated_rect["cy"]),
            "width": max(1.0, float(rotated_rect["width"])),
            "height": max(1.0, float(rotated_rect["height"])),
            "angle_deg": float(rotated_rect["angle_deg"]),
        }
        patch["source_quad"] = _rotated_rect_corners(patch["rotated_rect"])
        patch["angle_deg"] = float(patch["rotated_rect"]["angle_deg"])
        patch["angle_confidence"] = 1.0
        patch["angle_source"] = "manual"
        patch["status_flags"] = [flag for flag in patch["status_flags"] if flag != "low_confidence_angle"]
        self._mark_modified(session, version, patch_id)
        return patch

    def update_patch_edge(self, session: dict, patch_id: str, edge_points: list[list[float]]) -> dict:
        version = self._active_version(session)
        patch = self._find_patch(version, patch_id)
        if len(edge_points) != 2:
            raise ValueError("edge_points must contain exactly two points")
        (x1, y1), (x2, y2) = edge_points
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        patch["edge_points"] = [[float(x1), float(y1)], [float(x2), float(y2)]]
        patch["angle_deg"] = float(angle)
        patch["rotated_rect"]["angle_deg"] = float(angle)
        patch["source_quad"] = _rotated_rect_corners(patch["rotated_rect"])
        patch["angle_confidence"] = 1.0
        patch["angle_source"] = "manual"
        patch["status_flags"] = [flag for flag in patch["status_flags"] if flag != "low_confidence_angle"]
        self._mark_modified(session, version, patch_id)
        return patch

    def update_patch_angle(self, session: dict, patch_id: str, angle_deg: float) -> dict:
        version = self._active_version(session)
        patch = self._find_patch(version, patch_id)
        patch["angle_deg"] = float(angle_deg)
        patch["rotated_rect"]["angle_deg"] = float(angle_deg)
        patch["source_quad"] = _rotated_rect_corners(patch["rotated_rect"])
        patch["angle_confidence"] = 1.0
        patch["angle_source"] = "manual"
        patch["status_flags"] = [flag for flag in patch["status_flags"] if flag != "low_confidence_angle"]
        self._mark_modified(session, version, patch_id)
        return patch

    def update_patch_crop(self, session: dict, patch_id: str, crop_bbox: list[int] | None) -> dict:
        version = self._active_version(session)
        patch = self._find_patch(version, patch_id)
        patch["crop_bbox"] = None if crop_bbox is None else [int(v) for v in crop_bbox]
        self._mark_modified(session, version, patch_id)
        return patch

    def reorder_patches(self, session: dict, patch_ids: list[str]) -> dict:
        version = self._active_version(session)
        known = {patch["patch_id"] for patch in version["patches"]}
        if set(patch_ids) != known:
            raise ValueError("patch_ids must contain every patch exactly once")
        version["patch_order"] = list(patch_ids)
        self._sync_stack_assignments(version)
        self._mark_modified(session, version, None)
        return self.serialize_session(session)

    def update_patch_assignment(
        self,
        session: dict,
        patch_id: str,
        assignment_status: str,
        assigned_order: int | None = None,
    ) -> dict:
        if assignment_status not in {"assigned", "ignored_duplicate", "unassigned"}:
            raise ValueError("assignment_status must be assigned, ignored_duplicate, or unassigned")
        version = self._active_version(session)
        patch = self._find_patch(version, patch_id)
        if assignment_status == "assigned":
            if assigned_order is None:
                raise ValueError("assigned_order is required when assignment_status is assigned")
            self._assign_patch_to_order(version, patch, int(assigned_order))
        else:
            patch["assignment_status"] = assignment_status
            patch["assigned_order"] = None
            self._sync_stack_assignments(version)
        self._mark_modified(session, version, patch_id)
        return patch

    def export_session(self, session: dict) -> dict:
        output_dir = Path(session["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        session_payload = self._session_snapshot(session)
        current_payload = self.serialize_session(session)
        review_payload = {
            "session_id": session["session_id"],
            "version_id": current_payload["version_id"],
            "shot_id": session["shot_id"],
            "revision": current_payload["revision"],
            "assigned_patch_count": sum(1 for patch in current_payload["patches"] if patch["assignment_status"] == "assigned"),
            "processable_patch_count": sum(1 for patch in current_payload["patches"] if patch["process_in_current_batch"]),
            "ignored_patch_count": sum(1 for patch in current_payload["patches"] if patch["assignment_status"] == "ignored_duplicate"),
            "patches": current_payload["patches"],
        }
        session_file = output_dir / "gui_session.json"
        review_file = output_dir / "gui_review.json"
        save_json(session_file, session_payload)
        save_json(review_file, review_payload)
        return {"session_file": str(session_file), "review_file": str(review_file)}

    def _resolve_inputs(
        self,
        input_files: list[Path] | None,
        config_file: Path | None,
        output_dir: Path | None,
        stack_config_file: Path | None,
        detection_mode: str,
        shot_id: str | None,
        data_root: Path | None,
    ) -> dict:
        resolved_shot_id = shot_id
        resolved_data_root = data_root
        resolved_input_files = list(input_files or [])
        resolved_config_file = config_file
        resolved_output_dir = output_dir
        resolved_stack_config_file = stack_config_file

        if shot_id:
            if data_root is None:
                raise ValueError("shot_id mode requires data_root")
            resolved = resolve_shot_inputs(shot_id, data_root)
            resolved_shot_id = resolved["shot_id"]
            resolved_data_root = resolved["data_root"]
            resolved_input_files = resolved["input_files"]
            resolved_stack_config_file = resolved_stack_config_file or resolved["stack_config_file"]
            if resolved_config_file is None:
                resolved_config_file = resolved["config_file"]
            resolved_output_dir = resolved_output_dir or resolved["output_dir"]
            resolved_config_source = "explicit" if config_file is not None else resolved.get("config_source", "example")
        else:
            resolved_config_source = "explicit" if resolved_config_file is not None else None

        if not resolved_input_files:
            raise ValueError("input_files are required when shot_id is not provided")
        if resolved_config_file is None:
            raise ValueError("config_file is required")
        if resolved_output_dir is None:
            raise ValueError("output_dir is required")
        if detection_mode not in {"autocrop", "segment"}:
            raise ValueError("detection_mode must be autocrop or segment")

        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        if resolved_shot_id:
            identity_key = f"shot:{resolved_shot_id}|output:{resolved_output_dir}"
        else:
            identity_key = f"manual:{resolved_output_dir}"

        return {
            "shot_id": resolved_shot_id,
            "data_root": resolved_data_root,
            "input_files": resolved_input_files,
            "config_file": resolved_config_file,
            "output_dir": resolved_output_dir,
            "stack_config_file": resolved_stack_config_file,
            "detection_mode": detection_mode,
            "identity_key": identity_key,
            "config_source": resolved_config_source or "explicit",
        }

    def _new_session_record(self, resolved: dict) -> dict:
        session_id = uuid.uuid4().hex[:12]
        version = self._build_version(
            input_files=resolved["input_files"],
            config_file=resolved["config_file"],
            stack_config_file=resolved["stack_config_file"],
            detection_mode=resolved["detection_mode"],
            session_source="new_detection",
        )
        return {
            "session_id": session_id,
            "identity_key": resolved["identity_key"],
            "shot_id": resolved["shot_id"],
            "data_root": str(resolved["data_root"]) if resolved["data_root"] is not None else None,
            "input_files": [str(path) for path in resolved["input_files"]],
            "config_file": str(resolved["config_file"]),
            "config_source": resolved["config_source"],
            "output_dir": str(resolved["output_dir"]),
            "stack_config_file": str(resolved["stack_config_file"]) if resolved["stack_config_file"] is not None else None,
            "active_version_id": version["version_id"],
            "last_load_source": "new_detection",
            "versions": [version],
        }

    def _rebind_session_input_files(self, session: dict, input_files: list[Path]) -> None:
        scan_file_by_index = {index: str(path) for index, path in enumerate(input_files, start=1)}
        for version in session["versions"]:
            for scan in version.get("scans", []):
                scan_index = int(scan["scan_index"])
                rebound = scan_file_by_index.get(scan_index)
                if rebound is not None:
                    scan["scan_file"] = rebound
            for patch in version.get("patches", []):
                scan_index = int(patch["scan_index"])
                rebound = scan_file_by_index.get(scan_index)
                if rebound is not None:
                    patch["scan_file"] = rebound

    def _build_version(
        self,
        input_files: list[Path],
        config_file: Path,
        stack_config_file: Path | None,
        detection_mode: str,
        session_source: str,
    ) -> dict:
        config = load_config(config_file)
        stack_entries = load_stack_entries(stack_config_file) if stack_config_file is not None else []

        scans: list[dict] = []
        patches: list[dict] = []
        patch_order: list[str] = []
        for scan_index, input_file in enumerate(input_files, start=1):
            if detection_mode == "segment":
                raw_detection = detect_patches_path(
                    input_file,
                    {"min_area": 1000, "padding": 4, "sort_mode": "yx"},
                )
                detection = {
                    "patches": [
                        {
                            "order": int(patch["order"]),
                            "rotated_rect": _rotated_rect_from_bbox(
                                [int(value) for value in patch["bbox"]],
                                float(patch.get("angle_deg", 0.0)),
                            ),
                            "source_quad": _rotated_rect_corners(
                                _rotated_rect_from_bbox(
                                    [int(value) for value in patch["bbox"]],
                                    float(patch.get("angle_deg", 0.0)),
                                )
                            ),
                            "angle_confidence": float(patch.get("angle_confidence", 0.0)),
                            "angle_source": patch.get("angle_source", "segment"),
                            "detection_source": "segment",
                            "status_flags": list(patch.get("status_flags", [])),
                        }
                        for patch in raw_detection["patches"]
                    ]
                }
                with Image.open(input_file) as image_handle:
                    image_width, image_height = image_handle.size
            else:
                image = load_rgb_image(input_file)
                detection = self._detect_scan_patches(image=image, detection_mode=detection_mode, config=config)
                image_width = int(image.shape[1])
                image_height = int(image.shape[0])
            scan_patch_ids = []
            for patch in detection["patches"]:
                patch_id = f"scan{scan_index:02d}_patch{patch['order']:02d}"
                rotated_rect = patch["rotated_rect"]
                patch_record = {
                    "patch_id": patch_id,
                    "scan_index": scan_index,
                    "scan_file": str(input_file),
                    "rotated_rect": rotated_rect,
                    "source_quad": _normalize_quad(patch.get("source_quad")),
                    "angle_deg": float(rotated_rect["angle_deg"]),
                    "angle_confidence": float(patch.get("angle_confidence", 0.0)),
                    "angle_source": patch.get("angle_source", "manual"),
                    "detection_source": patch.get("detection_source", detection_mode),
                    "status_flags": list(patch.get("status_flags", [])),
                    "edge_points": [],
                    "crop_bbox": None,
                    "local_order": int(patch["order"]),
                    "global_order": None,
                    "assignment_status": "unassigned",
                    "assigned_order": None,
                    "process_in_current_batch": False,
                    "stack": None,
                    "stack_mapping": None,
                    "modified_revision": 0,
                }
                patches.append(patch_record)
                patch_order.append(patch_id)
                scan_patch_ids.append(patch_id)
            scans.append(
                {
                    "scan_index": scan_index,
                    "scan_file": str(input_file),
                    "width": image_width,
                    "height": image_height,
                    "patch_ids": scan_patch_ids,
                }
            )

        version = {
            "version_id": _version_id(),
            "created_at": _utc_now_iso(),
            "detection_mode": detection_mode,
            "qc_flags": [],
            "stack_entries": stack_entries,
            "scans": scans,
            "patches": patches,
            "patch_order": patch_order,
            "revision": 0,
            "last_modified_patch_id": None,
            "session_source": session_source,
            "autosaved_at": None,
        }
        self._sync_stack_assignments(version)
        return version

    def _session_file(self, output_dir: Path) -> Path:
        return output_dir / "gui_session.json"

    def _session_snapshot(self, session: dict) -> dict:
        active = self.serialize_session(session)
        return {
            "session_id": session["session_id"],
            "identity_key": session.get("identity_key"),
            "shot_id": session.get("shot_id"),
            "data_root": session.get("data_root"),
            "input_files": session.get("input_files", []),
            "config_file": session["config_file"],
            "config_source": session.get("config_source", "explicit"),
            "output_dir": session["output_dir"],
            "stack_config_file": session.get("stack_config_file"),
            "active_version_id": session["active_version_id"],
            "last_load_source": session.get("last_load_source", "new_detection"),
            "version_id": active["version_id"],
            "detection_mode": active["detection_mode"],
            "qc_flags": active["qc_flags"],
            "revision": active["revision"],
            "last_modified_patch_id": active["last_modified_patch_id"],
            "session_source": active["session_source"],
            "autosaved_at": active["autosaved_at"],
            "scans": active["scans"],
            "patches": active["patches"],
            "patch_order": list(self._active_version(session)["patch_order"]),
            "stack_entries": self._active_version(session)["stack_entries"],
            "versions": [self._version_snapshot(version) for version in session["versions"]],
        }

    def _version_snapshot(self, version: dict) -> dict:
        return {
            "version_id": version["version_id"],
            "created_at": version["created_at"],
            "detection_mode": version["detection_mode"],
            "qc_flags": version["qc_flags"],
            "stack_entries": version["stack_entries"],
            "scans": version["scans"],
            "patches": version["patches"],
            "patch_order": list(version["patch_order"]),
            "revision": version["revision"],
            "last_modified_patch_id": version["last_modified_patch_id"],
            "session_source": version.get("session_source", "new_detection"),
            "autosaved_at": version.get("autosaved_at"),
        }

    def _autosave_session(self, session: dict) -> None:
        active_version = self._active_version(session)
        active_version["autosaved_at"] = _utc_now_iso()
        save_json(self._session_file(Path(session["output_dir"])), self._session_snapshot(session))

    def _restore_session_file(self, session_file: Path) -> dict:
        payload = json.loads(session_file.read_text(encoding="utf-8"))
        if "versions" not in payload:
            legacy_version = {
                "version_id": payload.get("version_id") or _version_id(),
                "created_at": payload.get("autosaved_at") or _utc_now_iso(),
                "detection_mode": payload.get("detection_mode", "autocrop"),
                "qc_flags": payload.get("qc_flags", []),
                "stack_entries": payload.get("stack_entries", []),
                "scans": payload.get("scans", []),
                "patches": self._restore_patches(payload.get("patches", [])),
                "patch_order": payload.get("patch_order") or [patch["patch_id"] for patch in payload.get("patches", [])],
                "revision": int(payload.get("revision", 0)),
                "last_modified_patch_id": payload.get("last_modified_patch_id"),
                "session_source": payload.get("session_source", "new_detection"),
                "autosaved_at": payload.get("autosaved_at"),
            }
            versions = [legacy_version]
            active_version_id = legacy_version["version_id"]
        else:
            versions = []
            for item in payload["versions"]:
                version = dict(item)
                version["patches"] = self._restore_patches(item.get("patches", []))
                version["patch_order"] = item.get("patch_order") or [patch["patch_id"] for patch in version["patches"]]
                versions.append(version)
            active_version_id = payload.get("active_version_id") or versions[-1]["version_id"]

        session = {
            "session_id": payload["session_id"],
            "identity_key": payload.get("identity_key"),
            "shot_id": payload.get("shot_id"),
            "data_root": payload.get("data_root"),
            "input_files": payload.get("input_files", []),
            "config_file": payload["config_file"],
            "config_source": payload.get("config_source", "explicit"),
            "output_dir": payload["output_dir"],
            "stack_config_file": payload.get("stack_config_file"),
            "active_version_id": active_version_id,
            "last_load_source": payload.get("last_load_source", "restored"),
            "versions": versions,
        }
        for version in session["versions"]:
            self._sync_stack_assignments(version)
        return session

    def _restore_patches(self, patches: list[dict]) -> list[dict]:
        restored = []
        for saved_patch in patches:
            patch = dict(saved_patch)
            patch.pop("corners", None)
            patch.pop("display_bbox", None)
            restored.append(patch)
        return restored

    def _active_version(self, session: dict) -> dict:
        return self._get_version(session, session["active_version_id"])

    def _get_version(self, session: dict, version_id: str) -> dict:
        for version in session["versions"]:
            if version["version_id"] == version_id:
                return version
        raise KeyError(f"Unknown GUI version: {version_id}")

    def _serialize_version_summary(self, version: dict, number: int) -> dict:
        return {
            "version_id": version["version_id"],
            "version_number": number,
            "created_at": version["created_at"],
            "revision": version["revision"],
            "session_source": version.get("session_source", "new_detection"),
            "detection_mode": version["detection_mode"],
            "patch_count": len(version["patches"]),
        }

    def _detect_scan_patches(self, image, detection_mode: str, config: dict) -> dict:
        if detection_mode == "segment":
            detection = detect_patches(image, {"min_area": 1000, "padding": 4, "sort_mode": "yx"})
            patches = []
            for patch in detection["patches"]:
                bbox = [int(v) for v in patch["bbox"]]
                angle_deg = float(patch.get("angle_deg", 0.0))
                patches.append(
                    {
                        "order": int(patch["order"]),
                        "rotated_rect": _rotated_rect_from_bbox(bbox, angle_deg),
                        "source_quad": _rotated_rect_corners(_rotated_rect_from_bbox(bbox, angle_deg)),
                        "angle_confidence": float(patch.get("angle_confidence", 0.0)),
                        "angle_source": patch.get("angle_source", "segment"),
                        "detection_source": "segment",
                        "status_flags": list(patch.get("status_flags", [])),
                    }
                )
            return {"patches": patches}

        candidates, _, _ = detect_rcf_rectangles(
            image_bgr=image[:, :, ::-1],
            expected_count=None,
            min_area_ratio=0.01,
            max_area_ratio=0.50,
            min_side_px=70,
            max_aspect_ratio=1.8,
            blur_ksize=5,
            morph_ksize=9,
            manual_threshold=None,
        )
        patches = []
        for order, candidate in enumerate(candidates, start=1):
            bbox = [int(v) for v in candidate["bbox"]]
            angle_deg = float(candidate.get("rotation_angle_deg", 0.0))
            patches.append(
                {
                    "order": order,
                    "rotated_rect": _rotated_rect_from_bbox(bbox, angle_deg),
                    "source_quad": _normalize_quad(candidate.get("box")),
                    "angle_confidence": float(max(0.0, min(1.0, candidate.get("score", 0.0)))),
                    "angle_source": "autocrop_min_area_rect",
                    "detection_source": "autocrop",
                    "status_flags": [],
                }
            )
        if not patches:
            raise ValueError("No RCF patches were detected in the input scan")
        return {"patches": patches}

    def _ordered_patches(self, version: dict) -> list[dict]:
        patch_map = {patch["patch_id"]: patch for patch in version["patches"]}
        ordered = []
        for patch_id in version["patch_order"]:
            patch = dict(patch_map[patch_id])
            patch["corners"] = _rotated_rect_corners(patch["rotated_rect"])
            patch["display_bbox"] = _rotated_rect_aabb(patch["rotated_rect"])
            patch["display_label"] = _display_label(patch)
            ordered.append(patch)
        return ordered

    def _find_patch(self, version: dict, patch_id: str) -> dict:
        for patch in version["patches"]:
            if patch["patch_id"] == patch_id:
                return patch
        raise KeyError(f"Unknown patch: {patch_id}")

    def _sync_stack_assignments(self, version: dict) -> None:
        ordered_patches = self._ordered_patches(version)
        stack_entries = version["stack_entries"]
        qc_flags = []
        for patch in version["patches"]:
            patch["global_order"] = patch["assigned_order"]
            patch["process_in_current_batch"] = patch["assignment_status"] == "assigned"
            if patch["assignment_status"] == "assigned" and patch["assigned_order"] is not None:
                stack_mapping = stack_entries[patch["assigned_order"] - 1] if patch["assigned_order"] - 1 < len(stack_entries) else None
                patch["stack"] = stack_mapping
                patch["stack_mapping"] = stack_mapping
            else:
                patch["stack"] = None
                patch["stack_mapping"] = None
        assigned_count = sum(1 for patch in ordered_patches if patch["assignment_status"] == "assigned")
        if stack_entries and assigned_count > len(stack_entries):
            qc_flags.append("assigned_patch_count_exceeds_stack")
        version["qc_flags"] = qc_flags

    def _assign_patch_to_order(self, version: dict, patch: dict, assigned_order: int) -> None:
        assigned_order = max(1, int(assigned_order))
        displaced_patch = self._patch_with_assigned_order(version, assigned_order, exclude_patch_id=patch["patch_id"])
        if displaced_patch is not None:
            displaced_patch["assignment_status"] = "unassigned"
            displaced_patch["assigned_order"] = None
        patch["assignment_status"] = "assigned"
        patch["assigned_order"] = assigned_order
        self._sync_stack_assignments(version)

    def _patch_with_assigned_order(self, version: dict, assigned_order: int, exclude_patch_id: str | None = None) -> dict | None:
        for patch in version["patches"]:
            if patch["patch_id"] == exclude_patch_id:
                continue
            if patch["assignment_status"] == "assigned" and patch["assigned_order"] == assigned_order:
                return patch
        return None

    def _mark_modified(self, session: dict, version: dict, patch_id: str | None) -> None:
        version["revision"] += 1
        version["last_modified_patch_id"] = patch_id
        if patch_id is not None:
            patch = self._find_patch(version, patch_id)
            patch["modified_revision"] = version["revision"]
        self._autosave_session(session)
