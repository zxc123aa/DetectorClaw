from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import threading

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
import numpy as np
from PIL import Image
from pydantic import BaseModel

from .calibration import dose_from_patch
from .calibration import load_background_mean
from .config import load_config
from .config import resolve_background_paths
from .config import resolve_film_type
from .gui_state import SessionStore
from . import preview
from .runtime_cache import LRUCache
from .runtime_cache import file_signature

PREVIEW_CACHE: dict[tuple, tuple[bytes, str]] = {}
DOSE_STATS_CACHE: dict[tuple, dict] = {}
DOSE_ARRAY_CACHE = LRUCache[tuple[np.ndarray, dict]](64)
DOSE_PREVIEW_ARRAY_CACHE = LRUCache[tuple[np.ndarray, dict]](128)
DOSE_CONFIG_CACHE = LRUCache[dict](8)
DOSE_OVERVIEW_PREVIEW_DIM = 260
DOSE_SINGLE_PREVIEW_DIM = 320
DOSE_HIGH_RES_PREVIEW_DIM = 960
DOSE_OVERVIEW_PREFETCH_COUNT = 12
DOSE_PSEUDOCOLOR_PALETTES = ("turbo", "jet")
DOSE_NATIVE_BATCH_SIZE = 8
DOSE_PREVIEW_JPEG_QUALITY = 78
DOSE_ASSET_VARIANTS = (
    {"variant_id": "dose_single_320_turbo", "palette": "turbo", "max_dim": DOSE_SINGLE_PREVIEW_DIM, "format": "jpeg", "quality": DOSE_PREVIEW_JPEG_QUALITY},
    {"variant_id": "dose_single_320_jet", "palette": "jet", "max_dim": DOSE_SINGLE_PREVIEW_DIM, "format": "jpeg", "quality": DOSE_PREVIEW_JPEG_QUALITY},
    {"variant_id": "dose_single_960_turbo", "palette": "turbo", "max_dim": DOSE_HIGH_RES_PREVIEW_DIM, "format": "jpeg", "quality": DOSE_PREVIEW_JPEG_QUALITY},
    {"variant_id": "dose_single_960_jet", "palette": "jet", "max_dim": DOSE_HIGH_RES_PREVIEW_DIM, "format": "jpeg", "quality": DOSE_PREVIEW_JPEG_QUALITY},
    {"variant_id": "dose_overview_260_turbo", "palette": "turbo", "max_dim": DOSE_OVERVIEW_PREVIEW_DIM, "format": "jpeg", "quality": DOSE_PREVIEW_JPEG_QUALITY},
    {"variant_id": "dose_overview_260_jet", "palette": "jet", "max_dim": DOSE_OVERVIEW_PREVIEW_DIM, "format": "jpeg", "quality": DOSE_PREVIEW_JPEG_QUALITY},
)


class LoadSessionRequest(BaseModel):
    shot_id: str | None = None
    data_root: str | None = None
    input_files: list[str] | None = None
    config_file: str | None = None
    output_dir: str | None = None
    stack_config_file: str | None = None
    detection_mode: str = "autocrop"
    force_redetect: bool = False


class GeometryRequest(BaseModel):
    rotated_rect: dict


class EdgeRequest(BaseModel):
    edge_points: list[list[float]]


class AngleRequest(BaseModel):
    angle_deg: float


class CropRequest(BaseModel):
    crop_bbox: list[int] | None


class OrderRequest(BaseModel):
    patch_ids: list[str]


class AssignmentRequest(BaseModel):
    assignment_status: str
    assigned_order: int | None = None


def _normalize_dose_palette(value: str | None) -> str:
    palette = (value or "gray").lower()
    if palette in {"gray", "grey"}:
        return "gray"
    if palette in {"turbo", "pseudocolor", "pseudo"}:
        return "turbo"
    if palette == "jet":
        return "jet"
    raise ValueError("dose palette must be gray, turbo, or jet")


def _normalize_dose_export_format(value: str | None) -> str:
    normalized = (value or "tiff").lower()
    if normalized in {"tif", "tiff"}:
        return "tiff"
    if normalized == "png":
        return "png"
    raise ValueError("dose export format must be tiff or png")


def _resolve_dose_config_source(session: dict) -> str:
    source = session.get("config_source")
    if source:
        return str(source)
    config_name = Path(session["config_file"]).name.lower()
    if ".example." in config_name:
        return "example"
    return "explicit"


def _load_cached_dose_config(config_file: Path) -> dict:
    cache_key = file_signature(config_file)
    cached = DOSE_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    return DOSE_CONFIG_CACHE.set(cache_key, load_config(config_file))


def _load_dose_context(session: dict, patch: dict | None = None) -> dict:
    config = _load_cached_dose_config(Path(session["config_file"]))
    stack_mapping = None if patch is None else patch.get("stack") or patch.get("stack_mapping")
    film_type = resolve_film_type(config, stack_mapping)
    film_models = config["calibration"]["film_models"]
    if film_type not in film_models:
        raise ValueError(f"No calibration model configured for film_type {film_type}")
    film_model = film_models[film_type]
    background_quantile = float(config["calibration"]["background_quantile"])
    film_background_path, scanner_background_path = resolve_background_paths(config, film_type)
    if not film_background_path.exists():
        raise FileNotFoundError(f"Background file not found: {film_background_path}")
    if not scanner_background_path.exists():
        raise FileNotFoundError(f"Background file not found: {scanner_background_path}")
    film_background_mean = load_background_mean(film_background_path)
    scanner_background_mean = load_background_mean(scanner_background_path)
    return {
        "film_type": film_type,
        "film_model": film_model,
        "backend": str(config["calibration"].get("backend", "auto")),
        "background_quantile": background_quantile,
        "film_background_mean": film_background_mean,
        "scanner_background_mean": scanner_background_mean,
    }


def _dose_asset_id(
    patch_id: str,
    modified_revision: int,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> str:
    payload = {
        "patch_id": patch_id,
        "modified_revision": int(modified_revision),
        "palette": str(palette),
        "max_dim": int(max_dim),
        "format": str(preview_format),
        "quality": int(quality),
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    return f"dose_{encoded.rstrip('=')}"


def _parse_dose_asset_id(asset_id: str) -> dict:
    if not asset_id.startswith("dose_"):
        raise ValueError("unsupported asset id")
    encoded = asset_id[len("dose_") :]
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("invalid asset id") from exc
    required = {"patch_id", "modified_revision", "palette", "max_dim", "format", "quality"}
    if not isinstance(payload, dict) or not required.issubset(payload.keys()):
        raise ValueError("invalid asset id payload")
    return payload


def _dose_status(session: dict) -> tuple[bool, str | None]:
    try:
        _load_dose_context(session)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


def _dose_metadata(session: dict) -> dict:
    available, error = _dose_status(session)
    return {
        "dose_available": available,
        "dose_error": error,
        "dose_config_source": _resolve_dose_config_source(session),
    }


def _dose_array(session: dict, patch: dict) -> tuple[np.ndarray, dict]:
    stack = patch.get("stack") or {}
    dose_context = _load_dose_context(session, patch=patch)
    cache_key = _dose_array_cache_key(session, patch, dose_context)
    cached = DOSE_ARRAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    quad_points = patch.get("source_quad") or patch["corners"]
    patch_image = preview.extract_patch_image(
        scan_file=Path(patch["scan_file"]),
        quad_points=quad_points,
        crop_bbox=patch.get("crop_bbox"),
    )
    patch_rgb = np.asarray(patch_image.convert("RGB"), dtype=np.uint8)
    dose, _ = dose_from_patch(
        patch_rgb=patch_rgb,
        film_background_mean=dose_context["film_background_mean"],
        scanner_background_mean=dose_context["scanner_background_mean"],
        film_model=dose_context["film_model"],
        background_quantile=dose_context["background_quantile"],
        backend=dose_context.get("backend", "auto"),
    )
    return DOSE_ARRAY_CACHE.set(cache_key, (dose, dose_context))


def _dose_preview_array(session: dict, patch: dict, max_dim: int) -> tuple[np.ndarray, dict]:
    dose_context = _load_dose_context(session, patch=patch)
    cache_key = _dose_preview_array_cache_key(session, patch, dose_context, max_dim)
    cached = DOSE_PREVIEW_ARRAY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    quad_points = patch.get("source_quad") or patch["corners"]
    patch_image = preview.extract_patch_image(
        scan_file=Path(patch["scan_file"]),
        quad_points=quad_points,
        crop_bbox=patch.get("crop_bbox"),
        backend="auto",
        max_dim=max_dim,
    )
    patch_rgb = np.asarray(patch_image.convert("RGB"), dtype=np.uint8)
    dose, _ = dose_from_patch(
        patch_rgb=patch_rgb,
        film_background_mean=dose_context["film_background_mean"],
        scanner_background_mean=dose_context["scanner_background_mean"],
        film_model=dose_context["film_model"],
        background_quantile=dose_context["background_quantile"],
        backend=dose_context.get("backend", "auto"),
    )
    return DOSE_PREVIEW_ARRAY_CACHE.set(cache_key, (dose, dose_context))


def _pick_cached_dose_preview_array(
    session: dict,
    patch: dict,
    dose_context: dict,
) -> tuple[np.ndarray, dict] | None:
    stack = patch.get("stack") or patch.get("stack_mapping") or {}
    best_entry: tuple[np.ndarray, dict] | None = None
    best_max_dim = -1
    for key, value in DOSE_PREVIEW_ARRAY_CACHE.items():
        if not isinstance(key, tuple) or len(key) < 12:
            continue
        if key[0] != "dose-preview-array":
            continue
        if key[1] != session.get("version_id"):
            continue
        if key[2] != session.get("config_file"):
            continue
        if key[3] != patch["patch_id"]:
            continue
        if key[4] != dose_context.get("backend", "auto"):
            continue
        if key[6] != patch.get("modified_revision", 0):
            continue
        if key[7] != tuple(patch.get("crop_bbox") or []):
            continue
        if key[8] != patch.get("assignment_status"):
            continue
        if key[9] != patch.get("assigned_order"):
            continue
        if key[10] != stack.get("material_name"):
            continue
        if key[11] != stack.get("material_index"):
            continue
        candidate_max_dim = int(key[5])
        if candidate_max_dim > best_max_dim:
            best_entry = value
            best_max_dim = candidate_max_dim
    return best_entry


def _dose_array_cache_key(session: dict, patch: dict, dose_context: dict) -> tuple:
    stack = patch.get("stack") or {}
    return (
        session.get("version_id"),
        session.get("config_file"),
        patch["patch_id"],
        dose_context.get("backend", "auto"),
        patch.get("modified_revision", 0),
        tuple(patch.get("crop_bbox") or []),
        patch.get("assignment_status"),
        patch.get("assigned_order"),
        stack.get("material_name"),
        stack.get("material_index"),
    )


def _dose_stats_cache_key(state: dict, patch: dict) -> tuple:
    crop_bbox = patch.get("crop_bbox")
    return (
        "dose-stats",
        state["version_id"],
        patch["patch_id"],
        patch.get("modified_revision", 0),
        tuple(crop_bbox or []),
    )


def _dose_preview_array_cache_key(session: dict, patch: dict, dose_context: dict, max_dim: int) -> tuple:
    stack = patch.get("stack") or {}
    return (
        "dose-preview-array",
        session.get("version_id"),
        session.get("config_file"),
        patch["patch_id"],
        dose_context.get("backend", "auto"),
        int(max_dim),
        patch.get("modified_revision", 0),
        tuple(patch.get("crop_bbox") or []),
        patch.get("assignment_status"),
        patch.get("assigned_order"),
        stack.get("material_name"),
        stack.get("material_index"),
    )


def _dose_preview_cache_key(
    state: dict,
    patch: dict,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> tuple:
    crop_bbox = patch.get("crop_bbox")
    return (
        "dose-patch",
        state["version_id"],
        patch["patch_id"],
        patch.get("modified_revision", 0),
        tuple(crop_bbox or []),
        palette,
        max_dim,
        preview_format,
        quality,
    )


def _dose_disk_cache_paths(
    state: dict,
    patch: dict,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> tuple[Path, Path] | None:
    output_dir = state.get("output_dir")
    if not output_dir:
        return None
    payload = {
        "version_id": state.get("version_id"),
        "patch_id": patch.get("patch_id"),
        "modified_revision": patch.get("modified_revision", 0),
        "crop_bbox": list(patch.get("crop_bbox") or []),
        "palette": str(palette),
        "max_dim": int(max_dim),
        "format": str(preview_format),
        "quality": int(quality),
    }
    digest = hashlib.blake2b(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    root = Path(output_dir) / ".detectorclaw" / "dose_assets" / str(state.get("version_id"))
    return root / f"{digest}.bin", root / f"{digest}.meta.json"


def _dose_disk_cache_available(
    state: dict,
    patch: dict,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> bool:
    paths = _dose_disk_cache_paths(state, patch, palette, max_dim, preview_format, quality)
    if paths is None:
        return False
    data_path, meta_path = paths
    return data_path.exists() and meta_path.exists()


def _dose_disk_cache_load(
    state: dict,
    patch: dict,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> tuple[bytes, str] | None:
    paths = _dose_disk_cache_paths(state, patch, palette, max_dim, preview_format, quality)
    if paths is None:
        return None
    data_path, meta_path = paths
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        media_type = str(meta.get("media_type") or "")
        if not media_type.startswith("image/"):
            return None
        content = data_path.read_bytes()
    except Exception:  # noqa: BLE001
        return None
    return content, media_type


def _dose_disk_cache_store(
    state: dict,
    patch: dict,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
    content: bytes,
    media_type: str,
) -> None:
    paths = _dose_disk_cache_paths(state, patch, palette, max_dim, preview_format, quality)
    if paths is None:
        return
    data_path, meta_path = paths
    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_tmp = data_path.with_suffix(data_path.suffix + ".tmp")
        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        data_tmp.write_bytes(content)
        meta_tmp.write_text(
            json.dumps({"media_type": media_type, "size": len(content)}, ensure_ascii=False),
            encoding="utf-8",
        )
        data_tmp.replace(data_path)
        meta_tmp.replace(meta_path)
    except Exception:  # noqa: BLE001
        return


def _dose_stats_payload(session: dict, patch: dict) -> dict:
    dose_context = _load_dose_context(session, patch=patch)
    cached = _pick_cached_dose_preview_array(session, patch, dose_context)
    if cached is None:
        dose, _ = _dose_array(session, patch)
    else:
        dose, _ = cached
    return {
        "patch_id": patch["patch_id"],
        "film_type": dose_context["film_type"],
        "dose_min": round(float(dose.min()), 6),
        "dose_max": round(float(dose.max()), 6),
        "dose_mean": round(float(dose.mean()), 6),
    }


def _dose_preview_image_from_array(dose: np.ndarray, palette: str) -> Image.Image:
    max_value = float(dose.max())
    if max_value <= 0:
        preview_gray = np.zeros_like(dose, dtype=np.uint8)
    else:
        preview_gray = np.clip(dose / max_value * 255.0, 0, 255).astype(np.uint8)
    if palette == "gray":
        return Image.fromarray(preview_gray, mode="L")
    color_map = cv2.COLORMAP_TURBO if palette == "turbo" else cv2.COLORMAP_JET
    preview_rgb = cv2.cvtColor(cv2.applyColorMap(preview_gray, color_map), cv2.COLOR_BGR2RGB)
    return Image.fromarray(preview_rgb, mode="RGB")


def _dose_preview_image(session: dict, patch: dict, palette: str, max_dim: int | None = None) -> Image.Image:
    if max_dim is not None and max_dim > 0:
        dose, _ = _dose_preview_array(session, patch, max_dim)
    else:
        dose, _ = _dose_array(session, patch)
    return _dose_preview_image_from_array(dose, palette)


def _encode_dose_export_image(image: Image.Image, export_format: str) -> tuple[bytes, str]:
    normalized = _normalize_dose_export_format(export_format)
    buffer = io.BytesIO()
    if normalized == "tiff":
        image.save(buffer, format="TIFF", compression="tiff_lzw")
        return buffer.getvalue(), "image/tiff"
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def _chunked(items: list, size: int) -> list[list]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def _prewarm_dose_variants_native(
    session: dict,
    state: dict,
    variants: list[dict],
) -> tuple[int, bool]:
    pending: list[dict] = []
    for variant in variants:
        patch = variant["patch"]
        palette = variant["palette"]
        max_dim = int(variant["max_dim"])
        preview_format = str(variant["preview_format"])
        quality = int(variant["quality"])
        cache_key = _dose_preview_cache_key(state, patch, palette, max_dim, preview_format, quality)
        if PREVIEW_CACHE.get(cache_key) is not None:
            continue
        if _dose_disk_cache_available(state, patch, palette, max_dim, preview_format, quality):
            continue
        pending.append(variant)

    if not pending:
        return 0, False

    warmed = 0
    used_native = False
    for chunk in _chunked(pending, DOSE_NATIVE_BATCH_SIZE):
        tasks: list[dict] = []
        task_map: dict[str, dict] = {}
        for index, variant in enumerate(chunk, start=1):
            patch = variant["patch"]
            dose_context = _load_dose_context(session, patch=patch)
            task_id = (
                f"{patch['patch_id']}|{patch.get('modified_revision', 0)}|"
                f"{variant['palette']}|{int(variant['max_dim'])}|{index}"
            )
            task_map[task_id] = variant | {"dose_context": dose_context}
            tasks.append(
                {
                    "task_id": task_id,
                    "scan_file": str(patch["scan_file"]),
                    "quad_points": patch.get("source_quad") or patch["corners"],
                    "crop_bbox": patch.get("crop_bbox"),
                    "max_dim": int(variant["max_dim"]),
                    "preview_format": str(variant["preview_format"]),
                    "quality": int(variant["quality"]),
                    "palette": str(variant["palette"]),
                    "film_background_mean": float(dose_context["film_background_mean"]),
                    "scanner_background_mean": float(dose_context["scanner_background_mean"]),
                    "film_model": dose_context["film_model"],
                    "background_quantile": float(dose_context["background_quantile"]),
                }
            )
        payload = {"tasks": tasks}
        result = preview.run_native_dose_batch(payload)
        if result is None:
            return warmed, used_native
        used_native = True
        for item in result.get("results", []):
            task_id = str(item.get("task_id") or "")
            variant = task_map.get(task_id)
            if variant is None:
                continue
            patch = variant["patch"]
            palette = str(variant["palette"])
            max_dim = int(variant["max_dim"])
            preview_format = str(variant["preview_format"])
            quality = int(variant["quality"])
            media_type = str(item.get("media_type") or "")
            content_hex = str(item.get("content_hex") or "")
            if not media_type.startswith("image/") or not content_hex:
                continue
            content = bytes.fromhex(content_hex)
            cache_key = _dose_preview_cache_key(state, patch, palette, max_dim, preview_format, quality)
            PREVIEW_CACHE[cache_key] = (content, media_type)
            _dose_disk_cache_store(
                state,
                patch,
                palette,
                max_dim,
                preview_format,
                quality,
                content=content,
                media_type=media_type,
            )
            dose_context = variant["dose_context"]
            DOSE_STATS_CACHE[_dose_stats_cache_key(state, patch)] = {
                "patch_id": patch["patch_id"],
                "film_type": dose_context["film_type"],
                "dose_min": round(float(item.get("dose_min", 0.0)), 6),
                "dose_max": round(float(item.get("dose_max", 0.0)), 6),
                "dose_mean": round(float(item.get("dose_mean", 0.0)), 6),
            }
            warmed += 1
    return warmed, used_native


def _prewarm_dose_overview_cache(
    session: dict,
    state: dict,
    patches: list[dict],
    *,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> int:
    grouped: dict[str, list[dict]] = {}
    for patch in patches:
        grouped.setdefault(str(patch["scan_file"]), []).append(patch)

    selected: list[dict] = []
    for group in grouped.values():
        selected.extend(group[:DOSE_OVERVIEW_PREFETCH_COUNT])

    variants = [
        {
            "patch": patch,
            "palette": palette,
            "max_dim": max_dim,
            "preview_format": preview_format,
            "quality": quality,
        }
        for patch in selected
    ]
    warmed, used_native = _prewarm_dose_variants_native(session, state, variants)
    if used_native:
        return warmed

    warmed = 0
    for scan_file, group in grouped.items():
        group = group[:DOSE_OVERVIEW_PREFETCH_COUNT]
        quad_points_list = [patch.get("source_quad") or patch["corners"] for patch in group]
        crop_bboxes = [patch.get("crop_bbox") for patch in group]
        try:
            patch_images = preview.extract_patch_images(
                scan_file=Path(scan_file),
                quad_points_list=quad_points_list,
                crop_bboxes=crop_bboxes,
                backend="auto",
                max_dim=max_dim,
            )
        except TypeError:
            patch_images = preview.extract_patch_images(
                scan_file=Path(scan_file),
                quad_points_list=quad_points_list,
                crop_bboxes=crop_bboxes,
                backend="auto",
            )

        for patch, patch_image in zip(group, patch_images, strict=False):
            dose_context = _load_dose_context(session, patch=patch)
            patch_rgb = np.asarray(patch_image.convert("RGB"), dtype=np.uint8)
            dose, _ = dose_from_patch(
                patch_rgb=patch_rgb,
                film_background_mean=dose_context["film_background_mean"],
                scanner_background_mean=dose_context["scanner_background_mean"],
                film_model=dose_context["film_model"],
                background_quantile=dose_context["background_quantile"],
                backend=dose_context.get("backend", "auto"),
            )
            DOSE_PREVIEW_ARRAY_CACHE.set(
                _dose_preview_array_cache_key(session, patch, dose_context, max_dim=max_dim),
                (dose, dose_context),
            )
            DOSE_STATS_CACHE[_dose_stats_cache_key(state, patch)] = {
                "patch_id": patch["patch_id"],
                "film_type": dose_context["film_type"],
                "dose_min": round(float(dose.min()), 6),
                "dose_max": round(float(dose.max()), 6),
                "dose_mean": round(float(dose.mean()), 6),
            }
            image = _dose_preview_image_from_array(dose, palette)
            image = preview.resize_for_preview(image, max_dim)
            PREVIEW_CACHE[_dose_preview_cache_key(state, patch, palette, max_dim, preview_format, quality)] = (
                *preview.encode_preview_image(image, preview_format, quality),
            )
            content, media_type = PREVIEW_CACHE[_dose_preview_cache_key(state, patch, palette, max_dim, preview_format, quality)]
            _dose_disk_cache_store(
                state,
                patch,
                palette,
                max_dim,
                preview_format,
                quality,
                content=content,
                media_type=media_type,
            )
            warmed += 1
    return warmed


def _scan_preview_cache_key(
    state: dict,
    scan_index: int,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> tuple:
    return ("scan", state["version_id"], scan_index, max_dim, preview_format, quality)


def _patch_preview_cache_key(
    state: dict,
    patch: dict,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> tuple:
    return (
        "patch",
        state["version_id"],
        patch["patch_id"],
        patch.get("modified_revision", 0),
        tuple(patch.get("crop_bbox") or []),
        max_dim,
        preview_format,
        quality,
    )


def _raw_patch_preview_cache_key(
    state: dict,
    patch: dict,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> tuple:
    bbox = [int(value) for value in patch["display_bbox"]]
    return (
        "raw-patch",
        state["version_id"],
        patch["patch_id"],
        patch.get("modified_revision", 0),
        tuple(bbox),
        max_dim,
        preview_format,
        quality,
    )


def _ensure_scan_preview_cached(
    state: dict,
    scan: dict,
    *,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> bool:
    cache_key = _scan_preview_cache_key(state, scan["scan_index"], max_dim, preview_format, quality)
    if PREVIEW_CACHE.get(cache_key) is not None:
        return False
    PREVIEW_CACHE[cache_key] = preview.render_scan_preview(
        scan_file=Path(scan["scan_file"]),
        max_dim=max_dim,
        preview_format=preview_format,
        quality=quality,
    )
    return True


def _ensure_patch_preview_cached(
    state: dict,
    patch: dict,
    *,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> bool:
    cache_key = _patch_preview_cache_key(state, patch, max_dim, preview_format, quality)
    if PREVIEW_CACHE.get(cache_key) is not None:
        return False
    quad_points = patch.get("source_quad") or patch["corners"]
    PREVIEW_CACHE[cache_key] = preview.render_patch_preview(
        scan_file=Path(patch["scan_file"]),
        quad_points=quad_points,
        crop_bbox=patch.get("crop_bbox"),
        max_dim=max_dim,
        preview_format=preview_format,
        quality=quality,
    )
    return True


def _normalize_crop_bbox_for_patch(patch: dict, crop_bbox: list[int] | None) -> list[int] | None:
    if crop_bbox is None:
        return None
    if len(crop_bbox) != 4:
        raise ValueError("crop_bbox must contain x, y, width, height")
    x, y, width, height = [int(round(value)) for value in crop_bbox]
    if width < 1 or height < 1:
        raise ValueError("crop_bbox width and height must be positive")

    quad_points = patch.get("source_quad") or patch["corners"]
    patch_image = preview.extract_patch_image(
        scan_file=Path(patch["scan_file"]),
        quad_points=quad_points,
        max_dim=preview.DEFAULT_PATCH_MAX_DIM,
    )
    image_width, image_height = patch_image.size
    x0 = max(0, min(image_width - 1, x))
    y0 = max(0, min(image_height - 1, y))
    x1 = max(x0 + 1, min(image_width, x + width))
    y1 = max(y0 + 1, min(image_height, y + height))
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def _ensure_raw_patch_preview_cached(
    state: dict,
    patch: dict,
    *,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> bool:
    cache_key = _raw_patch_preview_cache_key(state, patch, max_dim, preview_format, quality)
    if PREVIEW_CACHE.get(cache_key) is not None:
        return False
    PREVIEW_CACHE[cache_key] = preview.render_bbox_preview(
        scan_file=Path(patch["scan_file"]),
        bbox=[int(value) for value in patch["display_bbox"]],
        max_dim=max_dim,
        preview_format=preview_format,
        quality=quality,
    )
    return True


def _ensure_dose_preview_cached(
    session: dict,
    state: dict,
    patch: dict,
    *,
    palette: str,
    max_dim: int,
    preview_format: str,
    quality: int,
) -> bool:
    cache_key = _dose_preview_cache_key(state, patch, palette, max_dim, preview_format, quality)
    if PREVIEW_CACHE.get(cache_key) is not None:
        return False
    disk_cached = _dose_disk_cache_load(state, patch, palette, max_dim, preview_format, quality)
    if disk_cached is not None:
        PREVIEW_CACHE[cache_key] = disk_cached
        return False
    image = _dose_preview_image(session, patch, palette, max_dim=max_dim)
    image = preview.resize_for_preview(image, max_dim)
    PREVIEW_CACHE[cache_key] = preview.encode_preview_image(image, preview_format, quality)
    content, media_type = PREVIEW_CACHE[cache_key]
    _dose_disk_cache_store(
        state,
        patch,
        palette,
        max_dim,
        preview_format,
        quality,
        content=content,
        media_type=media_type,
    )
    return True


def _ensure_dose_stats_cached(session: dict, state: dict, patch: dict) -> bool:
    cache_key = _dose_stats_cache_key(state, patch)
    if DOSE_STATS_CACHE.get(cache_key) is not None:
        return False
    DOSE_STATS_CACHE[cache_key] = _dose_stats_payload(session, patch)
    return True


def _gui_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RCF 在线处理台</title>
  <style>
    :root {
      --bg: #f4efe4;
      --panel: #fffdf7;
      --ink: #1f1a17;
      --muted: #77685b;
      --line: #d7c7b6;
      --accent: #b43f2f;
      --accent-soft: #f4d9d0;
      --olive: #59634c;
      --shadow: 0 20px 60px rgba(43, 31, 18, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180,63,47,0.08), transparent 26%),
        linear-gradient(180deg, #f8f3ea 0%, #efe6d7 100%);
      min-height: 100vh;
    }
    .shell {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
      padding: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 18px;
    }
    h1, h2, h3 { margin: 0 0 12px; font-weight: 700; }
    h1 { font-size: 32px; letter-spacing: -0.03em; }
    h2 { font-size: 20px; }
    h3 { font-size: 16px; }
    .eyebrow {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--accent);
      margin-bottom: 8px;
    }
    .muted { color: var(--muted); }
    .stack {
      display: grid;
      gap: 12px;
    }
    input, textarea, button, select {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 84px; resize: vertical; }
    button {
      cursor: pointer;
      background: var(--ink);
      color: #fff;
      border: none;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); opacity: 0.95; }
    button.secondary { background: var(--accent); }
    button.ghost { background: #f7f1e9; color: var(--ink); border: 1px solid var(--line); }
    .workspace {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 20px;
    }
    .workspace-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 20px;
    }
    .canvas-wrap {
      position: relative;
      border: 1px dashed var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, #fff, #fbf6ef);
      padding: 12px;
    }
    canvas {
      width: 100%;
      border-radius: 14px;
      display: block;
      background: #efe7da;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .patch-list {
      display: grid;
      gap: 8px;
      max-height: 280px;
      overflow: auto;
      padding-right: 4px;
    }
    .patch-item {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 14px;
      padding: 10px 12px;
      display: grid;
      gap: 6px;
    }
    .patch-item.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-soft);
    }
    .flash-assignment {
      box-shadow: 0 0 0 3px rgba(180, 63, 47, 0.22);
      transition: box-shadow 240ms ease;
    }
    .patch-actions, .order-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .patch-actions button, .order-actions button {
      width: auto;
      padding: 6px 10px;
      font-size: 13px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .grid-4 {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .grid-5 {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      background: #f4ecdf;
      color: var(--olive);
      font-size: 12px;
    }
    .workflow-box {
      display: grid;
      gap: 8px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #faf5ee;
    }
    .status {
      white-space: pre-wrap;
      font-size: 13px;
      color: var(--muted);
      min-height: 48px;
    }
    .view-switch {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .viewer-toolbar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .viewer-toolbar button {
      width: auto;
      padding: 8px 12px;
      font-size: 13px;
    }
    .viewer-toolbar select {
      width: auto;
      min-width: 116px;
      padding: 8px 10px;
      font-size: 13px;
    }
    .viewer-palette {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-left: 4px;
      color: var(--muted);
      font-size: 13px;
    }
    .viewer-nav {
      display: flex;
      gap: 8px;
      margin-left: auto;
    }
    .viewer-caption {
      margin: 10px 0 0;
      font-size: 13px;
      color: var(--muted);
    }
    .workflow-tab-pane {
      display: none;
      gap: 12px;
    }
    .workflow-tab-pane.active {
      display: grid;
    }
    .viewer-stage {
      min-height: 380px;
    }
    .scan-stage {
      min-height: 520px;
    }
    .viewer-image {
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      border-radius: 14px;
      display: none;
      background: linear-gradient(180deg, #fff, #fbf6ef);
    }
    .viewer-grid {
      display: none;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }
    .viewer-tile {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 14px;
      padding: 10px;
      display: grid;
      gap: 8px;
      cursor: pointer;
    }
    .viewer-tile.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-soft);
    }
    .viewer-tile img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      border-radius: 10px;
      background: linear-gradient(180deg, #fff, #fbf6ef);
    }
    .viewer-tile strong {
      font-size: 14px;
    }
    .viewer-empty {
      min-height: 320px;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
      padding: 24px;
    }
    .viewer-loading {
      opacity: 0.75;
    }
    .viewer-colorbar {
      display: none;
      gap: 8px;
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      min-height: 92px;
    }
    .viewer-colorbar-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
      color: var(--muted);
    }
    .viewer-colorbar-canvas {
      width: 100%;
      height: 18px;
      border-radius: 999px;
      display: block;
      border: 1px solid var(--line);
    }
    .viewer-colorbar-scale {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 12px;
      color: var(--muted);
    }
    .viewer-crop-toolbar {
      display: none;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .viewer-crop-toolbar.active {
      display: flex;
    }
    .viewer-crop-toolbar button {
      width: auto;
      padding: 8px 12px;
      font-size: 13px;
    }
    .viewer-crop-status {
      font-size: 13px;
      color: var(--muted);
      margin-left: auto;
    }
    .viewer-crop-editor {
      display: none;
      gap: 12px;
    }
    .viewer-crop-editor.active {
      display: grid;
    }
    .viewer-crop-surface {
      min-height: 360px;
      display: grid;
      place-items: center;
      padding: 16px;
      border: 1px dashed var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, #fff, #fbf6ef);
    }
    .viewer-crop-frame {
      position: relative;
      display: inline-block;
      max-width: 100%;
      line-height: 0;
    }
    .viewer-crop-frame img {
      display: block;
      max-width: 100%;
      max-height: 520px;
      width: auto;
      height: auto;
      border-radius: 14px;
      background: linear-gradient(180deg, #fff, #fbf6ef);
    }
    .viewer-crop-overlay {
      position: absolute;
      inset: 0;
      touch-action: none;
      cursor: default;
    }
    .viewer-crop-overlay.editing {
      cursor: crosshair;
    }
    .viewer-crop-box {
      position: absolute;
      border: 2px solid var(--accent);
      background: rgba(180, 63, 47, 0.12);
      display: none;
    }
    .viewer-crop-box.active {
      display: block;
    }
    .viewer-crop-handle {
      position: absolute;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 2px solid var(--accent);
      background: #fff;
      transform: translate(-50%, -50%);
      display: none;
    }
    .viewer-crop-handle.active {
      display: block;
    }
    body[data-view-mode="workflow"] .expert-only {
      display: none !important;
    }
    body[data-view-mode="expert"] .workflow-only {
      display: none !important;
    }
  </style>
</head>
<body data-view-mode="workflow">
  <div class="shell">
    <aside class="panel stack">
      <div>
        <div class="eyebrow">DetectorClaw</div>
        <h1>RCF 在线处理台</h1>
        <p class="muted">面向发次的在线 RCF 胶片处理与人工复核界面。</p>
      </div>
      <div class="view-switch">
        <button class="secondary" id="view-workflow" type="button">向导视图</button>
        <button class="ghost" id="view-expert" type="button">专家视图</button>
      </div>

      <div class="grid-2">
        <label>
          <div class="eyebrow">发次编号</div>
          <input id="shot-id" type="text" placeholder="001">
        </label>
        <label>
          <div class="eyebrow">数据根目录</div>
          <input id="data-root" type="text" placeholder="/mnt/c/Songtan/DetectorClaw">
        </label>
      </div>
      <label>
        <div class="eyebrow">检测模式</div>
        <select id="detection-mode">
          <option value="autocrop" selected>自动切片校正</option>
          <option value="segment">分割调试链</option>
        </select>
      </label>
      <label>
        <div class="eyebrow">配置 YAML</div>
        <input id="config-file" type="text" placeholder="可选覆盖">
      </label>
      <label>
        <div class="eyebrow">输出目录</div>
        <input id="output-dir" type="text" placeholder="可选覆盖">
      </label>
      <label>
        <div class="eyebrow">堆栈 JSON</div>
        <input id="stack-config-file" type="text" placeholder="可选覆盖">
      </label>
      <details>
        <summary class="eyebrow">手动输入覆盖</summary>
        <label>
          <div class="eyebrow">输入 TIFF 文件</div>
          <textarea id="input-files" placeholder="每行一个路径"></textarea>
        </label>
      </details>
      <div class="toolbar">
        <button class="secondary" id="load-session">恢复当前结果</button>
        <button class="ghost" id="redetect-session">重新检测本发</button>
      </div>
      <button class="ghost" id="export-session">导出处理结果</button>
      <div class="workflow-box">
        <div class="eyebrow">版本历史</div>
        <div id="version-list" class="patch-actions"></div>
      </div>
      <div id="status" class="status">尚未加载会话。</div>
    </aside>

    <main class="workspace">
      <div class="panel">
        <div class="eyebrow">扫描列表</div>
        <div id="scan-pills" class="patch-actions"></div>
      </div>

      <div class="workspace-grid">
        <section class="panel stack">
          <div>
            <div class="eyebrow">胶片编辑</div>
            <h2 id="patch-title">未选择胶片</h2>
            <div id="patch-stack" class="pill">无堆栈映射</div>
            <div id="patch-assignment" class="pill">未分配</div>
            <div id="patch-angle-meta" class="pill">角度元数据不可用</div>
          </div>

          <div class="workflow-box">
            <div class="eyebrow">人工标记流程</div>
            <strong id="workflow-step">步骤 1：修切割与旋转</strong>
            <span id="marking-progress" class="muted">尚未开始人工标记。</span>
            <span class="muted">步骤 1：修切割与旋转</span>
            <span class="muted">步骤 2：标记片序与重复片</span>
          </div>

          <div class="workflow-box workflow-only">
            <div class="eyebrow">主查看区</div>
            <div class="viewer-toolbar">
              <button id="viewer-mode-raw" class="secondary" type="button">原始扫描图</button>
              <button id="viewer-mode-corrected" class="ghost" type="button">修正图</button>
              <button id="viewer-mode-dose-overview" class="ghost" type="button">剂量计算图</button>
              <button id="viewer-mode-dose-pseudocolor" class="ghost" type="button">单片伪色</button>
              <label class="viewer-palette" for="viewer-dose-palette">
                <span>Palette</span>
                <select id="viewer-dose-palette">
                  <option value="gray">Gray</option>
                  <option value="turbo" selected>Turbo</option>
                  <option value="jet">Jet</option>
                </select>
              </label>
              <div class="viewer-nav">
                <button id="viewer-prev" class="ghost" type="button">↑ 上一片</button>
                <button id="viewer-next" class="ghost" type="button">↓ 下一片</button>
              </div>
            </div>
            <div id="scan-review-stage" class="workflow-tab-pane active">
              <div id="viewer-caption" class="viewer-caption">当前查看：原始扫描图</div>
              <h2 id="scan-title">未选择扫描</h2>
              <div class="canvas-wrap scan-stage">
                <canvas id="scan-canvas" width="960" height="760"></canvas>
              </div>
              <div class="workflow-box">
                <div class="eyebrow">直接输入片序</div>
                <label><div class="eyebrow">输入片序</div><input id="quick-assign-order" type="number" min="1" step="1"></label>
                <button id="quick-assign-apply" class="secondary" type="button">设为这个片序</button>
                <div class="patch-actions">
                  <button id="quick-mark-duplicate" class="ghost" type="button">标记重复胶片</button>
                  <button id="quick-clear-assignment" class="ghost" type="button">清除分配</button>
                </div>
              </div>
              <div class="toolbar">
                <button id="refresh-state" class="ghost">刷新状态</button>
                <button id="reset-geometry" class="ghost">重置当前几何</button>
              </div>
            </div>
            <div id="viewer-detail-stage" class="workflow-tab-pane">
            <div id="viewer-detail-caption" class="viewer-caption">当前查看：修正图</div>
            <div id="viewer-crop-toolbar" class="viewer-crop-toolbar">
              <button id="viewer-crop-start" class="secondary" type="button">开始框选</button>
              <button id="viewer-crop-apply" class="ghost" type="button">应用框选</button>
              <button id="viewer-crop-clear" class="ghost" type="button">清除框选</button>
              <button id="viewer-crop-back" class="ghost" type="button">返回总览</button>
              <span id="viewer-crop-status" class="viewer-crop-status">修正图总览模式</span>
            </div>
              <div id="viewer-colorbar" class="viewer-colorbar">
                <div class="viewer-colorbar-header">
                  <span>自适应色标 (Gy)</span>
                  <span id="viewer-colorbar-summary">等待剂量图</span>
                </div>
                <canvas id="viewer-colorbar-canvas" class="viewer-colorbar-canvas" width="320" height="18"></canvas>
                <div class="viewer-colorbar-scale">
                  <span id="viewer-colorbar-min">0</span>
                  <span id="viewer-colorbar-max">0</span>
                </div>
              </div>
              <div id="viewer-crop-editor" class="viewer-crop-editor">
                <div class="viewer-crop-surface">
                  <div id="viewer-crop-frame" class="viewer-crop-frame">
                    <img id="viewer-crop-image" alt="RCF corrected patch editor">
                    <div id="viewer-crop-overlay" class="viewer-crop-overlay">
                      <div id="viewer-crop-box" class="viewer-crop-box"></div>
                      <div id="viewer-crop-handle-nw" class="viewer-crop-handle"></div>
                      <div id="viewer-crop-handle-ne" class="viewer-crop-handle"></div>
                      <div id="viewer-crop-handle-se" class="viewer-crop-handle"></div>
                      <div id="viewer-crop-handle-sw" class="viewer-crop-handle"></div>
                    </div>
                  </div>
                </div>
              </div>
              <div class="canvas-wrap viewer-stage">
                <div id="viewer-empty" class="viewer-empty">加载会话后可在这里快速查看修正图和剂量图。</div>
                <img id="viewer-image" class="viewer-image" alt="RCF viewer image">
                <div id="viewer-grid" class="viewer-grid"></div>
              </div>
            </div>
          </div>

          <div class="grid-2 expert-only">
            <div class="canvas-wrap">
              <div class="eyebrow">当前扫描胶片</div>
              <label class="pill" style="margin-bottom: 10px;">
                <input id="patch-overlay-toggle" type="checkbox" style="width: auto; margin: 0 6px 0 0;">
                显示几何叠加
              </label>
              <canvas id="patch-canvas" width="420" height="300"></canvas>
            </div>
            <div class="canvas-wrap">
              <div class="eyebrow">校正后胶片</div>
              <canvas id="rotated-canvas" width="420" height="300"></canvas>
            </div>
          </div>

          <div class="grid-5 expert-only">
            <label><div class="eyebrow">中心 X</div><input id="geom-cx" type="number" step="0.1"></label>
            <label><div class="eyebrow">中心 Y</div><input id="geom-cy" type="number" step="0.1"></label>
            <label><div class="eyebrow">宽</div><input id="geom-width" type="number" step="0.1"></label>
            <label><div class="eyebrow">高</div><input id="geom-height" type="number" step="0.1"></label>
            <label><div class="eyebrow">角度</div><input id="geom-angle" type="number" step="0.1"></label>
          </div>
          <button id="apply-geometry" class="ghost expert-only">应用几何修改</button>

          <div class="grid-2 expert-only">
            <label><div class="eyebrow">角度滑杆</div><input id="angle-range" type="range" min="-90" max="90" step="0.1" value="0"></label>
            <label><div class="eyebrow">角度数值</div><input id="angle-number" type="number" step="0.1" value="0"></label>
          </div>
          <div class="toolbar expert-only">
            <button id="apply-angle" class="ghost">应用角度</button>
            <button id="capture-edge" class="secondary">采集 2 个边点</button>
          </div>

          <div class="grid-4 expert-only">
            <label><div class="eyebrow">精裁 X</div><input id="crop-x" type="number"></label>
            <label><div class="eyebrow">精裁 Y</div><input id="crop-y" type="number"></label>
            <label><div class="eyebrow">精裁宽</div><input id="crop-w" type="number"></label>
            <label><div class="eyebrow">精裁高</div><input id="crop-h" type="number"></label>
          </div>
          <button id="apply-crop" class="ghost expert-only">保存精裁框</button>

          <div>
            <div class="eyebrow">当前扫描胶片</div>
            <div id="patch-list" class="patch-list"></div>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    let sessionId = null;
    let state = null;
    let selectedScanIndex = 1;
    let selectedPatchId = null;
    let scanImage = null;
    let rawPatchPreviewImage = null;
    let patchPreviewImage = null;
    let pointCaptureMode = false;
    let edgePoints = [];
    let assignmentFlashToken = null;
    let pollingToken = null;
    let dragState = null;
    let viewMode = 'workflow';
    let viewerMode = 'raw';
    let dosePalette = 'turbo';
    let viewerImageToken = 0;
    let viewerDoseStatsToken = 0;
    let viewerDisplayedUrl = null;
    let viewerDisplayedPatchId = null;
    let viewerDisplayedMode = null;
    let viewerDisplayedPalette = null;
    let viewerDoseStats = null;
    let viewerDoseStatsLoading = false;
    let correctedEditorOpen = false;
    let cropEditMode = false;
    let cropEditDraft = null;
    let cropDragState = null;
    let cropEditorImageKey = null;
    const viewerHighResDelayMs = 120;
    const viewerDoseStatsDelayMs = 40;
    const DOSE_OVERVIEW_PREVIEW_DIM = 260;
    const DOSE_SINGLE_PREVIEW_DIM = 320;
    const DOSE_SINGLE_HIGH_PREVIEW_DIM = 960;
    const DOSE_PREVIEW_QUALITY = 78;
    const DOSE_OVERVIEW_PREFETCH_LIMIT = 12;
    let viewerHighResTimer = null;
    let viewerWarmRetryTimer = null;
    let viewerDoseStatsTimer = null;
    let viewerImageFetchController = null;
    let viewerDoseStatsController = null;
    const prefetchedViewerAssets = new Map();
    const prefetchedViewerAssetLimit = 32;
    const viewerObjectUrlCache = new Map();
    const viewerObjectUrlCacheLimit = 48;
    const prefetchedDoseStats = new Map();
    const prefetchedDoseStatsLimit = 16;
    const doseAssetManifestCache = new Map();
    const doseAssetManifestCacheLimit = 4;
    const HANDLE_RADIUS_PX = 9;
    const ROTATION_HANDLE_OFFSET_PX = 28;

    const scanCanvas = document.getElementById('scan-canvas');
    const scanCtx = scanCanvas.getContext('2d');
    const patchCanvas = document.getElementById('patch-canvas');
    const patchCtx = patchCanvas.getContext('2d');
    const rotatedCanvas = document.getElementById('rotated-canvas');
    const rotatedCtx = rotatedCanvas.getContext('2d');
    const patchOverlayToggle = document.getElementById('patch-overlay-toggle');
    const scanReviewStage = document.getElementById('scan-review-stage');
    const viewerDetailStage = document.getElementById('viewer-detail-stage');
    const viewerImage = document.getElementById('viewer-image');
    const viewerGrid = document.getElementById('viewer-grid');
    const viewerEmpty = document.getElementById('viewer-empty');
    const viewerCropToolbar = document.getElementById('viewer-crop-toolbar');
    const viewerCropStatus = document.getElementById('viewer-crop-status');
    const viewerCropEditor = document.getElementById('viewer-crop-editor');
    const viewerCropFrame = document.getElementById('viewer-crop-frame');
    const viewerCropImage = document.getElementById('viewer-crop-image');
    const viewerCropOverlay = document.getElementById('viewer-crop-overlay');
    const viewerCropBox = document.getElementById('viewer-crop-box');
    const viewerCropHandles = {
      nw: document.getElementById('viewer-crop-handle-nw'),
      ne: document.getElementById('viewer-crop-handle-ne'),
      se: document.getElementById('viewer-crop-handle-se'),
      sw: document.getElementById('viewer-crop-handle-sw'),
    };
    const viewerDosePalette = document.getElementById('viewer-dose-palette');
    const viewerColorbar = document.getElementById('viewer-colorbar');
    const viewerColorbarCanvas = document.getElementById('viewer-colorbar-canvas');
    const viewerColorbarSummary = document.getElementById('viewer-colorbar-summary');
    const viewerColorbarMin = document.getElementById('viewer-colorbar-min');
    const viewerColorbarMax = document.getElementById('viewer-colorbar-max');

    function setStatus(text) {
      document.getElementById('status').textContent = text;
    }

    function setViewMode(mode) {
      viewMode = mode === 'expert' ? 'expert' : 'workflow';
      document.body.dataset.viewMode = viewMode;
      document.getElementById('view-workflow').className = viewMode === 'workflow' ? 'secondary' : 'ghost';
      document.getElementById('view-expert').className = viewMode === 'expert' ? 'secondary' : 'ghost';
      if (viewMode === 'expert') {
        loadPatchImages().then(() => renderAll()).catch((error) => setStatus(error.message));
      }
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
        ...options,
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail);
      }
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) return response.json();
      return response.text();
    }

    function patchById(patchId) {
      return state?.patches?.find((patch) => patch.patch_id === patchId) || null;
    }

    function doseViewsAvailable() {
      return Boolean(state?.dose_available);
    }

    function doseUnavailableMessage() {
      return state?.dose_error || '缺少背景图或校准配置，无法计算物理剂量。';
    }

    function doseModesActive() {
      return viewerMode === 'dose_overview' || viewerMode === 'dose_pseudocolor';
    }

    function trimViewerObjectUrlCache() {
      while (viewerObjectUrlCache.size > viewerObjectUrlCacheLimit) {
        const oldestKey = viewerObjectUrlCache.keys().next().value;
        const oldestObjectUrl = viewerObjectUrlCache.get(oldestKey);
        if (oldestObjectUrl) URL.revokeObjectURL(oldestObjectUrl);
        viewerObjectUrlCache.delete(oldestKey);
      }
    }

    function clearViewerObjectUrlCache() {
      viewerObjectUrlCache.forEach((objectUrl) => URL.revokeObjectURL(objectUrl));
      viewerObjectUrlCache.clear();
    }

    function getCachedViewerObjectUrl(url) {
      const objectUrl = viewerObjectUrlCache.get(url);
      if (!objectUrl) return null;
      viewerObjectUrlCache.delete(url);
      viewerObjectUrlCache.set(url, objectUrl);
      return objectUrl;
    }

    function getOrCreateViewerObjectUrl(url, blob) {
      const cached = getCachedViewerObjectUrl(url);
      if (cached) return cached;
      const objectUrl = URL.createObjectURL(blob);
      viewerObjectUrlCache.set(url, objectUrl);
      trimViewerObjectUrlCache();
      return objectUrl;
    }

    function viewerMatchesCurrentSelection(patchId, mode, palette) {
      return selectedPatchId === patchId
        && viewerMode === mode
        && (!doseModesActive() || dosePalette === palette);
    }

    function doseStatsUrl(patch) {
      return `/api/rcf/session/${sessionId}/patch/${patch.patch_id}/dose-stats?revision=${encodeURIComponent(String(patch.modified_revision || 0))}&version_id=${encodeURIComponent(state?.version_id || '')}`;
    }

    function doseOverviewPrewarmUrlForPalette(palette) {
      return `/api/rcf/session/${sessionId}/dose-overview-prewarm?palette=${encodeURIComponent(palette)}&max_dim=${DOSE_OVERVIEW_PREVIEW_DIM}&format=jpeg&quality=${DOSE_PREVIEW_QUALITY}&version_id=${encodeURIComponent(state?.version_id || '')}`;
    }

    function trimDoseAssetManifestCache() {
      while (doseAssetManifestCache.size > doseAssetManifestCacheLimit) {
        const oldestKey = doseAssetManifestCache.keys().next().value;
        doseAssetManifestCache.delete(oldestKey);
      }
    }

    function doseAssetManifestCacheKey() {
      if (!sessionId || !state) return null;
      return `${sessionId}:${state.version_id}:${state.revision}`;
    }

    async function fetchDoseAssetManifest() {
      const cacheKey = doseAssetManifestCacheKey();
      if (!cacheKey || !doseViewsAvailable()) return null;
      const cached = doseAssetManifestCache.get(cacheKey);
      if (cached) return cached;
      const promise = api(`/api/rcf/session/${sessionId}/assets/manifest`)
        .catch((error) => {
          if (doseAssetManifestCache.get(cacheKey) === promise) {
            doseAssetManifestCache.delete(cacheKey);
          }
          throw error;
        });
      doseAssetManifestCache.set(cacheKey, promise);
      trimDoseAssetManifestCache();
      return promise;
    }

    function dosePseudoPalettes() {
      return ['turbo', 'jet'];
    }

    function formatDoseValue(value) {
      const numeric = Number(value || 0);
      if (!Number.isFinite(numeric)) return '0.000';
      if (Math.abs(numeric) >= 100) return numeric.toFixed(1);
      if (Math.abs(numeric) >= 10) return numeric.toFixed(2);
      return numeric.toFixed(3);
    }

    function formatDoseWithUnit(value) {
      return `${formatDoseValue(value)} Gy`;
    }

    function paletteStops(palette) {
      if (palette === 'gray') return ['#000000', '#ffffff'];
      if (palette === 'jet') {
        return ['#00007f', '#0000ff', '#007fff', '#00ffff', '#7fff7f', '#ffff00', '#ff7f00', '#ff0000', '#7f0000'];
      }
      return ['#30123b', '#4145ab', '#4675ed', '#39a2fc', '#1bcfd4', '#24eca6', '#61fc6c', '#a4fc3c', '#d1e834', '#f3c63a', '#fe9b2d', '#f36315', '#d93806', '#b11901', '#7a0403'];
    }

    function renderDoseColorbar() {
      const isVisible = doseModesActive() && doseViewsAvailable();
      viewerColorbar.style.display = isVisible ? 'grid' : 'none';
      if (!isVisible) return;
      const context = viewerColorbarCanvas.getContext('2d');
      const gradient = context.createLinearGradient(0, 0, viewerColorbarCanvas.width, 0);
      const stops = paletteStops(dosePalette);
      stops.forEach((color, index) => {
        gradient.addColorStop(index / Math.max(1, stops.length - 1), color);
      });
      context.clearRect(0, 0, viewerColorbarCanvas.width, viewerColorbarCanvas.height);
      context.fillStyle = gradient;
      context.fillRect(0, 0, viewerColorbarCanvas.width, viewerColorbarCanvas.height);
      if (!viewerDoseStats) {
        viewerColorbarMin.textContent = '—';
        viewerColorbarMax.textContent = '—';
        viewerColorbarSummary.textContent = viewerDoseStatsLoading
          ? '正在计算当前胶片的自适应剂量范围…'
          : '等待剂量图';
        return;
      }
      viewerColorbarMin.textContent = formatDoseWithUnit(viewerDoseStats.dose_min);
      viewerColorbarMax.textContent = formatDoseWithUnit(viewerDoseStats.dose_max);
      const statsPatch = patchById(viewerDoseStats.patch_id);
      const statsLabel = statsPatch ? primaryPatchLabel(statsPatch) : viewerDoseStats.patch_id;
      if (viewerDoseStatsLoading) {
        viewerColorbarSummary.textContent = `${statsLabel} · 自适应范围更新中…`;
        return;
      }
      viewerColorbarSummary.textContent = `${statsLabel} · min ${formatDoseWithUnit(viewerDoseStats.dose_min)} · max ${formatDoseWithUnit(viewerDoseStats.dose_max)} · mean ${formatDoseWithUnit(viewerDoseStats.dose_mean)}`;
    }

    function renderViewerPaletteControl() {
      viewerDosePalette.value = dosePalette;
      viewerDosePalette.disabled = !doseModesActive();
    }

    function setViewerMode(mode) {
      clearViewerHighResTimer();
      clearViewerWarmRetryTimer();
      viewerMode = ['raw', 'corrected_grid', 'dose_overview', 'dose_pseudocolor'].includes(mode) ? mode : 'raw';
      renderWorkflowViewer();
      renderDoseColorbar();
      syncWorkflowViewerAsync();
      prefetchWorkflowAssets();
    }

    function assignedPatchesInOrder() {
      if (!state) return [];
      return state.patches
        .filter((patch) => patch.assignment_status === 'assigned' && Number.isFinite(patch.assigned_order))
        .sort((left, right) => {
          if (left.assigned_order !== right.assigned_order) return left.assigned_order - right.assigned_order;
          return left.patch_id.localeCompare(right.patch_id);
        });
    }

    function currentScanPatches() {
      if (!state) return [];
      return state.patches.filter((patch) => patch.scan_index === selectedScanIndex);
    }

    function rawPatchImageUrl(patch) {
      return `/api/rcf/session/${sessionId}/patch/${patch.patch_id}/raw-image?max_dim=960&format=jpeg&quality=82&revision=${encodeURIComponent(String(patch.modified_revision || 0))}&version_id=${encodeURIComponent(state?.version_id || '')}`;
    }

    function correctedPatchImageUrl(patch, maxDim = 260) {
      return `/api/rcf/session/${sessionId}/patch/${patch.patch_id}/image?max_dim=${maxDim}&format=jpeg&quality=82&revision=${encodeURIComponent(String(patch.modified_revision || 0))}&version_id=${encodeURIComponent(state?.version_id || '')}`;
    }

    function dosePatchImageUrl(patch, palette, maxDim = DOSE_OVERVIEW_PREVIEW_DIM) {
      const cacheOnly = maxDim === DOSE_SINGLE_PREVIEW_DIM || maxDim === DOSE_SINGLE_HIGH_PREVIEW_DIM;
      return `/api/rcf/session/${sessionId}/patch/${patch.patch_id}/dose-image?palette=${encodeURIComponent(palette)}&max_dim=${maxDim}&format=jpeg&quality=${DOSE_PREVIEW_QUALITY}&cache_only=${cacheOnly ? 'true' : 'false'}&revision=${encodeURIComponent(String(patch.modified_revision || 0))}&version_id=${encodeURIComponent(state?.version_id || '')}`;
    }

    function clearViewerHighResTimer() {
      if (viewerHighResTimer !== null) {
        clearTimeout(viewerHighResTimer);
        viewerHighResTimer = null;
      }
    }

    function clearViewerWarmRetryTimer() {
      if (viewerWarmRetryTimer !== null) {
        clearTimeout(viewerWarmRetryTimer);
        viewerWarmRetryTimer = null;
      }
    }

    function clearViewerDoseStatsTimer() {
      if (viewerDoseStatsTimer !== null) {
        clearTimeout(viewerDoseStatsTimer);
        viewerDoseStatsTimer = null;
      }
    }

    function abortViewerImageFetch() {
      if (viewerImageFetchController) {
        viewerImageFetchController.abort();
        viewerImageFetchController = null;
      }
    }

    function abortViewerDoseStatsFetch() {
      if (viewerDoseStatsController) {
        viewerDoseStatsController.abort();
        viewerDoseStatsController = null;
      }
    }

    function trimPrefetchedViewerAssets() {
      while (prefetchedViewerAssets.size > prefetchedViewerAssetLimit) {
        const oldestKey = prefetchedViewerAssets.keys().next().value;
        prefetchedViewerAssets.delete(oldestKey);
      }
    }

    function hasPrefetchedViewerAsset(url) {
      return Boolean(url) && prefetchedViewerAssets.has(url);
    }

    function trimPrefetchedDoseStats() {
      while (prefetchedDoseStats.size > prefetchedDoseStatsLimit) {
        const oldestKey = prefetchedDoseStats.keys().next().value;
        prefetchedDoseStats.delete(oldestKey);
      }
    }

    function prefetchDoseStats(patch) {
      const url = doseStatsUrl(patch);
      if (!url || prefetchedDoseStats.has(url)) return;
      const promise = fetch(url)
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(await response.text());
          }
          return response.json();
        })
        .catch((error) => {
          if (prefetchedDoseStats.get(url) === promise) {
            prefetchedDoseStats.delete(url);
          }
          throw error;
        });
      prefetchedDoseStats.set(url, promise);
      trimPrefetchedDoseStats();
    }

    async function fetchViewerImageBlob(url, patchId, mode, palette, token) {
      const cachedObjectUrl = getCachedViewerObjectUrl(url);
      if (cachedObjectUrl) {
        if (token !== viewerImageToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) {
          return false;
        }
        viewerImage.src = cachedObjectUrl;
        viewerImage.style.display = 'block';
        viewerImage.dataset.viewerUrl = url;
        viewerEmpty.style.display = 'none';
        viewerEmpty.textContent = '';
        viewerDisplayedUrl = url;
        viewerDisplayedPatchId = patchId;
        viewerDisplayedMode = mode;
        viewerDisplayedPalette = palette;
        return true;
      }
      let blob = null;
      const prefetched = prefetchedViewerAssets.get(url);
      if (prefetched) {
        blob = await prefetched;
      } else {
        try {
          viewerImageFetchController = new AbortController();
          const response = await fetch(url, { signal: viewerImageFetchController.signal });
          if (response.status === 425) {
            viewerImageFetchController = null;
            return false;
          }
          if (!response.ok) {
            throw new Error(await response.text());
          }
          blob = await response.blob();
          prefetchedViewerAssets.set(url, Promise.resolve(blob));
          trimPrefetchedViewerAssets();
          viewerImageFetchController = null;
        } catch (error) {
          viewerImageFetchController = null;
          if (error?.name === 'AbortError') {
            return false;
          }
          throw error;
        }
      }
      if (token !== viewerImageToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) {
        return false;
      }
      const objectUrl = getOrCreateViewerObjectUrl(url, blob);
      if (token !== viewerImageToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) {
        return false;
      }
      viewerImage.src = objectUrl;
      viewerImage.style.display = 'block';
      viewerImage.dataset.viewerUrl = url;
      viewerEmpty.style.display = 'none';
      viewerEmpty.textContent = '';
      viewerDisplayedUrl = url;
      viewerDisplayedPatchId = patchId;
      viewerDisplayedMode = mode;
      viewerDisplayedPalette = palette;
      return true;
    }

    async function loadViewerSingleImage() {
      const patch = patchById(selectedPatchId);
      if (!state || !patch) return;
      if (viewerMode !== 'raw' && viewerMode !== 'dose_pseudocolor') return;
      if (viewerMode === 'dose_pseudocolor' && !doseViewsAvailable()) return;

      const patchId = patch.patch_id;
      const mode = viewerMode;
      const palette = dosePalette;
      const lowUrl = mode === 'raw'
        ? rawPatchImageUrl(patch)
        : dosePatchImageUrl(patch, palette, DOSE_SINGLE_PREVIEW_DIM);
      const highUrl = mode === 'raw'
        ? rawPatchImageUrl(patch)
        : dosePatchImageUrl(patch, palette, DOSE_SINGLE_HIGH_PREVIEW_DIM);
      const currentViewerUrl = viewerImage.dataset.viewerUrl || '';
      const frameAlreadyDisplayed = (
        viewerDisplayedPatchId === patchId
        && viewerDisplayedMode === mode
        && (!doseModesActive() || viewerDisplayedPalette === palette)
        && (viewerDisplayedUrl === lowUrl || viewerDisplayedUrl === highUrl || currentViewerUrl === lowUrl || currentViewerUrl === highUrl)
      );
      if (frameAlreadyDisplayed) {
        viewerImage.classList.remove('viewer-loading');
        viewerEmpty.style.display = 'none';
        viewerImage.style.display = 'block';
        return;
      }
      const lowPrefetched = hasPrefetchedViewerAsset(lowUrl) || Boolean(getCachedViewerObjectUrl(lowUrl));
      const token = ++viewerImageToken;
      clearViewerHighResTimer();
      clearViewerWarmRetryTimer();
      abortViewerImageFetch();
      viewerImage.classList.add('viewer-loading');
      viewerDisplayedUrl = null;
      viewerDisplayedPatchId = null;
      viewerDisplayedMode = null;
      viewerDisplayedPalette = null;
      if (lowPrefetched) {
        viewerImage.style.display = 'block';
        viewerEmpty.style.display = 'none';
        viewerEmpty.textContent = '';
      } else {
        viewerImage.style.display = 'none';
        viewerEmpty.style.display = 'grid';
        viewerEmpty.textContent = '正在加载当前胶片图像…';
      }

      const lowApplied = await fetchViewerImageBlob(lowUrl, patchId, mode, palette, token);
      if (token !== viewerImageToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) return;
      if (lowApplied) {
        viewerImage.classList.remove('viewer-loading');
      } else {
        viewerWarmRetryTimer = setTimeout(() => {
          viewerWarmRetryTimer = null;
          if (token !== viewerImageToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) return;
          loadViewerSingleImage().catch((error) => setStatus(error.message));
        }, 120);
        return;
      }
      if (highUrl === lowUrl) return;
      viewerHighResTimer = setTimeout(() => {
        viewerHighResTimer = null;
        abortViewerImageFetch();
        fetchViewerImageBlob(highUrl, patchId, mode, palette, token)
          .then((applied) => {
            if (!applied) return;
            if (token !== viewerImageToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) return;
            viewerImage.classList.remove('viewer-loading');
          })
          .catch((error) => {
            if (token !== viewerImageToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) return;
            setStatus(error.message);
          });
      }, viewerHighResDelayMs);
    }

    async function loadViewerDoseStats() {
      const patch = patchById(selectedPatchId);
      if (!state || !patch || !doseViewsAvailable() || !doseModesActive()) return;
      const patchId = patch.patch_id;
      const mode = viewerMode;
      const palette = dosePalette;
      const token = ++viewerDoseStatsToken;
      clearViewerDoseStatsTimer();
      abortViewerDoseStatsFetch();
      viewerDoseStatsLoading = true;
      renderDoseColorbar();
      viewerDoseStatsTimer = setTimeout(async () => {
        viewerDoseStatsTimer = null;
        try {
          const url = doseStatsUrl(patch);
          let payload = null;
          const prefetched = prefetchedDoseStats.get(url);
          if (prefetched) {
            payload = await prefetched;
          } else {
            viewerDoseStatsController = new AbortController();
            const response = await fetch(url, { signal: viewerDoseStatsController.signal });
            if (!response.ok) {
              throw new Error(await response.text());
            }
            payload = await response.json();
          }
          viewerDoseStatsController = null;
          if (token !== viewerDoseStatsToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) return;
          viewerDoseStats = payload;
          viewerDoseStatsLoading = false;
          renderDoseColorbar();
        } catch (error) {
          viewerDoseStatsController = null;
          if (error?.name === 'AbortError') return;
          if (token !== viewerDoseStatsToken || !viewerMatchesCurrentSelection(patchId, mode, palette)) return;
          viewerDoseStatsLoading = false;
          setStatus(error.message);
        }
      }, viewerDoseStatsDelayMs);
    }

    function syncWorkflowViewerAsync() {
      if (!state || !selectedPatchId) return;
      if (viewerMode === 'dose_pseudocolor') {
        loadViewerSingleImage().catch((error) => setStatus(error.message));
      } else {
        viewerImageToken += 1;
        viewerDisplayedUrl = null;
        viewerDisplayedPatchId = null;
        viewerDisplayedMode = null;
        viewerDisplayedPalette = null;
        viewerImage.removeAttribute('data-viewer-url');
        viewerImage.removeAttribute('src');
        viewerImage.style.display = 'none';
      }
      if (doseModesActive()) {
        loadViewerDoseStats().catch((error) => setStatus(error.message));
      } else {
        clearViewerDoseStatsTimer();
        abortViewerDoseStatsFetch();
        viewerDoseStatsToken += 1;
        viewerDoseStatsLoading = false;
        renderDoseColorbar();
      }
    }

    function renderViewerModeButtons() {
      renderViewerPaletteControl();
      const mapping = {
        'viewer-mode-raw': 'raw',
        'viewer-mode-corrected': 'corrected_grid',
        'viewer-mode-dose-overview': 'dose_overview',
        'viewer-mode-dose-pseudocolor': 'dose_pseudocolor',
      };
      Object.entries(mapping).forEach(([id, mode]) => {
        document.getElementById(id).className = viewerMode === mode ? 'secondary' : 'ghost';
      });
    }

    function renderViewerNavigation() {
      const previousButton = document.getElementById('viewer-prev');
      const nextButton = document.getElementById('viewer-next');
      const assigned = assignedPatchesInOrder();
      if (!assigned.length) {
        previousButton.disabled = true;
        nextButton.disabled = true;
        return;
      }
      const index = assigned.findIndex((patch) => patch.patch_id === selectedPatchId);
      if (index === -1) {
        previousButton.disabled = false;
        nextButton.disabled = false;
        return;
      }
      previousButton.disabled = index <= 0;
      nextButton.disabled = index >= assigned.length - 1;
    }

    function canEditCropForPatch(patch) {
      return Boolean(patch && patch.assignment_status === 'assigned');
    }

    function correctedEditorPatch() {
      if (!correctedEditorOpen || viewerMode !== 'corrected_grid') return null;
      return patchById(selectedPatchId);
    }

    function correctedEditorImageUrl(patch) {
      return `${correctedPatchImageUrl(patch, 640)}&ignore_crop=true`;
    }

    function normalizedCropRect(rect, maxWidth, maxHeight) {
      if (!rect) return null;
      const x = Math.max(0, Math.min(maxWidth - 1, Math.round(rect[0])));
      const y = Math.max(0, Math.min(maxHeight - 1, Math.round(rect[1])));
      const width = Math.max(1, Math.round(rect[2]));
      const height = Math.max(1, Math.round(rect[3]));
      const x1 = Math.max(x + 1, Math.min(maxWidth, x + width));
      const y1 = Math.max(y + 1, Math.min(maxHeight, y + height));
      return [x, y, x1 - x, y1 - y];
    }

    function cropRectFromPoints(startPoint, endPoint, maxWidth, maxHeight) {
      const x0 = Math.min(startPoint.x, endPoint.x);
      const y0 = Math.min(startPoint.y, endPoint.y);
      const x1 = Math.max(startPoint.x, endPoint.x);
      const y1 = Math.max(startPoint.y, endPoint.y);
      return normalizedCropRect([x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)], maxWidth, maxHeight);
    }

    function currentRenderedCropRect(patch) {
      if (!patch) return null;
      if (correctedEditorOpen && cropEditDraft && patch.patch_id === selectedPatchId) return cropEditDraft;
      return patch.crop_bbox || null;
    }

    function viewerCropLocalPoint(event) {
      if (!viewerCropImage.naturalWidth || !viewerCropImage.naturalHeight) return null;
      const rect = viewerCropOverlay.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      const offsetX = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
      const offsetY = Math.max(0, Math.min(rect.height, event.clientY - rect.top));
      const scaleX = viewerCropImage.naturalWidth / rect.width;
      const scaleY = viewerCropImage.naturalHeight / rect.height;
      return {
        x: offsetX * scaleX,
        y: offsetY * scaleY,
      };
    }

    function cropHandleAtPoint(point, rect) {
      if (!point || !rect) return null;
      const [x, y, width, height] = rect;
      const handles = {
        nw: { x, y },
        ne: { x: x + width, y },
        se: { x: x + width, y: y + height },
        sw: { x, y: y + height },
      };
      const threshold = 14;
      return Object.entries(handles).find(([, handlePoint]) => {
        const dx = point.x - handlePoint.x;
        const dy = point.y - handlePoint.y;
        return Math.sqrt(dx * dx + dy * dy) <= threshold;
      })?.[0] || null;
    }

    function renderViewerCropOverlay(patch) {
      const rect = currentRenderedCropRect(patch);
      const imageReady = Boolean(viewerCropImage.naturalWidth && viewerCropImage.naturalHeight);
      viewerCropOverlay.classList.toggle('editing', cropEditMode && imageReady);
      if (!imageReady || !rect) {
        viewerCropBox.classList.remove('active');
        Object.values(viewerCropHandles).forEach((handle) => handle.classList.remove('active'));
        return;
      }
      const displayWidth = viewerCropImage.clientWidth;
      const displayHeight = viewerCropImage.clientHeight;
      const scaleX = displayWidth / viewerCropImage.naturalWidth;
      const scaleY = displayHeight / viewerCropImage.naturalHeight;
      const [x, y, width, height] = rect;
      viewerCropBox.classList.add('active');
      viewerCropBox.style.left = `${x * scaleX}px`;
      viewerCropBox.style.top = `${y * scaleY}px`;
      viewerCropBox.style.width = `${width * scaleX}px`;
      viewerCropBox.style.height = `${height * scaleY}px`;

      const handlePositions = {
        nw: [x, y],
        ne: [x + width, y],
        se: [x + width, y + height],
        sw: [x, y + height],
      };
      Object.entries(viewerCropHandles).forEach(([key, handle]) => {
        if (cropEditMode) {
          handle.classList.add('active');
          handle.style.left = `${handlePositions[key][0] * scaleX}px`;
          handle.style.top = `${handlePositions[key][1] * scaleY}px`;
        } else {
          handle.classList.remove('active');
        }
      });
    }

    function syncCorrectedEditorImage(patch) {
      if (!patch) return;
      const nextKey = `${patch.patch_id}:${patch.modified_revision || 0}:${state?.version_id || ''}`;
      if (cropEditorImageKey === nextKey && viewerCropImage.getAttribute('src')) return;
      cropEditorImageKey = nextKey;
      viewerCropImage.src = correctedEditorImageUrl(patch);
    }

    function renderViewerCropToolbar(patch) {
      const active = Boolean(patch && correctedEditorOpen && viewerMode === 'corrected_grid');
      viewerCropToolbar.classList.toggle('active', active);
      if (!active) return;
      const editable = canEditCropForPatch(patch);
      document.getElementById('viewer-crop-start').disabled = !editable;
      document.getElementById('viewer-crop-apply').disabled = !editable || !cropEditMode || !cropEditDraft;
      document.getElementById('viewer-crop-clear').disabled = !editable || (!patch.crop_bbox && !cropEditDraft);
      document.getElementById('viewer-crop-start').textContent = cropEditMode ? '重新框选' : '开始框选';
      if (!editable) {
        viewerCropStatus.textContent = '先给这片分配片序后再手动框选。';
        return;
      }
      if (cropEditMode) {
        viewerCropStatus.textContent = cropEditDraft ? '拖动四角微调后点“应用框选”。' : '在单片修正图上拖出新的矩形 ROI。';
        return;
      }
      viewerCropStatus.textContent = patch.crop_bbox ? '已存在手动框选，可继续调整或清除。' : '尚未设置手动框选。';
    }

    function openCorrectedCropEditor(patchId) {
      const patch = patchById(patchId);
      if (!patch) return;
      correctedEditorOpen = true;
      cropEditMode = false;
      cropEditDraft = null;
      cropDragState = null;
      selectPatch(patchId);
    }

    function closeCorrectedCropEditor() {
      correctedEditorOpen = false;
      cropEditMode = false;
      cropEditDraft = null;
      cropDragState = null;
      renderAll();
    }

    function startCorrectedCropEditing() {
      const patch = correctedEditorPatch();
      if (!patch || !canEditCropForPatch(patch)) {
        setStatus('先给这片分配片序后再手动框选。');
        return;
      }
      cropEditMode = true;
      cropDragState = null;
      cropEditDraft = patch.crop_bbox ? [...patch.crop_bbox] : null;
      renderViewerCropToolbar(patch);
      renderViewerCropOverlay(patch);
      setStatus(cropEditDraft ? '拖动四角微调后点“应用框选”。' : '在单片修正图上拖出新的矩形 ROI。');
    }

    async function applyCorrectedCropEdit() {
      const patch = correctedEditorPatch();
      if (!patch || !canEditCropForPatch(patch)) {
        setStatus('先给这片分配片序后再手动框选。');
        return;
      }
      if (!cropEditDraft) {
        setStatus('请先拖出一个有效框选。');
        return;
      }
      cropEditMode = false;
      cropDragState = null;
      await api(`/api/rcf/session/${sessionId}/patch/${patch.patch_id}/crop`, {
        method: 'POST',
        body: JSON.stringify({ crop_bbox: cropEditDraft }),
      });
      cropEditDraft = null;
      await refreshState();
      setStatus(`已保存手动框选：${patch.patch_id}`);
    }

    async function clearCorrectedCropEdit() {
      const patch = correctedEditorPatch();
      if (!patch || !canEditCropForPatch(patch)) {
        setStatus('先给这片分配片序后再手动框选。');
        return;
      }
      cropEditMode = false;
      cropEditDraft = null;
      cropDragState = null;
      await api(`/api/rcf/session/${sessionId}/patch/${patch.patch_id}/crop`, {
        method: 'POST',
        body: JSON.stringify({ crop_bbox: null }),
      });
      await refreshState();
      setStatus(`已清除手动框选：${patch.patch_id}`);
    }

    function renderViewerTile(container, patch, imageUrl, subtitle, onClick = null) {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = `viewer-tile ${patch.patch_id === selectedPatchId ? 'active' : ''}`;
      item.innerHTML = `
        <img alt="${patch.patch_id}" class="viewer-loading" loading="lazy" decoding="async">
        <strong>${primaryPatchLabel(patch)}</strong>
        <span class="muted">${subtitle}</span>
      `;
      const image = item.querySelector('img');
      image.onload = () => image.classList.remove('viewer-loading');
      image.src = imageUrl;
      item.onclick = () => (onClick ? onClick(patch) : selectPatch(patch.patch_id));
      container.appendChild(item);
    }

    function renderWorkflowViewer() {
      renderViewerModeButtons();
      renderViewerNavigation();
      const patch = patchById(selectedPatchId);
      const rawCaption = document.getElementById('viewer-caption');
      const detailCaption = document.getElementById('viewer-detail-caption');
      viewerGrid.innerHTML = '';
      viewerGrid.style.display = 'none';
      viewerCropEditor.classList.remove('active');
      viewerCropToolbar.classList.remove('active');

      if (!state || !patch) {
        scanReviewStage.classList.remove('active');
        viewerDetailStage.classList.add('active');
        viewerImage.style.display = 'none';
        viewerEmpty.style.display = 'grid';
        viewerEmpty.textContent = '加载会话后可在这里快速查看修正图和剂量图。';
        rawCaption.textContent = '当前查看：原始扫描图';
        detailCaption.textContent = '当前查看：等待选择胶片';
        renderDoseColorbar();
        return;
      }

      if (viewerMode === 'raw') {
        scanReviewStage.classList.add('active');
        viewerDetailStage.classList.remove('active');
        rawCaption.textContent = `当前查看：扫描 ${String(selectedScanIndex).padStart(2, '0')} 原始扫描图`;
        renderDoseColorbar();
        return;
      }

      scanReviewStage.classList.remove('active');
      viewerDetailStage.classList.add('active');

      if (viewerMode === 'corrected_grid') {
        if (correctedEditorOpen) {
          detailCaption.textContent = `当前查看：${primaryPatchLabel(patch)} 单片修正图`;
          viewerImage.style.display = 'none';
          viewerEmpty.style.display = 'none';
          viewerGrid.style.display = 'none';
          viewerCropEditor.classList.add('active');
          renderViewerCropToolbar(patch);
          syncCorrectedEditorImage(patch);
          renderViewerCropOverlay(patch);
          return;
        }
        viewerImage.style.display = 'none';
        viewerEmpty.style.display = 'none';
        viewerGrid.style.display = 'grid';
        currentScanPatches().forEach((item) => {
          renderViewerTile(
            viewerGrid,
            item,
            correctedPatchImageUrl(item),
            `扫描 ${item.scan_index} · ${assignmentText(item)}`,
            (clickedPatch) => {
              if (!canEditCropForPatch(clickedPatch)) {
                selectPatch(clickedPatch.patch_id);
                setStatus('先给这片分配片序后再手动框选。');
                return;
              }
              openCorrectedCropEditor(clickedPatch.patch_id);
            },
          );
        });
        detailCaption.textContent = `当前查看：扫描 ${String(selectedScanIndex).padStart(2, '0')} 的修正图总览`;
        return;
      }

      if (viewerMode === 'dose_overview') {
        if (!doseViewsAvailable()) {
          viewerImage.style.display = 'none';
          viewerEmpty.style.display = 'grid';
          viewerEmpty.textContent = doseUnavailableMessage();
          detailCaption.textContent = '当前查看：剂量总览不可用';
          renderDoseColorbar();
          return;
        }
        const assigned = assignedPatchesInOrder();
        if (!assigned.length) {
          viewerImage.style.display = 'none';
          viewerEmpty.style.display = 'grid';
          viewerEmpty.textContent = '先给胶片分配片序，剂量总览只显示已分配片序的胶片。';
          detailCaption.textContent = '当前查看：剂量总览';
          renderDoseColorbar();
          return;
        }
        viewerEmpty.style.display = 'none';
        viewerImage.style.display = 'none';
        viewerGrid.style.display = 'grid';
        assigned.forEach((item) => {
          renderViewerTile(
            viewerGrid,
            item,
            dosePatchImageUrl(item, dosePalette),
            `扫描 ${item.scan_index} · ${primaryPatchLabel(item)}`,
          );
        });
        detailCaption.textContent = `当前查看：整发已分配片序剂量总览 · ${dosePalette.toUpperCase()}`;
        renderDoseColorbar();
        return;
      }

      if (viewerMode === 'dose_pseudocolor' && !doseViewsAvailable()) {
        viewerImage.style.display = 'none';
        viewerEmpty.style.display = 'grid';
        viewerEmpty.textContent = doseUnavailableMessage();
        detailCaption.textContent = '当前查看：单片剂量伪色不可用';
        renderDoseColorbar();
        return;
      }

      const lowUrl = viewerMode === 'raw'
        ? rawPatchImageUrl(patch)
        : dosePatchImageUrl(patch, dosePalette, DOSE_SINGLE_PREVIEW_DIM);
      const canKeepCurrentFrame = (
        viewerDisplayedPatchId === patch.patch_id
        && viewerDisplayedMode === viewerMode
        && (viewerMode !== 'dose_pseudocolor' || viewerDisplayedPalette === dosePalette)
      ) || (hasPrefetchedViewerAsset(lowUrl) && viewerImage.getAttribute('src'));

      detailCaption.textContent = `当前查看：${primaryPatchLabel(patch)} 单片剂量伪色 · ${dosePalette.toUpperCase()}`;
      if (canKeepCurrentFrame) {
        viewerEmpty.style.display = 'none';
        viewerImage.style.display = 'block';
      } else {
        viewerImage.style.display = 'none';
        viewerEmpty.style.display = 'grid';
        viewerEmpty.textContent = '正在加载当前胶片图像…';
      }
      renderDoseColorbar();
    }

    function prefetchImage(url) {
      if (!url || prefetchedViewerAssets.has(url)) return;
      const promise = fetch(url)
        .then((response) => {
          if (response.status === 425) {
            return null;
          }
          if (!response.ok) {
            throw new Error(`Prefetch failed: ${response.status}`);
          }
          return response.blob();
        })
        .catch((error) => {
          if (prefetchedViewerAssets.get(url) === promise) {
            prefetchedViewerAssets.delete(url);
          }
          throw error;
        });
      prefetchedViewerAssets.set(url, promise);
      trimPrefetchedViewerAssets();
    }

    async function prefetchImageAndHydrateObjectUrl(url) {
      if (!url) return false;
      const cachedObjectUrl = getCachedViewerObjectUrl(url);
      if (cachedObjectUrl) return true;
      prefetchImage(url);
      const prefetched = prefetchedViewerAssets.get(url);
      if (!prefetched) return false;
      try {
        const blob = await prefetched;
        if (!blob) return false;
        getOrCreateViewerObjectUrl(url, blob);
        return true;
      } catch {
        return false;
      }
    }

    async function warmReadyDoseAssetsFromManifest() {
      const manifest = await fetchDoseAssetManifest();
      if (!manifest?.patches?.length) return;
      await Promise.all(
        manifest.patches.flatMap((patchPayload) =>
          (patchPayload.variants || [])
            .filter((variant) => variant.ready && variant.url)
            .map((variant) => prefetchImageAndHydrateObjectUrl(variant.url))
        )
      );
    }

    function prefetchDoseImagesForPatch(patch, maxDim = DOSE_SINGLE_PREVIEW_DIM) {
      dosePseudoPalettes().forEach((palette) => {
        prefetchImage(dosePatchImageUrl(patch, palette, maxDim));
      });
    }

    function prefetchWorkflowAssets() {
      if (!state || !selectedPatchId) return;
      const patch = patchById(selectedPatchId);
      if (!patch) return;
      currentScanPatches().forEach((item) => {
        prefetchImageAndHydrateObjectUrl(correctedPatchImageUrl(item, DOSE_OVERVIEW_PREVIEW_DIM)).catch(() => {});
      });
      if (viewerMode === 'corrected_grid' && correctedEditorOpen) {
        prefetchImage(correctedEditorImageUrl(patch));
      }
      if (!doseViewsAvailable()) return;
      warmReadyDoseAssetsFromManifest().catch((error) => setStatus(error.message));
      if (viewerMode === 'dose_overview') {
        const assigned = assignedPatchesInOrder().slice(0, DOSE_OVERVIEW_PREFETCH_LIMIT);
        assigned.forEach((item) => {
          dosePseudoPalettes().forEach((palette) => {
            prefetchImage(dosePatchImageUrl(item, palette, DOSE_OVERVIEW_PREVIEW_DIM));
          });
        });
        dosePseudoPalettes().forEach((palette) => {
          api(doseOverviewPrewarmUrlForPalette(palette)).catch((error) => setStatus(error.message));
        });
        return;
      }
      if (viewerMode === 'raw') {
        prefetchImage(rawPatchImageUrl(patch));
        prefetchImage(correctedPatchImageUrl(patch, DOSE_SINGLE_PREVIEW_DIM));
        prefetchDoseImagesForPatch(patch, DOSE_SINGLE_PREVIEW_DIM);
        prefetchDoseImagesForPatch(patch, DOSE_SINGLE_HIGH_PREVIEW_DIM);
        prefetchDoseStats(patch);
        const assigned = assignedPatchesInOrder();
        const index = assigned.findIndex((item) => item.patch_id === selectedPatchId);
        [index - 1, index + 1]
          .filter((value) => value >= 0 && value < assigned.length)
          .forEach((value) => {
            prefetchDoseImagesForPatch(assigned[value], DOSE_SINGLE_PREVIEW_DIM);
            prefetchDoseStats(assigned[value]);
          });
        return;
      }
      prefetchDoseImagesForPatch(patch, DOSE_SINGLE_PREVIEW_DIM);
      prefetchDoseImagesForPatch(patch, DOSE_SINGLE_HIGH_PREVIEW_DIM);
      prefetchDoseStats(patch);
      const assigned = assignedPatchesInOrder();
      const index = assigned.findIndex((item) => item.patch_id === selectedPatchId);
      [index - 1, index + 1]
        .filter((value) => value >= 0 && value < assigned.length)
        .forEach((value) => {
          prefetchDoseImagesForPatch(assigned[value], DOSE_SINGLE_PREVIEW_DIM);
          prefetchDoseStats(assigned[value]);
        });
    }

    function loadExpertPatchImages() {
      if (viewMode !== 'expert') return Promise.resolve();
      return loadPatchImages();
    }

    function selectPatch(patchId) {
      if (!state) return;
      const patch = patchById(patchId);
      if (!patch) return;
      const scanChanged = selectedScanIndex !== patch.scan_index;
      const patchChanged = selectedPatchId !== patch.patch_id;
      if (patchChanged) {
        cropEditMode = false;
        cropEditDraft = null;
        cropDragState = null;
      }
      clearViewerHighResTimer();
      clearViewerDoseStatsTimer();
      abortViewerImageFetch();
      abortViewerDoseStatsFetch();
      selectedPatchId = patch.patch_id;
      selectedScanIndex = patch.scan_index;
      renderAll();
      syncWorkflowViewerAsync();
      if (scanChanged) {
        loadScanImage()
          .then(() => renderAll())
          .catch((error) => setStatus(error.message));
      }
      loadExpertPatchImages()
        .then(() => renderAll())
        .catch((error) => setStatus(error.message));
      prefetchWorkflowAssets();
    }

    function navigateAssignedPatch(direction) {
      const assigned = assignedPatchesInOrder();
      if (!assigned.length) {
        setStatus('还没有已分配片序的胶片。');
        return;
      }
      const index = assigned.findIndex((patch) => patch.patch_id === selectedPatchId);
      if (index === -1) {
        selectPatch(assigned[0].patch_id);
        return;
      }
      const nextIndex = index + direction;
      if (nextIndex < 0 || nextIndex >= assigned.length) return;
      selectPatch(assigned[nextIndex].patch_id);
    }

    function rotatedRectCorners(rotatedRect) {
      const cx = Number(rotatedRect.cx);
      const cy = Number(rotatedRect.cy);
      const width = Number(rotatedRect.width);
      const height = Number(rotatedRect.height);
      const angleRad = (Number(rotatedRect.angle_deg) * Math.PI) / 180;
      const cosA = Math.cos(angleRad);
      const sinA = Math.sin(angleRad);
      const halfW = width / 2;
      const halfH = height / 2;
      const local = [
        [-halfW, -halfH],
        [halfW, -halfH],
        [halfW, halfH],
        [-halfW, halfH],
      ];
      return local.map(([dx, dy]) => [
        cx + dx * cosA - dy * sinA,
        cy + dx * sinA + dy * cosA,
      ]);
    }

    function rotatedRectAabb(rotatedRect) {
      const corners = rotatedRectCorners(rotatedRect);
      const xs = corners.map((point) => point[0]);
      const ys = corners.map((point) => point[1]);
      const x0 = Math.floor(Math.min(...xs));
      const y0 = Math.floor(Math.min(...ys));
      const x1 = Math.ceil(Math.max(...xs));
      const y1 = Math.ceil(Math.max(...ys));
      return [x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0)];
    }

    function rotatedRectAxes(rotatedRect) {
      const angleRad = (Number(rotatedRect.angle_deg) * Math.PI) / 180;
      return {
        ux: { x: Math.cos(angleRad), y: Math.sin(angleRad) },
        uy: { x: -Math.sin(angleRad), y: Math.cos(angleRad) },
      };
    }

    function dot(a, b) {
      return a.x * b.x + a.y * b.y;
    }

    function subtractPoints(a, b) {
      return { x: a.x - b.x, y: a.y - b.y };
    }

    function syncClientPatchGeometry(patch, rotatedRect) {
      patch.rotated_rect = {
        cx: Number(rotatedRect.cx),
        cy: Number(rotatedRect.cy),
        width: Math.max(1, Number(rotatedRect.width)),
        height: Math.max(1, Number(rotatedRect.height)),
        angle_deg: Number(rotatedRect.angle_deg),
      };
      patch.angle_deg = patch.rotated_rect.angle_deg;
      patch.corners = rotatedRectCorners(patch.rotated_rect);
      patch.display_bbox = rotatedRectAabb(patch.rotated_rect);
    }

    function pointInPolygon(point, polygon) {
      let inside = false;
      for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0];
        const yi = polygon[i][1];
        const xj = polygon[j][0];
        const yj = polygon[j][1];
        const intersects = ((yi > point.y) !== (yj > point.y))
          && (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-6) + xi);
        if (intersects) inside = !inside;
      }
      return inside;
    }

    function scanImagePointToCanvasPoint(point) {
      const { scale, offsetX, offsetY } = scanScale();
      return {
        x: offsetX + point.x * scale,
        y: offsetY + point.y * scale,
      };
    }

    function rotationHandleForPatch(patch) {
      const rect = patch.rotated_rect;
      const { ux, uy } = rotatedRectAxes(rect);
      const halfH = rect.height / 2;
      const offset = ROTATION_HANDLE_OFFSET_PX / scanScale().scale;
      return {
        x: rect.cx - uy.x * (halfH + offset),
        y: rect.cy - uy.y * (halfH + offset),
      };
    }

    function handleHit(point, handleCenter, radiusPx = HANDLE_RADIUS_PX) {
      const handleCanvas = scanImagePointToCanvasPoint(handleCenter);
      const dx = point.x - handleCanvas.x;
      const dy = point.y - handleCanvas.y;
      return Math.sqrt(dx * dx + dy * dy) <= radiusPx;
    }

    function assignmentText(patch) {
      if (patch.assignment_status === 'assigned') return `第 ${patch.assigned_order} 片`;
      if (patch.assignment_status === 'ignored_duplicate') return '重复';
      return '未标记';
    }

    function assignmentLabelText(patch) {
      return `当前人工片序：${assignmentText(patch)}`;
    }

    function primaryPatchLabel(patch) {
      return patch.display_label || assignmentText(patch);
    }

    function secondaryPatchLabel(patch) {
      return assignmentText(patch);
    }

    function nextAssignableOrder() {
      if (!state) return 1;
      const assignedOrders = state.patches
        .filter((patch) => patch.assignment_status === 'assigned' && Number.isFinite(patch.assigned_order))
        .map((patch) => patch.assigned_order);
      let nextOrder = 1;
      const assignedSet = new Set(assignedOrders);
      while (assignedSet.has(nextOrder)) nextOrder += 1;
      return nextOrder;
    }

    function markingProgressText() {
      if (!state) return '尚未开始人工标记。';
      const assigned = state.patches.filter((patch) => patch.assignment_status === 'assigned').length;
      const ignored = state.patches.filter((patch) => patch.assignment_status === 'ignored_duplicate').length;
      const remaining = state.patches.length - assigned - ignored;
      return `已标记 ${assigned} 片，重复 ${ignored} 片，待处理 ${remaining} 片。`;
    }

    function workflowStepText() {
      if (!state) return '步骤 1：修切割与旋转';
      return state.patches.some((patch) => patch.assignment_status === 'unassigned')
        ? '步骤 2：标记片序与重复片'
        : '片序标记已完成，可继续微调';
    }

    function currentScan() {
      return state?.scans?.find((scan) => scan.scan_index === selectedScanIndex) || null;
    }

    function detectionModeText(mode) {
      if (mode === 'autocrop') return '自动切片校正';
      if (mode === 'segment') return '分割调试链';
      return mode || '未知模式';
    }

    function autoStatusPrefix() {
      if (!state) return '';
      const shot = state.shot_id ? `发次 ${state.shot_id}` : '手动会话';
      const version = state.available_versions.find((item) => item.version_id === state.version_id);
      const versionLabel = version ? `v${version.version_number}` : state.version_id;
      return `${shot} · ${versionLabel} · 修订 ${state.revision}`;
    }

    function sessionSourceText(source) {
      if (source === 'restored') return '恢复旧结果';
      if (source === 'new_detection') return '新检测';
      return source || '未知来源';
    }

    async function loadSession(forceRedetect = false) {
      const shotId = document.getElementById('shot-id').value.trim();
      const dataRoot = document.getElementById('data-root').value.trim();
      const inputFiles = document.getElementById('input-files').value
        .split('\\n')
        .map((value) => value.trim())
        .filter(Boolean);
      const payload = {
        shot_id: shotId || null,
        data_root: dataRoot || null,
        input_files: inputFiles.length ? inputFiles : null,
        config_file: document.getElementById('config-file').value.trim() || null,
        output_dir: document.getElementById('output-dir').value.trim() || null,
        stack_config_file: document.getElementById('stack-config-file').value.trim() || null,
        detection_mode: document.getElementById('detection-mode').value,
        force_redetect: forceRedetect,
      };
      const result = await api('/api/rcf/session/load', { method: 'POST', body: JSON.stringify(payload) });
      sessionId = result.session_id;
      await refreshState(false);
      setStatus(`已加载 ${autoStatusPrefix()}，共 ${result.patch_count} 片，模式 ${detectionModeText(result.detection_mode)} · ${sessionSourceText(result.session_source)}。`);
      startPolling();
    }

    async function redetectCurrentSession() {
      if (!sessionId) {
        await loadSession(true);
        return;
      }
      const result = await api(`/api/rcf/session/${sessionId}/redetect`, { method: 'POST' });
      sessionId = result.session_id;
      await refreshState(false);
      setStatus(`已重新检测 ${autoStatusPrefix()}，共 ${result.patch_count} 片，模式 ${detectionModeText(result.detection_mode)}。`);
      startPolling();
    }

    async function refreshState(announceChanges = true) {
      if (!sessionId) return;
      const previousRevision = state ? state.revision : null;
      const previousSelectedPatch = selectedPatchId;
      state = await api(`/api/rcf/session/${sessionId}/state`);
      if (!selectedPatchId && state.patches.length) selectedPatchId = state.patches[0].patch_id;
      const patch = patchById(selectedPatchId);
      if (!patch && previousSelectedPatch) selectedPatchId = state.patches[0]?.patch_id || null;
      const selectedPatch = patchById(selectedPatchId);
      if (selectedPatch) selectedScanIndex = selectedPatch.scan_index;
      await loadScanImage();
      if (viewMode === 'expert') {
        await loadPatchImages();
      } else {
        rawPatchPreviewImage = null;
        patchPreviewImage = null;
      }
      renderAll();
      syncWorkflowViewerAsync();
      prefetchWorkflowAssets();
      if (
        announceChanges &&
        previousRevision !== null &&
        state.revision !== previousRevision &&
        state.last_modified_patch_id
      ) {
        setStatus(`检测到手动修改：${state.last_modified_patch_id} · 修订 ${state.revision} · 已自动保存`);
      }
    }

    function startPolling() {
      stopPolling();
      pollingToken = window.setInterval(async () => {
        if (!sessionId) return;
        try {
          const nextState = await api(`/api/rcf/session/${sessionId}/state`);
          if (!state || nextState.revision !== state.revision || nextState.version_id !== state.version_id) {
            state = nextState;
            const patch = patchById(selectedPatchId);
            if (patch) selectedScanIndex = patch.scan_index;
            await loadScanImage();
            if (viewMode === 'expert') {
              await loadPatchImages();
            } else {
              rawPatchPreviewImage = null;
              patchPreviewImage = null;
            }
            renderAll();
            syncWorkflowViewerAsync();
            prefetchWorkflowAssets();
            if (nextState.last_modified_patch_id) {
              setStatus(`检测到手动修改：${nextState.last_modified_patch_id} · 修订 ${nextState.revision} · 已自动保存`);
            }
          }
        } catch (error) {
          setStatus(`轮询失败：${error.message}`);
        }
      }, 1500);
    }

    function stopPolling() {
      if (!pollingToken) return;
      window.clearInterval(pollingToken);
      pollingToken = null;
    }

    async function loadScanImage() {
      if (!sessionId || !state) return;
      const image = new Image();
      image.src = `/api/rcf/session/${sessionId}/scan/${selectedScanIndex}/image?max_dim=1600&format=jpeg&quality=80&version_id=${encodeURIComponent(state.version_id || '')}`;
      await image.decode();
      scanImage = image;
    }

    async function loadPatchPreviewImage() {
      patchPreviewImage = null;
      if (!sessionId || !selectedPatchId) return;
      const patch = patchById(selectedPatchId);
      const image = new Image();
      image.src = `/api/rcf/session/${sessionId}/patch/${selectedPatchId}/image?max_dim=640&format=jpeg&quality=80&revision=${encodeURIComponent(String(patch?.modified_revision || 0))}&version_id=${encodeURIComponent(state?.version_id || '')}`;
      await image.decode();
      patchPreviewImage = image;
    }

    async function loadRawPatchPreviewImage() {
      rawPatchPreviewImage = null;
      if (!sessionId || !selectedPatchId) return;
      const patch = patchById(selectedPatchId);
      const image = new Image();
      image.src = `/api/rcf/session/${sessionId}/patch/${selectedPatchId}/raw-image?max_dim=640&format=jpeg&quality=80&revision=${encodeURIComponent(String(patch?.modified_revision || 0))}&version_id=${encodeURIComponent(state?.version_id || '')}`;
      await image.decode();
      rawPatchPreviewImage = image;
    }

    async function loadPatchImages() {
      await Promise.all([loadRawPatchPreviewImage(), loadPatchPreviewImage()]);
    }

    function renderAll() {
      renderVersionList();
      renderScanPills();
      renderScanCanvas();
      renderPatchList();
      renderWorkflowViewer();
      renderPatchEditor();
    }

    function renderVersionList() {
      const container = document.getElementById('version-list');
      container.innerHTML = '';
      if (!state) return;
      state.available_versions.forEach((version) => {
        const button = document.createElement('button');
        button.className = version.version_id === state.version_id ? 'secondary' : 'ghost';
        button.textContent = `v${version.version_number} · rev ${version.revision}`;
        button.onclick = async () => {
          await api(`/api/rcf/session/${sessionId}/versions/${version.version_id}/activate`, { method: 'POST' });
          await refreshState(false);
          setStatus(`已切换到版本 v${version.version_number}。`);
        };
        container.appendChild(button);
      });
    }

    function renderScanPills() {
      const container = document.getElementById('scan-pills');
      container.innerHTML = '';
      if (!state) {
        document.getElementById('scan-title').textContent = '未选择扫描';
        return;
      }
      state.scans.forEach((scan) => {
        const button = document.createElement('button');
        button.className = scan.scan_index === selectedScanIndex ? 'secondary' : 'ghost';
        button.textContent = `扫描 ${String(scan.scan_index).padStart(2, '0')}`;
        button.onclick = () => {
          selectedScanIndex = scan.scan_index;
          const firstPatchId = scan.patch_ids[0];
          if (firstPatchId) {
            selectPatch(firstPatchId);
            return;
          }
          renderAll();
          loadScanImage().then(() => renderAll()).catch((error) => setStatus(error.message));
        };
        container.appendChild(button);
      });
      document.getElementById('scan-title').textContent = state ? `扫描 ${String(selectedScanIndex).padStart(2, '0')}` : '未选择扫描';
    }

    function currentScanDimensions() {
      const scan = state ? currentScan() : null;
      if (!scan) return { width: scanImage ? scanImage.width : 1, height: scanImage ? scanImage.height : 1 };
      return { width: scan.width, height: scan.height };
    }

    function previewScaleFactors() {
      if (!scanImage || !state) return { x: 1, y: 1 };
      const scan = currentScan();
      if (!scan) return { x: 1, y: 1 };
      return { x: scanImage.width / scan.width, y: scanImage.height / scan.height };
    }

    function scanScale() {
      if (!scanImage || !state) return { scale: 1, offsetX: 0, offsetY: 0, drawWidth: 0, drawHeight: 0 };
      const dims = currentScanDimensions();
      const scale = Math.min(scanCanvas.width / dims.width, scanCanvas.height / dims.height);
      const drawWidth = dims.width * scale;
      const drawHeight = dims.height * scale;
      const offsetX = (scanCanvas.width - drawWidth) / 2;
      const offsetY = (scanCanvas.height - drawHeight) / 2;
      return { scale, offsetX, offsetY, drawWidth, drawHeight };
    }

    function scanCanvasPointToImagePoint(point) {
      const { scale, offsetX, offsetY, drawWidth, drawHeight } = scanScale();
      if (!scale) return null;
      if (
        point.x < offsetX || point.x > offsetX + drawWidth ||
        point.y < offsetY || point.y > offsetY + drawHeight
      ) {
        return null;
      }
      return {
        x: (point.x - offsetX) / scale,
        y: (point.y - offsetY) / scale,
      };
    }

    function drawPolygon(ctx, corners, scale, offsetX, offsetY, active) {
      ctx.beginPath();
      corners.forEach((corner, index) => {
        const x = offsetX + corner[0] * scale;
        const y = offsetY + corner[1] * scale;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.lineWidth = active ? 4 : 2;
      ctx.strokeStyle = active ? '#b43f2f' : '#59634c';
      ctx.stroke();
    }

    function drawSelectedPatchHandles(ctx, patch, scale, offsetX, offsetY) {
      patch.corners.forEach((corner) => {
        ctx.beginPath();
        ctx.arc(offsetX + corner[0] * scale, offsetY + corner[1] * scale, HANDLE_RADIUS_PX / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#fffdf7';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#b43f2f';
        ctx.stroke();
      });
      const rotationHandle = rotationHandleForPatch(patch);
      const handleX = offsetX + rotationHandle.x * scale;
      const handleY = offsetY + rotationHandle.y * scale;
      ctx.beginPath();
      ctx.arc(handleX, handleY, HANDLE_RADIUS_PX / 2, 0, Math.PI * 2);
      ctx.fillStyle = '#b43f2f';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#fffdf7';
      ctx.stroke();
      const topMid = {
        x: patch.rotated_rect.cx - rotatedRectAxes(patch.rotated_rect).uy.x * (patch.rotated_rect.height / 2),
        y: patch.rotated_rect.cy - rotatedRectAxes(patch.rotated_rect).uy.y * (patch.rotated_rect.height / 2),
      };
      ctx.beginPath();
      ctx.moveTo(offsetX + topMid.x * scale, offsetY + topMid.y * scale);
      ctx.lineTo(handleX, handleY);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#b43f2f';
      ctx.stroke();
    }

    function renderScanCanvas() {
      scanCtx.clearRect(0, 0, scanCanvas.width, scanCanvas.height);
      if (!scanImage) return;
      const { scale, offsetX, offsetY, drawWidth, drawHeight } = scanScale();
      scanCtx.drawImage(scanImage, offsetX, offsetY, drawWidth, drawHeight);
      state.patches
        .filter((patch) => patch.scan_index === selectedScanIndex)
        .forEach((patch) => {
          drawPolygon(scanCtx, patch.corners, scale, offsetX, offsetY, patch.patch_id === selectedPatchId);
          if (patch.patch_id === selectedPatchId) {
            drawSelectedPatchHandles(scanCtx, patch, scale, offsetX, offsetY);
          }
          const displayBox = patch.display_bbox;
          const labelX = offsetX + displayBox[0] * scale + 8;
          const labelY = offsetY + displayBox[1] * scale + 16;
          scanCtx.font = '13px Georgia';
          scanCtx.lineWidth = 3;
          scanCtx.strokeStyle = 'rgba(255, 253, 247, 0.95)';
          const overlayLabel = primaryPatchLabel(patch);
          scanCtx.strokeText(overlayLabel, labelX, labelY);
          scanCtx.fillStyle = patch.patch_id === selectedPatchId ? '#b43f2f' : '#2f5440';
          scanCtx.fillText(overlayLabel, labelX, labelY);
        });
    }

    function renderPatchList() {
      const container = document.getElementById('patch-list');
      container.innerHTML = '';
      if (!state) return;
      state.patches
        .filter((patch) => patch.scan_index === selectedScanIndex)
        .forEach((patch) => {
          const item = document.createElement('div');
          item.className = `patch-item ${patch.patch_id === selectedPatchId ? 'active' : ''}`;
          item.innerHTML = `
            <strong>${primaryPatchLabel(patch)}</strong>
            <span class="muted">扫描 ${patch.scan_index}</span>
            <span class="muted">几何: ${patch.rotated_rect.width.toFixed(1)} × ${patch.rotated_rect.height.toFixed(1)} | 角度: ${patch.angle_deg.toFixed(2)}°</span>
            <span class="muted">状态：${assignmentText(patch)}</span>
            <div class="patch-actions">
              <button class="ghost">选中</button>
            </div>
          `;
          item.querySelector('button').onclick = () => selectPatch(patch.patch_id);
          container.appendChild(item);
        });
    }

    function renderPatchEditor() {
      const patch = patchById(selectedPatchId);
      patchCtx.clearRect(0, 0, patchCanvas.width, patchCanvas.height);
      rotatedCtx.clearRect(0, 0, rotatedCanvas.width, rotatedCanvas.height);
      if (!patch || !scanImage) return;

      document.getElementById('workflow-step').textContent = workflowStepText();
      document.getElementById('marking-progress').textContent = markingProgressText();
      document.getElementById('patch-title').textContent = primaryPatchLabel(patch);
      document.getElementById('patch-stack').textContent = patch.stack ? `片号 ${patch.stack.rcf_id} · ${patch.stack.material_name} · ${patch.stack.thickness}` : '无堆栈映射';
      document.getElementById('patch-assignment').textContent = assignmentLabelText(patch);
      const lowConfidence = (patch.status_flags || []).includes('low_confidence_angle') ? ' · 低置信度' : '';
      document.getElementById('patch-angle-meta').textContent =
        `${patch.angle_source || '未知来源'} · ${detectionModeText(patch.detection_source || state.detection_mode)} · 置信度 ${(patch.angle_confidence || 0).toFixed(2)}${lowConfidence}`;

      const geom = patch.rotated_rect;
      document.getElementById('geom-cx').value = geom.cx.toFixed(1);
      document.getElementById('geom-cy').value = geom.cy.toFixed(1);
      document.getElementById('geom-width').value = geom.width.toFixed(1);
      document.getElementById('geom-height').value = geom.height.toFixed(1);
      document.getElementById('geom-angle').value = geom.angle_deg.toFixed(1);
      document.getElementById('angle-range').value = patch.angle_deg;
      document.getElementById('angle-number').value = patch.angle_deg;
      document.getElementById('quick-assign-order').value = patch.assigned_order || '';
      const previewWidth = patchPreviewImage ? patchPreviewImage.width : Math.round(geom.width);
      const previewHeight = patchPreviewImage ? patchPreviewImage.height : Math.round(geom.height);
      const crop = patch.crop_bbox || [0, 0, previewWidth, previewHeight];
      document.getElementById('crop-x').value = crop[0];
      document.getElementById('crop-y').value = crop[1];
      document.getElementById('crop-w').value = crop[2];
      document.getElementById('crop-h').value = crop[3];

      const box = patch.display_bbox;
      const [x, y, w, h] = box;
      const rawPatch = document.createElement('canvas');
      rawPatch.width = Math.max(1, w);
      rawPatch.height = Math.max(1, h);
      const rawCtx = rawPatch.getContext('2d');
      if (rawPatchPreviewImage) {
        rawCtx.drawImage(rawPatchPreviewImage, 0, 0, Math.max(1, w), Math.max(1, h));
      } else {
        const previewScale = previewScaleFactors();
        rawCtx.drawImage(
          scanImage,
          x * previewScale.x,
          y * previewScale.y,
          Math.max(1, w * previewScale.x),
          Math.max(1, h * previewScale.y),
          0,
          0,
          Math.max(1, w),
          Math.max(1, h),
        );
      }

      const patchScale = Math.min(patchCanvas.width / Math.max(1, w), patchCanvas.height / Math.max(1, h));
      const pw = w * patchScale;
      const ph = h * patchScale;
      const px = (patchCanvas.width - pw) / 2;
      const py = (patchCanvas.height - ph) / 2;
      patchCtx.drawImage(rawPatch, px, py, pw, ph);

      if (patchOverlayToggle.checked) {
        patchCtx.strokeStyle = '#b43f2f';
        patchCtx.lineWidth = 2;
        patchCtx.beginPath();
        patch.corners.forEach((corner, index) => {
          const localX = px + (corner[0] - x) * patchScale;
          const localY = py + (corner[1] - y) * patchScale;
          if (index === 0) patchCtx.moveTo(localX, localY);
          else patchCtx.lineTo(localX, localY);
        });
        patchCtx.closePath();
        patchCtx.stroke();

        if (patch.edge_points && patch.edge_points.length === 2) {
          patchCtx.beginPath();
          patchCtx.moveTo(px + patch.edge_points[0][0] * patchScale, py + patch.edge_points[0][1] * patchScale);
          patchCtx.lineTo(px + patch.edge_points[1][0] * patchScale, py + patch.edge_points[1][1] * patchScale);
          patchCtx.stroke();
        }
      }

      if (patchPreviewImage) {
        const rectifiedScale = Math.min(
          rotatedCanvas.width / Math.max(1, patchPreviewImage.width),
          rotatedCanvas.height / Math.max(1, patchPreviewImage.height),
        );
        const rw = patchPreviewImage.width * rectifiedScale;
        const rh = patchPreviewImage.height * rectifiedScale;
        const rx = (rotatedCanvas.width - rw) / 2;
        const ry = (rotatedCanvas.height - rh) / 2;
        rotatedCtx.drawImage(patchPreviewImage, rx, ry, rw, rh);

        if (patch.crop_bbox) {
          const [cx, cy, cw, ch] = patch.crop_bbox;
          rotatedCtx.strokeStyle = '#59634c';
          rotatedCtx.lineWidth = 2;
          rotatedCtx.strokeRect(
            rx + cx * rectifiedScale,
            ry + cy * rectifiedScale,
            cw * rectifiedScale,
            ch * rectifiedScale,
          );
        }
      } else {
        rotatedCtx.save();
        rotatedCtx.translate(rotatedCanvas.width / 2, rotatedCanvas.height / 2);
        rotatedCtx.rotate((patch.angle_deg * Math.PI) / 180);
        rotatedCtx.drawImage(rawPatch, -pw / 2, -ph / 2, pw, ph);
        rotatedCtx.restore();

        if (patch.crop_bbox) {
          const [cx, cy, cw, ch] = patch.crop_bbox;
          rotatedCtx.strokeStyle = '#59634c';
          rotatedCtx.lineWidth = 2;
          rotatedCtx.strokeRect(
            (rotatedCanvas.width - pw) / 2 + cx * patchScale,
            (rotatedCanvas.height - ph) / 2 + cy * patchScale,
            cw * patchScale,
            ch * patchScale,
          );
        }
      }
    }

    function canvasPoint(event, canvas) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: ((event.clientX - rect.left) / rect.width) * canvas.width,
        y: ((event.clientY - rect.top) / rect.height) * canvas.height,
      };
    }

    function patchFromCanvasPoint(point) {
      if (!state) return null;
      const imagePoint = scanCanvasPointToImagePoint(point);
      if (!imagePoint) return null;
      const visiblePatches = state.patches.filter((patch) => patch.scan_index === selectedScanIndex);
      const selected = visiblePatches.find((patch) => patch.patch_id === selectedPatchId);
      if (selected && pointInPolygon(imagePoint, selected.corners)) return selected;
      const reversed = [...visiblePatches].reverse();
      const polygonHit = reversed.find((patch) => pointInPolygon(imagePoint, patch.corners));
      if (polygonHit) return polygonHit;
      return state.patches
        .filter((patch) => patch.scan_index === selectedScanIndex)
        .find((patch) => {
          const box = patch.display_bbox;
          return (
            imagePoint.x >= box[0] &&
            imagePoint.x <= box[0] + box[2] &&
            imagePoint.y >= box[1] &&
            imagePoint.y <= box[1] + box[3]
          );
        });
    }

    function selectedPatchDragTarget(point) {
      const patch = patchById(selectedPatchId);
      if (!patch || patch.scan_index !== selectedScanIndex) return null;
      const imagePoint = scanCanvasPointToImagePoint(point);
      if (!imagePoint) return null;
      const rotationHandle = rotationHandleForPatch(patch);
      if (handleHit(point, rotationHandle)) {
        return { mode: 'rotate', patch, imagePoint };
      }
      for (let index = 0; index < patch.corners.length; index += 1) {
        const corner = { x: patch.corners[index][0], y: patch.corners[index][1] };
        if (handleHit(point, corner)) {
          return { mode: 'resize-corner', patch, cornerIndex: index, imagePoint };
        }
      }
      if (pointInPolygon(imagePoint, patch.corners)) {
        return { mode: 'move', patch, imagePoint };
      }
      return null;
    }

    async function commitSelectedGeometry(statusText) {
      const patch = patchById(selectedPatchId);
      if (!patch) return;
      await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/geometry`, {
        method: 'POST',
        body: JSON.stringify({ rotated_rect: patch.rotated_rect }),
      });
      await refreshState(false);
      setStatus(statusText);
    }

    scanCanvas.addEventListener('pointerdown', async (event) => {
      if (!state || !scanImage) return;
      const pointer = canvasPoint(event, scanCanvas);
      const selectedTarget = selectedPatchDragTarget(pointer);
      if (selectedTarget) {
        selectedPatchId = selectedTarget.patch.patch_id;
        selectedScanIndex = selectedTarget.patch.scan_index;
        dragState = {
          patchId: selectedTarget.patch.patch_id,
          pointerId: event.pointerId,
          mode: selectedTarget.mode,
          cornerIndex: selectedTarget.cornerIndex ?? null,
          startPoint: selectedTarget.imagePoint,
          startRect: { ...selectedTarget.patch.rotated_rect },
          startCorners: selectedTarget.patch.corners.map((corner) => [...corner]),
          moved: false,
        };
        if (scanCanvas.setPointerCapture) scanCanvas.setPointerCapture(event.pointerId);
        const statusText = selectedTarget.mode === 'move'
          ? `正在拖动 ${selectedTarget.patch.patch_id}。松手后将自动更新校正后胶片。`
          : selectedTarget.mode === 'rotate'
            ? `正在旋转 ${selectedTarget.patch.patch_id}。松手后将自动更新校正后胶片。`
            : `正在缩放 ${selectedTarget.patch.patch_id}。松手后将自动更新校正后胶片。`;
        setStatus(statusText);
        renderAll();
        return;
      }
      const patch = patchFromCanvasPoint(pointer);
      if (!patch) return;
      selectPatch(patch.patch_id);
    });

    scanCanvas.addEventListener('pointermove', (event) => {
      if (!dragState || !state) return;
      const pointer = canvasPoint(event, scanCanvas);
      const imagePoint = scanCanvasPointToImagePoint(pointer);
      if (!imagePoint) return;
      const patch = patchById(dragState.patchId);
      if (!patch) return;
      const dx = imagePoint.x - dragState.startPoint.x;
      const dy = imagePoint.y - dragState.startPoint.y;
      if (Math.abs(dx) > 0.1 || Math.abs(dy) > 0.1) dragState.moved = true;
      if (dragState.mode === 'move') {
        syncClientPatchGeometry(patch, {
          ...dragState.startRect,
          cx: dragState.startRect.cx + dx,
          cy: dragState.startRect.cy + dy,
        });
      } else if (dragState.mode === 'rotate') {
        const center = { x: dragState.startRect.cx, y: dragState.startRect.cy };
        const startVector = subtractPoints(dragState.startPoint, center);
        const currentVector = subtractPoints(imagePoint, center);
        const startAngle = Math.atan2(startVector.y, startVector.x);
        const currentAngle = Math.atan2(currentVector.y, currentVector.x);
        const deltaDeg = ((currentAngle - startAngle) * 180) / Math.PI;
        syncClientPatchGeometry(patch, {
          ...dragState.startRect,
          angle_deg: dragState.startRect.angle_deg + deltaDeg,
        });
      } else if (dragState.mode === 'resize-corner') {
        const cornerIndex = dragState.cornerIndex ?? 0;
        const oppositeIndex = (cornerIndex + 2) % 4;
        const oppositeCorner = {
          x: dragState.startCorners[oppositeIndex][0],
          y: dragState.startCorners[oppositeIndex][1],
        };
        const center = {
          x: (oppositeCorner.x + imagePoint.x) / 2,
          y: (oppositeCorner.y + imagePoint.y) / 2,
        };
        const { ux, uy } = rotatedRectAxes(dragState.startRect);
        const diagonal = subtractPoints(imagePoint, oppositeCorner);
        const width = Math.max(1, Math.abs(dot(diagonal, ux)));
        const height = Math.max(1, Math.abs(dot(diagonal, uy)));
        syncClientPatchGeometry(patch, {
          ...dragState.startRect,
          cx: center.x,
          cy: center.y,
          width,
          height,
        });
      }
      if (patch.patch_id === selectedPatchId) {
        document.getElementById('geom-cx').value = patch.rotated_rect.cx.toFixed(1);
        document.getElementById('geom-cy').value = patch.rotated_rect.cy.toFixed(1);
        document.getElementById('geom-width').value = patch.rotated_rect.width.toFixed(1);
        document.getElementById('geom-height').value = patch.rotated_rect.height.toFixed(1);
        document.getElementById('geom-angle').value = patch.rotated_rect.angle_deg.toFixed(1);
        document.getElementById('angle-number').value = patch.rotated_rect.angle_deg.toFixed(1);
        document.getElementById('angle-range').value = patch.rotated_rect.angle_deg;
      }
      renderAll();
    });

    async function finishScanDrag(event) {
      if (!dragState) return;
      const patchId = dragState.patchId;
      const pointerId = dragState.pointerId;
      const didMove = dragState.moved;
      dragState = null;
      if (scanCanvas.releasePointerCapture && pointerId !== undefined) {
        try {
          scanCanvas.releasePointerCapture(pointerId);
        } catch (_) {}
      }
      selectedPatchId = patchId;
      if (!event || !("pointerId" in event) || !didMove) return;
      await commitSelectedGeometry(`已应用拖拽：${patchId}`);
    }

    scanCanvas.addEventListener('pointerup', (event) => {
      finishScanDrag(event).catch((error) => setStatus(error.message));
    });
    scanCanvas.addEventListener('pointercancel', (event) => {
      finishScanDrag(event).catch((error) => setStatus(error.message));
    });

    patchCanvas.addEventListener('click', async (event) => {
      if (!pointCaptureMode || !selectedPatchId) return;
      const patch = patchById(selectedPatchId);
      const box = patch.display_bbox;
      const w = Math.max(1, box[2]);
      const h = Math.max(1, box[3]);
      const patchScale = Math.min(patchCanvas.width / w, patchCanvas.height / h);
      const pw = w * patchScale;
      const ph = h * patchScale;
      const px = (patchCanvas.width - pw) / 2;
      const py = (patchCanvas.height - ph) / 2;
      const point = canvasPoint(event, patchCanvas);
      edgePoints.push([
        Math.max(0, Math.min(w, (point.x - px) / patchScale)),
        Math.max(0, Math.min(h, (point.y - py) / patchScale)),
      ]);
      if (edgePoints.length === 2) {
        await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/edge`, {
          method: 'POST',
          body: JSON.stringify({ edge_points: edgePoints }),
        });
        edgePoints = [];
        pointCaptureMode = false;
        await refreshState();
      }
    });

    viewerCropImage.addEventListener('load', () => {
      renderViewerCropOverlay(correctedEditorPatch());
    });

    viewerCropOverlay.addEventListener('pointerdown', (event) => {
      const patch = correctedEditorPatch();
      if (!patch || !cropEditMode || !canEditCropForPatch(patch)) return;
      const point = viewerCropLocalPoint(event);
      if (!point) return;
      const maxWidth = viewerCropImage.naturalWidth;
      const maxHeight = viewerCropImage.naturalHeight;
      const handle = cropHandleAtPoint(point, currentRenderedCropRect(patch));
      if (handle && cropEditDraft) {
        const [x, y, width, height] = cropEditDraft;
        const anchors = {
          nw: { x: x + width, y: y + height },
          ne: { x, y: y + height },
          se: { x, y },
          sw: { x: x + width, y },
        };
        cropDragState = { pointerId: event.pointerId, anchor: anchors[handle], mode: 'resize' };
      } else {
        cropEditDraft = normalizedCropRect([point.x, point.y, 1, 1], maxWidth, maxHeight);
        cropDragState = { pointerId: event.pointerId, anchor: point, mode: 'create' };
      }
      if (viewerCropOverlay.setPointerCapture) viewerCropOverlay.setPointerCapture(event.pointerId);
      renderViewerCropToolbar(patch);
      renderViewerCropOverlay(patch);
    });

    viewerCropOverlay.addEventListener('pointermove', (event) => {
      const patch = correctedEditorPatch();
      if (!patch || !cropDragState || !cropEditMode) return;
      const point = viewerCropLocalPoint(event);
      if (!point) return;
      cropEditDraft = cropRectFromPoints(
        cropDragState.anchor,
        point,
        viewerCropImage.naturalWidth,
        viewerCropImage.naturalHeight,
      );
      renderViewerCropToolbar(patch);
      renderViewerCropOverlay(patch);
    });

    function finishViewerCropDrag(event) {
      if (!cropDragState) return;
      const pointerId = cropDragState.pointerId;
      cropDragState = null;
      if (viewerCropOverlay.releasePointerCapture && pointerId !== undefined) {
        try {
          viewerCropOverlay.releasePointerCapture(pointerId);
        } catch (_) {}
      }
      renderViewerCropToolbar(correctedEditorPatch());
      renderViewerCropOverlay(correctedEditorPatch());
    }

    viewerCropOverlay.addEventListener('pointerup', finishViewerCropDrag);
    viewerCropOverlay.addEventListener('pointercancel', finishViewerCropDrag);

    document.getElementById('load-session').onclick = () => loadSession(false).catch((error) => setStatus(error.message));
    document.getElementById('redetect-session').onclick = () => redetectCurrentSession().catch((error) => setStatus(error.message));
    document.getElementById('refresh-state').onclick = () => refreshState().catch((error) => setStatus(error.message));
    document.getElementById('view-workflow').onclick = () => setViewMode('workflow');
    document.getElementById('view-expert').onclick = () => setViewMode('expert');
    document.getElementById('viewer-mode-raw').onclick = () => setViewerMode('raw');
    document.getElementById('viewer-mode-corrected').onclick = () => setViewerMode('corrected_grid');
    document.getElementById('viewer-mode-dose-overview').onclick = () => setViewerMode('dose_overview');
    document.getElementById('viewer-mode-dose-pseudocolor').onclick = () => setViewerMode('dose_pseudocolor');
    document.getElementById('viewer-crop-start').onclick = () => startCorrectedCropEditing();
    document.getElementById('viewer-crop-apply').onclick = () => applyCorrectedCropEdit().catch((error) => setStatus(error.message));
    document.getElementById('viewer-crop-clear').onclick = () => clearCorrectedCropEdit().catch((error) => setStatus(error.message));
    document.getElementById('viewer-crop-back').onclick = () => closeCorrectedCropEditor();
    viewerDosePalette.onchange = () => {
      dosePalette = viewerDosePalette.value || 'turbo';
      renderAll();
      syncWorkflowViewerAsync();
      prefetchWorkflowAssets();
    };
    document.getElementById('viewer-prev').onclick = () => navigateAssignedPatch(-1);
    document.getElementById('viewer-next').onclick = () => navigateAssignedPatch(1);
    document.getElementById('capture-edge').onclick = () => {
      pointCaptureMode = true;
      edgePoints = [];
      setStatus('请在左侧原始胶片上点击两个点来定义局部边缘。');
    };
    document.getElementById('apply-geometry').onclick = async () => {
      if (!selectedPatchId) return;
      const rotatedRect = {
        cx: Number(document.getElementById('geom-cx').value),
        cy: Number(document.getElementById('geom-cy').value),
        width: Number(document.getElementById('geom-width').value),
        height: Number(document.getElementById('geom-height').value),
        angle_deg: Number(document.getElementById('geom-angle').value),
      };
      const patch = patchById(selectedPatchId);
      if (!patch) return;
      syncClientPatchGeometry(patch, rotatedRect);
      await commitSelectedGeometry(`已应用几何修改：${selectedPatchId}`);
    };
    document.getElementById('reset-geometry').onclick = async () => {
      await refreshState();
    };
    document.getElementById('apply-angle').onclick = async () => {
      if (!selectedPatchId) return;
      const angleDeg = Number(document.getElementById('angle-number').value);
      await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/angle`, {
        method: 'POST',
        body: JSON.stringify({ angle_deg: angleDeg }),
      });
      await refreshState();
    };
    document.getElementById('apply-crop').onclick = async () => {
      if (!selectedPatchId) return;
      const crop_bbox = ['crop-x', 'crop-y', 'crop-w', 'crop-h'].map((id) => Number(document.getElementById(id).value));
      await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/crop`, {
        method: 'POST',
        body: JSON.stringify({ crop_bbox }),
      });
      await refreshState();
    };
    document.getElementById('quick-assign-apply').onclick = async () => {
      if (!selectedPatchId) return;
      const assignedOrder = Number(document.getElementById('quick-assign-order').value);
      if (!Number.isFinite(assignedOrder) || assignedOrder < 1) {
        setStatus('请先输入有效片序。');
        return;
      }
      await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/assignment`, {
        method: 'POST',
        body: JSON.stringify({ assignment_status: 'assigned', assigned_order: assignedOrder }),
      });
      await refreshState();
      flashAssignmentFeedback();
      setStatus(`已设定片序：${selectedPatchId} -> 第 ${assignedOrder} 片；若该片序已被占用，原占用者已改为未标记。`);
    };
    document.getElementById('quick-assign-order').addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      document.getElementById('quick-assign-apply').click();
    });
    document.getElementById('quick-mark-duplicate').onclick = async () => {
      if (!selectedPatchId) return;
      await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/assignment`, {
        method: 'POST',
        body: JSON.stringify({ assignment_status: 'ignored_duplicate' }),
      });
      await refreshState();
      setStatus(`已标记重复胶片：${selectedPatchId}`);
    };
    document.getElementById('quick-clear-assignment').onclick = async () => {
      if (!selectedPatchId) return;
      await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/assignment`, {
        method: 'POST',
        body: JSON.stringify({ assignment_status: 'unassigned' }),
      });
      await refreshState();
      setStatus(`已清除片序：${selectedPatchId}`);
    };
    document.getElementById('export-session').onclick = async () => {
      if (!sessionId) return;
      const payload = await api(`/api/rcf/session/${sessionId}/export`, { method: 'POST' });
      setStatus(`已导出\\n${payload.session_file}\\n${payload.review_file}`);
    };
    document.getElementById('angle-range').addEventListener('input', (event) => {
      document.getElementById('angle-number').value = event.target.value;
      document.getElementById('geom-angle').value = event.target.value;
    });
    document.getElementById('angle-number').addEventListener('input', (event) => {
      document.getElementById('angle-range').value = event.target.value;
      document.getElementById('geom-angle').value = event.target.value;
    });
    patchOverlayToggle.addEventListener('change', () => renderPatchEditor());
    setViewMode('workflow');
    window.addEventListener('beforeunload', () => {
      stopPolling();
      clearViewerObjectUrlCache();
    });

    function flashAssignmentFeedback() {
      const pill = document.getElementById('patch-assignment');
      pill.classList.add('flash-assignment');
      if (assignmentFlashToken) window.clearTimeout(assignmentFlashToken);
      assignmentFlashToken = window.setTimeout(() => {
        pill.classList.remove('flash-assignment');
        assignmentFlashToken = null;
      }, 900);
    }
    window.__detectorclawRcfGui = {
      getSessionId: () => sessionId,
      getState: () => state,
      refreshState: () => refreshState(false),
      debugMoveSelectedPatch: async (dx, dy) => {
        if (!state) return null;
        if (!selectedPatchId && state.patches.length) selectedPatchId = state.patches[0].patch_id;
        const patch = patchById(selectedPatchId);
        if (!patch) return null;
        syncClientPatchGeometry(patch, {
          ...patch.rotated_rect,
          cx: patch.rotated_rect.cx + Number(dx),
          cy: patch.rotated_rect.cy + Number(dy),
        });
        renderAll();
        await commitSelectedGeometry(`已应用拖拽：${selectedPatchId}`);
        return patch.rotated_rect;
      },
      debugRotateSelectedPatch: async (deltaAngleDeg) => {
        if (!state) return null;
        if (!selectedPatchId && state.patches.length) selectedPatchId = state.patches[0].patch_id;
        const patch = patchById(selectedPatchId);
        if (!patch) return null;
        syncClientPatchGeometry(patch, {
          ...patch.rotated_rect,
          angle_deg: patch.rotated_rect.angle_deg + Number(deltaAngleDeg),
        });
        renderAll();
        await commitSelectedGeometry(`已应用旋转：${selectedPatchId}`);
        return patch.rotated_rect;
      },
      debugResizeSelectedPatch: async (deltaWidth, deltaHeight) => {
        if (!state) return null;
        if (!selectedPatchId && state.patches.length) selectedPatchId = state.patches[0].patch_id;
        const patch = patchById(selectedPatchId);
        if (!patch) return null;
        syncClientPatchGeometry(patch, {
          ...patch.rotated_rect,
          width: patch.rotated_rect.width + Number(deltaWidth),
          height: patch.rotated_rect.height + Number(deltaHeight),
        });
        renderAll();
        await commitSelectedGeometry(`已应用缩放：${selectedPatchId}`);
        return patch.rotated_rect;
      },
      debugAssignSelectedPatchAsNext: async () => {
        if (!state) return null;
        if (!selectedPatchId && state.patches.length) selectedPatchId = state.patches[0].patch_id;
        const patch = patchById(selectedPatchId);
        if (!patch) return null;
        const assignedOrder = nextAssignableOrder();
        await api(`/api/rcf/session/${sessionId}/patch/${selectedPatchId}/assignment`, {
          method: 'POST',
          body: JSON.stringify({ assignment_status: 'assigned', assigned_order: assignedOrder }),
        });
        await refreshState();
        flashAssignmentFeedback();
        setStatus(`已设为下一片：${selectedPatchId} -> 第 ${assignedOrder} 片`);
        return assignedOrder;
      },
    };
  </script>
</body>
</html>"""


def create_app(*, disable_precompute: bool = False) -> FastAPI:
    app = FastAPI(title="DetectorClaw RCF GUI")
    store = SessionStore()
    app.state.store = store
    app.state.precompute_status = {}
    app.state.precompute_disabled = bool(disable_precompute)

    def session_summary_payload(state: dict) -> dict:
        summary = {
            "session_id": state["session_id"],
            "version_id": state["version_id"],
            "active_version_id": state["active_version_id"],
            "available_versions": state["available_versions"],
            "scan_count": len(state["scans"]),
            "patch_count": len(state["patches"]),
            "detection_mode": state["detection_mode"],
            "shot_id": state["shot_id"],
            "session_source": state.get("session_source", "new_detection"),
            "autosaved_at": state.get("autosaved_at"),
        }
        summary.update(
            {
                "config_file": state["config_file"],
                "config_source": state.get("config_source", "explicit"),
                "dose_available": state["dose_available"],
                "dose_error": state["dose_error"],
                "dose_config_source": state["dose_config_source"],
            }
        )
        summary["precompute_status"] = current_precompute_status(state)
        return summary

    def precompute_key(state: dict) -> tuple[str, str, int]:
        return (state["session_id"], state["version_id"], int(state["revision"]))

    def current_precompute_status(state: dict) -> dict:
        key = precompute_key(state)
        payload = app.state.precompute_status.get(key)
        if payload is not None:
            return dict(payload)
        return {
            "state": "idle",
            "stage": "idle",
            "session_id": state["session_id"],
            "version_id": state["version_id"],
            "revision": int(state["revision"]),
            "warmed_count": 0,
            "total_count": 0,
            "error": None,
            "interactive_queue": 0,
            "bulk_queue": 0,
            "inflight_batch": 0,
            "backend": "native",
        }

    def update_precompute_status(key: tuple[str, str, int], **fields: object) -> None:
        current = dict(
            app.state.precompute_status.get(
                key,
                {
                    "state": "idle",
                    "stage": "idle",
                    "session_id": key[0],
                    "version_id": key[1],
                    "revision": int(key[2]),
                    "warmed_count": 0,
                    "total_count": 0,
                    "error": None,
                    "interactive_queue": 0,
                    "bulk_queue": 0,
                    "inflight_batch": 0,
                    "backend": "native",
                },
            )
        )
        current.update(fields)
        app.state.precompute_status[key] = current

    def precompute_targets(state: dict) -> dict[str, list[dict]]:
        patches = list(state["patches"])
        scans = list(state["scans"])
        current_scan_index = patches[0]["scan_index"] if patches else (scans[0]["scan_index"] if scans else 1)
        current_scan_patches = [patch for patch in patches if patch["scan_index"] == current_scan_index]
        assigned = sorted(
            [
                patch
                for patch in patches
                if patch["assignment_status"] == "assigned" and patch.get("assigned_order") is not None
            ],
            key=lambda item: (item["assigned_order"], item["patch_id"]),
        )
        if assigned:
            dose_targets = assigned
            # For smooth navigation, fully warm all assigned dose single/high variants first.
            dose_single_targets = dose_targets
            high_res_targets = dose_targets
        else:
            dose_targets = current_scan_patches[: min(6, len(current_scan_patches))]
            # Keep unassigned sessions lightweight to avoid unnecessary background churn.
            dose_single_targets = dose_targets[: min(2, len(dose_targets))]
            high_res_targets = dose_targets[: min(1, len(dose_targets))]
        raw_targets = current_scan_patches[:1]
        return {
            "scans": scans,
            "corrected": current_scan_patches,
            "raw": raw_targets,
            "dose": dose_targets,
            "dose_single": dose_single_targets,
            "dose_high": high_res_targets,
        }

    def run_session_precompute(key: tuple[str, str, int]) -> None:
        update_precompute_status(key, state="running", stage="bootstrap", error=None, warmed_count=0, total_count=0)
        try:
            session = store.get_session(key[0])
            state = store.serialize_session(session)
            state.update(_dose_metadata(session))
            if precompute_key(state) != key:
                update_precompute_status(key, state="superseded")
                return
            targets = precompute_targets(state)
            total_count = (
                len(targets["scans"])
                + len(targets["corrected"])
                + len(targets["raw"])
                + len(targets["dose"])
                + len(targets["dose"]) * len(DOSE_PSEUDOCOLOR_PALETTES)
                + len(targets["dose_single"]) * len(DOSE_PSEUDOCOLOR_PALETTES)
                + len(targets["dose_high"]) * len(DOSE_PSEUDOCOLOR_PALETTES)
            )
            interactive_queue = (
                len(targets["dose_single"]) * len(DOSE_PSEUDOCOLOR_PALETTES)
                + len(targets["dose_high"]) * len(DOSE_PSEUDOCOLOR_PALETTES)
            )
            bulk_queue = len(targets["dose"]) + len(targets["dose"]) * len(DOSE_PSEUDOCOLOR_PALETTES)
            update_precompute_status(
                key,
                total_count=total_count,
                interactive_queue=interactive_queue,
                bulk_queue=bulk_queue,
                inflight_batch=0,
                backend="native",
            )
            warmed_count = 0

            def tick() -> None:
                nonlocal warmed_count
                warmed_count += 1
                update_precompute_status(key, warmed_count=warmed_count)

            update_precompute_status(key, stage="scan")
            for scan in targets["scans"]:
                _ensure_scan_preview_cached(state, scan, max_dim=1600, preview_format="jpeg", quality=80)
                tick()
            update_precompute_status(key, stage="corrected")
            for patch in targets["corrected"]:
                _ensure_patch_preview_cached(state, patch, max_dim=320, preview_format="jpeg", quality=82)
                tick()
            update_precompute_status(key, stage="raw")
            for patch in targets["raw"]:
                _ensure_raw_patch_preview_cached(state, patch, max_dim=960, preview_format="jpeg", quality=82)
                tick()

            if state["dose_available"]:
                update_precompute_status(key, stage="dose-single")
                single_variants = [
                    {
                        "patch": patch,
                        "palette": palette,
                        "max_dim": DOSE_SINGLE_PREVIEW_DIM,
                        "preview_format": "jpeg",
                        "quality": DOSE_PREVIEW_JPEG_QUALITY,
                    }
                    for patch in targets["dose_single"]
                    for palette in DOSE_PSEUDOCOLOR_PALETTES
                ]
                _single_native_warmed, single_used_native = _prewarm_dose_variants_native(
                    session,
                    state,
                    single_variants,
                )
                if not single_used_native:
                    for variant in single_variants:
                        _ensure_dose_preview_cached(
                            session,
                            state,
                            variant["patch"],
                            palette=variant["palette"],
                            max_dim=variant["max_dim"],
                            preview_format=variant["preview_format"],
                            quality=variant["quality"],
                        )
                for _ in single_variants:
                    tick()
                update_precompute_status(key, stage="dose-high")
                high_variants = [
                    {
                        "patch": patch,
                        "palette": palette,
                        "max_dim": DOSE_HIGH_RES_PREVIEW_DIM,
                        "preview_format": "jpeg",
                        "quality": DOSE_PREVIEW_JPEG_QUALITY,
                    }
                    for patch in targets["dose_high"]
                    for palette in DOSE_PSEUDOCOLOR_PALETTES
                ]
                _high_native_warmed, high_used_native = _prewarm_dose_variants_native(
                    session,
                    state,
                    high_variants,
                )
                if not high_used_native:
                    for variant in high_variants:
                        _ensure_dose_preview_cached(
                            session,
                            state,
                            variant["patch"],
                            palette=variant["palette"],
                            max_dim=variant["max_dim"],
                            preview_format=variant["preview_format"],
                            quality=variant["quality"],
                        )
                for _ in high_variants:
                    tick()
                update_precompute_status(key, stage="dose-overview")
                for patch in targets["dose"]:
                    _ensure_dose_stats_cached(session, state, patch)
                    tick()
                overview_variants = [
                    {
                        "patch": patch,
                        "palette": palette,
                        "max_dim": DOSE_OVERVIEW_PREVIEW_DIM,
                        "preview_format": "jpeg",
                        "quality": DOSE_PREVIEW_JPEG_QUALITY,
                    }
                    for patch in targets["dose"]
                    for palette in DOSE_PSEUDOCOLOR_PALETTES
                ]
                _overview_native_warmed, overview_used_native = _prewarm_dose_variants_native(
                    session,
                    state,
                    overview_variants,
                )
                if not overview_used_native:
                    for variant in overview_variants:
                        _ensure_dose_preview_cached(
                            session,
                            state,
                            variant["patch"],
                            palette=variant["palette"],
                            max_dim=variant["max_dim"],
                            preview_format=variant["preview_format"],
                            quality=variant["quality"],
                        )
                for _ in overview_variants:
                    tick()

            update_precompute_status(
                key,
                state="done",
                stage="done",
                warmed_count=total_count,
                total_count=total_count,
                error=None,
                interactive_queue=0,
                bulk_queue=0,
                inflight_batch=0,
            )
        except Exception as exc:  # noqa: BLE001
            update_precompute_status(
                key,
                state="error",
                stage="error",
                error=str(exc),
                interactive_queue=0,
                bulk_queue=0,
                inflight_batch=0,
            )

    def enqueue_session_precompute(session: dict) -> None:
        if app.state.precompute_disabled:
            return
        state = store.serialize_session(session)
        state.update(_dose_metadata(session))
        key = precompute_key(state)
        current = app.state.precompute_status.get(key)
        if current and current.get("state") in {"queued", "running", "done"}:
            return
        update_precompute_status(
            key,
            state="queued",
            stage="queued",
            error=None,
            warmed_count=0,
            total_count=0,
            interactive_queue=0,
            bulk_queue=0,
            inflight_batch=0,
            backend="native",
        )
        threading.Thread(target=run_session_precompute, args=(key,), daemon=True).start()

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/rcf/gui", response_class=HTMLResponse)
    def gui_page() -> str:
        return _gui_html()

    @app.post("/api/rcf/session/load")
    def load_session(request: LoadSessionRequest) -> dict:
        try:
            session = store.create_session(
                input_files=[Path(path) for path in request.input_files] if request.input_files else None,
                config_file=Path(request.config_file) if request.config_file else None,
                output_dir=Path(request.output_dir) if request.output_dir else None,
                stack_config_file=Path(request.stack_config_file) if request.stack_config_file else None,
                detection_mode=request.detection_mode,
                shot_id=request.shot_id,
                data_root=Path(request.data_root) if request.data_root else None,
                force_redetect=request.force_redetect,
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        state = store.serialize_session(session)
        state.update(_dose_metadata(session))
        enqueue_session_precompute(session)
        return session_summary_payload(state)

    @app.post("/api/rcf/session/{session_id}/redetect")
    def redetect_session(session_id: str) -> dict:
        try:
            session = store.redetect_session(session_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        state = store.serialize_session(session)
        state.update(_dose_metadata(session))
        enqueue_session_precompute(session)
        return session_summary_payload(state)

    @app.post("/api/rcf/session/{session_id}/versions/{version_id}/activate")
    def activate_version(session_id: str, version_id: str) -> dict:
        try:
            result = store.activate_version(session_id, version_id)
            enqueue_session_precompute(store.get_session(session_id))
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/rcf/session/{session_id}/state")
    def get_state(session_id: str) -> dict:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        state.update(_dose_metadata(session))
        state["precompute_status"] = current_precompute_status(state)
        return state

    @app.get("/api/rcf/session/{session_id}/precompute/status")
    def get_precompute_status(session_id: str) -> dict:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return current_precompute_status(state)

    @app.post("/api/rcf/session/{session_id}/precompute/start")
    def start_precompute(session_id: str) -> dict:
        try:
            session = store.get_session(session_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if app.state.precompute_disabled:
            raise HTTPException(status_code=409, detail="Session precompute is disabled for this app instance")
        enqueue_session_precompute(session)
        state = store.serialize_session(session)
        return current_precompute_status(state)

    @app.get("/api/rcf/session/{session_id}/assets/manifest")
    def get_asset_manifest(session_id: str) -> dict:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        assigned = sorted(
            [
                patch
                for patch in state["patches"]
                if patch.get("assignment_status") == "assigned" and patch.get("assigned_order") is not None
            ],
            key=lambda patch: (patch.get("assigned_order"), patch["patch_id"]),
        )
        patch_payload = []
        for patch in assigned:
            variants = []
            for variant in DOSE_ASSET_VARIANTS:
                asset_id = _dose_asset_id(
                    patch_id=patch["patch_id"],
                    modified_revision=patch.get("modified_revision", 0),
                    palette=variant["palette"],
                    max_dim=variant["max_dim"],
                    preview_format=variant["format"],
                    quality=variant["quality"],
                )
                cache_key = _dose_preview_cache_key(
                    state,
                    patch,
                    variant["palette"],
                    variant["max_dim"],
                    variant["format"],
                    variant["quality"],
                )
                ready_in_memory = PREVIEW_CACHE.get(cache_key) is not None
                ready_on_disk = _dose_disk_cache_available(
                    state,
                    patch,
                    variant["palette"],
                    variant["max_dim"],
                    variant["format"],
                    variant["quality"],
                )
                variants.append(
                    {
                        "variant_id": variant["variant_id"],
                        "ready": ready_in_memory or ready_on_disk,
                        "asset_id": asset_id,
                        "url": f"/api/rcf/session/{session_id}/assets/{asset_id}",
                    }
                )
            patch_payload.append(
                {
                    "patch_id": patch["patch_id"],
                    "assigned_order": patch.get("assigned_order"),
                    "display_label": patch.get("display_label") or (f"第 {patch.get('assigned_order')} 片"),
                    "variants": variants,
                }
            )
        return {
            "session_id": state["session_id"],
            "version_id": state["version_id"],
            "revision": int(state["revision"]),
            "assigned_patch_count": len(patch_payload),
            "precompute_status": current_precompute_status(state),
            "patches": patch_payload,
        }

    @app.get("/api/rcf/session/{session_id}/assets/{asset_id}")
    def get_asset(session_id: str, asset_id: str) -> Response:
        try:
            decoded = _parse_dose_asset_id(asset_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
            patch = next(item for item in state["patches"] if item["patch_id"] == decoded["patch_id"])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        if int(decoded["modified_revision"]) != int(patch.get("modified_revision", 0)):
            raise HTTPException(status_code=409, detail="Asset revision is stale")

        try:
            normalized_palette = _normalize_dose_palette(str(decoded["palette"]))
            preview_format = preview.normalize_preview_format(str(decoded["format"]), default="jpeg")
            max_dim = int(decoded["max_dim"])
            quality = int(decoded["quality"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        cache_key = _dose_preview_cache_key(state, patch, normalized_palette, max_dim, preview_format, quality)
        cached = PREVIEW_CACHE.get(cache_key)
        if cached is None:
            cached = _dose_disk_cache_load(state, patch, normalized_palette, max_dim, preview_format, quality)
            if cached is not None:
                PREVIEW_CACHE[cache_key] = cached
        if cached is None:
            raise HTTPException(status_code=425, detail="Dose asset is warming")
        content, media_type = cached
        return Response(content=content, media_type=media_type)

    @app.get("/api/rcf/session/{session_id}/scan/{scan_index}/image")
    def get_scan_image(
        session_id: str,
        scan_index: int,
        max_dim: int = preview.DEFAULT_SCAN_MAX_DIM,
        format: str = "png",
        quality: int = 80,
    ) -> Response:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            preview_format = preview.normalize_preview_format(format, default="png")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        for scan in state["scans"]:
            if scan["scan_index"] == scan_index:
                cache_key = ("scan", state["version_id"], scan_index, max_dim, preview_format, quality)
                cached = PREVIEW_CACHE.get(cache_key)
                if cached is not None:
                    content, media_type = cached
                    return Response(content=content, media_type=media_type)
                content, media_type = preview.render_scan_preview(
                    scan_file=Path(scan["scan_file"]),
                    max_dim=max_dim,
                    preview_format=preview_format,
                    quality=quality,
                )
                PREVIEW_CACHE[cache_key] = (content, media_type)
                return Response(content=content, media_type=media_type)
        raise HTTPException(status_code=404, detail=f"Unknown scan index: {scan_index}")

    @app.get("/api/rcf/session/{session_id}/patch/{patch_id}/image")
    def get_patch_image(
        session_id: str,
        patch_id: str,
        max_dim: int = preview.DEFAULT_PATCH_MAX_DIM,
        format: str = "png",
        quality: int = 80,
        ignore_crop: bool = False,
    ) -> Response:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
            patch = next(item for item in state["patches"] if item["patch_id"] == patch_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            preview_format = preview.normalize_preview_format(format, default="png")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        crop_bbox = None if ignore_crop else patch.get("crop_bbox")
        cache_key = (
            "patch",
            state["version_id"],
            patch_id,
            patch.get("modified_revision", 0),
            tuple(crop_bbox or []),
            max_dim,
            preview_format,
            quality,
            bool(ignore_crop),
        )
        cached = PREVIEW_CACHE.get(cache_key)
        if cached is not None:
            content, media_type = cached
            return Response(content=content, media_type=media_type)
        if patch.get("source_quad"):
            quad_points = patch["source_quad"]
        else:
            serialized_patch = next(item for item in state["patches"] if item["patch_id"] == patch_id)
            quad_points = serialized_patch["corners"]
        content, media_type = preview.render_patch_preview(
            scan_file=Path(patch["scan_file"]),
            quad_points=quad_points,
            crop_bbox=crop_bbox,
            max_dim=max_dim,
            preview_format=preview_format,
            quality=quality,
        )
        PREVIEW_CACHE[cache_key] = (content, media_type)
        return Response(content=content, media_type=media_type)

    @app.get("/api/rcf/session/{session_id}/patch/{patch_id}/raw-image")
    def get_raw_patch_image(
        session_id: str,
        patch_id: str,
        max_dim: int = preview.DEFAULT_RAW_PATCH_MAX_DIM,
        format: str = "png",
        quality: int = 80,
    ) -> Response:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
            patch = next(item for item in state["patches"] if item["patch_id"] == patch_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            preview_format = preview.normalize_preview_format(format, default="png")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        bbox = [int(value) for value in patch["display_bbox"]]
        cache_key = (
            "raw-patch",
            state["version_id"],
            patch_id,
            patch.get("modified_revision", 0),
            tuple(bbox),
            max_dim,
            preview_format,
            quality,
        )
        cached = PREVIEW_CACHE.get(cache_key)
        if cached is not None:
            content, media_type = cached
            return Response(content=content, media_type=media_type)

        content, media_type = preview.render_bbox_preview(
            scan_file=Path(patch["scan_file"]),
            bbox=bbox,
            max_dim=max_dim,
            preview_format=preview_format,
            quality=quality,
        )
        PREVIEW_CACHE[cache_key] = (content, media_type)
        return Response(content=content, media_type=media_type)

    @app.get("/api/rcf/session/{session_id}/patch/{patch_id}/dose-image")
    def get_dose_patch_image(
        session_id: str,
        patch_id: str,
        palette: str = "gray",
        max_dim: int = preview.DEFAULT_PATCH_MAX_DIM,
        format: str = "jpeg",
        quality: int = DOSE_PREVIEW_JPEG_QUALITY,
        cache_only: bool = False,
    ) -> Response:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
            patch = next(item for item in state["patches"] if item["patch_id"] == patch_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        dose_available, dose_error = _dose_status(session)
        if not dose_available:
            raise HTTPException(status_code=400, detail=dose_error or "Dose preview is unavailable")
        try:
            preview_format = preview.normalize_preview_format(format, default="jpeg")
            normalized_palette = _normalize_dose_palette(palette)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        cache_key = _dose_preview_cache_key(state, patch, normalized_palette, max_dim, preview_format, quality)
        cached = PREVIEW_CACHE.get(cache_key)
        if cached is not None:
            content, media_type = cached
            return Response(content=content, media_type=media_type)
        cached = _dose_disk_cache_load(state, patch, normalized_palette, max_dim, preview_format, quality)
        if cached is not None:
            PREVIEW_CACHE[cache_key] = cached
            content, media_type = cached
            return Response(content=content, media_type=media_type)
        if cache_only:
            raise HTTPException(status_code=425, detail="Dose preview cache miss")

        native_variant = {
            "patch": patch,
            "palette": normalized_palette,
            "max_dim": max_dim,
            "preview_format": preview_format,
            "quality": quality,
        }
        _native_warmed, native_used = _prewarm_dose_variants_native(session, state, [native_variant])
        if native_used:
            cached = PREVIEW_CACHE.get(cache_key)
            if cached is None:
                cached = _dose_disk_cache_load(state, patch, normalized_palette, max_dim, preview_format, quality)
                if cached is not None:
                    PREVIEW_CACHE[cache_key] = cached
            if cached is not None:
                content, media_type = cached
                return Response(content=content, media_type=media_type)

        try:
            image = _dose_preview_image(session, patch, normalized_palette, max_dim=max_dim)
            image = preview.resize_for_preview(image, max_dim)
            content, media_type = preview.encode_preview_image(image, preview_format, quality)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        PREVIEW_CACHE[cache_key] = (content, media_type)
        _dose_disk_cache_store(
            state,
            patch,
            normalized_palette,
            max_dim,
            preview_format,
            quality,
            content=content,
            media_type=media_type,
        )
        return Response(content=content, media_type=media_type)

    @app.get("/api/rcf/session/{session_id}/patch/{patch_id}/dose-export")
    def get_dose_patch_export(
        session_id: str,
        patch_id: str,
        palette: str = "gray",
        max_dim: int = 0,
        format: str = "tiff",
    ) -> Response:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
            patch = next(item for item in state["patches"] if item["patch_id"] == patch_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        dose_available, dose_error = _dose_status(session)
        if not dose_available:
            raise HTTPException(status_code=400, detail=dose_error or "Dose export is unavailable")
        try:
            normalized_palette = _normalize_dose_palette(palette)
            export_format = _normalize_dose_export_format(format)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            image = _dose_preview_image(session, patch, normalized_palette)
            image = preview.resize_for_preview(image, max_dim)
            content, media_type = _encode_dose_export_image(image, export_format)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(content=content, media_type=media_type)

    @app.get("/api/rcf/session/{session_id}/patch/{patch_id}/dose-stats")
    def get_dose_patch_stats(session_id: str, patch_id: str) -> dict:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
            patch = next(item for item in state["patches"] if item["patch_id"] == patch_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        cache_key = _dose_stats_cache_key(state, patch)
        cached = DOSE_STATS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            payload = _dose_stats_payload(session, patch)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        DOSE_STATS_CACHE[cache_key] = payload
        return payload

    @app.get("/api/rcf/session/{session_id}/dose-overview-prewarm")
    def prewarm_dose_overview(
        session_id: str,
        palette: str = "gray",
        max_dim: int = DOSE_OVERVIEW_PREVIEW_DIM,
        format: str = "jpeg",
        quality: int = DOSE_PREVIEW_JPEG_QUALITY,
    ) -> dict:
        try:
            session = store.get_session(session_id)
            state = store.serialize_session(session)
            assigned = [
                patch
                for patch in state["patches"]
                if patch["assignment_status"] == "assigned" and patch.get("assigned_order") is not None
            ]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        dose_available, dose_error = _dose_status(session)
        if not dose_available:
            raise HTTPException(status_code=400, detail=dose_error or "Dose preview is unavailable")
        try:
            preview_format = preview.normalize_preview_format(format, default="jpeg")
            normalized_palette = _normalize_dose_palette(palette)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        warmed = _prewarm_dose_overview_cache(
            session,
            state,
            assigned,
            palette=normalized_palette,
            max_dim=max_dim,
            preview_format=preview_format,
            quality=quality,
        )
        return {"patch_count": warmed, "palette": normalized_palette}

    @app.post("/api/rcf/session/{session_id}/patch/{patch_id}/geometry")
    def update_geometry(session_id: str, patch_id: str, request: GeometryRequest) -> dict:
        try:
            session = store.get_session(session_id)
            result = store.update_patch_geometry(session, patch_id, request.rotated_rect)
            enqueue_session_precompute(session)
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rcf/session/{session_id}/patch/{patch_id}/edge")
    def update_edge(session_id: str, patch_id: str, request: EdgeRequest) -> dict:
        try:
            session = store.get_session(session_id)
            result = store.update_patch_edge(session, patch_id, request.edge_points)
            enqueue_session_precompute(session)
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rcf/session/{session_id}/patch/{patch_id}/angle")
    def update_angle(session_id: str, patch_id: str, request: AngleRequest) -> dict:
        try:
            session = store.get_session(session_id)
            result = store.update_patch_angle(session, patch_id, request.angle_deg)
            enqueue_session_precompute(session)
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rcf/session/{session_id}/patch/{patch_id}/crop")
    def update_crop(session_id: str, patch_id: str, request: CropRequest) -> dict:
        try:
            session = store.get_session(session_id)
            active_state = store.serialize_session(session)
            patch = next((item for item in active_state["patches"] if item["patch_id"] == patch_id), None)
            if patch is None:
                raise KeyError(f"Unknown patch: {patch_id}")
            normalized_crop = _normalize_crop_bbox_for_patch(patch, request.crop_bbox)
            result = store.update_patch_crop(session, patch_id, normalized_crop)
            enqueue_session_precompute(session)
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rcf/session/{session_id}/patch/{patch_id}/assignment")
    def update_assignment(session_id: str, patch_id: str, request: AssignmentRequest) -> dict:
        try:
            session = store.get_session(session_id)
            result = store.update_patch_assignment(
                session,
                patch_id,
                request.assignment_status,
                request.assigned_order,
            )
            enqueue_session_precompute(session)
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rcf/session/{session_id}/order")
    def update_order(session_id: str, request: OrderRequest) -> dict:
        try:
            session = store.get_session(session_id)
            result = store.reorder_patches(session, request.patch_ids)
            enqueue_session_precompute(session)
            return result
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/rcf/session/{session_id}/export")
    def export_session(session_id: str) -> dict:
        try:
            return store.export_session(store.get_session(session_id))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
