from __future__ import annotations

import ctypes
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable
import copy

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.feature import canny
from skimage.measure import label as sk_label
from skimage.measure import regionprops
from skimage.transform import probabilistic_hough_line
try:
    import torch
    import torch.nn.functional as torch_f
except Exception:  # noqa: BLE001
    torch = None
    torch_f = None

from .io import load_rgb_image
from .runtime_cache import LRUCache
from .runtime_cache import file_signature

NATIVE_SEGMENT_ENV = "DETECTORCLAW_RCF_NATIVE_SEGMENT_BIN"
NATIVE_SEGMENT_LIB_ENV = "DETECTORCLAW_RCF_NATIVE_SEGMENT_LIB"
SEGMENT_BACKEND_ENV = "DETECTORCLAW_RCF_SEGMENT_BACKEND"
_NATIVE_SEGMENT_LIBRARY = None
_SEGMENT_RESULT_CACHE = LRUCache[dict](max_entries=64)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, component_count = ndimage.label(mask)
    if component_count <= 0:
        return np.zeros_like(mask, dtype=bool)
    component_sizes = ndimage.sum(mask, labels, index=range(1, component_count + 1))
    largest_id = int(np.argmax(component_sizes)) + 1
    return labels == largest_id


def _border_pixels(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    border = max(4, min(height, width) // 20)
    top = image_rgb[:border, :, :]
    bottom = image_rgb[-border:, :, :]
    left = image_rgb[:, :border, :]
    right = image_rgb[:, -border:, :]
    return np.concatenate((top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)))


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available() and torch_f is not None)


def _normalize_segment_backend(config: dict | None = None) -> str:
    configured = None if config is None else config.get("backend")
    backend = (configured or os.environ.get(SEGMENT_BACKEND_ENV) or "cpu").lower()
    if backend not in {"cpu", "auto", "cuda"}:
        raise ValueError("segmentation backend must be cpu, auto, or cuda")
    if backend == "cuda" and not _cuda_available():
        return "cpu"
    if backend == "auto":
        return "cuda" if _cuda_available() else "cpu"
    return backend


def _compute_sheet_mask_cpu(image_rgb: np.ndarray) -> np.ndarray:
    image_float = image_rgb.astype(np.float64)
    gray = image_float.mean(axis=2)

    border_pixels = _border_pixels(image_rgb).astype(np.float64)
    background_color = np.median(border_pixels, axis=0)
    border_distance = np.linalg.norm(border_pixels - background_color, axis=1)
    color_distance = np.linalg.norm(image_float - background_color, axis=2)
    color_threshold = max(8.0, float(np.percentile(border_distance, 99)) * 3.0)
    tone_mask = color_distance >= color_threshold

    smoothed = ndimage.gaussian_filter(gray, sigma=1.0)
    sobel_x = ndimage.sobel(smoothed, axis=1)
    sobel_y = ndimage.sobel(smoothed, axis=0)
    edge_magnitude = np.hypot(sobel_x, sobel_y)
    edge_threshold = max(4.0, float(np.percentile(edge_magnitude, 97)))
    edge_mask = edge_magnitude >= edge_threshold
    edge_mask = ndimage.binary_dilation(edge_mask, structure=np.ones((3, 3)), iterations=1)
    edge_mask = ndimage.binary_closing(edge_mask, structure=np.ones((9, 9)), iterations=1)
    edge_mask = ndimage.binary_fill_holes(edge_mask)

    mask = tone_mask | edge_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndimage.binary_fill_holes(mask)
    return mask


def _compute_sheet_mask_cuda(image_rgb: np.ndarray) -> np.ndarray:
    if not _cuda_available():
        return _compute_sheet_mask_cpu(image_rgb)

    device = torch.device("cuda")
    image_float = torch.as_tensor(np.array(image_rgb, copy=True), device=device, dtype=torch.float32)
    gray = image_float.mean(dim=2)

    border_pixels = torch.as_tensor(np.array(_border_pixels(image_rgb), copy=True), device=device, dtype=torch.float32)
    background_color = border_pixels.median(dim=0).values
    border_distance = torch.linalg.norm(border_pixels - background_color, dim=1)
    color_distance = torch.linalg.norm(image_float - background_color.view(1, 1, 3), dim=2)
    color_threshold = max(8.0, float(torch.quantile(border_distance, 0.99).item()) * 3.0)
    tone_mask = (color_distance >= color_threshold).detach().cpu().numpy()

    gaussian_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=torch.float32)
    gaussian_2d = (gaussian_1d[:, None] * gaussian_1d[None, :]) / gaussian_1d.sum().pow(2)
    gray_4d = gray.unsqueeze(0).unsqueeze(0)
    smoothed = torch_f.conv2d(gray_4d, gaussian_2d.view(1, 1, 5, 5), padding=2)

    sobel_x_kernel = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    sobel_y_kernel = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).view(1, 1, 3, 3)
    sobel_x = torch_f.conv2d(smoothed, sobel_x_kernel, padding=1)
    sobel_y = torch_f.conv2d(smoothed, sobel_y_kernel, padding=1)
    edge_magnitude = torch.sqrt(sobel_x.square() + sobel_y.square()).squeeze(0).squeeze(0)
    edge_threshold = max(4.0, float(torch.quantile(edge_magnitude.reshape(-1), 0.97).item()))
    edge_mask = (edge_magnitude >= edge_threshold).detach().cpu().numpy()
    edge_mask = ndimage.binary_dilation(edge_mask, structure=np.ones((3, 3)), iterations=1)
    edge_mask = ndimage.binary_closing(edge_mask, structure=np.ones((9, 9)), iterations=1)
    edge_mask = ndimage.binary_fill_holes(edge_mask)

    mask = tone_mask | edge_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndimage.binary_fill_holes(mask)
    return mask


def _compute_sheet_mask(image_rgb: np.ndarray, backend: str = "cpu") -> np.ndarray:
    if _normalize_segment_backend({"backend": backend}) == "cuda":
        return _compute_sheet_mask_cuda(image_rgb)
    return _compute_sheet_mask_cpu(image_rgb)


def _extract_patch(image_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, width, height = bbox
    return image_rgb[y : y + height, x : x + width]


def _clip_bbox(x: int, y: int, width: int, height: int, image_shape: tuple[int, int, int]) -> list[int]:
    max_height, max_width = image_shape[:2]
    x = max(0, x)
    y = max(0, y)
    width = min(width, max_width - x)
    height = min(height, max_height - y)
    return [int(x), int(y), int(width), int(height)]


def _sort_key(patch: dict, sort_mode: str) -> tuple[int, int]:
    x, y, _, _ = patch["bbox"]
    if sort_mode == "xy":
        return (x, y)
    return (y, x)


def _normalize_angle(angle_deg: float) -> float:
    normalized = ((angle_deg + 90.0) % 180.0) - 90.0
    if normalized == -90.0:
        return 90.0
    return normalized


def _fold_rect_angle(angle_deg: float) -> float:
    normalized = _normalize_angle(angle_deg)
    if normalized > 45.0:
        return normalized - 90.0
    if normalized < -45.0:
        return normalized + 90.0
    return normalized


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    midpoint = sorted_weights.sum() / 2.0
    index = int(np.searchsorted(cumulative, midpoint, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _moment_pca_angle(component_mask: np.ndarray) -> float:
    ys, xs = np.nonzero(component_mask)
    if len(xs) < 2:
        return 0.0
    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle_deg = np.degrees(np.arctan2(principal_vector[1], principal_vector[0]))
    return float(_fold_rect_angle(float(angle_deg)))


def _estimate_patch_background(patch_rgb: np.ndarray, border_width: int = 24) -> np.ndarray:
    height, width = patch_rgb.shape[:2]
    border_width = max(4, min(border_width, height // 4, width // 4))
    if border_width <= 0:
        return np.median(patch_rgb.reshape(-1, 3), axis=0).astype(np.float64)
    border_pixels = np.concatenate(
        (
            patch_rgb[:border_width, :, :].reshape(-1, 3),
            patch_rgb[-border_width:, :, :].reshape(-1, 3),
            patch_rgb[:, :border_width, :].reshape(-1, 3),
            patch_rgb[:, -border_width:, :].reshape(-1, 3),
        )
    )
    return np.median(border_pixels.astype(np.float64), axis=0)


def _compute_patch_film_mask(patch_rgb: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    patch_float = patch_rgb.astype(np.float64)
    background_rgb = np.asarray(background_rgb, dtype=np.float64)

    color_distance = np.linalg.norm(patch_float - background_rgb, axis=2)
    border_distance = np.linalg.norm(_border_pixels(patch_rgb).astype(np.float64) - background_rgb, axis=1)
    color_threshold = max(4.0, float(np.percentile(border_distance, 98)) * 2.5)
    tone_mask = color_distance >= color_threshold

    luminance = patch_float.mean(axis=2)
    local_contrast = np.abs(luminance - ndimage.gaussian_filter(luminance, sigma=2.0))
    contrast_threshold = max(1.5, float(np.percentile(local_contrast, 92)))
    contrast_mask = local_contrast >= contrast_threshold

    edge_mask = canny(color_distance / max(1.0, color_distance.max()), sigma=1.0)
    edge_mask = ndimage.binary_dilation(edge_mask, structure=np.ones((3, 3)), iterations=1)

    film_mask = tone_mask | (contrast_mask & ndimage.binary_dilation(tone_mask, structure=np.ones((5, 5))))
    film_mask |= edge_mask & ndimage.binary_dilation(tone_mask, structure=np.ones((7, 7)))
    film_mask = ndimage.binary_opening(film_mask, structure=np.ones((3, 3)))
    film_mask = ndimage.binary_closing(film_mask, structure=np.ones((5, 5)), iterations=2)
    film_mask = ndimage.binary_fill_holes(film_mask)
    film_mask = _largest_component(film_mask)
    return film_mask.astype(bool)


def _estimate_legacy_patch_geometry(component_mask: np.ndarray) -> dict:
    height, width = component_mask.shape
    shorter_side = max(1, min(height, width))
    aspect_ratio = max(height, width) / shorter_side

    labeled = sk_label(component_mask.astype(np.uint8))
    props = regionprops(labeled)
    if not props:
        return {
            "angle_deg": 0.0,
            "angle_confidence": 0.0,
            "angle_source": "low_confidence_zero",
            "status_flags": ["low_confidence_angle"],
        }

    prop = props[0]
    # skimage orientation is relative to image rows; fold to the rectangle tilt domain.
    region_angle = _fold_rect_angle(90.0 - np.degrees(prop.orientation))
    moment_angle = _moment_pca_angle(component_mask)
    region_confidence = float(np.clip((aspect_ratio - 1.0) / 0.35, 0.0, 1.0))

    boundary = np.logical_xor(component_mask, ndimage.binary_erosion(component_mask, structure=np.ones((3, 3))))
    edge_map = canny(boundary.astype(float), sigma=1.0)
    line_length = max(20, min(height, width) // 4)
    lines = probabilistic_hough_line(edge_map, threshold=5, line_length=line_length, line_gap=8)

    hough_angle = None
    hough_confidence = 0.0
    if lines:
        folded_angles = []
        lengths = []
        for point_0, point_1 in lines:
            delta_x = point_1[0] - point_0[0]
            delta_y = point_1[1] - point_0[1]
            length = float(np.hypot(delta_x, delta_y))
            if length <= 0:
                continue
            angle = _fold_rect_angle(np.degrees(np.arctan2(delta_y, delta_x)))
            folded_angles.append(angle)
            lengths.append(length)

        if folded_angles:
            angle_values = np.asarray(folded_angles, dtype=np.float64)
            length_values = np.asarray(lengths, dtype=np.float64)
            hough_angle = _weighted_median(angle_values, length_values)
            median_deviation = float(np.median(np.abs(angle_values - hough_angle)))
            consistency = float(np.clip(1.0 - median_deviation / 12.0, 0.0, 1.0))
            coverage = float(np.clip(length_values.sum() / (2.0 * (height + width)), 0.0, 1.0))
            hough_confidence = consistency * coverage

    if hough_angle is not None and hough_confidence >= 0.3:
        if aspect_ratio < 1.2 and abs(hough_angle) < 5.0:
            return {
                "angle_deg": 0.0,
                "angle_confidence": 0.0,
                "angle_source": "low_confidence_zero",
                "status_flags": ["low_confidence_angle"],
            }
        return {
            "angle_deg": round(float(hough_angle), 4),
            "angle_confidence": round(float(hough_confidence), 4),
            "angle_source": "hough",
            "status_flags": [],
        }

    fallback_angle = moment_angle if abs(moment_angle) >= abs(region_angle) else region_angle
    fallback_confidence = max(region_confidence, float(np.clip((abs(moment_angle) - 4.0) / 12.0, 0.0, 1.0)))

    if fallback_confidence >= 0.3:
        if aspect_ratio < 1.2 and abs(fallback_angle) < 5.0:
            return {
                "angle_deg": 0.0,
                "angle_confidence": 0.0,
                "angle_source": "low_confidence_zero",
                "status_flags": ["low_confidence_angle"],
            }
        return {
            "angle_deg": round(float(fallback_angle), 4),
            "angle_confidence": round(float(fallback_confidence), 4),
            "angle_source": "regionprops_fallback",
            "status_flags": [],
        }

    return {
        "angle_deg": 0.0,
        "angle_confidence": 0.0,
        "angle_source": "low_confidence_zero",
        "status_flags": ["low_confidence_angle"],
    }


def _estimate_contour_rect_geometry(film_mask: np.ndarray) -> dict:
    height, width = film_mask.shape
    area = int(film_mask.sum())
    if area <= 0:
        return {
            "angle_deg": 0.0,
            "angle_confidence": 0.0,
            "angle_source": "low_confidence_zero",
            "status_flags": ["low_confidence_angle"],
        }

    ys, xs = np.nonzero(film_mask)
    mask_height = int(ys.max() - ys.min() + 1)
    mask_width = int(xs.max() - xs.min() + 1)
    shorter_side = max(1, min(mask_height, mask_width))
    aspect_ratio = max(mask_height, mask_width) / shorter_side
    fill_ratio = area / float(mask_height * mask_width)

    boundary = np.logical_xor(film_mask, ndimage.binary_erosion(film_mask, structure=np.ones((3, 3))))
    lines = probabilistic_hough_line(boundary.astype(np.uint8), threshold=5, line_length=max(20, min(height, width) // 4), line_gap=8)

    if not lines:
        return _estimate_legacy_patch_geometry(film_mask)

    angles = []
    lengths = []
    for point_0, point_1 in lines:
        delta_x = point_1[0] - point_0[0]
        delta_y = point_1[1] - point_0[1]
        length = float(np.hypot(delta_x, delta_y))
        if length <= 0:
            continue
        angles.append(_fold_rect_angle(np.degrees(np.arctan2(delta_y, delta_x))))
        lengths.append(length)

    if not angles:
        return _estimate_legacy_patch_geometry(film_mask)

    angle_values = np.asarray(angles, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    contour_angle = _weighted_median(angle_values, length_values)
    median_deviation = float(np.median(np.abs(angle_values - contour_angle)))
    consistency = float(np.clip(1.0 - median_deviation / 12.0, 0.0, 1.0))
    coverage = float(np.clip(length_values.sum() / (2.0 * (mask_height + mask_width)), 0.0, 1.0))
    fill_support = float(np.clip((fill_ratio - 0.55) / 0.25, 0.0, 1.0))
    confidence = consistency * max(coverage, fill_support)

    if confidence < 0.3:
        return _estimate_legacy_patch_geometry(film_mask)

    if aspect_ratio < 1.12 and abs(contour_angle) < 5.0:
        return {
            "angle_deg": 0.0,
            "angle_confidence": 0.0,
            "angle_source": "low_confidence_zero",
            "status_flags": ["low_confidence_angle"],
        }

    return {
        "angle_deg": round(float(contour_angle), 4),
        "angle_confidence": round(float(confidence), 4),
        "angle_source": "contour_rect",
        "status_flags": [],
    }


def _estimate_patch_geometry(component_mask: np.ndarray, patch_rgb: np.ndarray | None = None) -> dict:
    if patch_rgb is not None and patch_rgb.size > 0:
        background = _estimate_patch_background(patch_rgb)
        film_mask = _compute_patch_film_mask(patch_rgb, background)
        if int(film_mask.sum()) >= max(64, int(component_mask.sum() * 0.2)):
            contour_geometry = _estimate_contour_rect_geometry(film_mask)
            if contour_geometry["angle_source"] != "low_confidence_zero":
                return contour_geometry
            if contour_geometry["status_flags"] == ["low_confidence_angle"] and abs(contour_geometry["angle_deg"]) < 1e-6:
                return contour_geometry
    return _estimate_legacy_patch_geometry(component_mask)


def _segment_to_bbox(
    component_mask: np.ndarray,
    xslice: slice,
    yslice: slice,
    image_shape: tuple[int, int, int],
    padding: int,
    image_rgb: np.ndarray,
) -> dict | None:
    ys, xs = np.nonzero(component_mask)
    if len(xs) == 0:
        return None

    bbox = _clip_bbox(
        xslice.start + int(xs.min()) - padding,
        yslice.start + int(ys.min()) - padding,
        int(xs.max() - xs.min() + 1) + 2 * padding,
        int(ys.max() - ys.min() + 1) + 2 * padding,
        image_shape,
    )
    patch_image = _extract_patch(image_rgb, bbox)
    return {
        "bbox": bbox,
        "area": int(component_mask.sum()),
        **_estimate_patch_geometry(component_mask, patch_image),
    }


def _split_component_patches(
    component_mask: np.ndarray,
    xslice: slice,
    yslice: slice,
    image_shape: tuple[int, int, int],
    padding: int,
    min_area: int,
    image_rgb: np.ndarray,
) -> list[dict]:
    component_height, component_width = component_mask.shape
    longer_side = max(component_height, component_width)
    shorter_side = max(1, min(component_height, component_width))
    ratio = longer_side / shorter_side
    split_count = max(1, int(round(ratio)))

    if ratio <= 1.6 or split_count == 1:
        patch = _segment_to_bbox(component_mask, xslice, yslice, image_shape, padding, image_rgb)
        return [patch] if patch is not None else []

    patches = []
    if component_height >= component_width:
        step = component_height / split_count
        for index in range(split_count):
            local_y0 = int(round(index * step))
            local_y1 = component_height if index == split_count - 1 else int(round((index + 1) * step))
            segment_mask = component_mask[local_y0:local_y1, :]
            if int(segment_mask.sum()) < max(1, min_area // 4):
                continue
            patch = _segment_to_bbox(
                segment_mask,
                xslice,
                slice(yslice.start + local_y0, yslice.start + local_y1),
                image_shape,
                padding,
                image_rgb,
            )
            if patch is not None:
                patches.append(patch)
    else:
        step = component_width / split_count
        for index in range(split_count):
            local_x0 = int(round(index * step))
            local_x1 = component_width if index == split_count - 1 else int(round((index + 1) * step))
            segment_mask = component_mask[:, local_x0:local_x1]
            if int(segment_mask.sum()) < max(1, min_area // 4):
                continue
            patch = _segment_to_bbox(
                segment_mask,
                slice(xslice.start + local_x0, xslice.start + local_x1),
                yslice,
                image_shape,
                padding,
                image_rgb,
            )
            if patch is not None:
                patches.append(patch)

    return patches or (
        [patch] if (patch := _segment_to_bbox(component_mask, xslice, yslice, image_shape, padding, image_rgb)) is not None else []
    )


def detect_patches(image_rgb: np.ndarray, config: dict) -> dict:
    backend = _normalize_segment_backend(config)
    mask = _compute_sheet_mask(image_rgb, backend=backend)
    labels, component_count = ndimage.label(mask)
    object_slices = ndimage.find_objects(labels)

    min_area = int(config["min_area"])
    padding = int(config["padding"])
    sort_mode = config.get("sort_mode", "yx")

    patches: list[dict] = []
    components: list[dict] = []
    for component_id in range(1, component_count + 1):
        component_slice = object_slices[component_id - 1] if component_id - 1 < len(object_slices) else None
        if component_slice is None:
            continue
        yslice, xslice = component_slice
        component_mask = labels[yslice, xslice] == component_id
        area = int(np.sum(component_mask))
        bbox = _clip_bbox(
            xslice.start - padding,
            yslice.start - padding,
            (xslice.stop - xslice.start) + 2 * padding,
            (yslice.stop - yslice.start) + 2 * padding,
            image_rgb.shape,
        )
        patch_image = _extract_patch(image_rgb, bbox)
        geometry = _estimate_patch_geometry(component_mask, patch_image)
        kept = area >= min_area
        components.append(
            {
                "component_id": component_id,
                "area": area,
                "bbox": bbox,
                **geometry,
                "kept": kept,
            }
        )
        if area < min_area:
            continue

        patches.extend(
            _split_component_patches(
                component_mask=component_mask,
                xslice=xslice,
                yslice=yslice,
                image_shape=image_rgb.shape,
                padding=padding,
                min_area=min_area,
                image_rgb=image_rgb,
            )
        )

    if not patches:
        raise ValueError("No RCF patches were detected in the input scan")

    patches.sort(key=lambda patch: _sort_key(patch, sort_mode))
    normalized: list[dict] = []
    for order, patch in enumerate(patches, start=1):
        normalized.append(
            {
                "order": order,
                "bbox": patch["bbox"],
                "angle_deg": round(float(patch["angle_deg"]), 4),
                "angle_confidence": round(float(patch["angle_confidence"]), 4),
                "angle_source": patch["angle_source"],
                "status_flags": list(patch["status_flags"]),
            }
        )
    return {
        "mask": mask,
        "components": components,
        "component_count": component_count,
        "patches": normalized,
    }


def locate_native_segment_binary() -> Path | None:
    env_path = os.environ.get(NATIVE_SEGMENT_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        return path if path.exists() else None

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "native" / "rcf_preview_core" / "target" / "release" / "rcf_preview_core",
        repo_root / "native" / "rcf_preview_core" / "target" / "debug" / "rcf_preview_core",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def locate_native_segment_library() -> Path | None:
    env_path = os.environ.get(NATIVE_SEGMENT_LIB_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        return path if path.exists() else None

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "native" / "rcf_preview_core" / "target" / "release" / "librcf_preview_core.so",
        repo_root / "native" / "rcf_preview_core" / "target" / "debug" / "librcf_preview_core.so",
        repo_root / "native" / "rcf_preview_core" / "target" / "release" / "rcf_preview_core.dll",
        repo_root / "native" / "rcf_preview_core" / "target" / "debug" / "rcf_preview_core.dll",
        repo_root / "native" / "rcf_preview_core" / "target" / "release" / "librcf_preview_core.dylib",
        repo_root / "native" / "rcf_preview_core" / "target" / "debug" / "librcf_preview_core.dylib",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_native_segment_library():
    global _NATIVE_SEGMENT_LIBRARY

    if _NATIVE_SEGMENT_LIBRARY is False:
        return None
    if _NATIVE_SEGMENT_LIBRARY is not None:
        return _NATIVE_SEGMENT_LIBRARY

    path = locate_native_segment_library()
    if path is None:
        _NATIVE_SEGMENT_LIBRARY = False
        return None

    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        _NATIVE_SEGMENT_LIBRARY = False
        return None

    library.rcf_segment_detect_json.argtypes = [ctypes.c_char_p]
    library.rcf_segment_detect_json.restype = ctypes.c_void_p
    library.rcf_last_error_message.argtypes = []
    library.rcf_last_error_message.restype = ctypes.c_void_p
    library.rcf_free_string.argtypes = [ctypes.c_void_p]
    library.rcf_free_string.restype = None
    _NATIVE_SEGMENT_LIBRARY = library
    return library


def _consume_native_string(library, pointer) -> str:
    if not pointer:
        return ""
    try:
        return ctypes.string_at(pointer).decode("utf-8")
    finally:
        library.rcf_free_string(pointer)


def _decode_native_segment_detection(decoded: dict) -> dict:
    mask = np.array(Image.open(io.BytesIO(bytes.fromhex(decoded["mask_png_hex"]))).convert("L")) > 0
    return {
        "mask": mask,
        "components": decoded["components"],
        "component_count": int(decoded["component_count"]),
        "patches": decoded["patches"],
    }


def _run_native_segment_detection_via_library(library, image_path: Path, config: dict) -> dict:
    payload = {
        "scan_file": str(image_path),
        "min_area": int(config["min_area"]),
        "padding": int(config["padding"]),
        "sort_mode": config.get("sort_mode", "yx"),
    }
    output_pointer = library.rcf_segment_detect_json(json.dumps(payload).encode("utf-8"))
    if not output_pointer:
        error_pointer = library.rcf_last_error_message()
        message = _consume_native_string(library, error_pointer) or "native segment library call failed"
        raise RuntimeError(message)
    return _decode_native_segment_detection(json.loads(_consume_native_string(library, output_pointer)))


def _run_native_segment_detection(image_path: Path, config: dict) -> dict | None:
    library = load_native_segment_library()
    if library is not None:
        return _run_native_segment_detection_via_library(library, image_path=image_path, config=config)

    binary = locate_native_segment_binary()
    if binary is None:
        return None
    payload = {
        "scan_file": str(image_path),
        "min_area": int(config["min_area"]),
        "padding": int(config["padding"]),
        "sort_mode": config.get("sort_mode", "yx"),
    }
    result = subprocess.run(
        [str(binary), "segment-detect"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return _decode_native_segment_detection(json.loads(result.stdout))


def _native_result_requires_python_fallback(result: dict) -> bool:
    patches = result.get("patches", [])
    return len(patches) == 1 and patches[0].get("angle_source") == "low_confidence_zero"


def _segment_cache_key(image_path: Path, config: dict) -> tuple:
    return (
        "segment-detect",
        file_signature(image_path),
        int(config["min_area"]),
        int(config["padding"]),
        config.get("sort_mode", "yx"),
        _normalize_segment_backend(config),
    )


def _copy_detection_result(result: dict) -> dict:
    return {
        "mask": result["mask"].copy(),
        "components": copy.deepcopy(result["components"]),
        "component_count": int(result["component_count"]),
        "patches": copy.deepcopy(result["patches"]),
    }


def detect_patches_path(image_path: Path, config: dict) -> dict:
    cache_key = _segment_cache_key(image_path, config)
    cached = _SEGMENT_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return _copy_detection_result(cached)

    backend = _normalize_segment_backend(config)
    if backend == "cpu":
        native_result = _run_native_segment_detection(image_path=image_path, config=config)
        if native_result is not None:
            if not _native_result_requires_python_fallback(native_result):
                _SEGMENT_RESULT_CACHE.set(cache_key, native_result)
                return _copy_detection_result(native_result)
    python_result = detect_patches(image_rgb=load_rgb_image(image_path), config=config)
    _SEGMENT_RESULT_CACHE.set(cache_key, python_result)
    return _copy_detection_result(python_result)


def auto_detect_patches(image_rgb: np.ndarray, config: dict) -> list[dict]:
    return detect_patches(image_rgb=image_rgb, config=config)["patches"]


def normalize_review_patches(patches: Iterable[dict]) -> list[dict]:
    normalized = []
    for patch in sorted(patches, key=lambda item: item["order"]):
        bbox = [int(value) for value in patch["bbox"]]
        normalized.append(
            {
                "order": int(patch["order"]),
                "bbox": bbox,
                "angle_deg": float(patch.get("angle_deg", 0.0)),
                "angle_confidence": float(patch.get("angle_confidence", 0.0)),
                "angle_source": patch.get("angle_source", "manual"),
                "status_flags": list(patch.get("status_flags", [])),
            }
        )
    if not normalized:
        raise ValueError("Review file did not contain any patch overrides")
    return normalized
