from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .calibration import dose_from_patch, load_background_mean
from .config import load_config
from .io import (
    ensure_output_dirs,
    load_rgb_image,
    save_debug_log,
    save_dose_preview,
    save_json,
    save_mask,
    save_overlay,
    save_patch_image,
)
from .segment import _compute_patch_film_mask
from .segment import _estimate_patch_background
from .segment import _fold_rect_angle
from .segment import detect_patches_path, normalize_review_patches
from .stack import load_stack_entries


def _crop_patch(image_rgb: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, width, height = bbox
    return image_rgb[y : y + height, x : x + width].copy()


def _contour_mask(mask: np.ndarray) -> np.ndarray:
    return np.logical_xor(mask, ndimage.binary_erosion(mask, structure=np.ones((3, 3))))


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def _rotate_rgb_image(image_rgb: np.ndarray, angle_deg: float, fill_rgb: np.ndarray) -> np.ndarray:
    channels = []
    for channel_index in range(image_rgb.shape[2]):
        channels.append(
            ndimage.rotate(
                image_rgb[:, :, channel_index],
                angle=-angle_deg,
                reshape=True,
                order=1,
                mode="constant",
                cval=float(fill_rgb[channel_index]),
            )
        )
    return np.stack(channels, axis=2).astype(np.uint8)


def _make_contour_overlay(patch_rgb: np.ndarray, contour: np.ndarray) -> np.ndarray:
    overlay = patch_rgb.copy()
    overlay[contour] = np.array([255, 0, 0], dtype=np.uint8)
    return overlay


def _crop_by_bbox(image_rgb: np.ndarray, bbox: list[int] | None) -> np.ndarray:
    if bbox is None:
        return image_rgb
    x, y, width, height = bbox
    return image_rgb[y : y + height, x : x + width].copy()


def _save_gray_preview(path: Path, image_gray: np.ndarray) -> None:
    Image.fromarray(np.clip(image_gray, 0, 255).astype(np.uint8), mode="L").save(path)


def _order_box_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _rect_angle_from_box_points(box_points: np.ndarray) -> float:
    ordered = _order_box_points(box_points)
    edge = ordered[1] - ordered[0]
    angle_deg = np.degrees(np.arctan2(float(edge[1]), float(edge[0])))
    return float(_fold_rect_angle(float(angle_deg)))


def _estimate_edge_residual_angle(rotated_patch: np.ndarray, background_rgb: np.ndarray) -> dict:
    import cv2

    patch_lab = cv2.cvtColor(rotated_patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_patch = np.asarray([[background_rgb]], dtype=np.uint8)
    background_lab = cv2.cvtColor(bg_patch, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    distance = np.linalg.norm(patch_lab - background_lab, axis=2)
    if float(distance.max()) <= 1e-6:
        distance_u8 = np.zeros(distance.shape, dtype=np.uint8)
    else:
        distance_u8 = np.clip(distance / distance.max() * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(distance_u8, (5, 5), 0), 10, 30)
    min_side = min(rotated_patch.shape[0], rotated_patch.shape[1])
    min_line_length = max(30, min_side // 5)
    max_line_gap = max(15, min_side // 15)
    threshold = max(15, min_side // 20)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return {
            "edge_residual_angle_deg": 0.0,
            "edge_line_confidence": 0.0,
            "edge_line_count": 0,
        }

    folded_angles = []
    lengths = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle_deg = float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
        if abs(angle_deg) < 25.0 or abs(abs(angle_deg) - 90.0) < 25.0:
            folded = angle_deg if abs(angle_deg) < 25.0 else (angle_deg - 90.0 if angle_deg > 0.0 else angle_deg + 90.0)
            folded_angles.append(float(folded))
            lengths.append(float(np.hypot(float(x2 - x1), float(y2 - y1))))

    if not folded_angles:
        return {
            "edge_residual_angle_deg": 0.0,
            "edge_line_confidence": 0.0,
            "edge_line_count": 0,
        }

    angle_values = np.asarray(folded_angles, dtype=np.float64)
    length_values = np.asarray(lengths, dtype=np.float64)
    weighted_angle = float(np.average(angle_values, weights=length_values))
    median_deviation = float(np.median(np.abs(angle_values - weighted_angle)))
    consistency = float(np.clip(1.0 - median_deviation / 12.0, 0.0, 1.0))
    coverage = float(np.clip(length_values.sum() / (2.0 * (rotated_patch.shape[0] + rotated_patch.shape[1])), 0.0, 1.0))
    confidence = round(consistency * coverage, 4)
    return {
        "edge_residual_angle_deg": round(weighted_angle, 4),
        "edge_line_confidence": confidence,
        "edge_line_count": int(len(angle_values)),
    }


def _opencv_patch_rectification(patch_rgb: np.ndarray, background_rgb: np.ndarray) -> dict:
    import cv2

    patch_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_patch = np.asarray([[background_rgb]], dtype=np.uint8)
    background_lab = cv2.cvtColor(bg_patch, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    lab_distance = np.linalg.norm(patch_lab - background_lab, axis=2)

    if float(lab_distance.max()) <= 1e-6:
        distance_u8 = np.zeros(lab_distance.shape, dtype=np.uint8)
    else:
        distance_u8 = np.clip(lab_distance / lab_distance.max() * 255.0, 0, 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(distance_u8, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), dtype=np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        film_mask = _compute_patch_film_mask(patch_rgb, background_rgb).astype(np.uint8) * 255
        contours, _ = cv2.findContours(film_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        threshold = film_mask

    if not contours:
        height, width = patch_rgb.shape[:2]
        box_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        warped = patch_rgb.copy()
        component_mask = np.zeros((height, width), dtype=np.uint8)
        contour_overlay = patch_rgb.copy()
        box_overlay = patch_rgb.copy()
        rect_angle = 0.0
        rectification_confidence = 0.0
        component_coverage_ratio = 0.0
    else:
        largest_contour = max(contours, key=cv2.contourArea)
        component_mask = np.zeros(threshold.shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        contour_overlay = patch_rgb.copy()
        cv2.drawContours(contour_overlay, [largest_contour], -1, (255, 0, 0), thickness=3)

        rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rect).astype(np.float32)
        ordered = _order_box_points(box_points)
        width_a = np.linalg.norm(ordered[2] - ordered[3])
        width_b = np.linalg.norm(ordered[1] - ordered[0])
        height_a = np.linalg.norm(ordered[1] - ordered[2])
        height_b = np.linalg.norm(ordered[0] - ordered[3])
        warp_width = max(1, int(round(max(width_a, width_b))))
        warp_height = max(1, int(round(max(height_a, height_b))))
        destination = np.array(
            [[0, 0], [warp_width - 1, 0], [warp_width - 1, warp_height - 1], [0, warp_height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(ordered, destination)
        warped = cv2.warpPerspective(
            patch_rgb,
            transform,
            (warp_width, warp_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(int(value) for value in background_rgb.tolist()),
        )

        box_overlay = patch_rgb.copy()
        cv2.polylines(box_overlay, [np.round(box_points).astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)
        rect_angle = _rect_angle_from_box_points(box_points)
        contour_area = max(1.0, float(cv2.contourArea(largest_contour)))
        rect_area = max(1.0, float(rect[1][0] * rect[1][1]))
        patch_area = float(patch_rgb.shape[0] * patch_rgb.shape[1])
        fill_ratio = float(np.clip(contour_area / rect_area, 0.0, 1.0))
        component_coverage_ratio = float(np.clip(contour_area / patch_area, 0.0, 1.0))
        coverage_score = float(np.clip(component_coverage_ratio / 0.25, 0.0, 1.0))
        rectification_confidence = round(0.7 * fill_ratio + 0.3 * coverage_score, 4)

    refined_crop_bbox = [0, 0, int(warped.shape[1]), int(warped.shape[0])]
    return {
        "lab_distance": distance_u8,
        "threshold": threshold,
        "component_mask": component_mask,
        "contour_overlay": contour_overlay,
        "box_overlay": box_overlay,
        "box_points": [[round(float(x), 4), round(float(y), 4)] for x, y in box_points.tolist()],
        "rect_angle_deg": round(rect_angle, 4),
        "rectification_confidence": rectification_confidence,
        "component_coverage_ratio": round(component_coverage_ratio, 4),
        "warped": warped,
        "rotated": warped,
        "refined_crop": warped,
        "refined_crop_bbox": refined_crop_bbox,
        "mask_bbox": _mask_bbox(component_mask > 0),
    }


def _largest_true_run(mask_1d: np.ndarray) -> tuple[int, int] | None:
    best = None
    start = None
    for index, value in enumerate(mask_1d):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if start is not None and (best is None or len(mask_1d) - start > best[1] - best[0]):
        best = (start, len(mask_1d))
    return best


def _refine_rotated_patch_crop(rotated_patch: np.ndarray, background_rgb: np.ndarray) -> dict:
    import cv2

    patch_lab = cv2.cvtColor(rotated_patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_patch = np.asarray([[background_rgb]], dtype=np.uint8)
    background_lab = cv2.cvtColor(bg_patch, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    distance = np.linalg.norm(patch_lab - background_lab, axis=2)
    row_signal = ndimage.gaussian_filter1d(np.percentile(distance, 95, axis=1), sigma=4.0)
    col_signal = ndimage.gaussian_filter1d(np.percentile(distance, 95, axis=0), sigma=4.0)

    border = max(8, min(rotated_patch.shape[0], rotated_patch.shape[1]) // 20)
    border_samples = np.concatenate(
        (
            row_signal[:border],
            row_signal[-border:],
            col_signal[:border],
            col_signal[-border:],
        )
    )
    projection_threshold = max(4.0, float(np.percentile(border_samples, 99)) + 2.0)
    row_run = _largest_true_run(row_signal > projection_threshold)
    col_run = _largest_true_run(col_signal > projection_threshold)

    if row_run is None or col_run is None:
        bbox = [0, 0, int(rotated_patch.shape[1]), int(rotated_patch.shape[0])]
        return {
            "refined_crop": rotated_patch,
            "refined_crop_bbox": bbox,
            "projection_threshold": round(projection_threshold, 4),
            "sheet_coverage_ratio": 1.0,
        }

    y0, y1 = row_run
    x0, x1 = col_run
    bbox = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
    refined_crop = rotated_patch[y0:y1, x0:x1].copy()
    sheet_coverage_ratio = float(np.clip((bbox[2] * bbox[3]) / float(rotated_patch.shape[0] * rotated_patch.shape[1]), 0.0, 1.0))
    return {
        "refined_crop": refined_crop,
        "refined_crop_bbox": bbox,
        "projection_threshold": round(projection_threshold, 4),
        "sheet_coverage_ratio": round(sheet_coverage_ratio, 4),
    }


def _summary_patch(
    patch: dict,
    patch_rgb: np.ndarray,
    dose: np.ndarray,
    patch_background_mean: float,
) -> dict:
    return {
        "order": patch["order"],
        "bbox": patch["bbox"],
        "angle_deg": round(float(patch.get("angle_deg", 0.0)), 4),
        "angle_confidence": round(float(patch.get("angle_confidence", 0.0)), 4),
        "angle_source": patch.get("angle_source", "manual"),
        "status_flags": list(patch.get("status_flags", [])),
        "shape": [int(v) for v in patch_rgb.shape[:2]],
        "patch_background_mean": round(float(patch_background_mean), 4),
        "dose_mean": round(float(dose.mean()), 6),
        "dose_std": round(float(dose.std()), 6),
        "dose_min": round(float(dose.min()), 6),
        "dose_max": round(float(dose.max()), 6),
        "status": "ok",
    }


def debug_patch_rectification(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    patch_order: int,
) -> dict:
    config = load_config(config_path)
    image_rgb = load_rgb_image(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    detection = detect_patches_path(input_path, config["segmentation"])
    target_patch = next((patch for patch in detection["patches"] if patch["order"] == patch_order), None)
    if target_patch is None:
        raise ValueError(f"Patch order {patch_order} was not detected in {input_path}")

    patch_rgb = _crop_patch(image_rgb, target_patch["bbox"])
    background_rgb = _estimate_patch_background(patch_rgb)
    opencv_rectification = _opencv_patch_rectification(patch_rgb, background_rgb)
    contour = _contour_mask(opencv_rectification["component_mask"] > 0)
    component_angle_deg = float(opencv_rectification["rect_angle_deg"])
    component_rotated_patch = _rotate_rgb_image(patch_rgb, component_angle_deg, background_rgb)
    edge_refinement = _estimate_edge_residual_angle(component_rotated_patch, background_rgb)
    residual_angle_deg = float(edge_refinement["edge_residual_angle_deg"]) if edge_refinement["edge_line_confidence"] > 0.2 else 0.0
    final_rotation_angle_deg = component_angle_deg + residual_angle_deg
    rotated_patch = _rotate_rgb_image(patch_rgb, final_rotation_angle_deg, background_rgb)
    rotated_refinement = _refine_rotated_patch_crop(rotated_patch, background_rgb)
    refined_crop = rotated_refinement["refined_crop"]
    refined_crop_bbox = rotated_refinement["refined_crop_bbox"]

    save_patch_image(output_dir / "patch_raw.png", patch_rgb)
    _save_gray_preview(output_dir / "patch_lab_distance.png", opencv_rectification["lab_distance"])
    save_mask(output_dir / "patch_threshold.png", opencv_rectification["threshold"] > 0)
    save_mask(output_dir / "patch_component.png", opencv_rectification["component_mask"] > 0)
    save_mask(output_dir / "patch_mask.png", opencv_rectification["component_mask"] > 0)
    save_patch_image(output_dir / "patch_contour.png", _make_contour_overlay(patch_rgb, contour))
    save_patch_image(output_dir / "patch_boxpoints.png", opencv_rectification["box_overlay"])
    save_patch_image(output_dir / "patch_rotated.png", rotated_patch)
    save_patch_image(output_dir / "patch_warped.png", opencv_rectification["warped"])
    save_patch_image(output_dir / "patch_refined_crop.png", refined_crop)

    debug_payload = {
        "input_file": str(input_path),
        "config_file": str(config_path),
        "patch_order": patch_order,
        "bbox": list(target_patch["bbox"]),
        "angle_deg": round(final_rotation_angle_deg, 4),
        "angle_confidence": round(float(target_patch.get("angle_confidence", 0.0)), 4),
        "angle_source": target_patch.get("angle_source", "manual"),
        "rectification_source": "opencv_min_area_rect",
        "status_flags": list(target_patch.get("status_flags", [])),
        "background_rgb": [round(float(value), 4) for value in background_rgb.tolist()],
        "mask_bbox": opencv_rectification["mask_bbox"],
        "box_points": opencv_rectification["box_points"],
        "rectification_confidence": opencv_rectification["rectification_confidence"],
        "component_coverage_ratio": opencv_rectification["component_coverage_ratio"],
        "component_angle_deg": round(component_angle_deg, 4),
        "edge_residual_angle_deg": edge_refinement["edge_residual_angle_deg"],
        "edge_line_confidence": edge_refinement["edge_line_confidence"],
        "edge_line_count": edge_refinement["edge_line_count"],
        "rotation_angle_deg": round(final_rotation_angle_deg, 4),
        "sheet_coverage_ratio": rotated_refinement["sheet_coverage_ratio"],
        "projection_threshold": rotated_refinement["projection_threshold"],
        "refined_crop_bbox": refined_crop_bbox,
        "raw_shape": [int(v) for v in patch_rgb.shape[:2]],
        "rotated_shape": [int(v) for v in rotated_patch.shape[:2]],
        "refined_shape": [int(v) for v in refined_crop.shape[:2]],
    }
    save_json(output_dir / "patch_debug.json", debug_payload)
    return debug_payload


def process_scan(
    input_path: Path,
    config_path: Path,
    output_dir: Path,
    review_path: Path | None = None,
) -> dict:
    config = load_config(config_path)
    image_rgb = load_rgb_image(input_path)
    patches_dir, dose_dir = ensure_output_dirs(output_dir)

    film_background_mean = load_background_mean(Path(config["background"]["film_path"]))
    scanner_background_mean = load_background_mean(Path(config["background"]["scanner_path"]))
    film_type = config["film_type"]
    film_model = config["calibration"]["film_models"][film_type]
    background_quantile = float(config["calibration"]["background_quantile"])
    dose_backend = str(config["calibration"].get("backend", "auto"))
    detection = detect_patches_path(input_path, config["segmentation"])
    auto_patches = detection["patches"]

    review_applied = review_path is not None
    if review_path is not None:
        with review_path.open("r", encoding="utf-8") as handle:
            review_payload = json.load(handle)
        patches = normalize_review_patches(review_payload["patches"])
        segmentation_status = "review_override"
    else:
        patches = auto_patches
        segmentation_status = "ok"

    overlay_path = output_dir / "overlay.png"
    overlay_raw_path = output_dir / "overlay_raw.png"
    overlay_final_path = output_dir / "overlay_final.png"
    review_output_path = output_dir / "review.json"
    save_overlay(overlay_raw_path, image_rgb, auto_patches)
    save_overlay(overlay_final_path, image_rgb, patches)
    save_overlay(overlay_path, image_rgb, patches)
    save_mask(output_dir / "mask.png", detection["mask"])
    save_json(
        output_dir / "components.json",
        {
            "component_count": detection["component_count"],
            "components": detection["components"],
        },
    )
    save_json(review_output_path, {"input_file": str(input_path), "patches": patches})

    summary_patches: list[dict] = []
    for patch in patches:
        patch_rgb = _crop_patch(image_rgb, patch["bbox"])
        dose, patch_background_mean = dose_from_patch(
            patch_rgb=patch_rgb,
            film_background_mean=film_background_mean,
            scanner_background_mean=scanner_background_mean,
            film_model=film_model,
            background_quantile=background_quantile,
            backend=dose_backend,
        )

        order = patch["order"]
        save_patch_image(patches_dir / f"patch_{order:02d}.png", patch_rgb)
        save_dose_preview(dose_dir / f"dose_{order:02d}.png", dose)
        summary_patches.append(_summary_patch(patch, patch_rgb, dose, patch_background_mean))

    summary = {
        "input_file": str(input_path),
        "config_file": str(config_path),
        "film_type": film_type,
        "segmentation_status": segmentation_status,
        "review_applied": review_applied,
        "calibration_status": "dose",
        "qc_flags": [],
        "patch_count": len(summary_patches),
        "patches": summary_patches,
    }
    save_json(output_dir / "summary.json", summary)
    save_debug_log(
        output_dir / "debug.log",
        [
            f"input_file={input_path}",
            f"film_type={film_type}",
            f"segmentation_status={segmentation_status}",
            f"review_applied={review_applied}",
            f"component_count={detection['component_count']}",
            f"kept_component_count={len(auto_patches)}",
            f"background_quantile={background_quantile}",
            f"film_background_mean={film_background_mean:.4f}",
            f"scanner_background_mean={scanner_background_mean:.4f}",
        ],
    )
    return summary


def process_scan_series(
    input_paths: list[Path],
    config_path: Path,
    output_dir: Path,
    stack_config_path: Path | None = None,
    review_path: Path | None = None,
) -> dict:
    if review_path is not None:
        raise ValueError("Multi-scan mode does not support a shared review file yet")

    scans_dir = output_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    per_scan_summaries = []
    combined_patches = []
    global_order = 1

    for scan_index, input_path in enumerate(input_paths, start=1):
        scan_output_dir = scans_dir / f"scan_{scan_index:02d}"
        scan_summary = process_scan(
            input_path=input_path,
            config_path=config_path,
            output_dir=scan_output_dir,
            review_path=None,
        )
        per_scan_summaries.append(scan_summary)
        for patch in scan_summary["patches"]:
            combined_patch = {
                **patch,
                "scan_index": scan_index,
                "scan_file": str(input_path),
                "local_order": patch["order"],
                "global_order": global_order,
            }
            combined_patches.append(combined_patch)
            global_order += 1

    qc_flags: list[str] = []
    stack_entries: list[dict] = []
    if stack_config_path is not None:
        stack_entries = load_stack_entries(stack_config_path)
        for index, patch in enumerate(combined_patches):
            patch["stack"] = stack_entries[index] if index < len(stack_entries) else None
        if len(combined_patches) != len(stack_entries):
            qc_flags.append("stack_config_patch_count_mismatch")
    else:
        for patch in combined_patches:
            patch["stack"] = None

    summary = {
        "input_files": [str(path) for path in input_paths],
        "config_file": str(config_path),
        "stack_config_file": str(stack_config_path) if stack_config_path is not None else None,
        "scan_count": len(input_paths),
        "patch_count": len(combined_patches),
        "qc_flags": qc_flags,
        "patches": combined_patches,
    }
    save_json(output_dir / "summary.json", summary)
    save_debug_log(
        output_dir / "debug.log",
        [
            f"scan_count={len(input_paths)}",
            f"patch_count={len(combined_patches)}",
            f"stack_config_file={stack_config_path if stack_config_path is not None else 'none'}",
            f"qc_flags={','.join(qc_flags) if qc_flags else 'none'}",
        ],
    )
    return summary
