from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected (4, 2) points, got {points.shape}")

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def rotation_angle_from_box(box: np.ndarray) -> float:
    ordered = order_points(box)
    top_edge = ordered[1] - ordered[0]
    return float(np.degrees(np.arctan2(float(top_edge[1]), float(top_edge[0]))))


def four_point_transform(
    image_bgr: np.ndarray,
    points: np.ndarray,
    fixed_size: tuple[int, int] | None = None,
) -> np.ndarray:
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = max(2, int(round(max(width_a, width_b))))
    max_height = max(2, int(round(max(height_a, height_b))))

    if fixed_size is not None:
        dst_w, dst_h = fixed_size
    else:
        dst_w, dst_h = max_width, max_height

    destination = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image_bgr, transform, (dst_w, dst_h))


def white_distance_map(image_bgr: np.ndarray) -> np.ndarray:
    diff = 255.0 - image_bgr.astype(np.float32)
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    if float(distance.max()) > 0:
        distance = distance / float(distance.max()) * 255.0
    return np.clip(distance, 0, 255).astype(np.uint8)


def build_mask(
    image_bgr: np.ndarray,
    blur_ksize: int = 5,
    morph_ksize: int = 9,
    manual_threshold: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    distance = white_distance_map(image_bgr)
    if blur_ksize > 1:
        kernel = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        distance = cv2.GaussianBlur(distance, (kernel, kernel), 0)

    if manual_threshold is None:
        _, mask = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(distance, manual_threshold, 255, cv2.THRESH_BINARY)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    return mask, distance


def contour_score(contour: np.ndarray, rect_w: float, rect_h: float) -> float:
    area = float(cv2.contourArea(contour))
    rect_area = max(rect_w * rect_h, 1e-6)
    fill_ratio = area / rect_area
    aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1e-6)
    return float(fill_ratio - 0.15 * abs(aspect - 1.0))


def detect_rcf_rectangles(
    image_bgr: np.ndarray,
    expected_count: int | None = None,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.50,
    min_side_px: int = 80,
    max_aspect_ratio: float = 1.8,
    blur_ksize: int = 5,
    morph_ksize: int = 9,
    manual_threshold: int | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    height, width = image_bgr.shape[:2]
    image_area = height * width

    mask, distance = build_mask(
        image_bgr,
        blur_ksize=blur_ksize,
        morph_ksize=morph_ksize,
        manual_threshold=manual_threshold,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[dict] = []
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_w, rect_h), angle = rect
        if min(rect_w, rect_h) < min_side_px:
            continue
        aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1e-6)
        if aspect > max_aspect_ratio:
            continue

        box = order_points(cv2.boxPoints(rect).astype(np.float32))
        x, y, box_w, box_h = cv2.boundingRect(contour)
        candidates.append(
            {
                "center": [float(center_x), float(center_y)],
                "rect_size": [float(rect_w), float(rect_h)],
                "angle": float(angle),
                "rotation_angle_deg": rotation_angle_from_box(box),
                "box": box.tolist(),
                "bbox": [int(x), int(y), int(box_w), int(box_h)],
                "contour_area": area,
                "score": contour_score(contour, rect_w, rect_h),
            }
        )

    candidates.sort(key=lambda item: (item["center"][1], item["center"][0]))
    if expected_count is not None and len(candidates) > expected_count:
        candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:expected_count]
        candidates.sort(key=lambda item: (item["center"][1], item["center"][0]))

    return candidates, mask, distance


def draw_overlay(image_bgr: np.ndarray, candidates: list[dict], title_text: str | None = None) -> np.ndarray:
    overlay = image_bgr.copy()
    for index, candidate in enumerate(candidates, start=1):
        box = np.array(candidate["box"], dtype=np.int32)
        cv2.polylines(overlay, [box], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)
        for point in box:
            cv2.circle(overlay, tuple(point), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        x, y, _, _ = candidate["bbox"]
        cv2.putText(
            overlay,
            f"{index} | score={candidate['score']:.3f}",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 220),
            2,
            lineType=cv2.LINE_AA,
        )
    if title_text:
        cv2.putText(
            overlay,
            title_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    return overlay


def make_montage(
    images: list[np.ndarray],
    cols: int = 2,
    cell_size: tuple[int, int] = (500, 500),
    pad: int = 12,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if not images:
        return np.full((200, 400, 3), 255, dtype=np.uint8)

    cell_w, cell_h = cell_size
    rows = math.ceil(len(images) / cols)
    canvas_h = rows * cell_h + (rows + 1) * pad
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + pad)
        height, width = image.shape[:2]
        scale = min(cell_w / width, cell_h / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        xx = x0 + (cell_w - new_w) // 2
        yy = y0 + (cell_h - new_h) // 2
        canvas[yy : yy + new_h, xx : xx + new_w] = resized
    return canvas


def list_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return [path for path in sorted(input_path.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTS]


def parse_fixed_size(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError("--fixed-size must look like 800x800")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("--fixed-size values must be positive")
    return (width, height)


def save_summary_csv(rows: list[dict], output_csv: Path) -> None:
    ensure_dir(output_csv.parent)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "detected_count"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image_path": row["image_path"], "detected_count": row["detected_count"]})


def process_one_image(
    image_path: Path,
    output_dir: Path,
    expected_count: int | None,
    fixed_size: tuple[int, int] | None,
    min_area_ratio: float,
    max_area_ratio: float,
    min_side_px: int,
    max_aspect_ratio: float,
    blur_ksize: int,
    morph_ksize: int,
    manual_threshold: int | None,
    save_debug: bool,
) -> dict:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    stem = image_path.stem
    patches_dir = output_dir / "patches"
    overlays_dir = output_dir / "overlays"
    metadata_dir = output_dir / "metadata"
    debug_dir = output_dir / "debug"
    ensure_dir(patches_dir)
    ensure_dir(overlays_dir)
    ensure_dir(metadata_dir)
    if save_debug:
        ensure_dir(debug_dir)

    candidates, mask, distance = detect_rcf_rectangles(
        image_bgr=image_bgr,
        expected_count=expected_count,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_side_px=min_side_px,
        max_aspect_ratio=max_aspect_ratio,
        blur_ksize=blur_ksize,
        morph_ksize=morph_ksize,
        manual_threshold=manual_threshold,
    )

    overlay = draw_overlay(image_bgr, candidates, title_text=f"{stem} | detected={len(candidates)}")
    cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"), overlay)
    if save_debug:
        cv2.imwrite(str(debug_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(debug_dir / f"{stem}_white_distance.png"), distance)

    patches: list[np.ndarray] = []
    metadata = {
        "image_path": str(image_path),
        "image_size": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])},
        "detected_count": len(candidates),
        "candidates": [],
    }
    for idx, candidate in enumerate(candidates, start=1):
        box = np.array(candidate["box"], dtype=np.float32)
        warped = four_point_transform(image_bgr, box, fixed_size=fixed_size)
        patch_path = (patches_dir / f"{stem}_rcf_{idx:02d}.png").resolve()
        cv2.imwrite(str(patch_path), warped)
        patches.append(warped)

        candidate_payload = dict(candidate)
        candidate_payload["patch_path"] = str(patch_path)
        candidate_payload["patch_size"] = {"width": int(warped.shape[1]), "height": int(warped.shape[0])}
        metadata["candidates"].append(candidate_payload)

    metadata_path = metadata_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "image_path": str(image_path),
        "detected_count": len(candidates),
        "patches": patches,
        "metadata": metadata,
    }


def autocrop_inputs(
    input_path: Path,
    output_dir: Path,
    expected_count: int | None = None,
    fixed_size: tuple[int, int] | None = None,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.50,
    min_side_px: int = 80,
    max_aspect_ratio: float = 1.8,
    blur_ksize: int = 5,
    morph_ksize: int = 9,
    manual_threshold: int | None = None,
    save_debug: bool = False,
    save_montage: bool = False,
) -> list[dict]:
    image_paths = list_input_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_path}")

    ensure_dir(output_dir)
    montage_dir = output_dir / "montage"
    if save_montage:
        ensure_dir(montage_dir)

    summary_rows: list[dict] = []
    results: list[dict] = []
    for image_path in image_paths:
        result = process_one_image(
            image_path=image_path,
            output_dir=output_dir,
            expected_count=expected_count,
            fixed_size=fixed_size,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            min_side_px=min_side_px,
            max_aspect_ratio=max_aspect_ratio,
            blur_ksize=blur_ksize,
            morph_ksize=morph_ksize,
            manual_threshold=manual_threshold,
            save_debug=save_debug,
        )
        results.append(result)
        summary_rows.append({"image_path": result["image_path"], "detected_count": result["detected_count"]})
        if save_montage and result["patches"]:
            montage = make_montage(result["patches"], cols=2, cell_size=(420, 420))
            cv2.imwrite(str(montage_dir / f"{Path(result['image_path']).stem}_montage.png"), montage)

    save_summary_csv(summary_rows, output_dir / "summary.csv")
    return results
