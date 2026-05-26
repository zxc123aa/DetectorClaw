from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected (4,2), got {pts.shape}")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def white_distance_map(image_bgr: np.ndarray) -> np.ndarray:
    diff = 255.0 - image_bgr.astype(np.float32)
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    if dist.max() > 0:
        dist = dist / dist.max() * 255.0
    return np.clip(dist, 0, 255).astype(np.uint8)


def find_film_quad(
    image_bgr: np.ndarray,
    blur_ksize: int = 5,
    morph_ksize: int = 7,
    manual_threshold: int | None = None,
    min_area_ratio: float = 0.4,
    max_area_ratio: float = 1.01,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    h, w = image_bgr.shape[:2]
    img_area = h * w

    dist = white_distance_map(image_bgr)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    if k > 1:
        dist = cv2.GaussianBlur(dist, (k, k), 0)

    if manual_threshold is None:
        adaptive_thr = max(20, int(np.percentile(dist, 5) * 0.7))
        _, mask = cv2.threshold(dist, adaptive_thr, 255, cv2.THRESH_BINARY)
        threshold_used = adaptive_thr
    else:
        _, mask = cv2.threshold(dist, manual_threshold, 255, cv2.THRESH_BINARY)
        threshold_used = int(manual_threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio or area > img_area * max_area_ratio:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        box = cv2.boxPoints(rect).astype(np.float32)
        box = order_points(box)
        rect_area = max(rw * rh, 1e-6)
        fill = area / rect_area
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        score = 2.0 * fill - 0.2 * abs(aspect - 1.0)
        candidates.append(
            {
                "contour": cnt,
                "area": float(area),
                "box": box,
                "center": [float(cx), float(cy)],
                "rect_size": [float(rw), float(rh)],
                "angle": float(angle),
                "fill_ratio": float(fill),
                "aspect_ratio": float(aspect),
                "score": float(score),
            }
        )

    if not candidates:
        raise RuntimeError("No valid film contour found. Try lowering min_area_ratio or setting manual_threshold.")

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    info = {k: v for k, v in best.items() if k != "contour"}
    if debug:
        info["threshold_used"] = int(threshold_used)
    return best["box"], mask, info


def warp_with_mask(image_bgr: np.ndarray, src_pts: np.ndarray, out_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_pts = order_points(src_pts)
    out_w, out_h = out_size
    dst_pts = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image_bgr, homography, (out_w, out_h))

    src_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(src_mask, src_pts.astype(np.int32), 255)
    warped_mask = cv2.warpPerspective(src_mask, homography, (out_w, out_h), flags=cv2.INTER_NEAREST)
    warped_mask = (warped_mask > 127).astype(np.uint8) * 255
    return warped, warped_mask, homography


def alpha_apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


def detect_valid_bbox(mask: np.ndarray, pad: int = 5) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(mask.shape[1] - 1, x1 + pad)
    y1 = min(mask.shape[0] - 1, y1 + pad)
    return x0, y0, x1 + 1, y1 + 1


def crop_by_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def draw_overlay(image_bgr: np.ndarray, quad: np.ndarray, info: Dict) -> np.ndarray:
    vis = image_bgr.copy()
    pts = quad.astype(np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    for i, point in enumerate(pts):
        cv2.circle(vis, tuple(point), 7, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            str(i),
            tuple(point + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    txt = f"score={info['score']:.3f} fill={info['fill_ratio']:.3f}"
    cv2.putText(vis, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    return vis


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def parse_size(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError("--out-size should look like 2000x2000")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Output size must be positive")
    return width, height


def process_one(
    image_path: Path,
    out_dir: Path,
    out_size: tuple[int, int],
    blur_ksize: int,
    morph_ksize: int,
    manual_threshold: int | None,
    min_area_ratio: float,
    crop_mode: str,
    save_debug: bool,
) -> Dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    ensure_dir(out_dir / "registered")
    ensure_dir(out_dir / "registered_rgba")
    ensure_dir(out_dir / "masks")
    ensure_dir(out_dir / "overlays")
    ensure_dir(out_dir / "metadata")
    if save_debug:
        ensure_dir(out_dir / "debug")

    quad, raw_mask, info = find_film_quad(
        image,
        blur_ksize=blur_ksize,
        morph_ksize=morph_ksize,
        manual_threshold=manual_threshold,
        min_area_ratio=min_area_ratio,
    )
    registered, reg_mask, homography = warp_with_mask(image, quad, out_size)

    if crop_mode == "tight":
        bbox = detect_valid_bbox(reg_mask)
        registered_out = crop_by_bbox(registered, bbox)
        reg_mask_out = crop_by_bbox(reg_mask, bbox)
    else:
        bbox = (0, 0, registered.shape[1], registered.shape[0])
        registered_out = registered
        reg_mask_out = reg_mask

    rgba_out = alpha_apply_mask(registered_out, reg_mask_out)
    overlay = draw_overlay(image, quad, info)

    stem = image_path.stem
    reg_path = out_dir / "registered" / f"{stem}_registered.png"
    rgba_path = out_dir / "registered_rgba" / f"{stem}_registered_rgba.png"
    mask_path = out_dir / "masks" / f"{stem}_mask.png"
    overlay_path = out_dir / "overlays" / f"{stem}_overlay.png"
    meta_path = out_dir / "metadata" / f"{stem}.json"

    cv2.imwrite(str(reg_path), registered_out)
    cv2.imwrite(str(rgba_path), rgba_out)
    cv2.imwrite(str(mask_path), reg_mask_out)
    cv2.imwrite(str(overlay_path), overlay)

    if save_debug:
        dist = white_distance_map(image)
        cv2.imwrite(str(out_dir / "debug" / f"{stem}_white_distance.png"), dist)
        cv2.imwrite(str(out_dir / "debug" / f"{stem}_initial_mask.png"), raw_mask)

    meta = {
        "image_path": str(image_path),
        "original_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "registered_size_before_crop": {"width": int(registered.shape[1]), "height": int(registered.shape[0])},
        "registered_size_after_crop": {"width": int(registered_out.shape[1]), "height": int(registered_out.shape[0])},
        "crop_mode": crop_mode,
        "crop_bbox_in_registered_plane": [int(v) for v in bbox],
        "source_quad_tl_tr_br_bl": quad.tolist(),
        "homography_3x3": homography.tolist(),
        "quality": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()},
        "outputs": {
            "registered": str(reg_path),
            "registered_rgba": str(rgba_path),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
        },
    }

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "image_path": str(image_path),
        "registered": str(reg_path),
        "mask": str(mask_path),
        "score": info["score"],
        "fill_ratio": info["fill_ratio"],
        "crop_mode": crop_mode,
    }


def save_summary(rows: list[Dict], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "registered", "mask", "score", "fill_ratio", "crop_mode"])
        writer.writeheader()
        writer.writerows(rows)


def register_inputs(
    input_path: Path,
    output_dir: Path,
    out_size: tuple[int, int],
    crop_mode: str = "fixed",
    blur_ksize: int = 5,
    morph_ksize: int = 7,
    manual_threshold: int | None = None,
    min_area_ratio: float = 0.4,
    save_debug: bool = False,
) -> list[Dict]:
    ensure_dir(output_dir)
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found: {input_path}")

    rows: list[Dict] = []
    for image_path in images:
        row = process_one(
            image_path=image_path,
            out_dir=output_dir,
            out_size=out_size,
            blur_ksize=blur_ksize,
            morph_ksize=morph_ksize,
            manual_threshold=manual_threshold,
            min_area_ratio=min_area_ratio,
            crop_mode=crop_mode,
            save_debug=save_debug,
        )
        rows.append(row)
    save_summary(rows, output_dir / "summary.csv")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RCF register: 输出 registered image + mask + homography")
    parser.add_argument("--input", required=True, type=Path, help="输入图像文件或目录")
    parser.add_argument("--output", required=True, type=Path, help="输出目录")
    parser.add_argument("--out-size", default="2000x2000", type=str, help="统一配准平面尺寸，如 2000x2000")
    parser.add_argument("--crop-mode", default="fixed", choices=["fixed", "tight"], help="fixed=保留统一坐标系尺寸; tight=按mask紧裁")
    parser.add_argument("--blur-ksize", default=5, type=int)
    parser.add_argument("--morph-ksize", default=7, type=int)
    parser.add_argument("--manual-threshold", default=None, type=int)
    parser.add_argument("--min-area-ratio", default=0.4, type=float, help="片区最小面积比例，针对已切片图默认大一些")
    parser.add_argument("--save-debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = register_inputs(
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
    for row in rows:
        print(f"[OK] {Path(row['image_path']).name}")
    print(f"Done. Saved to: {args.output}")
    return 0
