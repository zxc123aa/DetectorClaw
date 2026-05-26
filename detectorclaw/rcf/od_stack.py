from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EPS = 1e-6


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_roi(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid ROI: x1 must > x0 and y1 must > y0")
    return x0, y0, x1, y1


def find_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def read_mask(path: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def get_channel_float(img_bgr: np.ndarray, channel: str) -> np.ndarray:
    channel = channel.lower()
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    if channel == "blue":
        return b
    if channel == "green":
        return g
    if channel == "red":
        return r
    if channel == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if channel == "rgb-mean":
        return (r + g + b) / 3.0
    raise ValueError(f"Unsupported channel: {channel}")


def estimate_i0_from_reference(reference_img_bgr: np.ndarray, channel: str, mask: np.ndarray) -> float:
    arr = get_channel_float(reference_img_bgr, channel)
    vals = arr[mask > 0]
    if vals.size == 0:
        raise ValueError("Reference mask has no valid pixels")
    return float(np.median(vals))


def estimate_i0_per_image_percentile(img_bgr: np.ndarray, channel: str, mask: np.ndarray, percentile: float) -> float:
    arr = get_channel_float(img_bgr, channel)
    vals = arr[mask > 0]
    if vals.size == 0:
        raise ValueError("Image mask has no valid pixels")
    return float(np.percentile(vals, percentile))


def compute_od(img_bgr: np.ndarray, mask: np.ndarray, channel: str, i0: float) -> np.ndarray:
    intensity = get_channel_float(img_bgr, channel)
    intensity = np.clip(intensity, EPS, None)
    i0 = max(float(i0), EPS)
    od = np.log10(i0 / intensity).astype(np.float32)
    od[mask == 0] = np.nan
    return od


def roi_mask_from_rect(shape: tuple[int, int], roi: tuple[int, int, int, int] | None) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if roi is None:
        mask[:, :] = 1
        return mask
    x0, y0, x1, y1 = roi
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask


def summarize_values(vals: np.ndarray) -> Dict[str, float]:
    if vals.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p05": np.nan,
            "p95": np.nan,
        }
    return {
        "count": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
    }


def od_to_png(od: np.ndarray, mask: np.ndarray, vmin: float | None, vmax: float | None) -> np.ndarray:
    vals = od[np.isfinite(od)]
    if vals.size == 0:
        return np.full((*od.shape, 3), 255, np.uint8)
    if vmin is None:
        vmin = float(np.percentile(vals, 1))
    if vmax is None:
        vmax = float(np.percentile(vals, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-3

    norm = (od - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    color[mask == 0] = (255, 255, 255)
    return color


def overlay_roi(img_bgr: np.ndarray, roi: tuple[int, int, int, int] | None, text: str = "") -> np.ndarray:
    vis = img_bgr.copy()
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 3)
    if text:
        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    return vis


def match_mask_for_image(img_path: Path, masks_dir: Path) -> Path:
    base_stem = img_path.stem.removesuffix("_registered")
    candidates = [
        masks_dir / img_path.name,
        masks_dir / f"{img_path.stem}_mask{img_path.suffix}",
        masks_dir / f"{img_path.stem}.png",
        masks_dir / f"{base_stem}_mask{img_path.suffix}",
        masks_dir / f"{base_stem}_mask.png",
        masks_dir / f"{base_stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No matching mask found for {img_path.name} in {masks_dir}")


def intersection_bbox(mask_paths: list[Path], target_shape: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    masks = [read_mask(path, target_shape) for path in mask_paths]
    if not masks:
        raise ValueError("No masks available to compute automatic ROI")
    first_shape = masks[0].shape
    if any(mask.shape != first_shape for mask in masks[1:]):
        raise ValueError("Automatic ROI requires masks with identical shapes; use fixed registration crop or pass --roi")
    intersection = masks[0].astype(bool)
    for mask in masks[1:]:
        intersection &= mask.astype(bool)
    ys, xs = np.where(intersection)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Automatic ROI intersection is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resolve_roi(
    image_paths: list[Path],
    masks_dir: Path,
    manual_roi: tuple[int, int, int, int] | None,
    auto_roi_mode: str,
) -> tuple[tuple[int, int, int, int] | None, str]:
    if manual_roi is not None:
        return manual_roi, "manual"
    if auto_roi_mode != "intersection-bbox":
        raise ValueError(f"Unsupported auto ROI mode: {auto_roi_mode}")
    mask_paths = [match_mask_for_image(img_path, masks_dir) for img_path in image_paths]
    return intersection_bbox(mask_paths), "auto:intersection-bbox"


def process_one(
    img_path: Path,
    masks_dir: Path,
    output_dir: Path,
    channel: str,
    reference_mode: str,
    reference_img_bgr: np.ndarray | None,
    reference_percentile: float,
    roi: tuple[int, int, int, int] | None,
    roi_source: str,
    png_vmin: float | None,
    png_vmax: float | None,
) -> Dict:
    img = read_image(img_path)
    mask_path = match_mask_for_image(img_path, masks_dir)
    mask = read_mask(mask_path, img.shape[:2])

    roi_mask = roi_mask_from_rect(mask.shape, roi)
    analysis_mask = ((mask > 0) & (roi_mask > 0)).astype(np.uint8)

    if reference_mode == "reference-file":
        if reference_img_bgr is None:
            raise ValueError("reference-file mode requires reference image")
        if reference_img_bgr.shape[:2] != img.shape[:2]:
            raise ValueError("reference-file image must match registered image shape")
        i0 = estimate_i0_from_reference(reference_img_bgr, channel, mask)
    elif reference_mode == "per-image-percentile":
        i0 = estimate_i0_per_image_percentile(img, channel, mask, reference_percentile)
    else:
        raise ValueError(f"Unsupported reference_mode: {reference_mode}")

    od = compute_od(img, mask, channel, i0)
    roi_vals = od[np.isfinite(od) & (analysis_mask > 0)]
    full_vals = od[np.isfinite(od) & (mask > 0)]

    ensure_dir(output_dir / "od_npy")
    ensure_dir(output_dir / "od_png")
    ensure_dir(output_dir / "overlays")
    ensure_dir(output_dir / "metadata")

    stem = img_path.stem
    np.save(output_dir / "od_npy" / f"{stem}_od.npy", od)
    od_png = od_to_png(od, mask, png_vmin, png_vmax)
    cv2.imwrite(str(output_dir / "od_png" / f"{stem}_od.png"), od_png)

    overlay = overlay_roi(img, roi, text=f"{stem} | ch={channel} | I0={i0:.2f}")
    cv2.imwrite(str(output_dir / "overlays" / f"{stem}_overlay.png"), overlay)

    stats_full = summarize_values(full_vals)
    stats_roi = summarize_values(roi_vals)
    meta = {
        "image_path": str(img_path),
        "mask_path": str(mask_path),
        "channel": channel,
        "reference_mode": reference_mode,
        "reference_percentile": reference_percentile if reference_mode == "per-image-percentile" else None,
        "i0": float(i0),
        "roi": list(roi) if roi is not None else None,
        "roi_source": roi_source,
        "stats_full": stats_full,
        "stats_roi": stats_roi,
    }
    (output_dir / "metadata" / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "image_name": img_path.name,
        "channel": channel,
        "reference_mode": reference_mode,
        "roi_source": roi_source,
        "i0": float(i0),
        "full_count": stats_full["count"],
        "full_mean": stats_full["mean"],
        "full_median": stats_full["median"],
        "full_std": stats_full["std"],
        "full_p95": stats_full["p95"],
        "roi_count": stats_roi["count"],
        "roi_mean": stats_roi["mean"],
        "roi_median": stats_roi["median"],
        "roi_std": stats_roi["std"],
        "roi_p95": stats_roi["p95"],
    }


def save_summary(rows: list[Dict], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def od_stack_registered(
    registered_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    channel: str = "red",
    reference_mode: str = "per-image-percentile",
    reference_file: Path | None = None,
    reference_percentile: float = 99.5,
    roi: tuple[int, int, int, int] | None = None,
    auto_roi_mode: str = "intersection-bbox",
    png_vmin: float | None = None,
    png_vmax: float | None = None,
) -> list[Dict]:
    ensure_dir(output_dir)
    image_paths = find_images(registered_dir)
    if not image_paths:
        raise FileNotFoundError(f"No registered images found in: {registered_dir}")

    reference_img_bgr = None
    if reference_mode == "reference-file":
        if reference_file is None:
            raise ValueError("reference-file mode requires --reference-file")
        reference_img_bgr = read_image(reference_file)

    resolved_roi, roi_source = resolve_roi(image_paths, masks_dir, roi, auto_roi_mode)
    rows: list[Dict] = []
    for img_path in image_paths:
        row = process_one(
            img_path=img_path,
            masks_dir=masks_dir,
            output_dir=output_dir,
            channel=channel,
            reference_mode=reference_mode,
            reference_img_bgr=reference_img_bgr,
            reference_percentile=reference_percentile,
            roi=resolved_roi,
            roi_source=roi_source,
            png_vmin=png_vmin,
            png_vmax=png_vmax,
        )
        rows.append(row)
    save_summary(rows, output_dir / "summary.csv")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RCF registered stack OD extraction")
    parser.add_argument("--registered-dir", type=Path, required=True, help="registered 图像目录")
    parser.add_argument("--masks-dir", type=Path, required=True, help="masks 目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--channel", type=str, default="red", choices=["red", "green", "blue", "gray", "rgb-mean"], help="OD 通道")
    parser.add_argument("--reference-mode", type=str, default="per-image-percentile", choices=["per-image-percentile", "reference-file"], help="I0 参考模式")
    parser.add_argument("--reference-file", type=Path, default=None, help="参考空白图（registered 坐标系）")
    parser.add_argument("--reference-percentile", type=float, default=99.5, help="per-image-percentile 模式下的高分位数")
    parser.add_argument("--roi", type=str, default=None, help="统一 ROI: x0,y0,x1,y1")
    parser.add_argument("--auto-roi", type=str, default="intersection-bbox", choices=["intersection-bbox"], help="自动 ROI 策略")
    parser.add_argument("--png-vmin", type=float, default=None, help="OD 可视化下限")
    parser.add_argument("--png-vmax", type=float, default=None, help="OD 可视化上限")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = od_stack_registered(
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
    for row in rows:
        print(f"[OK] {row['image_name']}")
    print(f"Done. Results saved to: {args.output_dir}")
    return 0
