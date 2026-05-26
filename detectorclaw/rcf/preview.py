from __future__ import annotations

import ctypes
import io
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
try:
    import torch
    import torch.nn.functional as torch_f
except Exception:  # noqa: BLE001
    torch = None
    torch_f = None

from .autocrop import four_point_transform
from .autocrop import order_points
from .runtime_cache import LRUCache
from .runtime_cache import file_signature

DEFAULT_SCAN_MAX_DIM = 1600
DEFAULT_PATCH_MAX_DIM = 640
DEFAULT_RAW_PATCH_MAX_DIM = 640
NATIVE_PREVIEW_ENV = "DETECTORCLAW_RCF_NATIVE_PREVIEW_BIN"
NATIVE_PREVIEW_LIB_ENV = "DETECTORCLAW_RCF_NATIVE_PREVIEW_LIB"
PREVIEW_BACKEND_ENV = "DETECTORCLAW_RCF_PREVIEW_BACKEND"
_NATIVE_PREVIEW_LIBRARY = None
_PREVIEW_RESULT_CACHE = LRUCache[tuple[bytes, str]](max_entries=128)


def normalize_preview_format(value: str | None, default: str = "png") -> str:
    normalized = (value or default).lower()
    if normalized in {"jpg", "jpeg"}:
        return "jpeg"
    if normalized == "png":
        return "png"
    raise ValueError("preview format must be png or jpeg")


def normalize_preview_backend(value: str | None) -> str:
    backend = (value or os.environ.get(PREVIEW_BACKEND_ENV) or "auto").lower()
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("preview backend must be auto, cpu, or cuda")
    if backend == "cuda" and not _cuda_available():
        return "cpu"
    if backend == "auto":
        return "cuda" if _cuda_available() else "cpu"
    return backend


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def _preview_cache_key(
    command_name: str,
    scan_file: Path,
    *,
    max_dim: int,
    preview_format: str,
    quality: int,
    backend: str | None = None,
    bbox: list[int] | None = None,
    quad_points: list[list[float]] | None = None,
    crop_bbox: list[int] | None = None,
) -> tuple:
    normalized_quad = None
    if quad_points is not None:
        normalized_quad = tuple(tuple(round(float(value), 4) for value in point) for point in quad_points)
    normalized_crop = tuple(int(value) for value in crop_bbox) if crop_bbox is not None else None
    normalized_bbox = tuple(int(value) for value in bbox) if bbox is not None else None
    return (
        command_name,
        file_signature(scan_file),
        int(max_dim),
        normalize_preview_format(preview_format),
        int(quality),
        backend,
        normalized_bbox,
        normalized_quad,
        normalized_crop,
    )


def resize_for_preview(image: Image.Image, max_dim: int | None) -> Image.Image:
    if max_dim is None or max_dim <= 0:
        return image
    largest = max(image.size)
    if largest <= max_dim:
        return image
    scale = max_dim / largest
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _normalize_bbox(image: Image.Image, bbox: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = [int(round(value)) for value in bbox]
    image_width, image_height = image.size
    x = min(max(0, x), max(0, image_width - 1))
    y = min(max(0, y), max(0, image_height - 1))
    width = max(1, width)
    height = max(1, height)
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    return x, y, width, height


def crop_to_bbox(image: Image.Image, bbox: list[int] | tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = _normalize_bbox(image, bbox)
    return image.crop((x, y, x + width, y + height))


def encode_preview_image(image: Image.Image, preview_format: str, quality: int) -> tuple[bytes, str]:
    preview_format = normalize_preview_format(preview_format)
    buffer = io.BytesIO()
    if preview_format == "jpeg":
        image.convert("RGB").save(buffer, format="JPEG", quality=max(1, min(95, quality)), optimize=True)
        return buffer.getvalue(), "image/jpeg"
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"


def render_scan_preview(
    scan_file: Path,
    max_dim: int = DEFAULT_SCAN_MAX_DIM,
    preview_format: str = "png",
    quality: int = 80,
) -> tuple[bytes, str]:
    payload = {
        "scan_file": str(scan_file),
        "max_dim": int(max_dim),
        "preview_format": normalize_preview_format(preview_format),
        "quality": int(quality),
    }
    cache_key = _preview_cache_key(
        "scan-preview",
        scan_file,
        max_dim=max_dim,
        preview_format=payload["preview_format"],
        quality=quality,
    )
    cached = _PREVIEW_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    native_result = _run_native_preview("scan-preview", payload)
    if native_result is not None:
        return _PREVIEW_RESULT_CACHE.set(cache_key, native_result)

    image = Image.open(scan_file).convert("RGB")
    image = resize_for_preview(image, max_dim)
    return _PREVIEW_RESULT_CACHE.set(cache_key, encode_preview_image(image, payload["preview_format"], quality))


def render_bbox_preview(
    scan_file: Path,
    bbox: list[int],
    max_dim: int = DEFAULT_RAW_PATCH_MAX_DIM,
    preview_format: str = "png",
    quality: int = 80,
) -> tuple[bytes, str]:
    payload = {
        "scan_file": str(scan_file),
        "bbox": [int(value) for value in bbox],
        "max_dim": int(max_dim),
        "preview_format": normalize_preview_format(preview_format),
        "quality": int(quality),
    }
    cache_key = _preview_cache_key(
        "bbox-preview",
        scan_file,
        max_dim=max_dim,
        preview_format=payload["preview_format"],
        quality=quality,
        bbox=payload["bbox"],
    )
    cached = _PREVIEW_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    native_result = _run_native_preview("bbox-preview", payload)
    if native_result is not None:
        return _PREVIEW_RESULT_CACHE.set(cache_key, native_result)

    image = Image.open(scan_file).convert("RGB")
    image = crop_to_bbox(image, bbox)
    image = resize_for_preview(image, max_dim)
    return _PREVIEW_RESULT_CACHE.set(cache_key, encode_preview_image(image, payload["preview_format"], quality))


def render_patch_preview(
    scan_file: Path,
    quad_points: list[list[float]],
    crop_bbox: list[int] | None = None,
    max_dim: int = DEFAULT_PATCH_MAX_DIM,
    preview_format: str = "png",
    quality: int = 80,
    backend: str = "auto",
) -> tuple[bytes, str]:
    normalized_backend = normalize_preview_backend(backend)
    payload = {
        "scan_file": str(scan_file),
        "quad_points": [[float(x), float(y)] for x, y in quad_points],
        "crop_bbox": [int(value) for value in crop_bbox] if crop_bbox is not None else None,
        "max_dim": int(max_dim),
        "preview_format": normalize_preview_format(preview_format),
        "quality": int(quality),
        "backend": normalized_backend,
    }
    cache_key = _preview_cache_key(
        "patch-preview",
        scan_file,
        max_dim=max_dim,
        preview_format=payload["preview_format"],
        quality=quality,
        backend=normalized_backend,
        quad_points=payload["quad_points"],
        crop_bbox=payload["crop_bbox"],
    )
    cached = _PREVIEW_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if normalized_backend == "cpu":
        native_result = _run_native_preview("patch-preview", payload)
        if native_result is not None:
            return _PREVIEW_RESULT_CACHE.set(cache_key, native_result)

    image = extract_patch_image(
        scan_file,
        payload["quad_points"],
        crop_bbox=payload["crop_bbox"],
        backend=normalized_backend,
        max_dim=max_dim,
    )
    image = resize_for_preview(image, max_dim)
    return _PREVIEW_RESULT_CACHE.set(cache_key, encode_preview_image(image, payload["preview_format"], quality))


def _ordered_quad_and_size(
    quad_points: list[list[float]],
    max_dim: int | None = None,
    crop_bbox: list[int] | None = None,
) -> tuple[np.ndarray, int, int, float]:
    ordered = order_points(np.array([[float(x), float(y)] for x, y in quad_points], dtype=np.float32))
    tl, tr, br, bl = ordered
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    dst_w = max(2, int(round(max(width_a, width_b))))
    dst_h = max(2, int(round(max(height_a, height_b))))
    scale = 1.0
    if max_dim is not None and max_dim > 0:
        if crop_bbox is not None:
            largest = max(int(crop_bbox[2]), int(crop_bbox[3]))
        else:
            largest = max(dst_w, dst_h)
        if largest > max_dim:
            scale = float(max_dim) / float(largest)
            dst_w = max(2, int(round(dst_w * scale)))
            dst_h = max(2, int(round(dst_h * scale)))
    return ordered, dst_w, dst_h, scale


def _scale_crop_bbox(crop_bbox: list[int] | None, scale: float) -> list[int] | None:
    if crop_bbox is None:
        return None
    if abs(scale - 1.0) < 1e-6:
        return [int(value) for value in crop_bbox]
    x, y, width, height = [int(value) for value in crop_bbox]
    return [
        max(0, int(round(x * scale))),
        max(0, int(round(y * scale))),
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    ]


def _source_roi_for_quads(
    quad_points_list: list[list[list[float]]],
    image_width: int,
    image_height: int,
    padding: int = 4,
) -> tuple[int, int, int, int]:
    xs = [float(point[0]) for quad in quad_points_list for point in quad]
    ys = [float(point[1]) for quad in quad_points_list for point in quad]
    x0 = max(0, int(np.floor(min(xs))) - padding)
    y0 = max(0, int(np.floor(min(ys))) - padding)
    x1 = min(image_width, int(np.ceil(max(xs))) + padding)
    y1 = min(image_height, int(np.ceil(max(ys))) + padding)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _translate_quad_points(quad_points: list[list[float]], x0: int, y0: int) -> list[list[float]]:
    return [[float(x) - float(x0), float(y) - float(y0)] for x, y in quad_points]


def extract_patch_images(
    scan_file: Path,
    quad_points_list: list[list[list[float]]],
    crop_bboxes: list[list[int] | None] | None = None,
    backend: str = "auto",
    max_dim: int | None = None,
) -> list[Image.Image]:
    normalized_backend = normalize_preview_backend(backend)
    if crop_bboxes is None:
        crop_bboxes = [None] * len(quad_points_list)
    if len(crop_bboxes) != len(quad_points_list):
        raise ValueError("crop_bboxes must match quad_points_list length")
    if normalized_backend == "cuda":
        return _extract_patch_images_cuda(scan_file, quad_points_list, crop_bboxes, max_dim=max_dim)
    return _extract_patch_images_cpu(scan_file, quad_points_list, crop_bboxes, max_dim=max_dim)


def extract_patch_image(
    scan_file: Path,
    quad_points: list[list[float]],
    crop_bbox: list[int] | None = None,
    backend: str = "auto",
    max_dim: int | None = None,
) -> Image.Image:
    return extract_patch_images(
        scan_file=scan_file,
        quad_points_list=[quad_points],
        crop_bboxes=[crop_bbox],
        backend=backend,
        max_dim=max_dim,
    )[0]


def _extract_patch_image_cpu(
    scan_file: Path,
    quad_points: list[list[float]],
    crop_bbox: list[int] | None = None,
    max_dim: int | None = None,
) -> Image.Image:
    source_image = Image.open(scan_file).convert("RGB")
    roi_x, roi_y, roi_w, roi_h = _source_roi_for_quads([quad_points], source_image.width, source_image.height)
    source_image = source_image.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))
    source_bgr = np.array(source_image, dtype=np.uint8)[:, :, ::-1]
    translated_quad = _translate_quad_points(quad_points, roi_x, roi_y)
    ordered, dst_w, dst_h, scale = _ordered_quad_and_size(translated_quad, max_dim=max_dim, crop_bbox=crop_bbox)
    destination = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    patch_bgr = cv2.warpPerspective(source_bgr, transform, (dst_w, dst_h))
    patch_rgb = patch_bgr[:, :, ::-1]
    image = Image.fromarray(patch_rgb)
    if crop_bbox is not None:
        image = crop_to_bbox(image, _scale_crop_bbox(crop_bbox, scale))
    return image


def _extract_patch_images_cpu(
    scan_file: Path,
    quad_points_list: list[list[list[float]]],
    crop_bboxes: list[list[int] | None],
    max_dim: int | None = None,
) -> list[Image.Image]:
    return [
        _extract_patch_image_cpu(scan_file, quad_points, crop_bbox=crop_bbox, max_dim=max_dim)
        for quad_points, crop_bbox in zip(quad_points_list, crop_bboxes, strict=False)
    ]


def _extract_patch_image_cuda(
    scan_file: Path,
    quad_points: list[list[float]],
    crop_bbox: list[int] | None = None,
    max_dim: int | None = None,
) -> Image.Image:
    if torch is None or torch_f is None or not torch.cuda.is_available():
        return _extract_patch_image_cpu(scan_file, quad_points, crop_bbox=crop_bbox, max_dim=max_dim)

    source_image = Image.open(scan_file).convert("RGB")
    roi_x, roi_y, roi_w, roi_h = _source_roi_for_quads([quad_points], source_image.width, source_image.height)
    source_image = source_image.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))
    source_rgb = np.asarray(source_image, dtype=np.float32)
    translated_quad = _translate_quad_points(quad_points, roi_x, roi_y)
    ordered, dst_w, dst_h, scale = _ordered_quad_and_size(translated_quad, max_dim=max_dim, crop_bbox=crop_bbox)

    destination = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    inverse = cv2.getPerspectiveTransform(destination, ordered)

    device = torch.device("cuda")
    image_tensor = torch.from_numpy(np.ascontiguousarray(source_rgb / 255.0)).to(device=device, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    xs = torch.arange(dst_w, device=device, dtype=torch.float32)
    ys = torch.arange(dst_h, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    homography = torch.tensor(inverse, device=device, dtype=torch.float32)

    denom = homography[2, 0] * grid_x + homography[2, 1] * grid_y + homography[2, 2] * ones
    src_x = (homography[0, 0] * grid_x + homography[0, 1] * grid_y + homography[0, 2] * ones) / denom
    src_y = (homography[1, 0] * grid_x + homography[1, 1] * grid_y + homography[1, 2] * ones) / denom

    src_w = float(source_rgb.shape[1] - 1)
    src_h = float(source_rgb.shape[0] - 1)
    norm_x = (src_x / max(src_w, 1.0)) * 2.0 - 1.0
    norm_y = (src_y / max(src_h, 1.0)) * 2.0 - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)

    warped = torch_f.grid_sample(
        image_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    patch_rgb = (
        warped.squeeze(0)
        .permute(1, 2, 0)
        .mul(255.0)
        .clamp(0.0, 255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    image = Image.fromarray(patch_rgb, mode="RGB")
    if crop_bbox is not None:
        image = crop_to_bbox(image, _scale_crop_bbox(crop_bbox, scale))
    return image


def _extract_patch_images_cuda(
    scan_file: Path,
    quad_points_list: list[list[list[float]]],
    crop_bboxes: list[list[int] | None],
    max_dim: int | None = None,
) -> list[Image.Image]:
    if torch is None or torch_f is None or not torch.cuda.is_available():
        return _extract_patch_images_cpu(scan_file, quad_points_list, crop_bboxes, max_dim=max_dim)
    if not quad_points_list:
        return []

    source_image = Image.open(scan_file).convert("RGB")
    roi_x, roi_y, roi_w, roi_h = _source_roi_for_quads(quad_points_list, source_image.width, source_image.height)
    source_image = source_image.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))
    source_rgb = np.asarray(source_image, dtype=np.float32)
    translated_quads = [_translate_quad_points(quad, roi_x, roi_y) for quad in quad_points_list]

    specs: list[tuple[np.ndarray, int, int, float]] = [
        _ordered_quad_and_size(quad, max_dim=max_dim, crop_bbox=crop_bbox)
        for quad, crop_bbox in zip(translated_quads, crop_bboxes, strict=False)
    ]
    max_w = max(spec[1] for spec in specs)
    max_h = max(spec[2] for spec in specs)
    destination = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )

    device = torch.device("cuda")
    image_tensor = torch.from_numpy(np.ascontiguousarray(source_rgb / 255.0)).to(device=device, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).expand(len(specs), -1, -1, -1)

    xs = torch.arange(max_w, device=device, dtype=torch.float32)
    ys = torch.arange(max_h, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    src_w = float(source_rgb.shape[1] - 1)
    src_h = float(source_rgb.shape[0] - 1)

    grids = []
    for ordered, dst_w, dst_h, _scale in specs:
        scaled_destination = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32,
        )
        inverse = cv2.getPerspectiveTransform(scaled_destination, ordered)
        homography = torch.tensor(inverse, device=device, dtype=torch.float32)
        denom = homography[2, 0] * grid_x + homography[2, 1] * grid_y + homography[2, 2] * ones
        src_x = (homography[0, 0] * grid_x + homography[0, 1] * grid_y + homography[0, 2] * ones) / denom
        src_y = (homography[1, 0] * grid_x + homography[1, 1] * grid_y + homography[1, 2] * ones) / denom
        norm_x = (src_x / max(src_w, 1.0)) * 2.0 - 1.0
        norm_y = (src_y / max(src_h, 1.0)) * 2.0 - 1.0
        grids.append(torch.stack((norm_x, norm_y), dim=-1))

    grid = torch.stack(grids, dim=0)
    warped = torch_f.grid_sample(
        image_tensor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    images: list[Image.Image] = []
    for index, ((_, dst_w, dst_h, scale), crop_bbox) in enumerate(zip(specs, crop_bboxes, strict=False)):
        patch_rgb = (
            warped[index, :, :dst_h, :dst_w]
            .permute(1, 2, 0)
            .mul(255.0)
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        image = Image.fromarray(patch_rgb, mode="RGB")
        if crop_bbox is not None:
            image = crop_to_bbox(image, _scale_crop_bbox(crop_bbox, scale))
        images.append(image)
    return images


def locate_native_preview_library() -> Path | None:
    env_path = os.environ.get(NATIVE_PREVIEW_LIB_ENV)
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


def load_native_preview_library():
    global _NATIVE_PREVIEW_LIBRARY

    if _NATIVE_PREVIEW_LIBRARY is False:
        return None
    if _NATIVE_PREVIEW_LIBRARY is not None:
        return _NATIVE_PREVIEW_LIBRARY

    path = locate_native_preview_library()
    if path is None:
        _NATIVE_PREVIEW_LIBRARY = False
        return None

    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        _NATIVE_PREVIEW_LIBRARY = False
        return None

    library.rcf_preview_command_json.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    library.rcf_preview_command_json.restype = ctypes.c_void_p
    if hasattr(library, "rcf_dose_batch_json"):
        library.rcf_dose_batch_json.argtypes = [ctypes.c_char_p]
        library.rcf_dose_batch_json.restype = ctypes.c_void_p
    library.rcf_last_error_message.argtypes = []
    library.rcf_last_error_message.restype = ctypes.c_void_p
    library.rcf_free_string.argtypes = [ctypes.c_void_p]
    library.rcf_free_string.restype = None
    _NATIVE_PREVIEW_LIBRARY = library
    return library


def locate_native_preview_binary() -> Path | None:
    env_path = os.environ.get(NATIVE_PREVIEW_ENV)
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


def _consume_native_string(library, pointer) -> str:
    if not pointer:
        return ""
    try:
        return ctypes.string_at(pointer).decode("utf-8")
    finally:
        library.rcf_free_string(pointer)


def _run_native_preview_via_library(library, command_name: str, payload: dict) -> tuple[bytes, str]:
    output_pointer = library.rcf_preview_command_json(
        command_name.encode("utf-8"),
        json.dumps(payload).encode("utf-8"),
    )
    if not output_pointer:
        error_pointer = library.rcf_last_error_message()
        message = _consume_native_string(library, error_pointer) or "native preview library call failed"
        raise RuntimeError(message)
    decoded = json.loads(_consume_native_string(library, output_pointer))
    return bytes.fromhex(decoded["content_hex"]), decoded["media_type"]


def _run_native_preview(command_name: str, payload: dict) -> tuple[bytes, str] | None:
    library = load_native_preview_library()
    if library is not None:
        return _run_native_preview_via_library(library, command_name, payload)

    binary = locate_native_preview_binary()
    if binary is None:
        return None
    result = subprocess.run(
        [str(binary), command_name],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    decoded = json.loads(result.stdout)
    return bytes.fromhex(decoded["content_hex"]), decoded["media_type"]


def _run_native_dose_batch_via_library(library, payload: dict) -> dict:
    if not hasattr(library, "rcf_dose_batch_json"):
        raise RuntimeError("native preview library does not expose rcf_dose_batch_json")
    output_pointer = library.rcf_dose_batch_json(json.dumps(payload).encode("utf-8"))
    if not output_pointer:
        error_pointer = library.rcf_last_error_message()
        message = _consume_native_string(library, error_pointer) or "native dose batch library call failed"
        raise RuntimeError(message)
    return json.loads(_consume_native_string(library, output_pointer))


def run_native_dose_batch(payload: dict) -> dict | None:
    library = load_native_preview_library()
    if library is not None:
        try:
            return _run_native_dose_batch_via_library(library, payload)
        except Exception:  # noqa: BLE001
            pass

    binary = locate_native_preview_binary()
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [str(binary), "dose-batch"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return json.loads(result.stdout)
