"""Lightweight video feature extraction helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


def ensure_rgb_uint8(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError(f"Expected RGB/RGBA input, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _resolve_roi_bounds(roi: Optional[Dict[str, Any]], width: int, height: int) -> Tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    x = roi.get("x", 0)
    y = roi.get("y", 0)
    w = roi.get("w", width)
    h = roi.get("h", height)

    normalized = bool(roi.get("normalized", False))
    if not normalized:
        if any(isinstance(value, float) and 0.0 <= float(value) <= 1.0 for value in (x, y, w, h)):
            normalized = all(0.0 <= float(value) <= 1.0 for value in (x, y, w, h))

    if normalized:
        x = int(round(float(x) * width))
        y = int(round(float(y) * height))
        w = int(round(float(w) * width))
        h = int(round(float(h) * height))
    else:
        x = int(round(float(x)))
        y = int(round(float(y)))
        w = int(round(float(w)))
        h = int(round(float(h)))

    x0 = max(0, min(width - 1, x))
    y0 = max(0, min(height - 1, y))
    x1 = max(x0 + 1, min(width, x0 + max(1, w)))
    y1 = max(y0 + 1, min(height, y0 + max(1, h)))
    return x0, y0, x1, y1


def apply_camera_transforms(frame: Any, camera_cfg, roi_override: Optional[Dict[str, Any]] = None) -> np.ndarray:
    image = ensure_rgb_uint8(frame)
    roi = camera_cfg.roi if getattr(camera_cfg, "roi", None) is not None else roi_override
    x0, y0, x1, y1 = _resolve_roi_bounds(roi, image.shape[1], image.shape[0])
    image = image[y0:y1, x0:x1]

    rotation = int(getattr(camera_cfg, "rotation", 0) or 0) % 360
    if rotation in (90, 180, 270):
        image = np.rot90(image, k=rotation // 90).copy()
    elif rotation != 0:
        pil_image = Image.fromarray(image)
        image = np.asarray(pil_image.rotate(rotation, expand=True))

    resize = getattr(camera_cfg, "resize", None)
    if resize:
        target_w = int(resize[0])
        target_h = int(resize[1] if len(resize) > 1 else resize[0])
        pil_image = Image.fromarray(image)
        image = np.asarray(pil_image.resize((target_w, target_h), resample=Image.BILINEAR))

    return ensure_rgb_uint8(image)


def _grayscale(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32).mean(axis=2)


def extract_visual_features(
    current_frame: np.ndarray,
    prev_frame: Optional[np.ndarray],
    baseline_frame: Optional[np.ndarray],
    feature_cfg,
) -> tuple[np.ndarray, Dict[str, float]]:
    current = ensure_rgb_uint8(current_frame)
    current_gray = _grayscale(current)

    prev_gray = _grayscale(prev_frame) if prev_frame is not None else current_gray
    baseline_gray = _grayscale(baseline_frame) if baseline_frame is not None else current_gray

    diff_prev = np.abs(current_gray - prev_gray)
    diff_base = np.abs(current_gray - baseline_gray)

    motion_energy = float(diff_prev.mean() / 255.0)
    baseline_shift = float(diff_base.mean() / 255.0)
    brightness = float(current_gray.mean() / 255.0)
    contrast = float(current_gray.std() / 255.0)

    gx = np.diff(current_gray, axis=1, append=current_gray[:, -1:])
    gy = np.diff(current_gray, axis=0, append=current_gray[-1:, :])
    edge_mag = np.sqrt((gx * gx) + (gy * gy))
    edge_density = float(np.mean(edge_mag > float(feature_cfg.edge_density_threshold)))

    motion_mask = diff_prev > float(feature_cfg.frame_diff_threshold)
    if np.any(motion_mask):
        ys, xs = np.nonzero(motion_mask)
        motion_cx = float(xs.mean() / max(1, current.shape[1] - 1))
        motion_cy = float(ys.mean() / max(1, current.shape[0] - 1))
        motion_area = float(motion_mask.mean())
    else:
        motion_cx = 0.5
        motion_cy = 0.5
        motion_area = 0.0

    visibility_score = float(
        np.clip(
            (0.45 * min(1.0, contrast * 4.0))
            + (0.35 * min(1.0, edge_density * 8.0))
            + (0.20 * min(1.0, max(0.0, motion_area * 12.0))),
            0.0,
            1.0,
        )
    )

    vector = np.asarray(
        [
            brightness,
            contrast,
            edge_density,
            motion_energy,
            motion_area,
            motion_cx,
            motion_cy,
            baseline_shift,
        ],
        dtype=np.float32,
    )
    metadata = {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "motion_energy": motion_energy,
        "motion_area": motion_area,
        "motion_cx": motion_cx,
        "motion_cy": motion_cy,
        "baseline_shift": baseline_shift,
        "visibility_score": visibility_score,
    }
    return vector, metadata
