from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class _StereoFrameCache:
    x_coords_f32: np.ndarray
    y_coords_f32: np.ndarray


_FRAME_CACHE: dict[tuple[int, int], _StereoFrameCache] = {}


def _get_frame_cache(width: int, height: int) -> _StereoFrameCache:
    key = (height, width)
    cache = _FRAME_CACHE.get(key)
    if cache is not None:
        return cache

    x_coords = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    y_coords = np.broadcast_to(
        np.arange(height, dtype=np.float32).reshape(height, 1),
        (height, width),
    )
    cache = _StereoFrameCache(
        x_coords_f32=x_coords,
        y_coords_f32=y_coords,
    )
    _FRAME_CACHE.clear()
    _FRAME_CACHE[key] = cache
    return cache


def _prepare_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape[:2] != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    return np.clip(depth, 0.0, 1.0)


def _disparity_map(
    depth: np.ndarray,
    width: int,
    strength: float,
    max_disparity_px: int | None,
) -> np.ndarray:
    if max_disparity_px is None:
        max_shift = max(1.0, width * 0.03 * strength)
    else:
        if max_disparity_px <= 0:
            raise ValueError("max_disparity_px must be greater than zero.")
        max_shift = max_disparity_px * strength

    # Compress far-field disparity so background doesn't get dragged around foreground edges.
    near_weight = np.clip((depth - 0.45) / 0.55, 0.0, 1.0) ** 1.25
    return near_weight * max_shift


def _forward_warp_eye(
    frame: np.ndarray,
    depth: np.ndarray,
    shifted_x: np.ndarray,
) -> np.ndarray:
    height, width = frame.shape[:2]
    cache = _get_frame_cache(width, height)
    target_x = np.rint(shifted_x).astype(np.int32)

    valid = (target_x >= 0) & (target_x < width)
    if not np.any(valid):
        return frame.copy()

    src_y, src_x = np.nonzero(valid)
    dst_x = target_x[valid]
    dst_y = src_y
    depth_values = depth[valid].astype(np.float32, copy=False)
    dst_linear = (dst_y.astype(np.int64) * width) + dst_x.astype(np.int64)

    result = np.zeros_like(frame)
    top_depth = np.full(height * width, -np.inf, dtype=np.float32)
    np.maximum.at(top_depth, dst_linear, depth_values)
    visible = depth_values >= (top_depth[dst_linear] - 1e-6)
    result[dst_y[visible], dst_x[visible]] = frame[src_y[visible], src_x[visible]]

    occupied = np.zeros((height, width), dtype=bool)
    occupied[dst_y[visible], dst_x[visible]] = True

    holes = ~occupied
    if np.any(holes):
        fallback_map = np.clip(shifted_x, 0, width - 1).astype(np.float32)
        fallback = cv2.remap(
            frame,
            fallback_map,
            cache.y_coords_f32,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        result[holes] = fallback[holes]
    return result


def synthesize_stereo_views(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    stereo_strength: float,
    max_disparity_px: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    depth = _prepare_depth(depth, width, height)
    disparity = _disparity_map(depth, width, stereo_strength, max_disparity_px)
    cache = _get_frame_cache(width, height)

    x_coords = cache.x_coords_f32

    left_shifted_x = x_coords - (disparity * 0.5)
    right_shifted_x = x_coords + (disparity * 0.5)

    left_eye = _forward_warp_eye(frame_bgr, depth, left_shifted_x)
    right_eye = _forward_warp_eye(frame_bgr, depth, right_shifted_x)
    return left_eye, right_eye


def compose_sbs(
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "full":
        return np.concatenate((left_eye, right_eye), axis=1)
    if mode == "half":
        half_width = max(2, left_eye.shape[1] // 2)
        left_half = cv2.resize(left_eye, (half_width, left_eye.shape[0]), interpolation=cv2.INTER_AREA)
        right_half = cv2.resize(
            right_eye, (half_width, right_eye.shape[0]), interpolation=cv2.INTER_AREA
        )
        return np.concatenate((left_half, right_half), axis=1)
    raise ValueError(f"Unsupported SBS mode: {mode}")
