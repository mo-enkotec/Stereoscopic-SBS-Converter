from __future__ import annotations

import cv2
import numpy as np


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
    x_coords = np.tile(np.arange(width, dtype=np.int32), (height, 1))
    y_coords = np.tile(np.arange(height, dtype=np.int32).reshape(height, 1), (1, width))
    target_x = np.rint(shifted_x).astype(np.int32)

    valid = (target_x >= 0) & (target_x < width)
    if not np.any(valid):
        return frame.copy()

    src_x = x_coords[valid]
    src_y = y_coords[valid]
    dst_x = target_x[valid]
    dst_y = src_y
    depth_values = depth[valid]

    # Render far-to-near so closer pixels overwrite farther ones.
    order = np.argsort(depth_values, kind="stable")

    result = np.zeros_like(frame)
    occupied = np.zeros((height, width), dtype=bool)
    result[dst_y[order], dst_x[order]] = frame[src_y[order], src_x[order]]
    occupied[dst_y[order], dst_x[order]] = True

    holes = (~occupied).astype(np.uint8) * 255
    if np.any(holes):
        fallback_map = np.clip(shifted_x, 0, width - 1).astype(np.float32)
        y_coords = np.tile(np.arange(height, dtype=np.float32).reshape(height, 1), (1, width))
        fallback = cv2.remap(
            frame,
            fallback_map,
            y_coords,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mask = holes.astype(bool)
        result[mask] = fallback[mask]
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

    x_coords = np.tile(np.arange(width, dtype=np.float32), (height, 1))

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
