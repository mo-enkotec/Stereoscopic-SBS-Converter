from __future__ import annotations

import cv2
import numpy as np


def _prepare_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape[:2] != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    return np.clip(depth, 0.0, 1.0)


def _disparity_map(depth: np.ndarray, width: int, strength: float) -> np.ndarray:
    max_shift = max(1.0, width * 0.03 * strength)
    centered_depth = (depth - 0.5) * 2.0
    return centered_depth * max_shift


def _remap_eye(
    frame: np.ndarray,
    shifted_x: np.ndarray,
    y_coords: np.ndarray,
    width: int,
) -> np.ndarray:
    out_of_bounds = ((shifted_x < 0) | (shifted_x > (width - 1))).astype(np.uint8) * 255
    clipped = np.clip(shifted_x, 0, width - 1).astype(np.float32)
    remapped = cv2.remap(
        frame,
        clipped,
        y_coords,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if np.any(out_of_bounds):
        remapped = cv2.inpaint(remapped, out_of_bounds, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    return remapped


def synthesize_stereo_views(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    stereo_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    depth = _prepare_depth(depth, width, height)
    disparity = _disparity_map(depth, width, stereo_strength)

    x_coords = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    y_coords = np.tile(np.arange(height, dtype=np.float32).reshape(height, 1), (1, width))

    left_shifted_x = x_coords - disparity
    right_shifted_x = x_coords + disparity

    left_eye = _remap_eye(frame_bgr, left_shifted_x, y_coords, width)
    right_eye = _remap_eye(frame_bgr, right_shifted_x, y_coords, width)
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
