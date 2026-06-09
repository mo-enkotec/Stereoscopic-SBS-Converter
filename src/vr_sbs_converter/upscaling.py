from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class Upscaler(ABC):
    @abstractmethod
    def upscale(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        raise NotImplementedError


class InterpolationUpscaler(Upscaler):
    def __init__(self, interpolation: int) -> None:
        self._interpolation = interpolation

    def upscale(self, frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        if frame.shape[1] == target_width and frame.shape[0] == target_height:
            return frame
        return cv2.resize(frame, (target_width, target_height), interpolation=self._interpolation)


def create_default_upscaler() -> Upscaler:
    # Lanczos offers sharper reconstruction than bicubic for quality-first defaults.
    return InterpolationUpscaler(interpolation=cv2.INTER_LANCZOS4)


def compute_target_dimensions(
    source_width: int,
    source_height: int,
    target_height: int,
) -> tuple[int, int]:
    if target_height < source_height:
        raise ValueError(
            f"Target height {target_height} is smaller than source height {source_height}. "
            "Disable --upscale or provide a larger target."
        )
    aspect_ratio = source_width / source_height
    target_width = int(round(target_height * aspect_ratio))

    # Force even dimensions for broad encoder/player compatibility.
    if target_width % 2:
        target_width += 1
    if target_height % 2:
        target_height += 1

    return target_width, target_height
