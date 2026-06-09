from __future__ import annotations

from abc import ABC, abstractmethod
import warnings

import cv2
import numpy as np


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TemporalDepthSmoother:
    def __init__(self, alpha: float = 0.65) -> None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("Temporal smoothing alpha must be in range (0, 1].")
        self._alpha = alpha
        self._previous: np.ndarray | None = None

    def apply(self, depth: np.ndarray) -> np.ndarray:
        if self._previous is None:
            self._previous = depth
            return depth
        smoothed = (self._alpha * depth) + ((1 - self._alpha) * self._previous)
        self._previous = smoothed
        return smoothed


def normalize_depth_map(depth: np.ndarray) -> np.ndarray:
    min_value = float(depth.min())
    max_value = float(depth.max())
    spread = max_value - min_value
    if spread <= 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    normalized = (depth - min_value) / spread
    return normalized.astype(np.float32)


class LumaDepthEstimator(DepthEstimator):
    def __init__(self, smoothing_alpha: float = 0.65) -> None:
        self._smoother = TemporalDepthSmoother(alpha=smoothing_alpha)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.1, sigmaY=1.1)
        normalized = normalize_depth_map(blurred)
        return self._smoother.apply(normalized)


class MidasDepthEstimator(DepthEstimator):
    def __init__(self, device: str = "auto", model_name: str = "Intel/dpt-hybrid-midas") -> None:
        self._device = device
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._torch = None
        self._smoother = TemporalDepthSmoother(alpha=0.7)

    def _resolve_device(self) -> str:
        if self._device == "cpu":
            return "cpu"
        if self._device == "cuda":
            return "cuda"
        assert self._torch is not None
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        import torch  # type: ignore
        from transformers import AutoImageProcessor, DPTForDepthEstimation  # type: ignore

        self._torch = torch
        torch_device = self._resolve_device()
        self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._model = DPTForDepthEstimation.from_pretrained(self._model_name)
        self._model.to(torch_device)
        self._model.eval()

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._load()
        assert self._model is not None and self._processor is not None and self._torch is not None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        torch_device = self._resolve_device()
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}

        with self._torch.no_grad():
            predicted = self._model(**inputs).predicted_depth
            resized = self._torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = resized.detach().cpu().numpy()
        normalized = normalize_depth_map(depth.astype(np.float32))
        return self._smoother.apply(normalized)


class AutoDepthEstimator(DepthEstimator):
    def __init__(self, preferred: DepthEstimator, fallback: DepthEstimator) -> None:
        self._preferred = preferred
        self._fallback = fallback
        self._use_fallback = False

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._use_fallback:
            return self._fallback.estimate(frame_bgr)
        try:
            return self._preferred.estimate(frame_bgr)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f"Preferred depth backend failed ({exc}); switching to luma depth.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._use_fallback = True
            return self._fallback.estimate(frame_bgr)


def create_depth_estimator(backend: str, device: str) -> DepthEstimator:
    if backend == "luma":
        return LumaDepthEstimator()

    if backend == "midas":
        return MidasDepthEstimator(device=device)

    if backend == "auto":
        return AutoDepthEstimator(
            preferred=MidasDepthEstimator(device=device),
            fallback=LumaDepthEstimator(),
        )

    raise ValueError(f"Unsupported depth backend: {backend}")
