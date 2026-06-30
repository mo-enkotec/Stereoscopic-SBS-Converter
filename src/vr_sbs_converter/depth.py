from __future__ import annotations

from abc import ABC, abstractmethod
import warnings

import cv2
import numpy as np


class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def estimate_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        return [self.estimate(frame) for frame in frames_bgr]


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


def _edge_mask(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    max_value = float(magnitude.max())
    if max_value <= 1e-8:
        return np.zeros_like(magnitude, dtype=np.float32)
    return np.clip(magnitude / max_value, 0.0, 1.0).astype(np.float32)


def normalize_depth_map(depth: np.ndarray) -> np.ndarray:
    min_value = float(depth.min())
    max_value = float(depth.max())
    spread = max_value - min_value
    if spread <= 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    normalized = (depth - min_value) / spread
    return normalized.astype(np.float32)


def condition_depth_for_stereo(
    depth: np.ndarray,
    frame_bgr: np.ndarray,
    edge_protect_strength: float,
) -> np.ndarray:
    normalized = normalize_depth_map(depth.astype(np.float32))
    if edge_protect_strength <= 0:
        return normalized

    smoothed = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1.2, sigmaY=1.2)
    edge_weight = np.clip(_edge_mask(frame_bgr) * edge_protect_strength, 0.0, 1.0)
    conditioned = (edge_weight * normalized) + ((1.0 - edge_weight) * smoothed)
    return normalize_depth_map(conditioned)


class LumaDepthEstimator(DepthEstimator):
    def __init__(self, smoothing_alpha: float = 0.65, edge_protect_strength: float = 0.75) -> None:
        self._smoother = TemporalDepthSmoother(alpha=smoothing_alpha)
        self._edge_protect_strength = edge_protect_strength

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.1, sigmaY=1.1)
        conditioned = condition_depth_for_stereo(
            blurred,
            frame_bgr,
            edge_protect_strength=self._edge_protect_strength,
        )
        return self._smoother.apply(conditioned)


class MidasDepthEstimator(DepthEstimator):
    def __init__(
        self,
        device: str = "auto",
        model_name: str = "Intel/dpt-hybrid-midas",
        edge_protect_strength: float = 0.75,
        depth_process_scale: float = 1.0,
        use_fp16: bool = False,
    ) -> None:
        self._device = device
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._torch = None
        self._smoother = TemporalDepthSmoother(alpha=0.7)
        self._edge_protect_strength = edge_protect_strength
        self._depth_process_scale = depth_process_scale
        self._use_fp16 = use_fp16

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
        if self._use_fp16 and torch_device == "cuda":
            self._model.half()
            torch.backends.cudnn.benchmark = True
        self._model.eval()

    def _prepare_inference_rgb(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inference_rgb = rgb
        if self._depth_process_scale < 1.0:
            scaled_width = max(2, int(round(rgb.shape[1] * self._depth_process_scale)))
            scaled_height = max(2, int(round(rgb.shape[0] * self._depth_process_scale)))
            inference_rgb = cv2.resize(
                rgb,
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_AREA,
            )
        return rgb, inference_rgb

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._load()
        assert self._model is not None and self._processor is not None and self._torch is not None

        rgb, inference_rgb = self._prepare_inference_rgb(frame_bgr)

        inputs = self._processor(images=inference_rgb, return_tensors="pt")
        torch_device = self._resolve_device()
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}

        with self._torch.no_grad():
            use_autocast = self._use_fp16 and torch_device == "cuda"
            with self._torch.cuda.amp.autocast(enabled=use_autocast):
                predicted = self._model(**inputs).predicted_depth
                resized = self._torch.nn.functional.interpolate(
                    predicted.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
        depth = resized.detach().cpu().numpy()
        conditioned = condition_depth_for_stereo(
            depth.astype(np.float32),
            frame_bgr,
            edge_protect_strength=self._edge_protect_strength,
        )
        return self._smoother.apply(conditioned)

    def estimate_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        if not frames_bgr:
            return []

        self._load()
        assert self._model is not None and self._processor is not None and self._torch is not None

        prepared = [self._prepare_inference_rgb(frame) for frame in frames_bgr]
        rgb_frames = [item[0] for item in prepared]
        inference_rgbs = [item[1] for item in prepared]
        base_output_shape = rgb_frames[0].shape[:2]
        base_inference_shape = inference_rgbs[0].shape[:2]
        if any(frame.shape[:2] != base_output_shape for frame in rgb_frames):
            return super().estimate_batch(frames_bgr)
        if any(frame.shape[:2] != base_inference_shape for frame in inference_rgbs):
            return super().estimate_batch(frames_bgr)

        inputs = self._processor(images=inference_rgbs, return_tensors="pt")
        torch_device = self._resolve_device()
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}

        with self._torch.no_grad():
            use_autocast = self._use_fp16 and torch_device == "cuda"
            with self._torch.cuda.amp.autocast(enabled=use_autocast):
                predicted = self._model(**inputs).predicted_depth
                resized = self._torch.nn.functional.interpolate(
                    predicted.unsqueeze(1),
                    size=base_output_shape,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)
        depth_batch = resized.detach().cpu().numpy()
        conditioned_batch: list[np.ndarray] = []
        for depth, frame_bgr in zip(depth_batch, frames_bgr, strict=True):
            conditioned = condition_depth_for_stereo(
                depth.astype(np.float32),
                frame_bgr,
                edge_protect_strength=self._edge_protect_strength,
            )
            conditioned_batch.append(self._smoother.apply(conditioned))
        return conditioned_batch


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

    def estimate_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        if self._use_fallback:
            return self._fallback.estimate_batch(frames_bgr)
        try:
            return self._preferred.estimate_batch(frames_bgr)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f"Preferred depth backend failed ({exc}); switching to luma depth.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._use_fallback = True
            return self._fallback.estimate_batch(frames_bgr)


def create_depth_estimator(
    backend: str,
    device: str,
    edge_protect_strength: float = 0.75,
    depth_process_scale: float = 1.0,
    use_fp16: bool = False,
) -> DepthEstimator:
    if backend == "luma":
        return LumaDepthEstimator(edge_protect_strength=edge_protect_strength)

    if backend == "midas":
        return MidasDepthEstimator(
            device=device,
            edge_protect_strength=edge_protect_strength,
            depth_process_scale=depth_process_scale,
            use_fp16=use_fp16,
        )

    if backend == "auto":
        return AutoDepthEstimator(
            preferred=MidasDepthEstimator(
                device=device,
                edge_protect_strength=edge_protect_strength,
                depth_process_scale=depth_process_scale,
                use_fp16=use_fp16,
            ),
            fallback=LumaDepthEstimator(edge_protect_strength=edge_protect_strength),
        )

    raise ValueError(f"Unsupported depth backend: {backend}")
