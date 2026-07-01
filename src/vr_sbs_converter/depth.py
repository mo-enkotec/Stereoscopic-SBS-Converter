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


def _midas_torch_preprocess(
    rgb: np.ndarray,
    *,
    torch,
    device,
    mean,
    std,
    target_size: tuple[int, int],
    dtype=None,
):
    """Torch-native replacement for HuggingFace ``AutoImageProcessor``.

    Skips the PIL round-trip that dominates per-frame overhead. Uploads the
    numpy RGB frame straight to ``device`` as a float32 tensor, rescales to
    the model's expected input size with bicubic interpolation, then applies
    the same ``(x/255 - mean) / std`` normalization used by the DPT
    processor. If ``dtype`` is given (e.g. ``torch.float16``) the result is
    cast at the end so the model receives its expected precision.
    """
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    if tuple(tensor.shape[-2:]) != tuple(target_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(int(target_size[0]), int(target_size[1])),
            mode="bicubic",
            align_corners=False,
        )
    tensor = (tensor - mean.to(device=device, dtype=torch.float32)) / std.to(
        device=device, dtype=torch.float32
    )
    if dtype is not None and dtype != torch.float32:
        tensor = tensor.to(dtype=dtype)
    return tensor


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
        self._preproc_mean = None
        self._preproc_std = None
        self._preproc_size: tuple[int, int] | None = None

    def _resolve_device(self) -> str:
        if self._device == "cpu":
            return "cpu"
        if self._device == "cuda":
            return "cuda"
        assert self._torch is not None
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        if self._model is not None and self._preproc_size is not None:
            return
        import torch  # type: ignore
        from transformers import AutoImageProcessor, DPTForDepthEstimation  # type: ignore

        self._torch = torch
        torch_device = self._resolve_device()
        # Load the HF processor once purely to extract preprocessing constants
        # (image_mean, image_std, and target size), then discard the per-call
        # PIL/dict overhead by using a torch-native preprocess.
        processor = AutoImageProcessor.from_pretrained(self._model_name)
        self._processor = processor
        mean_values = getattr(processor, "image_mean", None) or [0.5, 0.5, 0.5]
        std_values = getattr(processor, "image_std", None) or [0.5, 0.5, 0.5]
        size_attr = getattr(processor, "size", None) or {"height": 384, "width": 384}
        if isinstance(size_attr, dict):
            input_h = int(size_attr.get("height", size_attr.get("shortest_edge", 384)))
            input_w = int(size_attr.get("width", input_h))
        else:
            input_h = input_w = int(size_attr)
        self._preproc_mean = torch.tensor(mean_values, dtype=torch.float32).view(1, 3, 1, 1).to(torch_device)
        self._preproc_std = torch.tensor(std_values, dtype=torch.float32).view(1, 3, 1, 1).to(torch_device)
        self._preproc_size = (input_h, input_w)

        self._model = DPTForDepthEstimation.from_pretrained(self._model_name)
        self._model.to(torch_device)
        if self._use_fp16 and torch_device == "cuda":
            self._model.half()
            torch.backends.cudnn.benchmark = True
        self._model.eval()

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        self._load()
        assert self._model is not None and self._torch is not None
        assert self._preproc_mean is not None and self._preproc_std is not None
        assert self._preproc_size is not None

        torch = self._torch
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

        torch_device = self._resolve_device()
        use_autocast = self._use_fp16 and torch_device == "cuda"
        model_dtype = torch.float16 if use_autocast else torch.float32
        pixel_values = _midas_torch_preprocess(
            inference_rgb,
            torch=torch,
            device=torch_device,
            mean=self._preproc_mean,
            std=self._preproc_std,
            target_size=self._preproc_size,
            dtype=model_dtype,
        )

        with self._torch.no_grad():
            with self._torch.cuda.amp.autocast(enabled=use_autocast):
                predicted = self._model(pixel_values=pixel_values).predicted_depth
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
