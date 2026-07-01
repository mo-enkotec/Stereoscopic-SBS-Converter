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


_DEPTH_CONDITIONING_KERNEL_CACHE = {}


def _normalize_depth_tensor(depth_tensor, *, torch):
    depth_tensor = depth_tensor.to(dtype=torch.float32)
    min_value = depth_tensor.amin()
    max_value = depth_tensor.amax()
    spread = max_value - min_value
    normalized = (depth_tensor - min_value) / spread.clamp_min(1e-8)
    return torch.where(spread <= 1e-8, torch.zeros_like(normalized), normalized).to(
        dtype=torch.float32
    )


def _conditioning_kernel_cache_key(device) -> tuple[str, int | None]:
    return (str(device.type), getattr(device, "index", None))


def _get_depth_conditioning_kernels(*, torch, device, kernel_cache=None):
    cache = kernel_cache if kernel_cache is not None else _DEPTH_CONDITIONING_KERNEL_CACHE
    key = _conditioning_kernel_cache_key(device)
    cached = cache.get(key)
    if cached is not None:
        return cached

    offsets = torch.arange(-3, 4, device=device, dtype=torch.float32)
    gaussian = torch.exp(-(offsets * offsets) / (2.0 * 1.2 * 1.2))
    gaussian = gaussian / gaussian.sum()
    kernels = {
        "gaussian_x": gaussian.view(1, 1, 1, 7),
        "gaussian_y": gaussian.view(1, 1, 7, 1),
        "sobel_x": torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3),
        "sobel_y": torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3),
    }
    cache[key] = kernels
    return kernels


def _pad_for_kernel(tensor, padding: tuple[int, int, int, int], *, torch):
    _, _, height, width = tensor.shape
    left, right, top, bottom = padding
    can_reflect = height > max(top, bottom) and width > max(left, right)
    mode = "reflect" if can_reflect else "replicate"
    return torch.nn.functional.pad(tensor, padding, mode=mode)


def _condition_depth_for_stereo_torch_impl(
    depth_tensor,
    frame_bgr_tensor,
    edge_protect_strength,
    *,
    torch,
    kernel_cache=None,
):
    """Torch/GPU equivalent of condition_depth_for_stereo.

    depth_tensor: float32 tensor [H, W] on any device.
    frame_bgr_tensor: uint8 or float tensor [H, W, 3] on the same device (BGR).
    Returns a torch tensor [H, W] float32 on the same device.
    """
    normalized = _normalize_depth_tensor(depth_tensor, torch=torch)
    if edge_protect_strength <= 0:
        return normalized

    kernels = _get_depth_conditioning_kernels(
        torch=torch,
        device=normalized.device,
        kernel_cache=kernel_cache,
    )
    image = normalized.view(1, 1, *normalized.shape)
    smoothed = torch.nn.functional.conv2d(
        _pad_for_kernel(image, (3, 3, 0, 0), torch=torch),
        kernels["gaussian_x"],
    )
    smoothed = torch.nn.functional.conv2d(
        _pad_for_kernel(smoothed, (0, 0, 3, 3), torch=torch),
        kernels["gaussian_y"],
    ).squeeze(0).squeeze(0)

    frame = frame_bgr_tensor.to(device=normalized.device, dtype=torch.float32) / 255.0
    gray = (
        (frame[..., 0] * 0.114)
        + (frame[..., 1] * 0.587)
        + (frame[..., 2] * 0.299)
    )
    gray_image = gray.view(1, 1, *gray.shape)
    padded_gray = _pad_for_kernel(gray_image, (1, 1, 1, 1), torch=torch)
    grad_x = torch.nn.functional.conv2d(padded_gray, kernels["sobel_x"]).squeeze(0).squeeze(0)
    grad_y = torch.nn.functional.conv2d(padded_gray, kernels["sobel_y"]).squeeze(0).squeeze(0)
    magnitude = torch.sqrt((grad_x * grad_x) + (grad_y * grad_y))
    max_magnitude = magnitude.amax()
    edge_mask = torch.clamp(magnitude / max_magnitude.clamp_min(1e-8), 0.0, 1.0)
    edge_mask = torch.where(max_magnitude <= 1e-8, torch.zeros_like(edge_mask), edge_mask).to(
        dtype=torch.float32
    )

    edge_weight = torch.clamp(edge_mask * float(edge_protect_strength), 0.0, 1.0)
    conditioned = (edge_weight * normalized) + ((1.0 - edge_weight) * smoothed)
    return _normalize_depth_tensor(conditioned, torch=torch)


def _condition_depth_for_stereo_torch(
    depth_tensor,
    frame_bgr_tensor,
    edge_protect_strength,
    *,
    torch,
):
    """Torch/GPU equivalent of condition_depth_for_stereo.

    depth_tensor: float32 tensor [H, W] on any device.
    frame_bgr_tensor: uint8 or float tensor [H, W, 3] on the same device (BGR).
    Returns a torch tensor [H, W] float32 on the same device.
    """
    return _condition_depth_for_stereo_torch_impl(
        depth_tensor,
        frame_bgr_tensor,
        edge_protect_strength,
        torch=torch,
    )


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
    pinned_buffer=None,
):
    """Torch-native replacement for HuggingFace ``AutoImageProcessor``.

    Skips the PIL round-trip that dominates per-frame overhead. Uploads the
    numpy RGB frame straight to ``device`` as a float32 tensor, rescales to
    the model's expected input size with bicubic interpolation, then applies
    the same ``(x/255 - mean) / std`` normalization used by the DPT
    processor. If ``dtype`` is given (e.g. ``torch.float16``) the result is
    cast at the end so the model receives its expected precision.
    """
    if pinned_buffer is None:
        tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    else:
        if tuple(pinned_buffer.shape) != tuple(rgb.shape):
            raise ValueError(
                f"pinned_buffer shape {tuple(pinned_buffer.shape)} must match rgb shape {tuple(rgb.shape)}"
            )
        pinned_buffer.copy_(torch.from_numpy(rgb))
        tensor = pinned_buffer.to(device=device, dtype=torch.float32, non_blocking=True)
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


def _triton_available() -> bool:
    """Return True only if a working ``triton`` install is importable.

    ``torch.compile(mode='reduce-overhead')`` requires triton for the
    inductor backend. If triton is missing, compile appears to succeed
    but every forward call raises. Pre-flight this so we skip cleanly.
    """
    try:
        import importlib

        importlib.import_module("triton")
        return True
    except Exception:
        return False


def _autocast_ctx(torch, *, device_type: str, enabled: bool):
    """Version-compatible autocast context.

    Prefers the new ``torch.amp.autocast`` API and falls back to the
    deprecated ``torch.cuda.amp.autocast`` on older torch.
    """
    amp = getattr(torch, "amp", None)
    autocast = getattr(amp, "autocast", None) if amp is not None else None
    if autocast is not None:
        try:
            return autocast(device_type=device_type, enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=enabled)


def _extract_processor_size(size_attr, *, default: int = 384) -> tuple[int, int]:
    """Extract ``(height, width)`` from a HuggingFace image processor size.

    HuggingFace transformers 4.x may return a plain ``dict``, a ``SizeDict``
    dataclass (with ``.height``/``.width``/``.shortest_edge`` attributes), a
    single ``int``, or ``None``. This helper handles all four forms without
    relying on ``isinstance(size_attr, dict)`` (``SizeDict`` is not a dict
    subclass in newer transformers releases).
    """
    if size_attr is None:
        return (default, default)

    def _as_int(value, fallback: int) -> int:
        if value is None:
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    if isinstance(size_attr, (int, float)):
        return (int(size_attr), int(size_attr))

    # Try attribute access first (SizeDict), then dict-style access.
    height = getattr(size_attr, "height", None)
    width = getattr(size_attr, "width", None)
    shortest = getattr(size_attr, "shortest_edge", None)
    if height is None or width is None:
        getter = getattr(size_attr, "get", None)
        if callable(getter):
            height = height if height is not None else getter("height")
            width = width if width is not None else getter("width")
            shortest = shortest if shortest is not None else getter("shortest_edge")

    if height is None and shortest is not None:
        height = shortest
    if width is None:
        width = height if height is not None else shortest

    return (_as_int(height, default), _as_int(width, default))


class MidasDepthEstimator(DepthEstimator):
    def __init__(
        self,
        device: str = "auto",
        model_name: str = "Intel/dpt-hybrid-midas",
        edge_protect_strength: float = 0.75,
        depth_process_scale: float = 1.0,
        use_fp16: bool = False,
        depth_compile: bool = False,
    ) -> None:
        self._device = device
        self._model_name = model_name
        self._model = None
        self._uncompiled_model = None
        self._processor = None
        self._torch = None
        self._smoother = TemporalDepthSmoother(alpha=0.7)
        self._edge_protect_strength = edge_protect_strength
        self._depth_process_scale = depth_process_scale
        self._use_fp16 = use_fp16
        self._depth_compile = depth_compile
        self._compiled = False
        self._preproc_mean = None
        self._preproc_std = None
        self._preproc_size: tuple[int, int] | None = None
        self._depth_conditioning_kernel_cache = {}
        self._pinned_rgb_buffers: dict[tuple[int, int, int], "torch.Tensor"] = {}

    def _resolve_device(self) -> str:
        if self._device == "cpu":
            return "cpu"
        if self._device == "cuda":
            return "cuda"
        assert self._torch is not None
        return "cuda" if self._torch.cuda.is_available() else "cpu"

    def _get_pinned_rgb_buffer(self, shape: tuple[int, int, int]):
        if self._resolve_device() != "cuda":
            return None
        assert self._torch is not None
        cached = self._pinned_rgb_buffers.get(shape)
        if cached is None:
            cached = self._torch.empty(shape, dtype=self._torch.uint8, pin_memory=True)
            self._pinned_rgb_buffers[shape] = cached
        return cached

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
        size_attr = getattr(processor, "size", None)
        input_h, input_w = _extract_processor_size(size_attr, default=384)
        self._preproc_mean = torch.tensor(mean_values, dtype=torch.float32).view(1, 3, 1, 1).to(torch_device)
        self._preproc_std = torch.tensor(std_values, dtype=torch.float32).view(1, 3, 1, 1).to(torch_device)
        self._preproc_size = (input_h, input_w)

        self._model = DPTForDepthEstimation.from_pretrained(self._model_name)
        self._model.to(torch_device)
        if self._use_fp16 and torch_device == "cuda":
            self._model.half()
            torch.backends.cudnn.benchmark = True
        self._model.eval()
        if self._depth_compile and torch_device == "cuda" and hasattr(torch, "compile"):
            if not _triton_available():
                warnings.warn(
                    "depth_compile requested but a working triton install was not found; "
                    "using uncompiled MiDaS. Install triton (pip install triton) to enable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._compiled = False
            else:
                uncompiled_model = self._model
                try:
                    self._model = torch.compile(
                        uncompiled_model,
                        mode="reduce-overhead",
                        fullgraph=False,
                        dynamic=False,
                    )
                    self._uncompiled_model = uncompiled_model
                    self._compiled = True
                except Exception as exc:
                    warnings.warn(
                        f"torch.compile failed for MiDaS ({exc}); using uncompiled MiDaS fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._compiled = False

    def _predict_depth(self, pixel_values):
        assert self._model is not None
        try:
            return self._model(pixel_values=pixel_values).predicted_depth
        except Exception as exc:
            if not self._compiled or self._uncompiled_model is None:
                raise
            warnings.warn(
                f"Compiled MiDaS forward failed ({exc}); retrying with uncompiled MiDaS fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._model = self._uncompiled_model
            self._uncompiled_model = None
            self._compiled = False
            return self._model(pixel_values=pixel_values).predicted_depth

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
        pinned_buffer = (
            self._get_pinned_rgb_buffer(inference_rgb.shape) if torch_device == "cuda" else None
        )
        pixel_values = _midas_torch_preprocess(
            inference_rgb,
            torch=torch,
            device=torch_device,
            mean=self._preproc_mean,
            std=self._preproc_std,
            target_size=self._preproc_size,
            dtype=model_dtype,
            pinned_buffer=pinned_buffer,
        )

        with torch.no_grad():
            with _autocast_ctx(torch, device_type="cuda", enabled=use_autocast):
                predicted = self._predict_depth(pixel_values)
                resized = torch.nn.functional.interpolate(
                    predicted.unsqueeze(1),
                    size=rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            if torch_device == "cuda":
                frame_bgr_writable = np.ascontiguousarray(frame_bgr)
                if not frame_bgr_writable.flags.writeable:
                    frame_bgr_writable = frame_bgr_writable.copy()
                frame_bgr_tensor = torch.from_numpy(frame_bgr_writable).to(device=torch_device)
                conditioned_tensor = _condition_depth_for_stereo_torch_impl(
                    resized,
                    frame_bgr_tensor,
                    self._edge_protect_strength,
                    torch=torch,
                    kernel_cache=self._depth_conditioning_kernel_cache,
                )
                conditioned = conditioned_tensor.detach().cpu().numpy()
            else:
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
    depth_compile: bool = False,
) -> DepthEstimator:
    if backend == "luma":
        return LumaDepthEstimator(edge_protect_strength=edge_protect_strength)

    if backend == "midas":
        return MidasDepthEstimator(
            device=device,
            edge_protect_strength=edge_protect_strength,
            depth_process_scale=depth_process_scale,
            use_fp16=use_fp16,
            depth_compile=depth_compile,
        )

    if backend == "auto":
        return AutoDepthEstimator(
            preferred=MidasDepthEstimator(
                device=device,
                edge_protect_strength=edge_protect_strength,
                depth_process_scale=depth_process_scale,
                use_fp16=use_fp16,
                depth_compile=depth_compile,
            ),
            fallback=LumaDepthEstimator(edge_protect_strength=edge_protect_strength),
        )

    raise ValueError(f"Unsupported depth backend: {backend}")
