from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Callable, Literal, TypeAlias

import numpy as np

from .stereo import _disparity_map, _prepare_depth, synthesize_stereo_views

StereoDevicePreference: TypeAlias = Literal["auto", "cpu", "cuda"]
StereoBackendName: TypeAlias = Literal["cpu", "torch-cuda"]
StereoSynthesisFn: TypeAlias = Callable[
    [np.ndarray, np.ndarray, float, int | None],
    tuple[np.ndarray, np.ndarray],
]


@dataclass(frozen=True, slots=True)
class StereoSynthesisBackend:
    name: StereoBackendName
    synthesize: StereoSynthesisFn


def _import_torch():
    try:
        return importlib.import_module("torch")
    except Exception:
        return None


def is_torch_cuda_stereo_available(*, torch_module=None) -> bool:
    torch = _import_torch() if torch_module is None else torch_module
    if torch is None:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _clamp_to_numpy_dtype(tensor, dtype: np.dtype):
    torch = importlib.import_module("torch")
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        tensor = tensor.round().clamp(info.min, info.max)
        torch_dtype = getattr(torch, dtype.name, None)
        if torch_dtype is not None:
            tensor = tensor.to(dtype=torch_dtype)
    return tensor


def _convert_back_to_numpy_dtype(tensor, dtype: np.dtype) -> np.ndarray:
    tensor = _clamp_to_numpy_dtype(tensor, dtype)
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype, copy=False)


def tensor_frame_to_numpy(payload, dtype: np.dtype = np.dtype(np.uint8)) -> np.ndarray:
    """Convert a GPU-resident frame tensor to a contiguous numpy array."""
    if isinstance(payload, np.ndarray):
        return payload
    torch = _import_torch()
    if torch is None or not isinstance(payload, torch.Tensor):
        return np.asarray(payload)
    return _convert_back_to_numpy_dtype(payload, dtype)


def _linear_horizontal_sample(frame, x_map):
    torch = importlib.import_module("torch")
    height, width = x_map.shape
    y_coords = torch.arange(height, device=frame.device, dtype=torch.int64).view(height, 1).expand(height, width)
    x0 = x_map.floor().to(dtype=torch.int64).clamp(0, width - 1)
    x1 = (x0 + 1).clamp(0, width - 1)

    weight = (x_map - x0.to(dtype=x_map.dtype)).unsqueeze(-1)
    left = frame[y_coords, x0]
    right = frame[y_coords, x1]
    return left * (1.0 - weight) + right * weight


def _has_scatter_reduce_support(torch) -> bool:
    tensor_type = getattr(torch, "Tensor", None)
    return bool(tensor_type is not None and hasattr(tensor_type, "scatter_reduce_"))


def _backward_warp_eye_torch(frame, disparity, direction: float, *, torch):
    """Backward warp one eye using a single grid_sample call.

    ``frame`` is expected as ``[H, W, C]`` float tensor on the same device as
    ``disparity``. ``disparity`` is a ``[H, W]`` non-negative shift in pixels.
    ``direction`` is ``-1.0`` for the left eye and ``+1.0`` for the right eye.

    This approximates the forward warp (``target = source ± disparity/2``) by
    sampling ``source_x = dest_x ∓ disparity(dest_x)/2`` — the standard
    trick used for real-time DIBR. Hole-filling is implicit thanks to
    ``padding_mode='border'``.
    """
    height, width = frame.shape[:2]
    device = frame.device
    dtype = torch.float32

    y = torch.arange(height, device=device, dtype=dtype).view(height, 1).expand(height, width)
    x = torch.arange(width, device=device, dtype=dtype).view(1, width).expand(height, width)

    source_x = x - float(direction) * disparity.to(dtype=dtype) * 0.5

    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    grid_x = (2.0 * source_x / denom_x) - 1.0
    grid_y = (2.0 * y / denom_y) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    frame_nchw = frame.to(dtype=dtype).permute(2, 0, 1).unsqueeze(0)
    warped = torch.nn.functional.grid_sample(
        frame_nchw,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(0).permute(1, 2, 0)


def _forward_warp_eye_torch(frame, depth, shifted_x, *, torch):
    if not _has_scatter_reduce_support(torch):
        raise RuntimeError(
            "Torch stereo backend requires torch.Tensor.scatter_reduce_ for deterministic collision handling."
        )

    height, width = frame.shape[:2]
    target_x = shifted_x.round().to(dtype=torch.int64)

    valid = (target_x >= 0) & (target_x < width)
    if not bool(valid.any()):
        return frame.clone()

    src_y, src_x = torch.where(valid)
    dst_x = target_x[valid]
    dst_y = src_y
    depth_values = depth[valid]
    dst_linear = (dst_y * width) + dst_x
    src_linear = (src_y * width) + src_x

    result = torch.zeros_like(frame)
    top_depth = torch.full((height * width,), float("-inf"), device=frame.device, dtype=depth.dtype)
    top_depth.scatter_reduce_(0, dst_linear, depth_values, reduce="amax", include_self=True)
    is_max_depth = depth_values == top_depth[dst_linear]

    max_linear_index = height * width
    tie_break_candidates = torch.where(
        is_max_depth,
        src_linear,
        torch.full_like(src_linear, max_linear_index),
    )
    winner_src_linear = torch.full(
        (height * width,),
        max_linear_index,
        device=frame.device,
        dtype=src_linear.dtype,
    )
    winner_src_linear.scatter_reduce_(0, dst_linear, tie_break_candidates, reduce="amin", include_self=True)
    visible = is_max_depth & (src_linear == winner_src_linear[dst_linear])

    result[dst_y[visible], dst_x[visible]] = frame[src_y[visible], src_x[visible]]

    occupied = torch.zeros((height, width), device=frame.device, dtype=torch.bool)
    occupied[dst_y[visible], dst_x[visible]] = True
    holes = ~occupied
    if bool(holes.any()):
        fallback_map = shifted_x.clamp(0, width - 1)
        fallback = _linear_horizontal_sample(frame, fallback_map)
        result[holes] = fallback[holes]
    return result


def synthesize_stereo_views_torch(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    stereo_strength: float,
    max_disparity_px: int | None = None,
):
    """Warp left/right eyes on the CUDA device.

    Returns a tuple of torch tensors that remain on the GPU so downstream
    composition (``compose_sbs``) can operate without an intermediate CPU
    roundtrip. Callers must convert to numpy via ``tensor_frame_to_numpy``
    before encoding.
    """
    torch = _import_torch()
    if torch is None or not is_torch_cuda_stereo_available(torch_module=torch):
        raise RuntimeError("Torch CUDA stereo backend is unavailable.")

    height, width = frame_bgr.shape[:2]
    prepared_depth = _prepare_depth(depth, width, height)
    disparity = _disparity_map(prepared_depth, width, stereo_strength, max_disparity_px)

    device = torch.device("cuda")
    frame_tensor = torch.from_numpy(frame_bgr).to(device=device, dtype=torch.float32)
    disparity_tensor = torch.from_numpy(disparity).to(device=device, dtype=torch.float32)

    left_eye = _backward_warp_eye_torch(frame_tensor, disparity_tensor, direction=-1.0, torch=torch)
    right_eye = _backward_warp_eye_torch(frame_tensor, disparity_tensor, direction=+1.0, torch=torch)

    left_eye = _clamp_to_numpy_dtype(left_eye, frame_bgr.dtype)
    right_eye = _clamp_to_numpy_dtype(right_eye, frame_bgr.dtype)
    return left_eye, right_eye


def select_stereo_synthesis_backend(
    device_preference: StereoDevicePreference = "auto",
    *,
    torch_module=None,
) -> StereoSynthesisBackend:
    if device_preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("Invalid stereo device preference.")

    if device_preference == "cpu":
        return StereoSynthesisBackend(name="cpu", synthesize=synthesize_stereo_views)

    if is_torch_cuda_stereo_available(torch_module=torch_module):
        return StereoSynthesisBackend(name="torch-cuda", synthesize=synthesize_stereo_views_torch)

    return StereoSynthesisBackend(name="cpu", synthesize=synthesize_stereo_views)
