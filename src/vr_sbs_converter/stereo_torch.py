from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, TypeAlias, overload

import numpy as np

from .stereo import _disparity_map, _prepare_depth, synthesize_stereo_views

if TYPE_CHECKING:
    import torch as _torch

    TorchTensor: TypeAlias = _torch.Tensor
else:
    TorchTensor: TypeAlias = Any

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


def _convert_back_to_numpy_dtype(tensor, dtype: np.dtype) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(array), info.min, info.max).astype(dtype)
    return array.astype(dtype, copy=False)


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


def _linear_horizontal_sample_batch(frame, x_map, *, torch):
    batch_size, height, width = x_map.shape
    batch_coords = (
        torch.arange(batch_size, device=frame.device, dtype=torch.int64)
        .view(batch_size, 1, 1)
        .expand(batch_size, height, width)
    )
    y_coords = (
        torch.arange(height, device=frame.device, dtype=torch.int64)
        .view(1, height, 1)
        .expand(batch_size, height, width)
    )
    x0 = x_map.floor().to(dtype=torch.int64).clamp(0, width - 1)
    x1 = (x0 + 1).clamp(0, width - 1)

    weight = (x_map - x0.to(dtype=x_map.dtype)).unsqueeze(-1)
    left = frame[batch_coords, y_coords, x0]
    right = frame[batch_coords, y_coords, x1]
    return left * (1.0 - weight) + right * weight


def _has_scatter_reduce_support(torch) -> bool:
    tensor_type = getattr(torch, "Tensor", None)
    return bool(tensor_type is not None and hasattr(tensor_type, "scatter_reduce_"))


def _disparity_map_torch(depth_tensor, width: int, strength: float, max_disparity_px: int | None, *, torch):
    if max_disparity_px is None:
        max_shift = max(1.0, width * 0.03 * strength)
    else:
        if max_disparity_px <= 0:
            raise ValueError("max_disparity_px must be greater than zero.")
        max_shift = max_disparity_px * strength
    near_weight = torch.clamp((depth_tensor - 0.45) / 0.55, 0.0, 1.0).pow(1.25)
    return near_weight * float(max_shift)


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


def _forward_warp_eye_torch_batch(frames, depths, shifted_x, *, torch):
    if not _has_scatter_reduce_support(torch):
        raise RuntimeError(
            "Torch stereo backend requires torch.Tensor.scatter_reduce_ for deterministic collision handling."
        )

    batch_size, height, width = depths.shape
    target_x = shifted_x.round().to(dtype=torch.int64)
    valid = (target_x >= 0) & (target_x < width)
    if not bool(valid.any()):
        return frames.clone()

    src_batch, src_y, src_x = torch.where(valid)
    dst_x = target_x[valid]
    dst_y = src_y
    dst_batch = src_batch
    depth_values = depths[valid]
    frame_pixels = height * width
    dst_linear = (dst_batch * frame_pixels) + (dst_y * width) + dst_x
    src_linear = (src_batch * frame_pixels) + (src_y * width) + src_x

    result = torch.zeros_like(frames)
    top_depth = torch.full((batch_size * frame_pixels,), float("-inf"), device=frames.device, dtype=depths.dtype)
    top_depth.scatter_reduce_(0, dst_linear, depth_values, reduce="amax", include_self=True)
    is_max_depth = depth_values == top_depth[dst_linear]

    max_linear_index = batch_size * frame_pixels
    tie_break_candidates = torch.where(
        is_max_depth,
        src_linear,
        torch.full_like(src_linear, max_linear_index),
    )
    winner_src_linear = torch.full(
        (batch_size * frame_pixels,),
        max_linear_index,
        device=frames.device,
        dtype=src_linear.dtype,
    )
    winner_src_linear.scatter_reduce_(0, dst_linear, tie_break_candidates, reduce="amin", include_self=True)
    visible = is_max_depth & (src_linear == winner_src_linear[dst_linear])

    result[dst_batch[visible], dst_y[visible], dst_x[visible]] = (
        frames[src_batch[visible], src_y[visible], src_x[visible]]
    )

    occupied = torch.zeros((batch_size, height, width), device=frames.device, dtype=torch.bool)
    occupied[dst_batch[visible], dst_y[visible], dst_x[visible]] = True
    holes = ~occupied
    if bool(holes.any()):
        fallback_map = shifted_x.clamp(0, width - 1)
        fallback = _linear_horizontal_sample_batch(frames, fallback_map, torch=torch)
        result[holes] = fallback[holes]
    return result


def synthesize_stereo_views_torch(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    stereo_strength: float,
    max_disparity_px: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    torch = _import_torch()
    if torch is None or not is_torch_cuda_stereo_available(torch_module=torch):
        raise RuntimeError("Torch CUDA stereo backend is unavailable.")

    height, width = frame_bgr.shape[:2]
    prepared_depth = _prepare_depth(depth, width, height)
    disparity = _disparity_map(prepared_depth, width, stereo_strength, max_disparity_px)

    device = torch.device("cuda")
    frame_tensor = torch.from_numpy(frame_bgr).to(device=device, dtype=torch.float32)
    depth_tensor = torch.from_numpy(prepared_depth).to(device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32).view(1, width).expand(height, width)
    disparity_tensor = torch.from_numpy(disparity).to(device=device, dtype=torch.float32)

    left_shifted_x = x_coords - (disparity_tensor * 0.5)
    right_shifted_x = x_coords + (disparity_tensor * 0.5)

    left_eye = _forward_warp_eye_torch(frame_tensor, depth_tensor, left_shifted_x, torch=torch)
    right_eye = _forward_warp_eye_torch(frame_tensor, depth_tensor, right_shifted_x, torch=torch)

    left_np = _convert_back_to_numpy_dtype(left_eye, frame_bgr.dtype)
    right_np = _convert_back_to_numpy_dtype(right_eye, frame_bgr.dtype)
    return left_np, right_np


@overload
def synthesize_stereo_views_torch_batch(
    frames_bgr: Sequence[np.ndarray],
    depths: Sequence[np.ndarray],
    stereo_strength: float,
    max_disparity_px: int | None = None,
    stream_overlap: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]: ...


@overload
def synthesize_stereo_views_torch_batch(
    frames_bgr: Sequence[TorchTensor],
    depths: Sequence[TorchTensor],
    stereo_strength: float,
    max_disparity_px: int | None = None,
    stream_overlap: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]: ...


def synthesize_stereo_views_torch_batch(
    frames_bgr: Sequence[np.ndarray] | Sequence[TorchTensor],
    depths: Sequence[np.ndarray] | Sequence[TorchTensor],
    stereo_strength: float,
    max_disparity_px: int | None = None,
    stream_overlap: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Synthesize stereo views from batched numpy arrays or CUDA torch tensors."""
    frames = list(frames_bgr)
    depth_maps = list(depths)
    if not frames:
        return []
    if len(frames) != len(depth_maps):
        raise ValueError("frames_bgr and depths must have the same batch length.")

    torch = _import_torch()
    if torch is None or not is_torch_cuda_stereo_available(torch_module=torch):
        raise RuntimeError("Torch CUDA stereo backend is unavailable.")

    try:
        device = torch.device("cuda")
        if isinstance(frames[0], np.ndarray):
            first_shape = frames[0].shape
            first_dtype = frames[0].dtype
            if any(frame.shape != first_shape for frame in frames):
                return [
                    synthesize_stereo_views_torch(frame, depth, stereo_strength, max_disparity_px)
                    for frame, depth in zip(frames, depth_maps, strict=True)
                ]
            if any(frame.dtype != first_dtype for frame in frames):
                return [
                    synthesize_stereo_views_torch(frame, depth, stereo_strength, max_disparity_px)
                    for frame, depth in zip(frames, depth_maps, strict=True)
                ]

            height, width = first_shape[:2]
            prepared_depths = [_prepare_depth(depth, width, height) for depth in depth_maps]
            disparities = [
                _disparity_map(prepared_depth, width, stereo_strength, max_disparity_px)
                for prepared_depth in prepared_depths
            ]
            frame_batch = torch.from_numpy(np.stack(frames, axis=0)).to(device=device, dtype=torch.float32)
            depth_batch = torch.from_numpy(np.stack(prepared_depths, axis=0)).to(device=device, dtype=torch.float32)
            disparity_batch = torch.from_numpy(np.stack(disparities, axis=0)).to(device=device, dtype=torch.float32)
        else:
            if any(not isinstance(frame, torch.Tensor) for frame in frames):
                raise TypeError("frames_bgr must be all numpy arrays or all torch tensors.")
            if any(not isinstance(depth, torch.Tensor) for depth in depth_maps):
                raise TypeError("depths must be torch tensors when frames_bgr are torch tensors.")
            if any(frame.ndim != 3 for frame in frames):
                raise ValueError("Frame tensors must be HxWxC.")
            first_shape = tuple(frames[0].shape)
            if any(tuple(frame.shape) != first_shape for frame in frames):
                raise ValueError("All frame tensors must share the same shape for batch stereo synthesis.")
            first_depth_shape = tuple(depth_maps[0].shape)
            if any(tuple(depth.shape) != first_depth_shape for depth in depth_maps):
                raise ValueError("All depth tensors must share the same shape for batch stereo synthesis.")
            height, width = first_shape[:2]
            if first_depth_shape != (height, width):
                raise ValueError("Depth tensor shape must match frame spatial dimensions.")
            frame_batch = torch.stack(
                [frame.to(device=device, dtype=torch.float32, non_blocking=True) for frame in frames],
                dim=0,
            )
            depth_batch = torch.stack(
                [
                    torch.clamp(depth.to(device=device, dtype=torch.float32, non_blocking=True), 0.0, 1.0)
                    for depth in depth_maps
                ],
                dim=0,
            )
            disparity_batch = _disparity_map_torch(
                depth_batch,
                width,
                stereo_strength,
                max_disparity_px,
                torch=torch,
            )
            first_dtype = np.uint8

        x_coords = (
            torch.arange(width, device=device, dtype=torch.float32)
            .view(1, 1, width)
            .expand(frame_batch.shape[0], height, width)
        )

        left_shifted_x = x_coords - (disparity_batch * 0.5)
        right_shifted_x = x_coords + (disparity_batch * 0.5)

        if stream_overlap:
            compute_stream = torch.cuda.Stream(device=device)
            compute_stream.wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(compute_stream):
                left_batch = _forward_warp_eye_torch_batch(frame_batch, depth_batch, left_shifted_x, torch=torch)
                right_batch = _forward_warp_eye_torch_batch(frame_batch, depth_batch, right_shifted_x, torch=torch)
            torch.cuda.current_stream(device=device).wait_stream(compute_stream)
        else:
            left_batch = _forward_warp_eye_torch_batch(frame_batch, depth_batch, left_shifted_x, torch=torch)
            right_batch = _forward_warp_eye_torch_batch(frame_batch, depth_batch, right_shifted_x, torch=torch)

        left_np_batch = _convert_back_to_numpy_dtype(left_batch, first_dtype)
        right_np_batch = _convert_back_to_numpy_dtype(right_batch, first_dtype)
        return list(zip(left_np_batch, right_np_batch, strict=True))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if len(frames) <= 1:
            raise
        torch.cuda.empty_cache()
        midpoint = max(1, len(frames) // 2)
        first_results = synthesize_stereo_views_torch_batch(
            frames[:midpoint],
            depth_maps[:midpoint],
            stereo_strength,
            max_disparity_px,
            stream_overlap=False,
        )
        second_results = synthesize_stereo_views_torch_batch(
            frames[midpoint:],
            depth_maps[midpoint:],
            stereo_strength,
            max_disparity_px,
            stream_overlap=False,
        )
        return first_results + second_results


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
