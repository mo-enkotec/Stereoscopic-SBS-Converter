from __future__ import annotations

import inspect
import numpy as np
import pytest

from vr_sbs_converter.stereo import _disparity_map, _prepare_depth, synthesize_stereo_views
from vr_sbs_converter.stereo_torch import (
    create_encoder_readback_stream,
    _forward_warp_eye_torch,
    _import_torch,
    is_torch_cuda_stereo_available,
    select_stereo_synthesis_backend,
    tensor_frame_to_numpy_async,
)


def _sample_inputs() -> tuple[np.ndarray, np.ndarray]:
    frame = np.full((24, 32, 3), 96, dtype=np.uint8)
    depth = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (24, 1))
    return frame, depth


def test_backend_selector_falls_back_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import vr_sbs_converter.stereo_torch as stereo_torch_module

    def _raise_import_error(_: str):
        raise ImportError("torch not installed")

    monkeypatch.setattr(stereo_torch_module.importlib, "import_module", _raise_import_error)

    backend = select_stereo_synthesis_backend(device_preference="auto")
    assert backend.name == "cpu"
    assert backend.synthesize is synthesize_stereo_views


def test_backend_selector_falls_back_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    import vr_sbs_converter.stereo_torch as stereo_torch_module

    monkeypatch.setattr(stereo_torch_module, "_import_torch", lambda: _FakeTorch())

    backend = select_stereo_synthesis_backend(device_preference="cuda")
    assert backend.name == "cpu"
    assert backend.synthesize is synthesize_stereo_views


def test_backend_selector_respects_device_preferences_when_cuda_is_available() -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class _FakeTorch:
        cuda = _FakeCuda()

    assert select_stereo_synthesis_backend("cpu", torch_module=_FakeTorch()).name == "cpu"
    assert select_stereo_synthesis_backend("auto", torch_module=_FakeTorch()).name == "torch-cuda"
    assert select_stereo_synthesis_backend("cuda", torch_module=_FakeTorch()).name == "torch-cuda"


def test_backend_selector_respects_device_preferences_when_cuda_is_unavailable() -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    assert select_stereo_synthesis_backend("cpu", torch_module=_FakeTorch()).name == "cpu"
    assert select_stereo_synthesis_backend("auto", torch_module=_FakeTorch()).name == "cpu"
    assert select_stereo_synthesis_backend("cuda", torch_module=_FakeTorch()).name == "cpu"


def test_selected_cpu_backend_preserves_shape_dtype_and_range() -> None:
    frame, depth = _sample_inputs()

    backend = select_stereo_synthesis_backend(device_preference="cpu")
    left_eye, right_eye = backend.synthesize(frame, depth, stereo_strength=0.8, max_disparity_px=8)

    assert left_eye.shape == frame.shape
    assert right_eye.shape == frame.shape
    assert left_eye.dtype == frame.dtype
    assert right_eye.dtype == frame.dtype
    assert int(left_eye.min()) >= 0 and int(left_eye.max()) <= 255
    assert int(right_eye.min()) >= 0 and int(right_eye.max()) <= 255


def test_tensor_frame_to_numpy_async_falls_back_for_numpy_input() -> None:
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    result = tensor_frame_to_numpy_async(frame)

    assert result is frame


def test_tensor_frame_to_numpy_async_returns_numpy_from_cpu_tensor() -> None:
    torch = _import_torch()
    if torch is None:
        pytest.skip("torch unavailable")

    tensor = torch.tensor(
        [[[0.0, 127.4, 255.0], [300.0, -10.0, 12.6]]],
        dtype=torch.float32,
    )

    result = tensor_frame_to_numpy_async(tensor)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.array([[[0, 127, 255], [255, 0, 13]]], dtype=np.uint8))


def test_tensor_frame_to_numpy_async_uses_provided_stream_and_synchronizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vr_sbs_converter.stereo_torch as stereo_torch_module

    events: list[str] = []
    default_stream = object()

    class _FakeDevice:
        type = "cuda"

    class _SpyStream:
        def wait_stream(self, stream) -> None:
            assert stream is default_stream
            events.append("wait_stream")

        def synchronize(self) -> None:
            events.append("synchronize")

    class _FakeCudaStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            events.append("enter_stream")
            return self.stream

        def __exit__(self, exc_type, exc, traceback) -> None:
            events.append("exit_stream")

    class _FakeTensor:
        device = _FakeDevice()
        shape = (1, 2, 3)

        def __init__(self, *, dtype="float32") -> None:
            self.dtype = dtype

        def round(self):
            events.append("round")
            return self

        def clamp(self, minimum, maximum):
            assert (minimum, maximum) == (0, 255)
            events.append("clamp")
            return self

        def to(self, *, dtype):
            self.dtype = dtype
            events.append("to_uint8")
            return self

        def detach(self):
            events.append("detach")
            return self

    class _FakeDestTensor:
        def __init__(self, shape, dtype, device, pin_memory) -> None:
            assert shape == _FakeTensor.shape
            assert dtype == _FakeTorch.uint8
            assert device == "cpu"
            assert pin_memory is True

        def copy_(self, src, *, non_blocking):
            assert isinstance(src, _FakeTensor)
            assert non_blocking is True
            events.append("copy")
            return self

        def numpy(self):
            events.append("numpy")
            return np.arange(6, dtype=np.uint8).reshape(1, 2, 3)

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_stream():
            return default_stream

        @staticmethod
        def stream(stream):
            return _FakeCudaStreamContext(stream)

    class _FakeTorch:
        Tensor = _FakeTensor
        cuda = _FakeCuda()
        uint8 = "uint8"

        @staticmethod
        def empty(shape, *, dtype, device, pin_memory):
            events.append("empty")
            return _FakeDestTensor(shape, dtype, device, pin_memory)

    stream = _SpyStream()
    monkeypatch.setattr(stereo_torch_module, "_import_torch", lambda: _FakeTorch())

    result = tensor_frame_to_numpy_async(_FakeTensor(), stream=stream)

    assert result.shape == (1, 2, 3)
    assert result.dtype == np.uint8
    assert events.index("wait_stream") < events.index("copy")
    assert events.index("copy") < events.index("synchronize")


def test_create_encoder_readback_stream_returns_none_without_cuda() -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    assert create_encoder_readback_stream(torch_module=_FakeTorch()) is None


def test_torch_cuda_backend_executes_or_skips_cleanly() -> None:
    if not is_torch_cuda_stereo_available():
        pytest.skip("torch+cuda stereo backend unavailable")

    torch = _import_torch()
    frame, depth = _sample_inputs()
    backend = select_stereo_synthesis_backend(device_preference="cuda")

    assert backend.name == "torch-cuda"
    left_eye, right_eye = backend.synthesize(frame, depth, stereo_strength=0.8, max_disparity_px=8)
    assert isinstance(left_eye, torch.Tensor)
    assert isinstance(right_eye, torch.Tensor)
    assert left_eye.device.type == "cuda"
    assert right_eye.device.type == "cuda"
    assert tuple(left_eye.shape) == frame.shape
    assert tuple(right_eye.shape) == frame.shape


def test_forward_warp_collision_is_deterministic_with_depth_and_tiebreak() -> None:
    torch = _import_torch()
    if torch is None:
        pytest.skip("torch unavailable")

    frame = torch.tensor(
        [[[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0], [40.0, 40.0, 40.0]]],
        dtype=torch.float32,
    )
    shifted_x = torch.tensor([[1.0, 1.0, 2.0, 3.0]], dtype=torch.float32)

    depth_prefers_second = torch.tensor([[0.3, 0.8, 0.2, 0.1]], dtype=torch.float32)
    warped_depth = _forward_warp_eye_torch(frame, depth_prefers_second, shifted_x, torch=torch)
    assert float(warped_depth[0, 1, 0]) == pytest.approx(20.0)

    depth_tie = torch.tensor([[0.8, 0.8, 0.2, 0.1]], dtype=torch.float32)
    first = _forward_warp_eye_torch(frame, depth_tie, shifted_x, torch=torch)
    second = _forward_warp_eye_torch(frame, depth_tie, shifted_x, torch=torch)
    assert float(first[0, 1, 0]) == pytest.approx(10.0)
    assert torch.equal(first, second)


def test_torch_forward_warp_cpu_envelope_matches_numpy_path_when_available() -> None:
    torch = _import_torch()
    if torch is None:
        pytest.skip("torch unavailable")
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        pytest.skip("torch scatter_reduce unavailable")

    height, width = 10, 14
    x_gradient = np.tile(np.linspace(0, 255, width, dtype=np.float32), (height, 1))
    frame = np.stack(
        [x_gradient, np.flip(x_gradient, axis=1), np.full_like(x_gradient, 96.0)],
        axis=-1,
    ).astype(np.uint8)
    depth = np.tile(np.linspace(0.15, 0.85, width, dtype=np.float32), (height, 1))
    stereo_strength = 0.6
    max_disparity_px = 3

    expected_left, expected_right = synthesize_stereo_views(
        frame,
        depth,
        stereo_strength=stereo_strength,
        max_disparity_px=max_disparity_px,
    )

    prepared_depth = _prepare_depth(depth, width, height)
    disparity = _disparity_map(prepared_depth, width, stereo_strength, max_disparity_px)
    frame_tensor = torch.from_numpy(frame).to(dtype=torch.float32)
    depth_tensor = torch.from_numpy(prepared_depth).to(dtype=torch.float32)
    x_coords = torch.arange(width, dtype=torch.float32).view(1, width).expand(height, width)
    disparity_tensor = torch.from_numpy(disparity).to(dtype=torch.float32)

    left_tensor = _forward_warp_eye_torch(
        frame_tensor,
        depth_tensor,
        x_coords - (disparity_tensor * 0.5),
        torch=torch,
    )
    right_tensor = _forward_warp_eye_torch(
        frame_tensor,
        depth_tensor,
        x_coords + (disparity_tensor * 0.5),
        torch=torch,
    )

    left_eye = np.clip(np.rint(left_tensor.detach().cpu().numpy()), 0, 255).astype(np.uint8)
    right_eye = np.clip(np.rint(right_tensor.detach().cpu().numpy()), 0, 255).astype(np.uint8)

    assert left_eye.shape == expected_left.shape
    assert right_eye.shape == expected_right.shape
    assert left_eye.dtype == expected_left.dtype
    assert right_eye.dtype == expected_right.dtype

    left_diff = np.abs(left_eye.astype(np.int16) - expected_left.astype(np.int16))
    right_diff = np.abs(right_eye.astype(np.int16) - expected_right.astype(np.int16))
    assert float(left_diff.mean()) <= 3.0
    assert float(right_diff.mean()) <= 3.0


def test_forward_warp_fails_fast_without_scatter_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    import vr_sbs_converter.stereo_torch as stereo_torch_module

    class _FakeTorchWithoutScatter:
        class Tensor:
            pass

    monkeypatch.setattr(stereo_torch_module, "_has_scatter_reduce_support", lambda _: False)

    with pytest.raises(RuntimeError, match="scatter_reduce"):
        _forward_warp_eye_torch(None, None, None, torch=_FakeTorchWithoutScatter())


def test_forward_warp_has_no_python_tolist_fallback() -> None:
    source = inspect.getsource(_forward_warp_eye_torch)
    assert ".tolist(" not in source


def test_backward_warp_shape_and_shift_direction_on_cpu() -> None:
    torch = _import_torch()
    if torch is None:
        pytest.skip("torch unavailable")

    from vr_sbs_converter.stereo_torch import _backward_warp_eye_torch

    height, width = 8, 24
    # Uniform gray with a bright vertical bar at x=12
    frame = torch.zeros((height, width, 3), dtype=torch.float32)
    frame[:, 12, :] = 200.0
    # Uniform disparity of 4 px across the image
    disparity = torch.full((height, width), 4.0, dtype=torch.float32)

    left_eye = _backward_warp_eye_torch(frame, disparity, direction=-1.0, torch=torch)
    right_eye = _backward_warp_eye_torch(frame, disparity, direction=+1.0, torch=torch)

    assert tuple(left_eye.shape) == (height, width, 3)
    assert tuple(right_eye.shape) == (height, width, 3)

    # Left eye: source_x = dest_x + disp/2 = dest_x + 2 → bright bar moves to dest_x = 10
    assert float(left_eye[0, 10, 0].item()) > float(left_eye[0, 12, 0].item())
    # Right eye: source_x = dest_x - disp/2 = dest_x - 2 → bright bar moves to dest_x = 14
    assert float(right_eye[0, 14, 0].item()) > float(right_eye[0, 12, 0].item())


def test_backward_warp_zero_disparity_is_identity() -> None:
    torch = _import_torch()
    if torch is None:
        pytest.skip("torch unavailable")

    from vr_sbs_converter.stereo_torch import _backward_warp_eye_torch

    height, width = 6, 10
    frame = torch.arange(height * width * 3, dtype=torch.float32).view(height, width, 3)
    disparity = torch.zeros((height, width), dtype=torch.float32)

    left_eye = _backward_warp_eye_torch(frame, disparity, direction=-1.0, torch=torch)
    right_eye = _backward_warp_eye_torch(frame, disparity, direction=+1.0, torch=torch)

    assert torch.allclose(left_eye, frame, atol=1e-4)
    assert torch.allclose(right_eye, frame, atol=1e-4)


def test_backward_warp_fp16_matches_fp32_within_tolerance() -> None:
    torch = _import_torch()
    if torch is None:
        pytest.skip("torch unavailable")

    from vr_sbs_converter.stereo_torch import _backward_warp_eye_torch

    height, width = 12, 32
    torch.manual_seed(0)
    frame = (torch.rand((height, width, 3), dtype=torch.float32) * 200.0)
    disparity = torch.rand((height, width), dtype=torch.float32) * 4.0

    ref = _backward_warp_eye_torch(frame, disparity, direction=-1.0, torch=torch, dtype=torch.float32)
    approx = _backward_warp_eye_torch(
        frame.to(dtype=torch.float16),
        disparity.to(dtype=torch.float16),
        direction=-1.0,
        torch=torch,
        dtype=torch.float16,
    ).to(dtype=torch.float32)

    diff = (ref - approx).abs()
    # Cast + interpolation in fp16 should be near-identical for values <255.
    assert float(diff.mean().item()) < 0.6
    assert float(diff.max().item()) < 4.0

