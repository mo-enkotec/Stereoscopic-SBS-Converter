import cv2
import inspect
import numpy as np

from vr_sbs_converter.depth import condition_depth_for_stereo


def _import_torch_or_skip():
    try:
        import torch

        return torch
    except Exception:
        import pytest

        pytest.skip("torch unavailable")


def test_edge_aware_depth_filter_preserves_hard_boundary() -> None:
    height, width = 64, 96
    depth = np.full((height, width), 0.2, dtype=np.float32)
    depth[:, width // 2 :] = 0.8

    rng = np.random.default_rng(7)
    noise = (rng.normal(0, 0.03, size=(height, width))).astype(np.float32)
    noisy_depth = np.clip(depth + noise, 0, 1)

    guide = np.zeros((height, width, 3), dtype=np.uint8)
    guide[:, : width // 2] = (30, 30, 30)
    guide[:, width // 2 :] = (220, 220, 220)

    naive = cv2.GaussianBlur(noisy_depth, (0, 0), sigmaX=1.2, sigmaY=1.2)
    conditioned = condition_depth_for_stereo(noisy_depth, guide, edge_protect_strength=0.9)

    left_slice = slice(width // 2 - 6, width // 2 - 1)
    right_slice = slice(width // 2 + 1, width // 2 + 6)
    naive_contrast = float(naive[:, right_slice].mean() - naive[:, left_slice].mean())
    conditioned_contrast = float(
        conditioned[:, right_slice].mean() - conditioned[:, left_slice].mean()
    )
    assert conditioned_contrast > naive_contrast + 0.04


def test_conditioned_depth_stays_normalized() -> None:
    depth = np.linspace(0, 1, 120, dtype=np.float32).reshape(10, 12)
    guide = np.full((10, 12, 3), 127, dtype=np.uint8)
    conditioned = condition_depth_for_stereo(depth, guide, edge_protect_strength=0.75)
    assert float(conditioned.min()) >= 0.0
    assert float(conditioned.max()) <= 1.0


def test_condition_depth_does_not_require_bilateral_filter(monkeypatch) -> None:
    depth = np.linspace(0, 1, 80, dtype=np.float32).reshape(8, 10)
    guide = np.full((8, 10, 3), 127, dtype=np.uint8)

    def _fail_bilateral(*args, **kwargs):
        raise AssertionError("bilateralFilter should not be called in optimized path")

    monkeypatch.setattr(cv2, "bilateralFilter", _fail_bilateral)
    conditioned = condition_depth_for_stereo(depth, guide, edge_protect_strength=0.75)
    assert conditioned.shape == depth.shape


def test_midas_torch_preprocess_normalizes_with_mean_and_std_on_target_size() -> None:
    import pytest

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception:
        pytest.skip("torch unavailable")

    from vr_sbs_converter.depth import _midas_torch_preprocess

    rgb = np.full((10, 20, 3), 128, dtype=np.uint8)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

    out = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(6, 12),
        dtype=torch.float32,
    )

    assert tuple(out.shape) == (1, 3, 6, 12)
    assert out.dtype == torch.float32
    # (128/255 - 0.5) / 0.5 ≈ 0.00392
    assert float(out.mean().item()) == pytest.approx(0.00392, abs=1e-3)


def test_midas_torch_preprocess_supports_fp16_output() -> None:
    import pytest

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception:
        pytest.skip("torch unavailable")

    from vr_sbs_converter.depth import _midas_torch_preprocess

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

    out = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(4, 4),
        dtype=torch.float16,
    )

    assert out.dtype == torch.float16


def test_midas_torch_preprocess_with_pinned_buffer_matches_unpinned() -> None:
    from vr_sbs_converter.depth import _midas_torch_preprocess

    torch = _import_torch_or_skip()
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    pinned_buffer = torch.empty(rgb.shape, dtype=torch.uint8)

    unpinned = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(6, 10),
        dtype=torch.float32,
    )
    staged = _midas_torch_preprocess(
        rgb,
        torch=torch,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
        target_size=(6, 10),
        dtype=torch.float32,
        pinned_buffer=pinned_buffer,
    )

    assert torch.allclose(staged, unpinned, atol=1e-5)


def test_midas_torch_preprocess_pinned_buffer_shape_mismatch_raises_or_falls_back() -> None:
    import pytest

    from vr_sbs_converter.depth import _midas_torch_preprocess

    torch = _import_torch_or_skip()
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
    wrong_shape_buffer = torch.empty((8, 11, 3), dtype=torch.uint8)

    with pytest.raises(ValueError, match="pinned_buffer shape .* must match rgb shape"):
        _midas_torch_preprocess(
            rgb,
            torch=torch,
            device=torch.device("cpu"),
            mean=mean,
            std=std,
            target_size=(8, 12),
            pinned_buffer=wrong_shape_buffer,
        )


def test_get_pinned_rgb_buffer_caches_by_shape(monkeypatch) -> None:
    from vr_sbs_converter.depth import MidasDepthEstimator

    calls = []

    class FakeTorch:
        uint8 = object()

        @staticmethod
        def empty(shape, *, dtype, pin_memory):
            calls.append((shape, dtype, pin_memory))
            return {"shape": shape}

    estimator = MidasDepthEstimator(device="cuda")
    estimator._torch = FakeTorch()
    monkeypatch.setattr(estimator, "_resolve_device", lambda: "cuda")

    first = estimator._get_pinned_rgb_buffer((10, 20, 3))
    second = estimator._get_pinned_rgb_buffer((10, 20, 3))

    assert second is first
    assert calls == [((10, 20, 3), FakeTorch.uint8, True)]


def test_get_pinned_rgb_buffer_creates_separate_buffers_for_different_shapes(monkeypatch) -> None:
    from vr_sbs_converter.depth import MidasDepthEstimator

    calls = []

    class FakeTorch:
        uint8 = object()

        @staticmethod
        def empty(shape, *, dtype, pin_memory):
            calls.append((shape, dtype, pin_memory))
            return {"shape": shape}

    estimator = MidasDepthEstimator(device="cuda")
    estimator._torch = FakeTorch()
    monkeypatch.setattr(estimator, "_resolve_device", lambda: "cuda")

    first = estimator._get_pinned_rgb_buffer((10, 20, 3))
    second = estimator._get_pinned_rgb_buffer((12, 20, 3))

    assert second is not first
    assert calls == [
        ((10, 20, 3), FakeTorch.uint8, True),
        ((12, 20, 3), FakeTorch.uint8, True),
    ]


def test_condition_depth_torch_matches_numpy_within_tolerance() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    height, width = 16, 24
    depth = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
    frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    frame_bgr[:, : width // 2] = (20, 40, 60)
    frame_bgr[:, width // 2 :] = (220, 200, 180)

    expected = condition_depth_for_stereo(depth, frame_bgr, edge_protect_strength=0.75)
    actual = _condition_depth_for_stereo_torch(
        torch.from_numpy(depth),
        torch.from_numpy(frame_bgr),
        0.75,
        torch=torch,
    )

    difference = torch.abs(actual - torch.from_numpy(expected))
    assert float(difference.mean().item()) <= 0.05
    assert float(difference.max().item()) <= 0.20


def test_condition_depth_torch_returns_normalized_range() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    depth = torch.linspace(5.0, 9.0, 120, dtype=torch.float32).reshape(10, 12)
    frame_bgr = torch.full((10, 12, 3), 127, dtype=torch.uint8)

    conditioned = _condition_depth_for_stereo_torch(
        depth,
        frame_bgr,
        0.75,
        torch=torch,
    )

    assert float(conditioned.min().item()) >= 0.0
    assert float(conditioned.max().item()) <= 1.0


def test_condition_depth_torch_skips_edge_protect_when_strength_zero() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    depth = torch.tensor([[2.0, 4.0], [6.0, 10.0]], dtype=torch.float32)
    frame_bgr = torch.zeros((2, 2, 3), dtype=torch.uint8)

    conditioned = _condition_depth_for_stereo_torch(
        depth,
        frame_bgr,
        0.0,
        torch=torch,
    )

    expected = (depth - depth.min()) / (depth.max() - depth.min())
    assert torch.allclose(conditioned, expected, atol=1e-5)


def test_condition_depth_torch_handles_zero_spread() -> None:
    from vr_sbs_converter.depth import _condition_depth_for_stereo_torch

    torch = _import_torch_or_skip()
    depth = torch.full((6, 8), 3.0, dtype=torch.float32)
    frame_bgr = torch.full((6, 8, 3), 64, dtype=torch.uint8)

    conditioned = _condition_depth_for_stereo_torch(
        depth,
        frame_bgr,
        0.75,
        torch=torch,
    )

    assert torch.isfinite(conditioned).all()
    assert torch.count_nonzero(conditioned).item() == 0


def test_condition_depth_torch_avoids_cuda_scalar_synchronization() -> None:
    import vr_sbs_converter.depth as depth_module

    source = inspect.getsource(depth_module)

    assert "bool(" not in source
