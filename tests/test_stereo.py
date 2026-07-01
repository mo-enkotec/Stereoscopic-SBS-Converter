import numpy as np
from pathlib import Path

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import dry_run_example_frame
from vr_sbs_converter.stereo import compose_sbs, synthesize_stereo_views
import vr_sbs_converter.stereo as stereo_module


def test_synthesize_stereo_views_shape() -> None:
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)
    depth = np.tile(np.linspace(0, 1, 48, dtype=np.float32), (32, 1))

    left, right = synthesize_stereo_views(frame, depth, stereo_strength=0.8)
    assert left.shape == frame.shape
    assert right.shape == frame.shape


def test_compose_sbs_modes() -> None:
    left = np.zeros((20, 40, 3), dtype=np.uint8)
    right = np.ones((20, 40, 3), dtype=np.uint8)

    full = compose_sbs(left, right, "full")
    half = compose_sbs(left, right, "half")

    assert full.shape == (20, 80, 3)
    assert half.shape == (20, 40, 3)


def test_dry_run_example_frame_returns_sbs_shape() -> None:
    frame = np.full((24, 36, 3), 64, dtype=np.uint8)
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        sbs_mode="full",
        stereo_strength=0.9,
    )
    sbs = dry_run_example_frame(frame, config)
    assert sbs.shape == (24, 72, 3)


def test_occlusion_aware_synthesis_limits_color_bleed() -> None:
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    frame[:, :48] = (255, 0, 0)
    frame[:, 48:] = (0, 255, 0)
    frame[20:44, 34:58] = (0, 0, 255)

    depth = np.full((64, 96), 0.25, dtype=np.float32)
    depth[20:44, 34:58] = 0.95

    left_eye, _ = synthesize_stereo_views(
        frame,
        depth,
        stereo_strength=1.0,
        max_disparity_px=8,
    )

    inner_edge_band = left_eye[22:42, 54:58]
    assert float(inner_edge_band[..., 2].mean()) > 150.0
    assert float(inner_edge_band[..., 1].mean()) < 90.0


def test_synthesize_stereo_views_does_not_require_global_argsort(monkeypatch) -> None:
    frame = np.full((24, 32, 3), 127, dtype=np.uint8)
    depth = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (24, 1))

    def _fail_argsort(*args, **kwargs):
        raise AssertionError("argsort should not be required in optimized stereo synthesis")

    monkeypatch.setattr(np, "argsort", _fail_argsort)
    left, right = synthesize_stereo_views(frame, depth, stereo_strength=0.8)
    assert left.shape == frame.shape
    assert right.shape == frame.shape


def test_forward_warp_prefers_higher_depth_on_collision() -> None:
    frame = np.zeros((1, 2, 3), dtype=np.uint8)
    frame[0, 0] = (0, 0, 255)
    frame[0, 1] = (0, 255, 0)
    depth = np.array([[0.52, 0.51]], dtype=np.float32)
    shifted_x = np.array([[0.0, 0.0]], dtype=np.float32)

    warped = stereo_module._forward_warp_eye(frame, depth, shifted_x)
    assert int(warped[0, 0, 2]) > int(warped[0, 0, 1])


def test_compose_sbs_accepts_torch_tensors_full_mode() -> None:
    import pytest

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception:
        pytest.skip("torch unavailable")

    left = torch.zeros((6, 8, 3), dtype=torch.uint8)
    right = torch.ones((6, 8, 3), dtype=torch.uint8)

    composed = compose_sbs(left, right, "full")

    assert isinstance(composed, torch.Tensor)
    assert tuple(composed.shape) == (6, 16, 3)
    assert int(composed[0, 0, 0].item()) == 0
    assert int(composed[0, 8, 0].item()) == 1


def test_compose_sbs_accepts_torch_tensors_half_mode() -> None:
    import pytest

    torch = None
    try:
        import torch as _torch
        torch = _torch
    except Exception:
        pytest.skip("torch unavailable")

    left = torch.zeros((6, 8, 3), dtype=torch.uint8)
    right = torch.ones((6, 8, 3), dtype=torch.uint8)

    composed = compose_sbs(left, right, "half")

    assert isinstance(composed, torch.Tensor)
    assert tuple(composed.shape) == (6, 8, 3)

