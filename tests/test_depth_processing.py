import cv2
import numpy as np

from vr_sbs_converter.depth import condition_depth_for_stereo


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
