from __future__ import annotations

import warnings

import numpy as np

from vr_sbs_converter.depth import AutoDepthEstimator, DepthEstimator


class _AffineDepthEstimator(DepthEstimator):
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.calls.append(frame_bgr)
        return (frame_bgr.astype(np.float32) * 2.0) + 1.0


class _FailingBatchDepthEstimator(DepthEstimator):
    def __init__(self) -> None:
        self.batch_calls = 0

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        raise RuntimeError("preferred depth failed")

    def estimate_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        self.batch_calls += 1
        raise RuntimeError("preferred depth failed")


class _FallbackBatchDepthEstimator(DepthEstimator):
    def __init__(self) -> None:
        self.batch_calls = 0

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr.astype(np.float32)

    def estimate_batch(self, frames_bgr: list[np.ndarray]) -> list[np.ndarray]:
        self.batch_calls += 1
        return [frame.astype(np.float32) for frame in frames_bgr]


def test_depth_estimator_default_batch_maps_estimate() -> None:
    estimator = _AffineDepthEstimator()
    frames = [
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[3.0, 4.0]], dtype=np.float32),
    ]

    results = estimator.estimate_batch(frames)

    assert len(results) == 2
    assert estimator.calls == frames
    assert np.array_equal(results[0], np.array([[3.0, 5.0]], dtype=np.float32))
    assert np.array_equal(results[1], np.array([[7.0, 9.0]], dtype=np.float32))


def test_auto_depth_estimator_batch_switches_to_fallback_after_failure() -> None:
    preferred = _FailingBatchDepthEstimator()
    fallback = _FallbackBatchDepthEstimator()
    estimator = AutoDepthEstimator(preferred=preferred, fallback=fallback)
    frames = [np.array([[0.2, 0.4]], dtype=np.float32)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = estimator.estimate_batch(frames)
        second = estimator.estimate_batch(frames)

    assert preferred.batch_calls == 1
    assert fallback.batch_calls == 2
    assert len(caught) == 1
    assert "switching to luma depth" in str(caught[0].message)
    assert np.array_equal(first[0], frames[0])
    assert np.array_equal(second[0], frames[0])
