import numpy as np
from pathlib import Path

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import dry_run_example_frame
from vr_sbs_converter.stereo import compose_sbs, synthesize_stereo_views


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
