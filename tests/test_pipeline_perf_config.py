from pathlib import Path

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import resolve_runtime_plan


def test_gpu_balanced_profile_sets_depth_scale_and_encoder() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="gpu-balanced",
        encoder="auto",
        device="cuda",
    )
    runtime = resolve_runtime_plan(config)
    assert runtime.depth_process_scale == 0.75
    assert runtime.use_fp16 is True
    assert runtime.preferred_encoder == "h264_nvenc"


def test_quality_mode_uses_full_depth_scale() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="quality",
        encoder="libx264",
        device="cpu",
    )
    runtime = resolve_runtime_plan(config)
    assert runtime.depth_process_scale == 1.0
    assert runtime.use_fp16 is False
    assert runtime.preferred_encoder == "libx264"
