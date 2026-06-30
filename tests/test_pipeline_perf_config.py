from pathlib import Path

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import resolve_parallel_queue_config, resolve_runtime_plan


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


def test_gpu_balanced_auto_device_uses_cuda_runtime_when_available(monkeypatch) -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="gpu-balanced",
        encoder="auto",
        device="auto",
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline._is_cuda_runtime_available", lambda: True)
    runtime = resolve_runtime_plan(config)
    assert runtime.use_fp16 is True
    assert runtime.preferred_encoder == "h264_nvenc"


def test_gpu_balanced_auto_device_uses_cpu_plan_when_cuda_runtime_unavailable(monkeypatch) -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="gpu-balanced",
        encoder="auto",
        device="auto",
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline._is_cuda_runtime_available", lambda: False)
    runtime = resolve_runtime_plan(config)
    assert runtime.use_fp16 is False
    assert runtime.preferred_encoder == "libx264"


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


def test_resolve_parallel_queue_config_uses_perf_mode_defaults() -> None:
    quality = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="quality",
    )
    gpu_balanced = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="gpu-balanced",
    )
    max_speed = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="max-speed",
    )

    assert resolve_parallel_queue_config(quality).decode_queue_size == 4
    assert resolve_parallel_queue_config(gpu_balanced).decode_queue_size == 8
    assert resolve_parallel_queue_config(max_speed).decode_queue_size == 12


def test_resolve_parallel_queue_config_respects_override() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        parallel_queue_size=16,
    )

    queue_config = resolve_parallel_queue_config(config)
    assert queue_config.decode_queue_size == 16
    assert queue_config.depth_queue_size == 16
    assert queue_config.stereo_queue_size == 16
    assert queue_config.encode_queue_size == 16
