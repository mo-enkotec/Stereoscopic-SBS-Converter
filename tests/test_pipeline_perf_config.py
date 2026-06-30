from pathlib import Path

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import (
    _cap_cuda_batch_size_for_resolution,
    _cap_gpu_depth_queue,
    _should_enable_gpu_stereo,
    resolve_parallel_queue_config,
    resolve_runtime_plan,
)
from vr_sbs_converter.pipeline_parallel import ParallelQueueConfig


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


def test_quality_4k_upscale_uses_adaptive_depth_scale_for_throughput() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="quality",
        device="cuda",
        upscale=True,
        target_height=2160,
    )

    runtime = resolve_runtime_plan(config)
    assert runtime.depth_process_scale == 0.7


def test_quality_4k_upscale_respects_explicit_depth_scale_override() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="quality",
        device="cuda",
        upscale=True,
        target_height=2160,
        depth_process_scale=1.0,
        depth_process_scale_overridden=True,
    )

    runtime = resolve_runtime_plan(config)
    assert runtime.depth_process_scale == 1.0


def test_quality_4k_upscale_respects_direct_config_depth_scale() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="quality",
        device="cuda",
        upscale=True,
        target_height=2160,
        depth_process_scale=1.0,
    )

    runtime = resolve_runtime_plan(config)
    assert runtime.depth_process_scale == 1.0


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


def test_cap_gpu_depth_queue_limits_depth_stage_to_two() -> None:
    queue_config = ParallelQueueConfig(
        decode_queue_size=12,
        depth_queue_size=12,
        stereo_queue_size=12,
        encode_queue_size=12,
    )

    capped = _cap_gpu_depth_queue(queue_config)

    assert capped.decode_queue_size == 12
    assert capped.depth_queue_size == 2
    assert capped.stereo_queue_size == 12
    assert capped.encode_queue_size == 12


def test_cap_gpu_depth_queue_keeps_small_depth_queue() -> None:
    queue_config = ParallelQueueConfig(
        decode_queue_size=4,
        depth_queue_size=1,
        stereo_queue_size=4,
        encode_queue_size=4,
    )

    capped = _cap_gpu_depth_queue(queue_config)
    assert capped.depth_queue_size == 1


def test_cap_cuda_batch_size_for_resolution_prefers_single_batch_on_1080p_plus() -> None:
    assert _cap_cuda_batch_size_for_resolution(4, 1920, 1080) == 1
    assert _cap_cuda_batch_size_for_resolution(4, 3840, 2160) == 1


def test_cap_cuda_batch_size_for_resolution_allows_small_batches_for_low_res() -> None:
    assert _cap_cuda_batch_size_for_resolution(4, 1280, 720) == 2
    assert _cap_cuda_batch_size_for_resolution(4, 640, 360) == 4


def test_quality_4k_prefers_cpu_stereo_when_effective_batch_is_one() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="quality",
        device="cuda",
    )

    assert (
        _should_enable_gpu_stereo(
            config=config,
            stereo_backend_name="torch-cuda",
            effective_depth_batch_size=1,
            width=3840,
            height=2160,
        )
        is False
    )


def test_gpu_balanced_keeps_gpu_stereo_with_batch_two() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="gpu-balanced",
        device="cuda",
    )

    assert (
        _should_enable_gpu_stereo(
            config=config,
            stereo_backend_name="torch-cuda",
            effective_depth_batch_size=2,
            width=3840,
            height=2160,
        )
        is True
    )


def test_quality_1080p_does_not_auto_bypass_gpu_stereo() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        perf_mode="quality",
        device="cuda",
    )

    assert (
        _should_enable_gpu_stereo(
            config=config,
            stereo_backend_name="torch-cuda",
            effective_depth_batch_size=1,
            width=1920,
            height=1080,
        )
        is True
    )
