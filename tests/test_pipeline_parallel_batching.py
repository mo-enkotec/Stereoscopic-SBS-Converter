from __future__ import annotations

import pytest

from vr_sbs_converter.pipeline_parallel import DepthFramePayload, run_parallel_conversion_configured


def test_parallel_pipeline_uses_depth_batch_callback_when_configured() -> None:
    frames = iter(["f0", "f1", "f2", None])
    batch_calls: list[list[str]] = []
    writes: list[tuple[int, str]] = []

    def read_frame():
        return next(frames)

    def estimate_depth_batch(batch: list[str]) -> list[str]:
        batch_calls.append(list(batch))
        return [f"d:{item}" for item in batch]

    result = run_parallel_conversion_configured(
        read_frame=read_frame,
        estimate_depth=lambda frame: f"d-single:{frame}",
        estimate_depth_batch=estimate_depth_batch,
        depth_batch_size=3,
        synthesize_stereo=lambda frame, depth: (frame, depth),
        compose_sbs=lambda stereo_payload: f"sbs:{stereo_payload[0]}:{stereo_payload[1]}",
        write_frame=lambda frame_index, payload: writes.append((frame_index, payload)),
        total_frames=3,
    )

    assert batch_calls
    assert [item for batch in batch_calls for item in batch] == ["f0", "f1", "f2"]
    assert all(1 <= len(batch) <= 3 for batch in batch_calls)
    assert writes == [
        (0, "sbs:f0:d:f0"),
        (1, "sbs:f1:d:f1"),
        (2, "sbs:f2:d:f2"),
    ]
    assert result["frames_written"] == 3
    assert result["telemetry"]["depth_batches"] == len(batch_calls)
    assert sum(
        batch_size * count
        for batch_size, count in result["telemetry"]["depth_batch_histogram"].items()
    ) == 3
    assert 1.0 <= result["telemetry"]["depth_batch_avg"] <= 3.0
    assert set(result["telemetry"]["queue_max_depth"]) == {"decode", "depth", "stereo", "encode"}


def test_parallel_pipeline_rejects_non_positive_depth_batch_size() -> None:
    def read_frame():
        return None

    with pytest.raises(ValueError, match="depth_batch_size must be >= 1"):
        run_parallel_conversion_configured(
            read_frame=read_frame,
            estimate_depth=lambda frame: frame,
            synthesize_stereo=lambda frame, depth: (frame, depth),
            compose_sbs=lambda stereo_payload: stereo_payload,
            write_frame=lambda _index, _payload: None,
            depth_batch_size=0,
            total_frames=0,
        )


def test_parallel_pipeline_reports_partial_batch_distribution() -> None:
    frames = iter(["f0", "f1", "f2", None])

    result = run_parallel_conversion_configured(
        read_frame=lambda: next(frames),
        estimate_depth=lambda frame: f"d:{frame}",
        depth_batch_size=2,
        synthesize_stereo=lambda frame, depth: (frame, depth),
        compose_sbs=lambda stereo_payload: stereo_payload,
        write_frame=lambda _index, _payload: None,
        total_frames=3,
    )

    histogram = result["telemetry"]["depth_batch_histogram"]
    assert result["telemetry"]["depth_batches"] == sum(histogram.values())
    assert sum(batch_size * count for batch_size, count in histogram.items()) == 3
    assert all(1 <= batch_size <= 2 for batch_size in histogram)
    assert 1.0 <= result["telemetry"]["depth_batch_avg"] <= 2.0


def test_parallel_pipeline_uses_stereo_batch_callback_when_configured() -> None:
    frames = iter(["f0", None])
    stereo_batch_calls: list[list[tuple[str, str]]] = []

    def synthesize_stereo(_frame: str, _depth: str):
        raise AssertionError("single stereo callback should not be used when batch callback is provided")

    def synthesize_stereo_batch(frames_payload: list[str], depths_payload: list[str]) -> list[tuple[str, str]]:
        stereo_batch_calls.append(list(zip(frames_payload, depths_payload, strict=True)))
        return [(frame, f"stereo:{depth}") for frame, depth in zip(frames_payload, depths_payload, strict=True)]

    result = run_parallel_conversion_configured(
        read_frame=lambda: next(frames),
        estimate_depth=lambda frame: f"d:{frame}",
        synthesize_stereo=synthesize_stereo,
        synthesize_stereo_batch=synthesize_stereo_batch,
        stereo_batch_size=4,
        compose_sbs=lambda stereo_payload: stereo_payload,
        write_frame=lambda _index, _payload: None,
        total_frames=1,
    )

    assert stereo_batch_calls == [[("f0", "d:f0")]]
    assert result["telemetry"]["stereo_batches"] == 1
    assert result["telemetry"]["stereo_batch_histogram"] == {1: 1}


def test_parallel_pipeline_allows_depth_stage_to_replace_frame_payload() -> None:
    frames = iter(["f0", None])
    writes: list[tuple[int, tuple[str, str]]] = []

    result = run_parallel_conversion_configured(
        read_frame=lambda: next(frames),
        estimate_depth=lambda frame: DepthFramePayload(
            frame_payload=f"gpu:{frame}",
            depth_payload=f"d:{frame}",
        ),
        synthesize_stereo=lambda frame, depth: (frame, depth),
        compose_sbs=lambda stereo_payload: stereo_payload,
        write_frame=lambda index, payload: writes.append((index, payload)),
        total_frames=1,
    )

    assert writes == [(0, ("gpu:f0", "d:f0"))]
    assert result["frames_written"] == 1
