from pathlib import Path
import subprocess
import threading

import numpy as np
import pytest

from vr_sbs_converter.ffmpeg_utils import FFmpegError, VideoMetadata
from vr_sbs_converter.checkpointing import (
    append_segment,
    build_checkpoint_identity,
    checkpoint_id_from_identity,
    create_manifest,
    load_manifest,
    save_manifest,
)
from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import ConversionCallbacks, ConversionCancelledError, run_conversion


def _make_sample_video(path: Path) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=96x54:rate=5",
        "-t",
        "1",
        str(path),
    ]
    subprocess.run(command, check=True)


def test_run_conversion_emits_callbacks(tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    _make_sample_video(input_video)

    events: dict[str, list] = {
        "start": [],
        "progress": [],
        "preview": [],
        "status": [],
        "complete": [],
    }
    callbacks = ConversionCallbacks(
        on_start=lambda payload: events["start"].append(payload),
        on_progress=lambda payload: events["progress"].append(payload),
        on_frame_preview=lambda frame: events["preview"].append(frame.shape),
        on_status=lambda message: events["status"].append(message),
        on_complete=lambda payload: events["complete"].append(payload),
        preview_enabled=True,
        preview_every_n=1,
    )

    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="strict",
    )
    run_conversion(config, callbacks=callbacks)

    assert output_video.exists()
    assert len(events["start"]) == 1
    assert len(events["complete"]) == 1
    assert len(events["progress"]) > 0
    assert len(events["preview"]) > 0
    assert {"input_path", "output_path", "total_frames", "fps", "width", "height"} <= set(events["start"][0].keys())
    assert {"frame_index", "total_frames", "percent", "stage"} <= set(events["progress"][0].keys())
    assert {"frames_processed", "effective_fps", "encoder", "output_path"} <= set(events["complete"][0].keys())
    assert any("Runtime summary:" in message for message in events["status"])


def test_run_conversion_honors_cancel_callback(tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    _make_sample_video(input_video)

    callbacks = ConversionCallbacks(should_cancel=lambda: True)
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="strict",
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        run_conversion(config, callbacks=callbacks)


def test_run_conversion_emits_complete_when_no_frames(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        def estimate(self, frame):
            return frame

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=16,
            height=16,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=0,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.read_raw_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)

    complete_events: list[dict] = []
    callbacks = ConversionCallbacks(on_complete=lambda payload: complete_events.append(payload))
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )

    run_conversion(config, callbacks=callbacks)

    assert len(complete_events) == 1
    assert complete_events[0]["frames_processed"] == 0
    assert complete_events[0]["effective_fps"] == 0.0


def test_cancelled_conversion_not_overridden_by_close_errors(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        def estimate(self, frame):
            return frame

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=16,
            height=16,
            fps=24.0,
            duration_seconds=1.0,
            total_frames=10,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.read_raw_frame", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.close_reader",
        lambda _reader: (_ for _ in ()).throw(FFmpegError("decoder close failure")),
    )
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.close_writer",
        lambda _writer: (_ for _ in ()).throw(BrokenPipeError("encoder close broken pipe")),
    )

    callbacks = ConversionCallbacks(should_cancel=lambda: True)
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        run_conversion(config, callbacks=callbacks)


def test_run_conversion_routes_default_to_parallel_orchestrator(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        def estimate(self, frame):
            return frame

    parallel_calls: list[dict] = []

    def _fake_parallel(**kwargs):
        parallel_calls.append(kwargs)
        return {"frames_written": 0, "all_workers_joined": True, "failure": None, "cancel_requested": False}

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=16,
            height=16,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=0,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.run_parallel_conversion_configured", _fake_parallel, raising=False)

    complete_events: list[dict] = []
    callbacks = ConversionCallbacks(on_complete=lambda payload: complete_events.append(payload))
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )
    run_conversion(config, callbacks=callbacks)

    assert len(parallel_calls) == 1
    assert len(complete_events) == 1
    assert complete_events[0]["frames_processed"] == 0


def test_run_conversion_parallel_path_uses_selected_stereo_backend(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        def estimate(self, frame):
            return np.ones(frame.shape[:2], dtype=np.float32)

    backend_calls: list[str] = []
    selected_backends: list[str] = []

    class _FakeBackend:
        name = "torch-cuda"

        @staticmethod
        def synthesize(frame_bgr, depth, stereo_strength, max_disparity_px):
            _ = depth, stereo_strength, max_disparity_px
            backend_calls.append("torch-cuda")
            return frame_bgr, frame_bgr

    def _fake_select_backend(device_preference: str, **_kwargs):
        selected_backends.append(device_preference)
        return _FakeBackend()

    def _fake_parallel(**kwargs):
        sample = np.zeros((8, 8, 3), dtype=np.uint8)
        left_eye, right_eye = kwargs["synthesize_stereo"](sample, np.ones((8, 8), dtype=np.float32))
        sbs_frame = kwargs["compose_sbs"]((left_eye, right_eye))
        kwargs["write_frame"](0, sbs_frame)
        return {"frames_written": 1, "all_workers_joined": True, "failure": None, "cancel_requested": False}

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=8,
            height=8,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=1,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.read_raw_frame",
        lambda *_args, **_kwargs: next(frames),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.write_raw_frame", lambda _writer, _frame: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.concat_video_segments", lambda **_kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.select_stereo_synthesis_backend",
        _fake_select_backend,
        raising=False,
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.run_parallel_conversion_configured", _fake_parallel, raising=False)

    frames = iter([np.zeros((8, 8, 3), dtype=np.uint8), None])
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        device="cuda",
        overwrite=True,
        compat_profile="off",
    )
    run_conversion(config)

    assert selected_backends == ["cuda"]
    assert backend_calls == ["torch-cuda"]


def test_run_conversion_parallel_cancel_does_not_emit_complete(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        def estimate(self, frame):
            return frame

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=16,
            height=16,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=0,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.run_parallel_conversion_configured",
        lambda **_kwargs: (_ for _ in ()).throw(ConversionCancelledError("Conversion cancelled by user.")),
        raising=False,
    )

    complete_events: list[dict] = []
    callbacks = ConversionCallbacks(on_complete=lambda payload: complete_events.append(payload))
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        run_conversion(config, callbacks=callbacks)
    assert complete_events == []


def test_run_conversion_parallel_callbacks_stay_on_caller_thread(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        @staticmethod
        def estimate(frame):
            return np.ones(frame.shape[:2], dtype=np.float32)

    class _FakeBackend:
        @staticmethod
        def synthesize(frame_bgr, depth, stereo_strength, max_disparity_px):
            _ = depth, stereo_strength, max_disparity_px
            return frame_bgr, frame_bgr

    frames = iter([np.zeros((8, 8, 3), dtype=np.uint8), None])
    main_thread_id = threading.get_ident()
    progress_thread_ids: list[int] = []
    progress_events: list[dict[str, float | int | str | None]] = []
    preview_thread_ids: list[int] = []
    preview_shapes: list[tuple[int, ...]] = []
    cancel_thread_ids: list[int] = []
    complete_thread_ids: list[int] = []
    complete_events: list[dict] = []

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=8,
            height=8,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=1,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.read_raw_frame",
        lambda *_args, **_kwargs: next(frames),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.write_raw_frame", lambda _writer, _frame: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.concat_video_segments", lambda **_kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.select_stereo_synthesis_backend",
        lambda _device, **_kwargs: _FakeBackend(),
        raising=False,
    )

    callbacks = ConversionCallbacks(
        on_progress=lambda payload: (
            progress_thread_ids.append(threading.get_ident()),
            progress_events.append(payload),
        ),
        on_frame_preview=lambda frame: (
            preview_thread_ids.append(threading.get_ident()),
            preview_shapes.append(frame.shape),
        ),
        should_cancel=lambda: cancel_thread_ids.append(threading.get_ident()) or False,
        on_complete=lambda payload: (
            complete_thread_ids.append(threading.get_ident()),
            complete_events.append(payload),
        ),
        preview_enabled=True,
        preview_every_n=1,
    )
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )

    run_conversion(config, callbacks=callbacks)

    assert progress_thread_ids
    assert preview_thread_ids
    assert cancel_thread_ids
    assert set(progress_thread_ids) == {main_thread_id}
    assert set(preview_thread_ids) == {main_thread_id}
    assert set(cancel_thread_ids) == {main_thread_id}
    assert set(complete_thread_ids) == {main_thread_id}
    assert {"frame_index", "total_frames", "percent", "stage"} <= set(progress_events[0].keys())
    assert progress_events[0]["stage"] == "converting"
    assert preview_shapes == [(8, 16, 3)]
    assert {"frames_processed", "effective_fps", "encoder", "output_path"} <= set(complete_events[0].keys())


def test_run_conversion_parallel_effective_fps_uses_wall_clock(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        @staticmethod
        def estimate(frame):
            return frame

    perf_ticks = [100.0, 102.0]

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=16,
            height=16,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=4,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.concat_video_segments", lambda **_kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.run_parallel_conversion_configured",
        lambda **_kwargs: {
            "frames_written": 4,
            "all_workers_joined": True,
            "failure": None,
            "cancel_requested": False,
        },
        raising=False,
    )
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.perf_counter",
        lambda: perf_ticks.pop(0) if perf_ticks else 102.0,
    )

    status_events: list[str] = []
    complete_events: list[dict] = []
    callbacks = ConversionCallbacks(
        on_status=lambda message: status_events.append(message),
        on_complete=lambda payload: complete_events.append(payload),
    )
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )

    run_conversion(config, callbacks=callbacks)

    assert len(complete_events) == 1
    assert complete_events[0]["frames_processed"] == 4
    assert complete_events[0]["effective_fps"] == pytest.approx(2.0)
    assert any("effective_fps=2.00" in message for message in status_events)


def test_run_conversion_live_top5_does_not_burst_when_callbacks_are_sparse(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    class _DummyEstimator:
        @staticmethod
        def estimate(frame):
            return frame

    tick = {"count": 0}

    def _fake_perf_counter() -> float:
        tick["count"] += 1
        if tick["count"] <= 2:
            return 0.0
        return 10.0 + (tick["count"] - 3) * 0.1

    def _fake_parallel(**kwargs):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        kwargs["write_frame"](0, frame)
        progress_cb = kwargs["callbacks"].on_progress
        assert progress_cb is not None
        for frame_index in (1, 2, 3, 4):
            progress_cb(
                {
                    "frame_index": frame_index,
                    "total_frames": 4,
                    "percent": frame_index * 25.0,
                    "stage": "converting",
                }
            )
        return {"frames_written": 4, "all_workers_joined": True, "failure": None, "cancel_requested": False}

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=8,
            height=8,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=4,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.write_raw_frame", lambda _writer, _frame: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.concat_video_segments", lambda **_kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.perf_counter", _fake_perf_counter)
    monkeypatch.setattr("vr_sbs_converter.pipeline.run_parallel_conversion_configured", _fake_parallel, raising=False)

    status_events: list[str] = []
    progress_events: list[dict] = []
    callbacks = ConversionCallbacks(
        on_status=lambda message: status_events.append(message),
        on_progress=lambda payload: progress_events.append(payload),
    )
    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )
    run_conversion(config, callbacks=callbacks)

    live_top5_events = [message for message in status_events if "Function timing top-5:" in message]
    assert progress_events
    assert len(live_top5_events) == 1


def test_run_conversion_auto_resumes_from_saved_checkpoint(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    progress_root = tmp_path / "progress"
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frames = iter([frame, frame, frame, frame, None])
    progress_events: list[dict[str, float | int | str]] = []
    complete_events: list[dict] = []

    class _DummyEstimator:
        @staticmethod
        def estimate(_frame):
            return np.ones((8, 8), dtype=np.float32)

    monkeypatch.setattr("vr_sbs_converter.pipeline._progress_root", lambda: progress_root)
    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=8,
            height=8,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=4,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.read_raw_frame", lambda *_args, **_kwargs: next(frames))
    monkeypatch.setattr("vr_sbs_converter.pipeline.write_raw_frame", lambda _writer, _frame: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.concat_video_segments", lambda **_kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)

    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )
    identity = build_checkpoint_identity(config)
    checkpoint_id = checkpoint_id_from_identity(identity)
    manifest = create_manifest(checkpoint_id=checkpoint_id, identity=identity)
    save_manifest(progress_root, manifest)
    append_segment(
        progress_root,
        manifest,
        "segment_000000.mp4",
        frames_written=2,
        status="canceled",
    )

    callbacks = ConversionCallbacks(
        on_progress=lambda payload: progress_events.append(payload),
        on_complete=lambda payload: complete_events.append(payload),
    )
    run_conversion(config, callbacks=callbacks, use_parallel=False)

    assert progress_events
    assert int(progress_events[0]["frame_index"]) == 3
    assert len(complete_events) == 1
    assert complete_events[0]["frames_processed"] == 4

    reloaded = load_manifest(progress_root, checkpoint_id)
    assert reloaded is not None
    assert reloaded.status == "complete"
    assert reloaded.next_frame_index == 4


def test_run_conversion_emits_live_top5_function_stats_during_run(monkeypatch, tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frames = iter([frame, frame, frame, frame, None])
    status_events: list[str] = []

    class _DummyEstimator:
        @staticmethod
        def estimate(_frame):
            return np.ones((8, 8), dtype=np.float32)

    now = 0.0

    def _fake_perf_counter() -> float:
        nonlocal now
        now += 0.5
        return now

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=8,
            height=8,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=4,
            has_audio=False,
        ),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.create_depth_estimator", lambda *args, **kwargs: _DummyEstimator())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.read_raw_frame", lambda *_args, **_kwargs: next(frames))
    monkeypatch.setattr("vr_sbs_converter.pipeline.write_raw_frame", lambda _writer, _frame: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.concat_video_segments", lambda **_kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.perf_counter", _fake_perf_counter)

    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        overwrite=True,
        compat_profile="off",
    )
    callbacks = ConversionCallbacks(on_status=lambda message: status_events.append(message))
    run_conversion(config, callbacks=callbacks, use_parallel=False)

    assert any("Function timing top-5:" in message for message in status_events)
    assert any("Function timing summary:" in message for message in status_events)


def test_run_conversion_estimates_depth_on_source_resolution_when_upscaling(
    monkeypatch, tmp_path: Path
) -> None:
    input_video = tmp_path / "in.mp4"
    output_video = tmp_path / "out.mp4"
    input_video.write_bytes(b"fake")

    source_frame = np.zeros((120, 240, 3), dtype=np.uint8)
    frames = iter([source_frame, None])
    depth_input_shapes: list[tuple[int, ...]] = []
    stereo_input_shapes: list[tuple[int, ...]] = []

    class _RecordingDepthEstimator:
        @staticmethod
        def estimate(frame):
            depth_input_shapes.append(frame.shape)
            return np.ones(frame.shape[:2], dtype=np.float32)

    def _fake_synthesize(**kwargs):
        stereo_input_shapes.append(kwargs["frame_bgr"].shape)
        frame = kwargs["frame_bgr"]
        return frame.copy(), frame.copy()

    monkeypatch.setattr("vr_sbs_converter.pipeline.ensure_ffmpeg_installed", lambda: None)
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.probe_video",
        lambda _path: VideoMetadata(
            width=240,
            height=120,
            fps=24.0,
            duration_seconds=0.0,
            total_frames=1,
            has_audio=False,
        ),
    )
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.create_depth_estimator",
        lambda *args, **kwargs: _RecordingDepthEstimator(),
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_reader", lambda _path: object())
    monkeypatch.setattr("vr_sbs_converter.pipeline.open_frame_writer", lambda **kwargs: object())
    monkeypatch.setattr(
        "vr_sbs_converter.pipeline.read_raw_frame", lambda *_args, **_kwargs: next(frames)
    )
    monkeypatch.setattr("vr_sbs_converter.pipeline.write_raw_frame", lambda _writer, _frame: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_reader", lambda _reader: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.close_writer", lambda _writer: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.concat_video_segments", lambda **_kwargs: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.shutil.move", lambda _src, _dst: None)
    monkeypatch.setattr("vr_sbs_converter.pipeline.synthesize_stereo_views", _fake_synthesize)

    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
        upscale=True,
        target_height=480,
        overwrite=True,
        compat_profile="off",
    )
    run_conversion(config, callbacks=None, use_parallel=False)

    assert depth_input_shapes == [(120, 240, 3)]
    assert stereo_input_shapes and stereo_input_shapes[0] == (480, 960, 3)
