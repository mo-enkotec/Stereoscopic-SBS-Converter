from pathlib import Path
import subprocess

import pytest

from vr_sbs_converter.ffmpeg_utils import FFmpegError, VideoMetadata
from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import ConversionCallbacks, run_conversion


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
