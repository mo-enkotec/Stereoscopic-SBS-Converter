from pathlib import Path

import pytest

import vr_sbs_converter.video_io as video_io
from vr_sbs_converter.ffmpeg_utils import FFmpegError
from vr_sbs_converter.video_io import build_mux_audio_command, mux_audio_track


def test_build_mux_audio_command_copy_mode() -> None:
    command = build_mux_audio_command(
        source_video=Path("/tmp/src.mp4"),
        silent_video=Path("/tmp/silent.mp4"),
        destination=Path("/tmp/out.mp4"),
        overwrite=True,
        transcode_audio=False,
    )
    text = " ".join(command)
    assert "-c:a copy" in text
    assert "-c:v copy" in text


def test_build_mux_audio_command_aac_mode() -> None:
    command = build_mux_audio_command(
        source_video=Path("/tmp/src.mp4"),
        silent_video=Path("/tmp/silent.mp4"),
        destination=Path("/tmp/out.mp4"),
        overwrite=True,
        transcode_audio=True,
    )
    text = " ".join(command)
    assert "-c:a aac" in text
    assert "-b:a 192k" in text


def test_mux_audio_track_falls_back_to_aac(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run_command(command: list[str]):  # type: ignore[no-untyped-def]
        calls.append(command)
        if len(calls) == 1:
            raise FFmpegError("audio copy failed")
        return object()

    monkeypatch.setattr(video_io, "run_command", fake_run_command)
    mux_audio_track(
        source_video=Path("/tmp/src.mp4"),
        silent_video=Path("/tmp/silent.mp4"),
        destination=Path("/tmp/out.mp4"),
        overwrite=True,
        audio_fallback="copy-aac",
    )
    assert len(calls) == 2
    assert "-c:a" in calls[1] and "aac" in calls[1]
