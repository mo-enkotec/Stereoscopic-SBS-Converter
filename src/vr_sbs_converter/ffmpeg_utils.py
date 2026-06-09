from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


class FFmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe commands fail."""


@dataclass(slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration_seconds: float
    total_frames: int
    has_audio: bool


def ensure_ffmpeg_installed() -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg executable was not found in PATH.")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe executable was not found in PATH.")


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip()
        raise FFmpegError(
            f"Command failed with exit code {process.returncode}: {' '.join(command)}\n{stderr}"
        )
    return process


def _parse_fps(token: str | None) -> float:
    if not token:
        return 30.0
    try:
        value = Fraction(token)
        if value.numerator == 0 or value.denominator == 0:
            return 30.0
        fps = float(value)
        return fps if fps > 0 else 30.0
    except (ValueError, ZeroDivisionError):
        return 30.0


def probe_video(path: Path) -> VideoMetadata:
    ensure_ffmpeg_installed()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    format_section = payload.get("format", {})

    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video_stream is None:
        raise FFmpegError(f"No video stream found in input file: {path}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise FFmpegError("Could not determine source video dimensions.")

    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    duration_token = video_stream.get("duration") or format_section.get("duration")
    try:
        duration = float(duration_token)
    except (TypeError, ValueError):
        duration = 0.0

    frames_token = video_stream.get("nb_frames")
    total_frames: int
    if frames_token and str(frames_token).isdigit():
        total_frames = int(frames_token)
    elif duration > 0:
        total_frames = max(1, int(round(duration * fps)))
    else:
        total_frames = 0

    has_audio = any(item.get("codec_type") == "audio" for item in streams)
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
        total_frames=total_frames,
        has_audio=has_audio,
    )
