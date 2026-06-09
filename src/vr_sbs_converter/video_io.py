from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConversionConfig
from .ffmpeg_utils import FFmpegError, run_command


class VideoIOError(RuntimeError):
    """Raised when frame read/write operations fail."""


@dataclass(slots=True)
class FrameStream:
    process: subprocess.Popen[bytes]
    width: int
    height: int
    fps: float


def open_frame_reader(input_path: Path) -> FrameStream:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return FrameStream(process=process, width=0, height=0, fps=0.0)


def read_raw_frame(stream: FrameStream, width: int, height: int) -> np.ndarray | None:
    frame_size = width * height * 3
    if stream.process.stdout is None:
        raise VideoIOError("Frame reader stdout is unavailable.")
    raw = stream.process.stdout.read(frame_size)
    if not raw:
        return None
    if len(raw) != frame_size:
        raise VideoIOError(
            f"Incomplete frame from decoder. Expected {frame_size} bytes, got {len(raw)}."
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


def close_reader(stream: FrameStream) -> None:
    if stream.process.stdout:
        stream.process.stdout.close()
    return_code = stream.process.wait()
    if return_code != 0:
        stderr = stream.process.stderr.read().decode("utf-8", errors="replace")
        raise FFmpegError(f"ffmpeg decoder failed with exit code {return_code}: {stderr}")


def open_frame_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    config: ConversionConfig,
) -> subprocess.Popen[bytes]:
    overwrite_flag = "-y" if config.overwrite else "-n"
    command = [
        "ffmpeg",
        "-v",
        "error",
        overwrite_flag,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        config.codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def write_raw_frame(writer: subprocess.Popen[bytes], frame: np.ndarray) -> None:
    if writer.stdin is None:
        raise VideoIOError("Frame writer stdin is unavailable.")
    if frame.dtype != np.uint8:
        raise VideoIOError("Frame must use uint8 pixel format.")
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    writer.stdin.write(frame.tobytes())


def close_writer(writer: subprocess.Popen[bytes]) -> None:
    if writer.stdin:
        writer.stdin.flush()
        writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        stderr = writer.stderr.read().decode("utf-8", errors="replace")
        raise FFmpegError(f"ffmpeg encoder failed with exit code {return_code}: {stderr}")


def mux_audio_track(
    source_video: Path,
    silent_video: Path,
    destination: Path,
    overwrite: bool,
) -> None:
    overwrite_flag = "-y" if overwrite else "-n"
    command = [
        "ffmpeg",
        "-v",
        "error",
        overwrite_flag,
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(destination),
    ]
    run_command(command)
