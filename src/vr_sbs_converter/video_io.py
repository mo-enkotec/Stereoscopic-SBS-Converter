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


def ffmpeg_supports_encoder(encoder_name: str) -> bool:
    command = ["ffmpeg", "-v", "error", "-hide_banner", "-encoders"]
    try:
        result = run_command(command)
    except FFmpegError:
        return False
    return encoder_name in result.stdout


def ffmpeg_nvenc_usable() -> bool:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=16x16:rate=1:color=black",
        "-frames:v",
        "1",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    return process.returncode == 0


def _nvenc_preset_from_x264(preset: str) -> str:
    mapping = {
        "veryslow": "p7",
        "slower": "p6",
        "slow": "p5",
        "medium": "p4",
        "fast": "p3",
        "faster": "p2",
        "veryfast": "p1",
    }
    return mapping.get(preset, "p4")


def _compat_output_flags(config: ConversionConfig) -> list[str]:
    if config.compat_profile != "strict":
        return []
    return [
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-movflags",
        "+faststart",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
    ]


def build_frame_writer_command(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    config: ConversionConfig,
    codec_override: str | None = None,
) -> list[str]:
    overwrite_flag = "-y" if config.overwrite else "-n"
    codec_name = codec_override or config.codec
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
    ]

    if codec_name == "h264_nvenc":
        command += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            _nvenc_preset_from_x264(config.preset),
            "-cq:v",
            str(config.crf),
            "-b:v",
            "0",
        ]
    else:
        command += [
            "-c:v",
            codec_name,
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
        ]

    command += _compat_output_flags(config)
    command.append(str(output_path))
    return command


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
    codec_override: str | None = None,
) -> subprocess.Popen[bytes]:
    command = build_frame_writer_command(
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        config=config,
        codec_override=codec_override,
    )
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
    audio_fallback: str = "copy-aac",
) -> None:
    copy_command = build_mux_audio_command(
        source_video=source_video,
        silent_video=silent_video,
        destination=destination,
        overwrite=overwrite,
        transcode_audio=False,
    )
    try:
        run_command(copy_command)
        return
    except FFmpegError:
        if audio_fallback != "copy-aac":
            raise

    fallback_command = build_mux_audio_command(
        source_video=source_video,
        silent_video=silent_video,
        destination=destination,
        overwrite=overwrite,
        transcode_audio=True,
    )
    run_command(fallback_command)


def build_mux_audio_command(
    source_video: Path,
    silent_video: Path,
    destination: Path,
    overwrite: bool,
    transcode_audio: bool,
) -> list[str]:
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
        "-shortest",
    ]
    if transcode_audio:
        command += ["-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-c:a", "copy"]

    command.append(str(destination))
    return command
