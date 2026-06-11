from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ffmpeg_utils import FFmpegError, run_command


@dataclass(slots=True)
class VideoStreamInfo:
    codec_name: str
    profile: str
    pix_fmt: str
    width: int
    height: int


def probe_output_video_stream(path: Path) -> VideoStreamInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video_stream is None:
        raise FFmpegError(f"No video stream found in output file: {path}")

    return VideoStreamInfo(
        codec_name=str(video_stream.get("codec_name") or ""),
        profile=str(video_stream.get("profile") or ""),
        pix_fmt=str(video_stream.get("pix_fmt") or ""),
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
    )


def evaluate_player_compatibility(info: VideoStreamInfo) -> list[str]:
    warnings: list[str] = []
    profile_lower = info.profile.lower()
    pix_fmt_lower = info.pix_fmt.lower()
    codec_lower = info.codec_name.lower()

    if codec_lower not in {"h264", "hevc"}:
        warnings.append(
            f"Codec '{info.codec_name}' is less portable; H.264/H.265 are safer for VR players."
        )

    if "4:4:4" in profile_lower:
        warnings.append(
            f"Profile '{info.profile}' is often unsupported by hardware decoders; use H.264 High (4:2:0)."
        )

    if pix_fmt_lower != "yuv420p":
        warnings.append(
            f"Pixel format '{info.pix_fmt}' may fail in strict players; yuv420p is recommended."
        )

    if info.width > 5760:
        warnings.append(
            f"Output width {info.width} may exceed decoder limits on some devices (decoder limit warning)."
        )

    return warnings
