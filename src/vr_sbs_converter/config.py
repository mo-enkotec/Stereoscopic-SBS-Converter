from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SbsMode = Literal["full", "half"]
DeviceMode = Literal["auto", "cpu", "cuda"]
DepthBackend = Literal["auto", "midas", "luma"]
ProfileMode = Literal["halo-safe", "balanced", "fast"]
PerfMode = Literal["quality", "gpu-balanced", "max-speed"]
EncoderMode = Literal["auto", "libx264", "h264_nvenc"]
CompatProfile = Literal["strict", "off"]
AudioFallbackMode = Literal["copy-aac"]

_RESOLUTION_ALIASES = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "2160p": 2160,
    "4k": 2160,
    "8k": 4320,
}


@dataclass(slots=True)
class ConversionConfig:
    input_path: Path
    output_path: Path
    sbs_mode: SbsMode = "full"
    upscale: bool = False
    target_height: int | None = None
    codec: str = "libx264"
    preset: str = "slow"
    crf: int = 18
    device: DeviceMode = "auto"
    depth_backend: DepthBackend = "auto"
    profile: ProfileMode = "halo-safe"
    perf_mode: PerfMode = "quality"
    encoder: EncoderMode = "auto"
    compat_profile: CompatProfile = "strict"
    audio_fallback: AudioFallbackMode = "copy-aac"
    max_disparity_px: int | None = None
    depth_process_scale: float | None = None
    depth_compile: bool = False
    edge_protect_strength: float | None = None
    stereo_strength: float = 0.8
    overwrite: bool = False
    keep_temp: bool = False
    temp_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.crf < 0 or self.crf > 51:
            raise ValueError("--crf must be between 0 and 51.")
        if self.target_height is not None and self.target_height < 240:
            raise ValueError("Target height must be at least 240 pixels.")
        if self.stereo_strength <= 0 or self.stereo_strength > 3:
            raise ValueError("--stereo-strength must be in range (0, 3].")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid --device value.")
        if self.depth_backend not in {"auto", "midas", "luma"}:
            raise ValueError("Invalid --depth-backend value.")
        if self.sbs_mode not in {"full", "half"}:
            raise ValueError("Invalid --sbs-mode value.")
        if self.profile not in {"halo-safe", "balanced", "fast"}:
            raise ValueError("Invalid --profile value.")
        if self.perf_mode not in {"quality", "gpu-balanced", "max-speed"}:
            raise ValueError("Invalid --perf-mode value.")
        if self.encoder not in {"auto", "libx264", "h264_nvenc"}:
            raise ValueError("Invalid --encoder value.")
        if self.compat_profile not in {"strict", "off"}:
            raise ValueError("Invalid --compat-profile value.")
        if self.audio_fallback not in {"copy-aac"}:
            raise ValueError("Invalid --audio-fallback value.")

        if self.max_disparity_px is None:
            self.max_disparity_px = {"halo-safe": 12, "balanced": 16, "fast": 22}[self.profile]
        if self.depth_process_scale is None:
            self.depth_process_scale = {
                "quality": 1.0,
                "gpu-balanced": 0.75,
                "max-speed": 0.6,
            }[self.perf_mode]
        if self.edge_protect_strength is None:
            self.edge_protect_strength = {
                "halo-safe": 0.9,
                "balanced": 0.75,
                "fast": 0.55,
            }[self.profile]

        if self.max_disparity_px <= 0 or self.max_disparity_px > 128:
            raise ValueError("--max-disparity-px must be in range [1, 128].")
        if self.depth_process_scale <= 0 or self.depth_process_scale > 1:
            raise ValueError("--depth-process-scale must be in range (0, 1].")
        if self.edge_protect_strength < 0 or self.edge_protect_strength > 1:
            raise ValueError("--edge-protect-strength must be in range [0, 1].")


def parse_target_height(value: str) -> int:
    parsed = value.strip().lower()
    if parsed in _RESOLUTION_ALIASES:
        return _RESOLUTION_ALIASES[parsed]
    if parsed.endswith("p") and parsed[:-1].isdigit():
        return int(parsed[:-1])
    if parsed.isdigit():
        return int(parsed)
    if "x" in parsed:
        width_token, height_token = parsed.split("x", 1)
        if width_token.isdigit() and height_token.isdigit():
            return int(height_token)
    raise ValueError(
        f"Invalid --target value '{value}'. Use values like 1080p, 4k, 2160, or 3840x2160."
    )
