from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SbsMode = Literal["full", "half"]
DeviceMode = Literal["auto", "cpu", "cuda"]
DepthBackend = Literal["auto", "midas", "luma"]

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
