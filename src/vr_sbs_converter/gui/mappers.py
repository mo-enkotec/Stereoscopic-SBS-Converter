from __future__ import annotations

from pathlib import Path
from typing import Any

from vr_sbs_converter.config import ConversionConfig

SIMPLE_PRESETS: dict[str, dict[str, str]] = {
    "quality-safe": {
        "label": "Quality Safe",
        "profile": "halo-safe",
        "perf_mode": "quality",
    },
    "balanced": {
        "label": "Balanced",
        "profile": "balanced",
        "perf_mode": "gpu-balanced",
    },
    "fast": {
        "label": "Fast",
        "profile": "fast",
        "perf_mode": "max-speed",
    },
}


def build_simple_config(
    input_path: Path,
    output_path: Path,
    preset_key: str,
    upscale_4k: bool,
) -> ConversionConfig:
    if preset_key not in SIMPLE_PRESETS:
        raise ValueError(f"Unknown simple preset '{preset_key}'.")

    preset = SIMPLE_PRESETS[preset_key]
    return ConversionConfig(
        input_path=input_path,
        output_path=output_path,
        upscale=upscale_4k,
        target_height=2160 if upscale_4k else None,
        profile=preset["profile"],  # type: ignore[arg-type]
        perf_mode=preset["perf_mode"],  # type: ignore[arg-type]
        compat_profile="strict",
        audio_fallback="copy-aac",
        overwrite=True,
    )


def build_advanced_config(
    input_path: Path,
    output_path: Path,
    options: dict[str, Any],
) -> ConversionConfig:
    payload: dict[str, Any] = {
        "input_path": input_path,
        "output_path": output_path,
        "overwrite": options.get("overwrite", True),
    }
    allowed_keys = {
        "sbs_mode",
        "upscale",
        "target_height",
        "codec",
        "preset",
        "crf",
        "device",
        "depth_backend",
        "profile",
        "perf_mode",
        "encoder",
        "compat_profile",
        "audio_fallback",
        "max_disparity_px",
        "depth_process_scale",
        "edge_protect_strength",
        "stereo_strength",
        "parallel_queue_size",
        "gpu_batch_size",
        "gpu_stream_overlap",
        "overwrite",
        "keep_temp",
        "temp_dir",
    }
    for key, value in options.items():
        if key in allowed_keys and value is not None:
            payload[key] = value
    return ConversionConfig(**payload)
