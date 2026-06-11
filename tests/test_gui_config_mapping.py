from pathlib import Path

from vr_sbs_converter.gui.mappers import (
    SIMPLE_PRESETS,
    build_advanced_config,
    build_simple_config,
)


def test_build_simple_config_uses_preset_mapping() -> None:
    config = build_simple_config(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        preset_key="quality-safe",
        frame_preview_enabled=False,
    )
    assert config.profile == SIMPLE_PRESETS["quality-safe"]["profile"]
    assert config.perf_mode == SIMPLE_PRESETS["quality-safe"]["perf_mode"]
    assert config.compat_profile == "strict"


def test_build_advanced_config_honors_option_overrides() -> None:
    config = build_advanced_config(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        options={
            "sbs_mode": "full",
            "upscale": True,
            "target_height": 2160,
            "profile": "balanced",
            "perf_mode": "gpu-balanced",
            "depth_backend": "luma",
            "device": "cuda",
            "encoder": "h264_nvenc",
            "compat_profile": "strict",
            "audio_fallback": "copy-aac",
            "stereo_strength": 0.7,
            "overwrite": True,
        },
    )
    assert config.upscale is True
    assert config.target_height == 2160
    assert config.profile == "balanced"
    assert config.perf_mode == "gpu-balanced"
    assert config.encoder == "h264_nvenc"
    assert config.depth_backend == "luma"
    assert config.device == "cuda"
