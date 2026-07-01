import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from vr_sbs_converter.gui.advanced_panel import AdvancedPanel
from vr_sbs_converter.gui.mappers import (
    SIMPLE_PRESETS,
    build_advanced_config,
    build_simple_config,
)
from vr_sbs_converter.gui.simple_panel import SimplePanel


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_build_simple_config_uses_preset_mapping() -> None:
    config = build_simple_config(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        preset_key="quality-safe",
        upscale_4k=False,
    )
    assert config.profile == SIMPLE_PRESETS["quality-safe"]["profile"]
    assert config.perf_mode == SIMPLE_PRESETS["quality-safe"]["perf_mode"]
    assert config.compat_profile == "strict"
    assert config.upscale is False
    assert config.target_height is None


def test_build_simple_config_enables_fixed_4k_upscale_when_requested() -> None:
    config = build_simple_config(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        preset_key="quality-safe",
        upscale_4k=True,
    )

    assert config.upscale is True
    assert config.target_height == 2160


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
            "depth_compile": True,
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
    assert config.depth_compile is True
    assert config.device == "cuda"


def test_simple_panel_state_does_not_include_frame_preview_flag() -> None:
    _ensure_app()
    panel = SimplePanel()

    state = panel.get_state()

    assert "frame_preview_enabled" not in state
    assert "upscale_4k" in state
    assert state["upscale_4k"] is False


def test_advanced_panel_state_does_not_include_frame_preview_flag() -> None:
    _ensure_app()
    panel = AdvancedPanel()

    state = panel.get_state()

    assert "frame_preview_enabled" not in state


def test_advanced_panel_state_does_not_include_parallel_queue_size() -> None:
    _ensure_app()
    panel = AdvancedPanel()

    state = panel.get_state()

    assert "parallel_queue_size" not in state


def test_advanced_panel_state_includes_depth_compile_toggle() -> None:
    _ensure_app()
    panel = AdvancedPanel()

    panel.depth_compile.setChecked(True)

    state = panel.get_state()
    assert state["depth_compile"] is True
