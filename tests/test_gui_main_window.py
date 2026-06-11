from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from vr_sbs_converter.gui.main_window import MainWindow


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_builds_simple_config_from_default_preset() -> None:
    _ensure_app()
    window = MainWindow()
    window._input_edit.setText(str(Path("/tmp/in.mp4")))
    window._output_edit.setText(str(Path("/tmp/out.mp4")))

    config = window.build_config_from_ui()

    assert config.profile == "halo-safe"
    assert config.perf_mode == "quality"
    assert config.compat_profile == "strict"


def test_main_window_builds_advanced_config_from_panel_values() -> None:
    _ensure_app()
    window = MainWindow()
    window._input_edit.setText(str(Path("/tmp/in.mp4")))
    window._output_edit.setText(str(Path("/tmp/out.mp4")))
    window._tabs.setCurrentIndex(1)

    panel = window._advanced_panel
    panel.upscale.setChecked(True)
    panel.target_height.setValue(2160)
    panel.profile.setCurrentText("balanced")
    panel.perf_mode.setCurrentText("gpu-balanced")
    panel.device.setCurrentText("cuda")
    panel.depth_backend.setCurrentText("luma")
    panel.encoder.setCurrentText("h264_nvenc")

    config = window.build_config_from_ui()

    assert config.upscale is True
    assert config.target_height == 2160
    assert config.profile == "balanced"
    assert config.perf_mode == "gpu-balanced"
    assert config.device == "cuda"
    assert config.depth_backend == "luma"
    assert config.encoder == "h264_nvenc"

