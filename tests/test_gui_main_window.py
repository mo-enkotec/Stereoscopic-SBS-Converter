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


def test_main_window_pick_input_file_always_refreshes_output_path(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window._output_edit.setText(str(Path("/videos/old_output.mp4")))

    selected_input = Path("/videos/new_clip.mkv")
    monkeypatch.setattr(
        "vr_sbs_converter.gui.main_window.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(selected_input), ""),
    )

    window._pick_input_file()

    assert window._input_edit.text() == str(selected_input)
    assert window._output_edit.text() == str(selected_input.with_name("new_clip_sbs.mp4"))


def test_main_window_builds_simple_config_from_default_preset() -> None:
    _ensure_app()
    window = MainWindow()
    window._input_edit.setText(str(Path("/tmp/in.mp4")))
    window._output_edit.setText(str(Path("/tmp/out.mp4")))

    config = window.build_config_from_ui()

    assert config.profile == "halo-safe"
    assert config.perf_mode == "quality"
    assert config.compat_profile == "strict"
    assert config.upscale is False
    assert config.target_height is None


def test_main_window_builds_simple_config_with_4k_upscale_toggle() -> None:
    _ensure_app()
    window = MainWindow()
    window._input_edit.setText(str(Path("/tmp/in.mp4")))
    window._output_edit.setText(str(Path("/tmp/out.mp4")))
    window._simple_panel.upscale_4k_checkbox.setChecked(True)

    config = window.build_config_from_ui()

    assert config.upscale is True
    assert config.target_height == 2160


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


def test_main_window_has_no_preview_ui_or_callbacks() -> None:
    _ensure_app()
    window = MainWindow()

    assert not hasattr(window, "_preview_label")
    assert not hasattr(window, "_current_preview_enabled")
    assert not hasattr(window, "_on_preview_frame")


def test_main_window_progress_status_includes_eta() -> None:
    _ensure_app()
    window = MainWindow()

    window._on_worker_started({"total_frames": 100})
    window._on_worker_progress(
        {"percent": 25.0, "frame_index": 25, "total_frames": 100, "eta_seconds": 90.0}
    )

    assert window._status_label.text() == "Converting frame 25/100 • ETA 01:30"


def test_main_window_progress_status_handles_unknown_eta() -> None:
    _ensure_app()
    window = MainWindow()

    window._on_worker_started({"total_frames": 0})
    window._on_worker_progress({"percent": 0.0, "frame_index": 0, "total_frames": 0})

    assert window._status_label.text() == "Converting frame 0/0 • ETA --:--"


def test_main_window_auto_fits_on_tab_switch(monkeypatch) -> None:
    _ensure_app()
    fit_calls: list[int] = []

    def _fake_auto_fit(self) -> None:
        fit_calls.append(self._tabs.currentIndex())

    monkeypatch.setattr(MainWindow, "_auto_fit_window_to_current_tab", _fake_auto_fit, raising=False)
    window = MainWindow()
    window._tabs.setCurrentIndex(1)
    window._tabs.setCurrentIndex(0)

    assert fit_calls
    assert 1 in fit_calls
    assert fit_calls[-1] == 0
