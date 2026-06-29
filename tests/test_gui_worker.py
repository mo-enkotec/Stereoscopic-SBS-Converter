from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.gui.worker import ConversionWorker
from vr_sbs_converter.pipeline import ConversionCancelledError


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_worker_emits_callback_signals(monkeypatch) -> None:
    _ensure_app()
    captured: dict[str, list] = {
        "start": [],
        "progress": [],
        "status": [],
        "finished": [],
        "failed": [],
    }

    def fake_run_conversion(_config, callbacks=None) -> None:
        assert callbacks is not None
        assert callbacks.on_frame_preview is None
        assert callbacks.preview_enabled is False
        if callbacks.on_start:
            callbacks.on_start({"total_frames": 5})
        if callbacks.on_progress:
            callbacks.on_progress({"frame_index": 1, "total_frames": 5, "percent": 20.0, "stage": "converting"})
        if callbacks.on_status:
            callbacks.on_status("runtime summary")
        if callbacks.on_complete:
            callbacks.on_complete({"output_path": "/tmp/out.mp4"})

    monkeypatch.setattr("vr_sbs_converter.gui.worker.run_conversion", fake_run_conversion)

    worker = ConversionWorker(ConversionConfig(input_path=Path("/tmp/in.mp4"), output_path=Path("/tmp/out.mp4")))
    worker.started.connect(lambda payload: captured["start"].append(payload))
    worker.progress.connect(lambda payload: captured["progress"].append(payload))
    worker.status.connect(lambda message: captured["status"].append(message))
    worker.finished.connect(lambda payload: captured["finished"].append(payload))
    worker.failed.connect(lambda message: captured["failed"].append(message))

    worker.run()
    assert len(captured["start"]) == 1
    assert len(captured["progress"]) == 1
    assert captured["status"] == ["runtime summary"]
    assert len(captured["finished"]) == 1
    assert captured["failed"] == []


def test_worker_emits_failed_signal_on_exception(monkeypatch) -> None:
    _ensure_app()
    captured: list[str] = []

    def fake_run_conversion(_config, callbacks=None) -> None:
        _ = callbacks
        raise RuntimeError("conversion failed")

    monkeypatch.setattr("vr_sbs_converter.gui.worker.run_conversion", fake_run_conversion)

    worker = ConversionWorker(ConversionConfig(input_path=Path("/tmp/in.mp4"), output_path=Path("/tmp/out.mp4")))
    worker.failed.connect(captured.append)

    worker.run()

    assert captured == ["conversion failed"]


def test_worker_emits_canceled_signal_on_cancel_exception(monkeypatch) -> None:
    _ensure_app()
    captured: list[str] = []

    def fake_run_conversion(_config, callbacks=None) -> None:
        _ = callbacks
        raise ConversionCancelledError("Conversion cancelled by user.")

    monkeypatch.setattr("vr_sbs_converter.gui.worker.run_conversion", fake_run_conversion)

    worker = ConversionWorker(ConversionConfig(input_path=Path("/tmp/in.mp4"), output_path=Path("/tmp/out.mp4")))
    worker.canceled.connect(lambda: captured.append("cancelled"))

    worker.run()

    assert captured == ["cancelled"]
