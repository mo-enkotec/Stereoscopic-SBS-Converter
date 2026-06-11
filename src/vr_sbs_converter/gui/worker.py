from __future__ import annotations

from threading import Event
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.pipeline import ConversionCallbacks, ConversionCancelledError, run_conversion


class ConversionWorker(QObject):
    started = Signal(dict)
    progress = Signal(dict)
    status = Signal(str)
    preview_frame = Signal(object)
    finished = Signal(dict)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, config: ConversionConfig, preview_enabled: bool) -> None:
        super().__init__()
        self._config = config
        self._preview_enabled = preview_enabled
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        callbacks = ConversionCallbacks(
            on_start=self._emit_start,
            on_progress=self._emit_progress,
            on_status=self._emit_status,
            on_complete=self._emit_finished,
            on_frame_preview=self._emit_preview,
            should_cancel=self._cancel_event.is_set,
            preview_enabled=self._preview_enabled,
            preview_every_n=5,
        )
        try:
            run_conversion(self._config, callbacks=callbacks)
        except ConversionCancelledError:
            self.canceled.emit()
        except RuntimeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - defensive guard
            self.failed.emit(str(exc))

    def _emit_start(self, payload: dict[str, Any]) -> None:
        self.started.emit(payload)

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        self.progress.emit(payload)

    def _emit_status(self, message: str) -> None:
        self.status.emit(message)

    def _emit_finished(self, payload: dict[str, Any]) -> None:
        self.finished.emit(payload)

    def _emit_preview(self, frame: np.ndarray) -> None:
        self.preview_frame.emit(frame)
