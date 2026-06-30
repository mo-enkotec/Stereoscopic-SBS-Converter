from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QMainWindow,
    QProgressBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vr_sbs_converter.config import ConversionConfig

from .advanced_panel import AdvancedPanel
from .mappers import build_advanced_config, build_simple_config
from .simple_panel import SimplePanel
from .worker import ConversionWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VR SBS Converter")
        self.resize(1080, 760)

        self._input_edit = QLineEdit()
        self._output_edit = QLineEdit()
        self._input_browse_button = QPushButton("Browse…")
        self._output_browse_button = QPushButton("Browse…")
        self._input_browse_button.clicked.connect(self._pick_input_file)
        self._output_browse_button.clicked.connect(self._pick_output_file)

        self._start_button = QPushButton("Convert")
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._start_button.clicked.connect(self._start_conversion)
        self._cancel_button.clicked.connect(self._cancel_conversion)

        self._status_label = QLabel("Ready")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)

        self._simple_panel = SimplePanel()
        self._advanced_panel = AdvancedPanel()
        self._tabs = QTabWidget()
        self._tabs.addTab(self._simple_panel, "Simple")
        self._tabs.addTab(self._advanced_panel, "Advanced")

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setPlaceholderText("Runtime status and warnings will appear here.")

        self._thread: QThread | None = None
        self._worker: ConversionWorker | None = None

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(self._build_path_row("Input video", self._input_edit, self._input_browse_button))
        layout.addLayout(self._build_path_row("Output video", self._output_edit, self._output_browse_button))
        layout.addWidget(self._tabs)
        layout.addWidget(self._log_output)
        controls = QHBoxLayout()
        controls.addWidget(self._start_button)
        controls.addWidget(self._cancel_button)
        layout.addLayout(controls)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._status_label)
        self.setCentralWidget(root)

    @staticmethod
    def _build_path_row(label_text: str, edit: QLineEdit, browse_button: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        row.addWidget(edit, 1)
        row.addWidget(browse_button)
        return row

    def _pick_input_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select input video")
        if file_path:
            self._input_edit.setText(file_path)
            input_path = Path(file_path)
            stem = input_path.stem
            self._output_edit.setText(str(input_path.with_name(f"{stem}_sbs.mp4")))

    def _pick_output_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Select output path", filter="Video files (*.mp4)")
        if file_path:
            self._output_edit.setText(file_path)

    def build_config_from_ui(self) -> ConversionConfig:
        input_path = Path(self._input_edit.text().strip()).expanduser()
        output_path = Path(self._output_edit.text().strip()).expanduser()
        if self._tabs.currentIndex() == 0:
            state = self._simple_panel.get_state()
            config = build_simple_config(
                input_path=input_path,
                output_path=output_path,
                preset_key=str(state["preset_key"]),
            )
            config.compat_profile = str(state["compat_profile"])
            return config

        advanced_state = self._advanced_panel.get_state()
        return build_advanced_config(
            input_path=input_path,
            output_path=output_path,
            options=advanced_state,
        )

    def _start_conversion(self) -> None:
        if self._thread is not None:
            self._append_status("A conversion is already running.")
            return

        input_text = self._input_edit.text().strip()
        output_text = self._output_edit.text().strip()
        if not input_text or not output_text:
            self._append_status("Input and output paths are required.")
            self._status_label.setText("Missing input/output path")
            return
        if not Path(input_text).expanduser().exists():
            self._append_status("Input video does not exist.")
            self._status_label.setText("Invalid input path")
            return

        config = self.build_config_from_ui()
        self._worker = ConversionWorker(config=config)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_worker_started)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.status.connect(self._append_status)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.canceled.connect(self._on_worker_canceled)

        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.canceled.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)

        self._start_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Starting conversion...")
        self._thread.start()

    def _cancel_conversion(self) -> None:
        if self._worker is None:
            return
        self._append_status("Cancel requested.")
        self._status_label.setText("Cancel requested...")
        self._worker.cancel()

    def _on_worker_started(self, payload: dict[str, object]) -> None:
        total_frames = payload.get("total_frames", 0)
        self._append_status(f"Started conversion ({total_frames} frames).")
        self._status_label.setText("Converting...")

    def _on_worker_progress(self, payload: dict[str, object]) -> None:
        percent = int(float(payload.get("percent", 0.0)))
        self._progress_bar.setValue(max(0, min(100, percent)))
        frame_index = int(payload.get("frame_index", 0))
        total_frames = int(payload.get("total_frames", 0))
        eta_raw = payload.get("eta_seconds")
        eta_seconds = float(eta_raw) if isinstance(eta_raw, (int, float)) else None
        eta_text = self._format_eta(eta_seconds)
        self._status_label.setText(f"Converting frame {frame_index}/{total_frames} • ETA {eta_text}")

    def _on_worker_finished(self, payload: dict[str, object]) -> None:
        output_path = payload.get("output_path", "")
        self._status_label.setText("Completed")
        self._progress_bar.setValue(100)
        self._append_status(f"Completed: {output_path}")

    def _on_worker_failed(self, message: str) -> None:
        self._status_label.setText("Failed")
        self._append_status(f"Failed: {message}")

    def _on_worker_canceled(self) -> None:
        self._status_label.setText("Canceled")
        self._append_status("Conversion canceled.")

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._start_button.setEnabled(True)
        self._cancel_button.setEnabled(False)

    def _append_status(self, message: str) -> None:
        self._log_output.append(message)

    @staticmethod
    def _format_eta(remaining_seconds: float | None) -> str:
        if remaining_seconds is None:
            return "--:--"
        remaining = max(0, int(remaining_seconds))
        hours, remainder = divmod(remaining, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{minutes:02}:{seconds:02}"
