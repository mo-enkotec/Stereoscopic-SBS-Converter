from __future__ import annotations

from typing import Sequence

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .theme import build_dark_stylesheet


def launch_gui(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else [])
    app.setStyleSheet(build_dark_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()

