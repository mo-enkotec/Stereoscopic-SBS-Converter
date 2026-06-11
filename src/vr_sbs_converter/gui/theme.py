from __future__ import annotations


def build_dark_stylesheet() -> str:
    return """
QWidget {
    background-color: #121212;
    color: #EDEDED;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2A2A2A;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #AFAFAF;
}
QPushButton {
    background-color: #1E1E1E;
    border: 1px solid #2D2D2D;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #2A2A2A;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
    background-color: #1A1A1A;
    border: 1px solid #2E2E2E;
    border-radius: 4px;
    padding: 4px;
}
QProgressBar {
    border: 1px solid #2E2E2E;
    border-radius: 4px;
    text-align: center;
    background-color: #1A1A1A;
}
QProgressBar::chunk {
    background-color: #3D8BFF;
}
QTabWidget::pane {
    border: 1px solid #2A2A2A;
}
QTabBar::tab {
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-bottom: none;
    padding: 8px 12px;
}
QTabBar::tab:selected {
    background: #262626;
}
"""

