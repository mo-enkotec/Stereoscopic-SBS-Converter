from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QWidget

from .mappers import SIMPLE_PRESETS


class SimplePanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout(self)

        self.preset_combo = QComboBox()
        for key, meta in SIMPLE_PRESETS.items():
            self.preset_combo.addItem(meta["label"], userData=key)

        self.compat_strict_checkbox = QCheckBox("Use strict playback compatibility")
        self.compat_strict_checkbox.setChecked(True)

        layout.addRow("Overall profile", self.preset_combo)
        layout.addRow("", self.compat_strict_checkbox)

    def get_state(self) -> dict[str, object]:
        return {
            "preset_key": self.preset_combo.currentData(),
            "compat_profile": "strict" if self.compat_strict_checkbox.isChecked() else "off",
        }
