from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)


class AdvancedPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QFormLayout(self)

        self.sbs_mode = QComboBox()
        self.sbs_mode.addItems(["full", "half"])

        self.upscale = QCheckBox("Enable upscaling")
        self.target_height = QSpinBox()
        self.target_height.setRange(240, 4320)
        self.target_height.setValue(2160)

        self.profile = QComboBox()
        self.profile.addItems(["halo-safe", "balanced", "fast"])
        self.perf_mode = QComboBox()
        self.perf_mode.addItems(["quality", "gpu-balanced", "max-speed"])

        self.encoder = QComboBox()
        self.encoder.addItems(["auto", "libx264", "h264_nvenc"])
        self.codec = QLineEdit("libx264")
        self.preset = QLineEdit("slow")
        self.crf = QSpinBox()
        self.crf.setRange(0, 51)
        self.crf.setValue(18)

        self.compat_profile = QComboBox()
        self.compat_profile.addItems(["strict", "off"])
        self.audio_fallback = QComboBox()
        self.audio_fallback.addItems(["copy-aac"])

        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        self.depth_backend = QComboBox()
        self.depth_backend.addItems(["auto", "midas", "luma"])

        self.max_disparity_px = QSpinBox()
        self.max_disparity_px.setRange(1, 128)
        self.max_disparity_px.setValue(12)

        self.depth_process_scale = QDoubleSpinBox()
        self.depth_process_scale.setRange(0.1, 1.0)
        self.depth_process_scale.setSingleStep(0.05)
        self.depth_process_scale.setValue(1.0)

        self.edge_protect_strength = QDoubleSpinBox()
        self.edge_protect_strength.setRange(0.0, 1.0)
        self.edge_protect_strength.setSingleStep(0.05)
        self.edge_protect_strength.setValue(0.9)

        self.stereo_strength = QDoubleSpinBox()
        self.stereo_strength.setRange(0.1, 3.0)
        self.stereo_strength.setSingleStep(0.05)
        self.stereo_strength.setValue(0.8)

        self.overwrite = QCheckBox("Overwrite output if exists")
        self.overwrite.setChecked(True)
        self.keep_temp = QCheckBox("Keep temporary files")
        self.temp_dir = QLineEdit()

        layout.addRow("SBS mode", self.sbs_mode)
        layout.addRow("", self.upscale)
        layout.addRow("Target height", self.target_height)
        layout.addRow("Profile", self.profile)
        layout.addRow("Performance mode", self.perf_mode)
        layout.addRow("Encoder", self.encoder)
        layout.addRow("Codec", self.codec)
        layout.addRow("Preset", self.preset)
        layout.addRow("CRF", self.crf)
        layout.addRow("Compatibility", self.compat_profile)
        layout.addRow("Audio fallback", self.audio_fallback)
        layout.addRow("Device", self.device)
        layout.addRow("Depth backend", self.depth_backend)
        layout.addRow("Max disparity px", self.max_disparity_px)
        layout.addRow("Depth scale", self.depth_process_scale)
        layout.addRow("Edge protect strength", self.edge_protect_strength)
        layout.addRow("Stereo strength", self.stereo_strength)
        layout.addRow("", self.overwrite)
        layout.addRow("", self.keep_temp)
        layout.addRow("Temp directory", self.temp_dir)

    def get_state(self) -> dict[str, object]:
        temp_dir_value = self.temp_dir.text().strip()
        return {
            "sbs_mode": self.sbs_mode.currentText(),
            "upscale": self.upscale.isChecked(),
            "target_height": self.target_height.value() if self.upscale.isChecked() else None,
            "profile": self.profile.currentText(),
            "perf_mode": self.perf_mode.currentText(),
            "encoder": self.encoder.currentText(),
            "codec": self.codec.text().strip() or "libx264",
            "preset": self.preset.text().strip() or "slow",
            "crf": self.crf.value(),
            "compat_profile": self.compat_profile.currentText(),
            "audio_fallback": self.audio_fallback.currentText(),
            "device": self.device.currentText(),
            "depth_backend": self.depth_backend.currentText(),
            "max_disparity_px": self.max_disparity_px.value(),
            "depth_process_scale": self.depth_process_scale.value(),
            "edge_protect_strength": self.edge_protect_strength.value(),
            "stereo_strength": self.stereo_strength.value(),
            "overwrite": self.overwrite.isChecked(),
            "keep_temp": self.keep_temp.isChecked(),
            "temp_dir": Path(temp_dir_value).expanduser() if temp_dir_value else None,
        }
