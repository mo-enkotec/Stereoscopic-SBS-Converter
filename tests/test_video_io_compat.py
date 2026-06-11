from pathlib import Path

from vr_sbs_converter.config import ConversionConfig
from vr_sbs_converter.video_io import build_frame_writer_command


def _cmd_to_string(command: list[str]) -> str:
    return " ".join(command)


def test_strict_compat_profile_adds_player_friendly_flags() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        compat_profile="strict",
        codec="libx264",
        overwrite=True,
    )
    command = build_frame_writer_command(
        output_path=Path("/tmp/out.mp4"),
        width=3840,
        height=1080,
        fps=30.0,
        config=config,
        codec_override="libx264",
    )
    cmd_text = _cmd_to_string(command)
    assert "-pix_fmt yuv420p" in cmd_text
    assert "-profile:v high" in cmd_text
    assert "-movflags +faststart" in cmd_text
    assert "-colorspace bt709" in cmd_text
    assert "-color_trc bt709" in cmd_text
    assert "-color_primaries bt709" in cmd_text


def test_compat_profile_off_keeps_old_encode_behavior() -> None:
    config = ConversionConfig(
        input_path=Path("/tmp/in.mp4"),
        output_path=Path("/tmp/out.mp4"),
        compat_profile="off",
        codec="libx264",
        overwrite=True,
    )
    command = build_frame_writer_command(
        output_path=Path("/tmp/out.mp4"),
        width=3840,
        height=1080,
        fps=30.0,
        config=config,
        codec_override="libx264",
    )
    cmd_text = _cmd_to_string(command)
    assert "-profile:v high" not in cmd_text
    assert "-movflags +faststart" not in cmd_text
