from pathlib import Path

import pytest

from vr_sbs_converter.config import ConversionConfig, parse_target_height


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1080p", 1080),
        ("4k", 2160),
        ("2160", 2160),
        ("3840x2160", 2160),
    ],
)
def test_parse_target_height_valid_tokens(token: str, expected: int) -> None:
    assert parse_target_height(token) == expected


def test_parse_target_height_invalid_token() -> None:
    with pytest.raises(ValueError):
        parse_target_height("big")


def test_conversion_config_rejects_invalid_crf(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ConversionConfig(
            input_path=tmp_path / "in.mp4",
            output_path=tmp_path / "out.mp4",
            crf=99,
        )


def test_conversion_config_rejects_invalid_depth_scale(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ConversionConfig(
            input_path=tmp_path / "in.mp4",
            output_path=tmp_path / "out.mp4",
            depth_process_scale=1.2,
        )


def test_conversion_config_rejects_invalid_compat_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ConversionConfig(
            input_path=tmp_path / "in.mp4",
            output_path=tmp_path / "out.mp4",
            compat_profile="broken",  # type: ignore[arg-type]
        )

