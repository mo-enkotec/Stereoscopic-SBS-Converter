from pathlib import Path

import pytest

from vr_sbs_converter.cli import build_config, build_parser, infer_default_output


def test_infer_default_output() -> None:
    assert infer_default_output(Path("/tmp/demo.mp4")) == Path("/tmp/demo.sbs.mp4")


def test_build_config_from_args(tmp_path: Path) -> None:
    input_file = tmp_path / "input.mp4"
    input_file.write_bytes(b"fake")
    parser = build_parser()
    args = parser.parse_args(
        [
            str(input_file),
            "--upscale",
            "--target",
            "2160p",
            "--stereo-strength",
            "1.0",
            "--overwrite",
        ]
    )

    config = build_config(args)
    assert config.input_path == input_file.resolve()
    assert config.output_path.name == "input.sbs.mp4"
    assert config.upscale is True
    assert config.target_height == 2160


def test_build_config_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    input_file = tmp_path / "clip.mp4"
    input_file.write_bytes(b"fake")
    output_file = tmp_path / "clip.sbs.mp4"
    output_file.write_bytes(b"old")

    parser = build_parser()
    args = parser.parse_args([str(input_file)])

    with pytest.raises(FileExistsError):
        build_config(args)


def test_build_config_supports_profile_and_perf_mode(tmp_path: Path) -> None:
    input_file = tmp_path / "scene.mp4"
    input_file.write_bytes(b"fake")
    parser = build_parser()
    args = parser.parse_args(
        [
            str(input_file),
            "--profile",
            "halo-safe",
            "--perf-mode",
            "gpu-balanced",
            "--encoder",
            "auto",
            "--max-disparity-px",
            "16",
            "--depth-process-scale",
            "0.75",
            "--edge-protect-strength",
            "0.85",
            "--overwrite",
        ]
    )
    config = build_config(args)
    assert config.profile == "halo-safe"
    assert config.perf_mode == "gpu-balanced"
    assert config.encoder == "auto"
    assert config.max_disparity_px == 16
    assert config.depth_process_scale == 0.75
    assert config.edge_protect_strength == 0.85
