from __future__ import annotations

from pathlib import Path

from vr_sbs_converter.checkpointing import (
    CheckpointIdentity,
    CheckpointManifest,
    append_segment,
    build_checkpoint_identity,
    checkpoint_directory,
    checkpoint_id_from_identity,
    load_manifest,
    manifest_path,
    save_manifest,
)
from vr_sbs_converter.config import ConversionConfig


def test_checkpoint_identity_uses_path_size_mtime_and_config(tmp_path: Path) -> None:
    input_video = tmp_path / "in.mp4"
    input_video.write_bytes(b"abc")
    output_video = tmp_path / "out.mp4"

    config = ConversionConfig(
        input_path=input_video,
        output_path=output_video,
        depth_backend="luma",
    )
    identity = build_checkpoint_identity(config)

    assert identity.input_path == str(input_video.resolve())
    assert identity.input_size_bytes == 3
    assert identity.input_mtime_ns > 0
    assert identity.config_fingerprint


def test_manifest_roundtrip_and_append_segment(tmp_path: Path) -> None:
    identity = CheckpointIdentity(
        input_path="/tmp/in.mp4",
        input_size_bytes=123,
        input_mtime_ns=456,
        config_fingerprint="cfg",
    )
    checkpoint_id = checkpoint_id_from_identity(identity)

    manifest = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        identity=identity,
        status="running",
        next_frame_index=42,
        segments=[],
    )
    save_manifest(tmp_path, manifest)
    loaded = load_manifest(tmp_path, checkpoint_id)

    assert loaded is not None
    assert loaded.checkpoint_id == checkpoint_id
    assert loaded.next_frame_index == 42
    assert loaded.status == "running"

    append_segment(tmp_path, loaded, "segment_000000.mp4", frames_written=10, status="canceled")
    reloaded = load_manifest(tmp_path, checkpoint_id)
    assert reloaded is not None
    assert reloaded.status == "canceled"
    assert reloaded.next_frame_index == 52
    assert reloaded.segments == ["segment_000000.mp4"]


def test_checkpoint_paths_are_stable(tmp_path: Path) -> None:
    identity = CheckpointIdentity(
        input_path="/tmp/in.mp4",
        input_size_bytes=1,
        input_mtime_ns=2,
        config_fingerprint="x",
    )
    checkpoint_id = checkpoint_id_from_identity(identity)
    directory = checkpoint_directory(tmp_path, checkpoint_id)
    path = manifest_path(tmp_path, checkpoint_id)

    assert directory == tmp_path / checkpoint_id
    assert path == directory / "manifest.json"
