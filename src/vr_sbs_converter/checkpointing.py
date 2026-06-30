from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from .config import ConversionConfig

ManifestStatus = Literal["running", "canceled", "complete"]


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    input_path: str
    input_size_bytes: int
    input_mtime_ns: int
    config_fingerprint: str


@dataclass(slots=True)
class CheckpointManifest:
    checkpoint_id: str
    identity: CheckpointIdentity
    status: ManifestStatus
    next_frame_index: int
    segments: list[str]


def _stable_config_payload(config: ConversionConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "sbs_mode": config.sbs_mode,
        "upscale": config.upscale,
        "target_height": config.target_height,
        "codec": config.codec,
        "preset": config.preset,
        "crf": config.crf,
        "device": config.device,
        "depth_backend": config.depth_backend,
        "profile": config.profile,
        "perf_mode": config.perf_mode,
        "encoder": config.encoder,
        "compat_profile": config.compat_profile,
        "audio_fallback": config.audio_fallback,
        "max_disparity_px": config.max_disparity_px,
        "depth_process_scale": config.depth_process_scale,
        "depth_process_scale_overridden": config.depth_process_scale_overridden,
        "edge_protect_strength": config.edge_protect_strength,
        "stereo_strength": config.stereo_strength,
    }
    return payload


def _config_fingerprint(config: ConversionConfig) -> str:
    payload = _stable_config_payload(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_checkpoint_identity(config: ConversionConfig) -> CheckpointIdentity:
    resolved_input = config.input_path.expanduser().resolve()
    stats = resolved_input.stat()
    return CheckpointIdentity(
        input_path=str(resolved_input),
        input_size_bytes=int(stats.st_size),
        input_mtime_ns=int(stats.st_mtime_ns),
        config_fingerprint=_config_fingerprint(config),
    )


def checkpoint_id_from_identity(identity: CheckpointIdentity) -> str:
    payload = {
        "input_path": identity.input_path,
        "input_size_bytes": identity.input_size_bytes,
        "input_mtime_ns": identity.input_mtime_ns,
        "config_fingerprint": identity.config_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def checkpoint_directory(progress_root: Path, checkpoint_id: str) -> Path:
    return progress_root / checkpoint_id


def manifest_path(progress_root: Path, checkpoint_id: str) -> Path:
    return checkpoint_directory(progress_root, checkpoint_id) / "manifest.json"


def create_manifest(
    checkpoint_id: str,
    identity: CheckpointIdentity,
    *,
    status: ManifestStatus = "running",
    next_frame_index: int = 0,
    segments: list[str] | None = None,
) -> CheckpointManifest:
    return CheckpointManifest(
        checkpoint_id=checkpoint_id,
        identity=identity,
        status=status,
        next_frame_index=next_frame_index,
        segments=list(segments or []),
    )


def _manifest_to_json(manifest: CheckpointManifest) -> dict[str, object]:
    return {
        "checkpoint_id": manifest.checkpoint_id,
        "status": manifest.status,
        "next_frame_index": manifest.next_frame_index,
        "segments": manifest.segments,
        "identity": {
            "input_path": manifest.identity.input_path,
            "input_size_bytes": manifest.identity.input_size_bytes,
            "input_mtime_ns": manifest.identity.input_mtime_ns,
            "config_fingerprint": manifest.identity.config_fingerprint,
        },
    }


def _manifest_from_json(payload: dict[str, object]) -> CheckpointManifest:
    identity_payload = payload["identity"]
    if not isinstance(identity_payload, dict):
        raise ValueError("Invalid manifest identity payload.")
    identity = CheckpointIdentity(
        input_path=str(identity_payload["input_path"]),
        input_size_bytes=int(identity_payload["input_size_bytes"]),
        input_mtime_ns=int(identity_payload["input_mtime_ns"]),
        config_fingerprint=str(identity_payload["config_fingerprint"]),
    )
    return CheckpointManifest(
        checkpoint_id=str(payload["checkpoint_id"]),
        identity=identity,
        status=str(payload["status"]),  # type: ignore[arg-type]
        next_frame_index=int(payload["next_frame_index"]),
        segments=[str(item) for item in payload.get("segments", [])],
    )


def save_manifest(progress_root: Path, manifest: CheckpointManifest) -> None:
    checkpoint_dir = checkpoint_directory(progress_root, manifest.checkpoint_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_path = manifest_path(progress_root, manifest.checkpoint_id)
    payload = _manifest_to_json(manifest)
    with NamedTemporaryFile("w", encoding="utf-8", dir=checkpoint_dir, delete=False) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        temp_path = Path(handle.name)
    temp_path.replace(target_path)


def load_manifest(progress_root: Path, checkpoint_id: str) -> CheckpointManifest | None:
    target_path = manifest_path(progress_root, checkpoint_id)
    if not target_path.exists():
        return None
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid checkpoint manifest payload.")
    return _manifest_from_json(payload)


def is_manifest_compatible(manifest: CheckpointManifest, identity: CheckpointIdentity) -> bool:
    return manifest.identity == identity


def segment_filename(segment_index: int) -> str:
    return f"segment_{segment_index:06d}.mp4"


def segment_path(progress_root: Path, checkpoint_id: str, segment_index: int) -> Path:
    return checkpoint_directory(progress_root, checkpoint_id) / segment_filename(segment_index)


def append_segment(
    progress_root: Path,
    manifest: CheckpointManifest,
    segment_name: str,
    *,
    frames_written: int,
    status: ManifestStatus,
) -> CheckpointManifest:
    if frames_written < 0:
        raise ValueError("frames_written must be non-negative.")
    manifest.segments.append(segment_name)
    manifest.next_frame_index += frames_written
    manifest.status = status
    save_manifest(progress_root, manifest)
    return manifest
