from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from tqdm import tqdm

from .compatibility import evaluate_player_compatibility, probe_output_video_stream
from .config import ConversionConfig
from .depth import create_depth_estimator
from .ffmpeg_utils import FFmpegError, ensure_ffmpeg_installed, probe_video
from .pipeline_parallel import run_parallel_conversion_configured
from .stereo import compose_sbs, synthesize_stereo_views
from .stereo_torch import select_stereo_synthesis_backend
from .upscaling import compute_target_dimensions, create_default_upscaler
from .video_io import (
    VideoIOError,
    close_reader,
    close_writer,
    ffmpeg_nvenc_usable,
    ffmpeg_supports_encoder,
    mux_audio_track,
    open_frame_reader,
    open_frame_writer,
    read_raw_frame,
    write_raw_frame,
)


@dataclass(slots=True)
class RuntimePlan:
    depth_process_scale: float
    use_fp16: bool
    preferred_encoder: str


class ConversionCancelledError(RuntimeError):
    """Raised when conversion is cancelled by the user."""


@dataclass(slots=True)
class ConversionCallbacks:
    on_start: Callable[[dict[str, Any]], None] | None = None
    on_progress: Callable[[dict[str, Any]], None] | None = None
    on_frame_preview: Callable[[np.ndarray], None] | None = None
    on_status: Callable[[str], None] | None = None
    on_complete: Callable[[dict[str, Any]], None] | None = None
    should_cancel: Callable[[], bool] | None = None
    preview_enabled: bool = False
    preview_every_n: int = 10


def resolve_runtime_plan(config: ConversionConfig) -> RuntimePlan:
    depth_scale = config.depth_process_scale if config.depth_process_scale is not None else 1.0
    use_fp16 = config.device == "cuda" and config.perf_mode in {"gpu-balanced", "max-speed"}

    if config.encoder != "auto":
        preferred_encoder = config.encoder
    elif config.device == "cuda" and config.perf_mode in {"gpu-balanced", "max-speed"}:
        preferred_encoder = "h264_nvenc"
    else:
        preferred_encoder = config.codec

    return RuntimePlan(
        depth_process_scale=depth_scale,
        use_fp16=use_fp16,
        preferred_encoder=preferred_encoder,
    )


@contextmanager
def _working_directory(config: ConversionConfig) -> Iterator[Path]:
    if config.temp_dir is not None:
        config.temp_dir.mkdir(parents=True, exist_ok=True)
        yield config.temp_dir
        return

    if config.keep_temp:
        tmp_path = Path(tempfile.mkdtemp(prefix="vr_sbs_"))
        yield tmp_path
        return

    with tempfile.TemporaryDirectory(prefix="vr_sbs_") as tmp_dir:
        yield Path(tmp_dir)


def _resolve_processing_dimensions(
    source_width: int,
    source_height: int,
    config: ConversionConfig,
) -> tuple[int, int]:
    if not config.upscale:
        return source_width, source_height

    if config.target_height is None:
        raise ValueError("Upscaling is enabled but no target height was provided.")
    return compute_target_dimensions(source_width, source_height, config.target_height)


def _prepare_sbs_dimensions(width: int, height: int, mode: str) -> tuple[int, int]:
    if mode == "full":
        return width * 2, height
    if mode == "half":
        return width, height
    raise ValueError(f"Unsupported SBS mode: {mode}")


def run_conversion(
    config: ConversionConfig,
    callbacks: ConversionCallbacks | None = None,
    *,
    use_parallel: bool = True,
) -> None:
    ensure_ffmpeg_installed()
    runtime_plan = resolve_runtime_plan(config)
    metadata = probe_video(config.input_path)
    process_width, process_height = _resolve_processing_dimensions(
        metadata.width, metadata.height, config
    )
    output_width, output_height = _prepare_sbs_dimensions(
        process_width, process_height, config.sbs_mode
    )

    upscaler = create_default_upscaler() if config.upscale else None
    depth_estimator = create_depth_estimator(
        config.depth_backend,
        config.device,
        edge_protect_strength=config.edge_protect_strength or 0.75,
        depth_process_scale=runtime_plan.depth_process_scale,
        use_fp16=runtime_plan.use_fp16,
    )

    with _working_directory(config) as work_dir:
        silent_output = work_dir / "sbs_silent.mp4"
        reader = open_frame_reader(config.input_path)
        if callbacks and callbacks.on_start:
            callbacks.on_start(
                {
                    "input_path": str(config.input_path),
                    "output_path": str(config.output_path),
                    "total_frames": metadata.total_frames,
                    "fps": metadata.fps,
                    "width": metadata.width,
                    "height": metadata.height,
                }
            )
        writer_codec = runtime_plan.preferred_encoder
        if writer_codec == "h264_nvenc":
            if not ffmpeg_supports_encoder("h264_nvenc") or not ffmpeg_nvenc_usable():
                writer_codec = config.codec
                message = "h264_nvenc unavailable at runtime; falling back to CPU encoder."
                print(message)
                if callbacks and callbacks.on_status:
                    callbacks.on_status(message)

        writer = open_frame_writer(
            output_path=silent_output,
            width=output_width,
            height=output_height,
            fps=metadata.fps,
            config=config,
            codec_override=writer_codec,
        )

        progress = tqdm(
            total=metadata.total_frames if metadata.total_frames > 0 else None,
            unit="frame",
            desc="Converting",
            leave=True,
        )
        decode_time = 0.0
        depth_time = 0.0
        stereo_time = 0.0
        encode_time = 0.0
        frames_processed = 0
        conversion_failed = False
        processing_elapsed = 0.0
        timing_lock = Lock()

        def _emit_preview(frame_payload: np.ndarray) -> None:
            if (
                callbacks
                and callbacks.preview_enabled
                and callbacks.on_frame_preview
                and frames_processed % max(1, callbacks.preview_every_n) == 0
            ):
                callbacks.on_frame_preview(frame_payload)

        def _run_sequential_frames() -> None:
            nonlocal decode_time, depth_time, stereo_time, encode_time, frames_processed
            while True:
                if callbacks and callbacks.should_cancel and callbacks.should_cancel():
                    raise ConversionCancelledError("Conversion cancelled by user.")
                decode_started = perf_counter()
                frame = read_raw_frame(reader, metadata.width, metadata.height)
                with timing_lock:
                    decode_time += perf_counter() - decode_started
                if frame is None:
                    break
                if upscaler is not None:
                    frame = upscaler.upscale(frame, process_width, process_height)

                depth_started = perf_counter()
                depth = depth_estimator.estimate(frame)
                with timing_lock:
                    depth_time += perf_counter() - depth_started

                stereo_started = perf_counter()
                left_eye, right_eye = synthesize_stereo_views(
                    frame_bgr=frame,
                    depth=depth,
                    stereo_strength=config.stereo_strength,
                    max_disparity_px=config.max_disparity_px,
                )
                sbs_frame = compose_sbs(left_eye, right_eye, config.sbs_mode)
                with timing_lock:
                    stereo_time += perf_counter() - stereo_started

                encode_started = perf_counter()
                write_raw_frame(writer, sbs_frame)
                with timing_lock:
                    encode_time += perf_counter() - encode_started
                frames_processed += 1
                progress.update(1)
                if callbacks and callbacks.on_progress:
                    percent = (
                        (frames_processed / metadata.total_frames) * 100.0
                        if metadata.total_frames > 0
                        else 0.0
                    )
                    callbacks.on_progress(
                        {
                            "frame_index": frames_processed,
                            "total_frames": metadata.total_frames,
                            "percent": percent,
                            "stage": "converting",
                        }
                    )
                _emit_preview(sbs_frame)

        def _run_parallel_frames() -> None:
            nonlocal decode_time, depth_time, stereo_time, encode_time, frames_processed

            stereo_backend = select_stereo_synthesis_backend(config.device)

            def _read_frame() -> np.ndarray | None:
                nonlocal decode_time
                decode_started = perf_counter()
                frame_payload = read_raw_frame(reader, metadata.width, metadata.height)
                with timing_lock:
                    decode_time += perf_counter() - decode_started
                if frame_payload is not None and upscaler is not None:
                    frame_payload = upscaler.upscale(frame_payload, process_width, process_height)
                return frame_payload

            def _estimate_depth(frame_payload: np.ndarray) -> np.ndarray:
                nonlocal depth_time
                depth_started = perf_counter()
                depth_payload = depth_estimator.estimate(frame_payload)
                with timing_lock:
                    depth_time += perf_counter() - depth_started
                return depth_payload

            def _synthesize_stereo(frame_payload: np.ndarray, depth_payload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                nonlocal stereo_time
                stereo_started = perf_counter()
                stereo_payload = stereo_backend.synthesize(
                    frame_payload,
                    depth_payload,
                    config.stereo_strength,
                    config.max_disparity_px,
                )
                with timing_lock:
                    stereo_time += perf_counter() - stereo_started
                return stereo_payload

            def _compose_sbs(stereo_payload: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
                nonlocal stereo_time
                compose_started = perf_counter()
                left_eye, right_eye = stereo_payload
                sbs_frame_payload = compose_sbs(left_eye, right_eye, config.sbs_mode)
                with timing_lock:
                    stereo_time += perf_counter() - compose_started
                return sbs_frame_payload

            def _write_frame(_frame_index: int, sbs_frame_payload: np.ndarray) -> None:
                nonlocal encode_time, frames_processed
                encode_started = perf_counter()
                write_raw_frame(writer, sbs_frame_payload)
                with timing_lock:
                    encode_time += perf_counter() - encode_started
                frames_processed += 1
                progress.update(1)
                _emit_preview(sbs_frame_payload)

            parallel_callbacks = None
            if callbacks is not None:
                parallel_callbacks = ConversionCallbacks(
                    on_progress=callbacks.on_progress,
                    should_cancel=callbacks.should_cancel,
                )

            parallel_result = run_parallel_conversion_configured(
                read_frame=_read_frame,
                estimate_depth=_estimate_depth,
                synthesize_stereo=_synthesize_stereo,
                compose_sbs=_compose_sbs,
                write_frame=_write_frame,
                callbacks=parallel_callbacks,
                total_frames=metadata.total_frames,
            )
            frames_processed = max(
                frames_processed,
                int(parallel_result.get("frames_written", frames_processed)),
            )

        processing_started = perf_counter()
        try:
            if use_parallel:
                _run_parallel_frames()
            else:
                _run_sequential_frames()
        except KeyboardInterrupt as exc:
            conversion_failed = True
            raise ConversionCancelledError("Conversion interrupted by user.") from exc
        except (BrokenPipeError, VideoIOError) as exc:
            conversion_failed = True
            raise RuntimeError(f"Frame pipeline failed: {exc}") from exc
        except Exception:
            conversion_failed = True
            raise
        finally:
            processing_elapsed = perf_counter() - processing_started
            progress.close()
            try:
                close_reader(reader)
            except (FFmpegError, BrokenPipeError):
                if not conversion_failed:
                    raise
            try:
                close_writer(writer)
            except (FFmpegError, BrokenPipeError):
                if not conversion_failed:
                    raise
            if conversion_failed and not config.keep_temp and silent_output.exists():
                silent_output.unlink()

        if metadata.has_audio:
            mux_audio_track(
                source_video=config.input_path,
                silent_video=silent_output,
                destination=config.output_path,
                overwrite=config.overwrite,
                audio_fallback=config.audio_fallback,
            )
        else:
            if config.output_path.exists():
                if not config.overwrite:
                    raise FileExistsError(
                        f"Output file already exists: {config.output_path}. Use --overwrite."
                    )
                config.output_path.unlink()
            shutil.move(str(silent_output), str(config.output_path))

        if not config.keep_temp and config.temp_dir is not None:
            temp_silent = config.temp_dir / "sbs_silent.mp4"
            if temp_silent.exists():
                temp_silent.unlink()

        if config.compat_profile == "strict":
            stream_info = probe_output_video_stream(config.output_path)
            compatibility_warnings = evaluate_player_compatibility(stream_info)
            for warning in compatibility_warnings:
                print(f"Compatibility warning: {warning}")
                if callbacks and callbacks.on_status:
                    callbacks.on_status(f"Compatibility warning: {warning}")

        effective_fps = 0.0
        if frames_processed > 0:
            if use_parallel:
                fps = frames_processed / max(1e-6, processing_elapsed)
            else:
                fps = frames_processed / max(1e-6, decode_time + depth_time + stereo_time + encode_time)
            effective_fps = fps
            summary_message = (
                "Runtime summary: "
                f"profile={config.profile}, perf_mode={config.perf_mode}, "
                f"encoder={writer_codec}, depth_scale={runtime_plan.depth_process_scale:.2f}, "
                f"frames={frames_processed}, effective_fps={fps:.2f}, "
                f"decode_ms={decode_time * 1000 / frames_processed:.2f}, "
                f"depth_ms={depth_time * 1000 / frames_processed:.2f}, "
                f"stereo_ms={stereo_time * 1000 / frames_processed:.2f}, "
                f"encode_ms={encode_time * 1000 / frames_processed:.2f}"
            )
            print(summary_message)
            if callbacks and callbacks.on_status:
                callbacks.on_status(summary_message)
        if callbacks and callbacks.on_complete:
            callbacks.on_complete(
                {
                    "frames_processed": frames_processed,
                    "effective_fps": effective_fps,
                    "encoder": writer_codec,
                    "output_path": str(config.output_path),
                }
            )


def dry_run_example_frame(
    frame_bgr: np.ndarray,
    config: ConversionConfig,
) -> np.ndarray:
    process_width, process_height = frame_bgr.shape[1], frame_bgr.shape[0]
    upscaler = create_default_upscaler() if config.upscale else None
    if upscaler is not None:
        if config.target_height is None:
            raise ValueError("Upscale enabled but target height missing.")
        process_width, process_height = compute_target_dimensions(
            frame_bgr.shape[1], frame_bgr.shape[0], config.target_height
        )
        frame_bgr = upscaler.upscale(frame_bgr, process_width, process_height)

    runtime_plan = resolve_runtime_plan(config)
    depth_estimator = create_depth_estimator(
        "luma",
        config.device,
        edge_protect_strength=config.edge_protect_strength or 0.75,
        depth_process_scale=runtime_plan.depth_process_scale,
        use_fp16=runtime_plan.use_fp16,
    )
    depth = depth_estimator.estimate(frame_bgr)
    left_eye, right_eye = synthesize_stereo_views(
        frame_bgr,
        depth,
        config.stereo_strength,
        max_disparity_px=config.max_disparity_px,
    )
    return compose_sbs(left_eye, right_eye, config.sbs_mode)
