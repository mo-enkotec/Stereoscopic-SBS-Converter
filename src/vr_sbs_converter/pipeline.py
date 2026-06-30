from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from threading import Lock
from time import perf_counter
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from tqdm import tqdm

from .checkpointing import (
    append_segment,
    build_checkpoint_identity,
    checkpoint_directory,
    checkpoint_id_from_identity,
    create_manifest,
    load_manifest,
    save_manifest,
    segment_filename,
    segment_path,
)
from .compatibility import evaluate_player_compatibility, probe_output_video_stream
from .config import ConversionConfig
from .depth import create_depth_estimator
from .ffmpeg_utils import FFmpegError, ensure_ffmpeg_installed, probe_video
from .pipeline_parallel import DepthFramePayload, ParallelQueueConfig, run_parallel_conversion_configured
from .stereo import compose_sbs, synthesize_stereo_views
from .stereo_torch import select_stereo_synthesis_backend, synthesize_stereo_views_torch_batch
from .upscaling import compute_target_dimensions, create_default_upscaler
from .video_io import (
    VideoIOError,
    close_reader,
    close_writer,
    ffmpeg_nvenc_usable,
    ffmpeg_supports_encoder,
    concat_video_segments,
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


def _is_cuda_runtime_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_runtime_plan(config: ConversionConfig) -> RuntimePlan:
    depth_scale = config.depth_process_scale if config.depth_process_scale is not None else 1.0
    use_cuda_runtime = config.device == "cuda" or (
        config.device == "auto" and _is_cuda_runtime_available()
    )
    use_fp16 = use_cuda_runtime and config.perf_mode in {"gpu-balanced", "max-speed"}

    if config.encoder != "auto":
        preferred_encoder = config.encoder
    elif use_cuda_runtime and config.perf_mode in {"gpu-balanced", "max-speed"}:
        preferred_encoder = "h264_nvenc"
    else:
        preferred_encoder = config.codec

    return RuntimePlan(
        depth_process_scale=depth_scale,
        use_fp16=use_fp16,
        preferred_encoder=preferred_encoder,
    )


def resolve_parallel_queue_config(config: ConversionConfig) -> ParallelQueueConfig:
    queue_size = config.parallel_queue_size
    if queue_size is None:
        queue_size = {
            "quality": 4,
            "gpu-balanced": 8,
            "max-speed": 12,
        }[config.perf_mode]

    return ParallelQueueConfig(
        decode_queue_size=queue_size,
        depth_queue_size=queue_size,
        stereo_queue_size=queue_size,
        encode_queue_size=queue_size,
    )


def resolve_gpu_batch_size(config: ConversionConfig) -> int:
    if config.gpu_batch_size is not None:
        return config.gpu_batch_size

    return {
        "quality": 1,
        "gpu-balanced": 2,
        "max-speed": 4,
    }[config.perf_mode]


def _cap_gpu_depth_queue(queue_config: ParallelQueueConfig) -> ParallelQueueConfig:
    capped_depth = min(queue_config.depth_queue_size, 2)
    if capped_depth == queue_config.depth_queue_size:
        return queue_config
    return ParallelQueueConfig(
        decode_queue_size=queue_config.decode_queue_size,
        depth_queue_size=capped_depth,
        stereo_queue_size=queue_config.stereo_queue_size,
        encode_queue_size=queue_config.encode_queue_size,
    )


def _cap_cuda_batch_size_for_resolution(batch_size: int, width: int, height: int) -> int:
    pixels = width * height
    if pixels >= (1920 * 1080):
        return min(batch_size, 1)
    if pixels >= (1280 * 720):
        return min(batch_size, 2)
    return batch_size


def _should_enable_gpu_stereo(
    *,
    config: ConversionConfig,
    stereo_backend_name: str,
    effective_depth_batch_size: int,
    width: int,
    height: int,
) -> bool:
    if stereo_backend_name != "torch-cuda":
        return False
    if (
        config.perf_mode == "quality"
        and effective_depth_batch_size <= 1
        and (width * height) >= (3840 * 2160)
    ):
        return False
    return True


def _progress_root() -> Path:
    return Path(__file__).resolve().parents[2] / "progress"


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

    checkpoint_root = _progress_root()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    identity = build_checkpoint_identity(config)
    checkpoint_id = checkpoint_id_from_identity(identity)
    checkpoint_dir = checkpoint_directory(checkpoint_root, checkpoint_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(checkpoint_root, checkpoint_id)
    if manifest is None or manifest.status == "complete":
        manifest = create_manifest(checkpoint_id=checkpoint_id, identity=identity)
        save_manifest(checkpoint_root, manifest)

    resume_start_frame = max(0, int(manifest.next_frame_index))
    if callbacks and callbacks.on_status:
        if resume_start_frame > 0:
            callbacks.on_status(
                f"Resuming from progress/{checkpoint_id} at frame {resume_start_frame}."
            )
        else:
            callbacks.on_status(f"Starting fresh checkpoint in progress/{checkpoint_id}.")

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
        segment_index = len(manifest.segments)
        current_segment_name = segment_filename(segment_index)
        current_segment_path = segment_path(checkpoint_root, checkpoint_id, segment_index)
        if current_segment_path.exists():
            current_segment_path.unlink()

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
                    "resume_start_frame": resume_start_frame,
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
            output_path=current_segment_path,
            width=output_width,
            height=output_height,
            fps=metadata.fps,
            config=config,
            codec_override=writer_codec,
        )

        progress = tqdm(
            total=metadata.total_frames if metadata.total_frames > 0 else None,
            initial=min(resume_start_frame, metadata.total_frames) if metadata.total_frames > 0 else 0,
            unit="frame",
            desc="Converting",
            leave=True,
        )
        decode_time = 0.0
        depth_time = 0.0
        stereo_time = 0.0
        encode_time = 0.0
        frames_processed = resume_start_frame
        frames_processed_this_run = 0
        processing_elapsed = 0.0
        timing_lock = Lock()
        pending_error: Exception | None = None
        conversion_canceled = False

        def _emit_progress(frame_index: int) -> None:
            if callbacks and callbacks.on_progress:
                percent = (
                    (frame_index / metadata.total_frames) * 100.0
                    if metadata.total_frames > 0
                    else 0.0
                )
                callbacks.on_progress(
                    {
                        "frame_index": frame_index,
                        "total_frames": metadata.total_frames,
                        "percent": percent,
                        "stage": "converting",
                    }
                )

        def _emit_preview(frame_payload: np.ndarray) -> None:
            if (
                callbacks
                and callbacks.preview_enabled
                and callbacks.on_frame_preview
                and frames_processed % max(1, callbacks.preview_every_n) == 0
            ):
                callbacks.on_frame_preview(frame_payload)

        def _skip_frames_to_resume(target_frame_index: int) -> None:
            nonlocal decode_time
            for _ in range(target_frame_index):
                if callbacks and callbacks.should_cancel and callbacks.should_cancel():
                    raise ConversionCancelledError("Conversion cancelled by user.")
                decode_started = perf_counter()
                frame_payload = read_raw_frame(reader, metadata.width, metadata.height)
                with timing_lock:
                    decode_time += perf_counter() - decode_started
                if frame_payload is None:
                    raise RuntimeError(
                        "Cannot resume from checkpoint; source ended before resume frame index."
                    )

        def _run_sequential_frames() -> None:
            nonlocal decode_time, depth_time, stereo_time, encode_time, frames_processed, frames_processed_this_run
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
                frames_processed_this_run += 1
                progress.update(1)
                _emit_progress(frames_processed)
                _emit_preview(sbs_frame)

        def _run_parallel_frames() -> None:
            nonlocal decode_time, depth_time, stereo_time, encode_time, frames_processed, frames_processed_this_run

            stereo_backend = select_stereo_synthesis_backend(config.device)
            stereo_backend_name = getattr(stereo_backend, "name", "cpu")
            torch_module = None
            if stereo_backend_name == "torch-cuda":
                try:
                    torch_module = importlib.import_module("torch")
                except Exception:
                    torch_module = None
            queue_config = resolve_parallel_queue_config(config)
            depth_batch_size = resolve_gpu_batch_size(config)
            stereo_batch_size = depth_batch_size if stereo_backend_name == "torch-cuda" else 1
            auto_bypass_reason = None
            if stereo_backend_name == "torch-cuda":
                if torch_module is None:
                    use_gpu_stereo = False
                    auto_bypass_reason = "torch_unavailable"
                else:
                    queue_config = _cap_gpu_depth_queue(queue_config)
                    depth_batch_size = min(depth_batch_size, queue_config.depth_queue_size)
                    depth_batch_size = _cap_cuda_batch_size_for_resolution(
                        depth_batch_size,
                        process_width,
                        process_height,
                    )
                    stereo_batch_size = min(stereo_batch_size, queue_config.depth_queue_size, depth_batch_size)
                    use_gpu_stereo = _should_enable_gpu_stereo(
                        config=config,
                        stereo_backend_name=stereo_backend_name,
                        effective_depth_batch_size=depth_batch_size,
                        width=process_width,
                        height=process_height,
                    )
                    if not use_gpu_stereo:
                        auto_bypass_reason = "quality_4k_low_batch"
                if not use_gpu_stereo:
                    stereo_backend = select_stereo_synthesis_backend("cpu")
                    stereo_backend_name = "cpu"
                    torch_module = None
                    stereo_batch_size = 1
            if callbacks and callbacks.on_status:
                bypass_suffix = (
                    f", auto_bypass={auto_bypass_reason}" if auto_bypass_reason is not None else ""
                )
                callbacks.on_status(
                    "Parallel runtime: "
                    f"queue={queue_config.decode_queue_size}, "
                    f"depth_batch={depth_batch_size}, stereo_batch={stereo_batch_size}, "
                    f"stereo_backend={stereo_backend_name}{bypass_suffix}"
                )

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

            def _estimate_depth_batch(frame_payloads: list[np.ndarray]) -> list[Any]:
                nonlocal depth_time
                depth_started = perf_counter()
                estimate_batch = getattr(depth_estimator, "estimate_batch", None)
                if callable(estimate_batch):
                    batch_results = estimate_batch(frame_payloads)
                else:
                    batch_results = [depth_estimator.estimate(item) for item in frame_payloads]
                if stereo_backend_name == "torch-cuda" and torch_module is not None:
                    batch_results = [
                        DepthFramePayload(
                            frame_payload=torch_module.from_numpy(frame_payload).to(
                                device="cuda",
                                dtype=torch_module.float32,
                            ),
                            depth_payload=torch_module.from_numpy(
                                np.clip(np.asarray(depth_payload, dtype=np.float32), 0.0, 1.0)
                            ).to(
                                device="cuda",
                                dtype=torch_module.float32,
                            ),
                        )
                        for frame_payload, depth_payload in zip(frame_payloads, batch_results, strict=True)
                    ]
                with timing_lock:
                    depth_time += perf_counter() - depth_started
                return batch_results

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

            def _synthesize_stereo_batch(
                frame_payloads: list[np.ndarray],
                depth_payloads: list[np.ndarray],
            ) -> list[tuple[np.ndarray, np.ndarray]]:
                nonlocal stereo_time
                stereo_started = perf_counter()
                if stereo_backend_name == "torch-cuda":
                    stereo_payloads = synthesize_stereo_views_torch_batch(
                        frame_payloads,
                        depth_payloads,
                        config.stereo_strength,
                        config.max_disparity_px,
                        stream_overlap=config.gpu_stream_overlap,
                    )
                else:
                    stereo_payloads = [
                        stereo_backend.synthesize(frame, depth, config.stereo_strength, config.max_disparity_px)
                        for frame, depth in zip(frame_payloads, depth_payloads, strict=True)
                    ]
                with timing_lock:
                    stereo_time += perf_counter() - stereo_started
                return stereo_payloads

            def _compose_sbs(stereo_payload: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
                nonlocal stereo_time
                compose_started = perf_counter()
                left_eye, right_eye = stereo_payload
                sbs_frame_payload = compose_sbs(left_eye, right_eye, config.sbs_mode)
                with timing_lock:
                    stereo_time += perf_counter() - compose_started
                return sbs_frame_payload

            def _write_frame(_frame_index: int, sbs_frame_payload: np.ndarray) -> None:
                nonlocal encode_time, frames_processed, frames_processed_this_run
                encode_started = perf_counter()
                write_raw_frame(writer, sbs_frame_payload)
                with timing_lock:
                    encode_time += perf_counter() - encode_started
                frames_processed += 1
                frames_processed_this_run += 1
                progress.update(1)
                _emit_preview(sbs_frame_payload)

            def _on_parallel_progress(payload: dict[str, Any]) -> None:
                if callbacks is None or callbacks.on_progress is None:
                    return
                current_written = int(payload.get("frame_index", 0))
                absolute_frame = resume_start_frame + current_written
                percent = (
                    (absolute_frame / metadata.total_frames) * 100.0
                    if metadata.total_frames > 0
                    else 0.0
                )
                callbacks.on_progress(
                    {
                        "frame_index": absolute_frame,
                        "total_frames": metadata.total_frames,
                        "percent": percent,
                        "stage": str(payload.get("stage", "converting")),
                    }
                )

            parallel_callbacks = None
            if callbacks is not None:
                parallel_callbacks = ConversionCallbacks(
                    on_progress=_on_parallel_progress,
                    should_cancel=callbacks.should_cancel,
                )

            parallel_result = run_parallel_conversion_configured(
                read_frame=_read_frame,
                estimate_depth=_estimate_depth,
                estimate_depth_batch=_estimate_depth_batch,
                depth_batch_size=depth_batch_size,
                synthesize_stereo=_synthesize_stereo,
                synthesize_stereo_batch=_synthesize_stereo_batch if stereo_backend_name == "torch-cuda" else None,
                stereo_batch_size=stereo_batch_size,
                compose_sbs=_compose_sbs,
                write_frame=_write_frame,
                queue_config=queue_config,
                callbacks=parallel_callbacks,
                total_frames=metadata.total_frames,
            )
            current_frames = int(parallel_result.get("frames_written", frames_processed_this_run))
            frames_processed_this_run = max(frames_processed_this_run, current_frames)
            frames_processed = max(frames_processed, resume_start_frame + current_frames)

        processing_started = perf_counter()
        try:
            if resume_start_frame > 0:
                _skip_frames_to_resume(resume_start_frame)
            if use_parallel:
                _run_parallel_frames()
            else:
                _run_sequential_frames()
        except ConversionCancelledError as exc:
            conversion_canceled = True
            pending_error = exc
        except KeyboardInterrupt as exc:
            conversion_canceled = True
            pending_error = ConversionCancelledError("Conversion interrupted by user.")
        except (BrokenPipeError, VideoIOError) as exc:
            pending_error = RuntimeError(f"Frame pipeline failed: {exc}")
        except Exception as exc:
            pending_error = exc
        finally:
            processing_elapsed = perf_counter() - processing_started
            progress.close()
            try:
                close_reader(reader)
            except (FFmpegError, BrokenPipeError):
                if pending_error is None:
                    raise
            try:
                close_writer(writer)
            except (FFmpegError, BrokenPipeError):
                if pending_error is None:
                    raise

        if frames_processed_this_run > 0:
            append_segment(
                checkpoint_root,
                manifest,
                current_segment_name,
                frames_written=frames_processed_this_run,
                status="canceled" if pending_error is not None else "running",
            )
        else:
            if pending_error is not None and current_segment_path.exists():
                current_segment_path.unlink()
            manifest.status = "canceled" if pending_error is not None else manifest.status
            save_manifest(checkpoint_root, manifest)

        if pending_error is not None:
            if callbacks and callbacks.on_status:
                callbacks.on_status(
                    f"Progress saved to progress/{checkpoint_id} at frame {manifest.next_frame_index}."
                )
            raise pending_error

        segment_paths = [checkpoint_dir / item for item in manifest.segments]
        if segment_paths:
            concat_video_segments(
                segments=segment_paths,
                destination=silent_output,
                overwrite=True,
            )
        elif current_segment_path.exists():
            shutil.move(str(current_segment_path), str(silent_output))
        else:
            silent_output.touch()
        manifest.status = "complete"
        save_manifest(checkpoint_root, manifest)

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
        if frames_processed_this_run > 0:
            if use_parallel:
                fps = frames_processed_this_run / max(1e-6, processing_elapsed)
            else:
                fps = frames_processed_this_run / max(
                    1e-6, decode_time + depth_time + stereo_time + encode_time
                )
            effective_fps = fps
            summary_message = (
                "Runtime summary: "
                f"profile={config.profile}, perf_mode={config.perf_mode}, "
                f"encoder={writer_codec}, depth_scale={runtime_plan.depth_process_scale:.2f}, "
                f"frames={frames_processed_this_run}, effective_fps={fps:.2f}, "
                f"decode_ms={decode_time * 1000 / frames_processed_this_run:.2f}, "
                f"depth_ms={depth_time * 1000 / frames_processed_this_run:.2f}, "
                f"stereo_ms={stereo_time * 1000 / frames_processed_this_run:.2f}, "
                f"encode_ms={encode_time * 1000 / frames_processed_this_run:.2f}"
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
