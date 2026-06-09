from __future__ import annotations

from contextlib import contextmanager
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm

from .config import ConversionConfig
from .depth import create_depth_estimator
from .ffmpeg_utils import ensure_ffmpeg_installed, probe_video
from .stereo import compose_sbs, synthesize_stereo_views
from .upscaling import compute_target_dimensions, create_default_upscaler
from .video_io import (
    VideoIOError,
    close_reader,
    close_writer,
    mux_audio_track,
    open_frame_reader,
    open_frame_writer,
    read_raw_frame,
    write_raw_frame,
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


def run_conversion(config: ConversionConfig) -> None:
    ensure_ffmpeg_installed()
    metadata = probe_video(config.input_path)
    process_width, process_height = _resolve_processing_dimensions(
        metadata.width, metadata.height, config
    )
    output_width, output_height = _prepare_sbs_dimensions(
        process_width, process_height, config.sbs_mode
    )

    upscaler = create_default_upscaler() if config.upscale else None
    depth_estimator = create_depth_estimator(config.depth_backend, config.device)

    with _working_directory(config) as work_dir:
        silent_output = work_dir / "sbs_silent.mp4"
        reader = open_frame_reader(config.input_path)
        writer = open_frame_writer(
            output_path=silent_output,
            width=output_width,
            height=output_height,
            fps=metadata.fps,
            config=config,
        )

        progress = tqdm(
            total=metadata.total_frames if metadata.total_frames > 0 else None,
            unit="frame",
            desc="Converting",
            leave=True,
        )
        conversion_failed = False
        try:
            while True:
                frame = read_raw_frame(reader, metadata.width, metadata.height)
                if frame is None:
                    break
                if upscaler is not None:
                    frame = upscaler.upscale(frame, process_width, process_height)

                depth = depth_estimator.estimate(frame)
                left_eye, right_eye = synthesize_stereo_views(
                    frame_bgr=frame,
                    depth=depth,
                    stereo_strength=config.stereo_strength,
                )
                sbs_frame = compose_sbs(left_eye, right_eye, config.sbs_mode)
                write_raw_frame(writer, sbs_frame)
                progress.update(1)
        except KeyboardInterrupt as exc:
            conversion_failed = True
            raise RuntimeError("Conversion interrupted by user.") from exc
        except (BrokenPipeError, VideoIOError) as exc:
            conversion_failed = True
            raise RuntimeError(f"Frame pipeline failed: {exc}") from exc
        finally:
            progress.close()
            close_reader(reader)
            close_writer(writer)
            if conversion_failed and not config.keep_temp and silent_output.exists():
                silent_output.unlink()

        if metadata.has_audio:
            mux_audio_track(
                source_video=config.input_path,
                silent_video=silent_output,
                destination=config.output_path,
                overwrite=config.overwrite,
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

    depth_estimator = create_depth_estimator("luma", config.device)
    depth = depth_estimator.estimate(frame_bgr)
    left_eye, right_eye = synthesize_stereo_views(frame_bgr, depth, config.stereo_strength)
    return compose_sbs(left_eye, right_eye, config.sbs_mode)
