from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import ConversionConfig, parse_target_height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vr-sbs-convert",
        description="Convert a regular 2D video into stereoscopic side-by-side (SBS) VR video.",
    )
    parser.add_argument("input", help="Path to input 2D video file.")
    parser.add_argument("-o", "--output", help="Output path for converted SBS video.")
    parser.add_argument(
        "--sbs-mode",
        default="full",
        choices=["full", "half"],
        help="Output SBS format: 'full' keeps each eye at full source width, 'half' halves each eye width.",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Enable upscaling before stereo synthesis.",
    )
    parser.add_argument(
        "--target",
        default="2160p",
        help="Target resolution when --upscale is set (examples: 2160p, 4k, 3840x2160).",
    )
    parser.add_argument("--codec", default="libx264", help="Video codec for final encoding.")
    parser.add_argument("--preset", default="slow", help="ffmpeg encoding preset.")
    parser.add_argument("--crf", type=int, default=18, help="ffmpeg CRF value (0-51).")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device preference for depth estimation.",
    )
    parser.add_argument(
        "--depth-backend",
        choices=["auto", "midas", "luma"],
        default="auto",
        help="Depth backend. 'auto' prefers MiDaS when available.",
    )
    parser.add_argument(
        "--profile",
        choices=["halo-safe", "balanced", "fast"],
        default="halo-safe",
        help="Artifact/quality profile. 'halo-safe' prioritizes reduced edge pull artifacts.",
    )
    parser.add_argument(
        "--perf-mode",
        choices=["quality", "gpu-balanced", "max-speed"],
        default="quality",
        help="Performance mode. 'gpu-balanced' is recommended for GTX-class GPUs.",
    )
    parser.add_argument(
        "--encoder",
        choices=["auto", "libx264", "h264_nvenc"],
        default="auto",
        help="Encoder selection. 'auto' picks NVENC when available.",
    )
    parser.add_argument(
        "--max-disparity-px",
        type=int,
        help="Maximum horizontal disparity in pixels per eye. Lower values reduce halo risk.",
    )
    parser.add_argument(
        "--depth-process-scale",
        type=float,
        help="Scale factor for depth inference resolution (0-1]. Lower is faster.",
    )
    parser.add_argument(
        "--edge-protect-strength",
        type=float,
        help="Edge protection intensity for depth/stereo processing (0-1).",
    )
    parser.add_argument(
        "--stereo-strength",
        type=float,
        default=0.8,
        help="Stereo disparity strength. Typical range: 0.4-1.2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files created during conversion.",
    )
    parser.add_argument(
        "--temp-dir",
        help="Directory for intermediate files. Defaults to a generated temporary directory.",
    )
    return parser


def infer_default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.sbs.mp4")


def build_config(args: argparse.Namespace) -> ConversionConfig:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else infer_default_output(input_path)
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    target_height = parse_target_height(args.target) if args.upscale else None

    temp_dir = Path(args.temp_dir).expanduser().resolve() if args.temp_dir else None
    return ConversionConfig(
        input_path=input_path,
        output_path=output_path,
        sbs_mode=args.sbs_mode,
        upscale=args.upscale,
        target_height=target_height,
        codec=args.codec,
        preset=args.preset,
        crf=args.crf,
        device=args.device,
        depth_backend=args.depth_backend,
        profile=args.profile,
        perf_mode=args.perf_mode,
        encoder=args.encoder,
        max_disparity_px=args.max_disparity_px,
        depth_process_scale=args.depth_process_scale,
        edge_protect_strength=args.edge_protect_strength,
        stereo_strength=args.stereo_strength,
        overwrite=args.overwrite,
        keep_temp=args.keep_temp,
        temp_dir=temp_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = build_config(args)
        from .pipeline import run_conversion

        run_conversion(config)
        print(f"Done: {config.output_path}")
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
