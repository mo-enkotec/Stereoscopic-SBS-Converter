# VR SBS Converter (CLI)

Convert a regular 2D video into a stereoscopic side-by-side (SBS) video that can be played in VR video players.

## What it does

- Accepts common video formats through ffmpeg decode.
- Builds left/right eye views from RGB + depth.
- Exports SBS video (`full` or `half` SBS).
- Optional upscaling before stereo synthesis (for example to 4K).
- Preserves source audio when present.

## Requirements

1. Python 3.11+
2. ffmpeg + ffprobe available in `PATH`
3. Python dependencies:

```bash
pip install -r requirements.txt
```

### Optional: model-based depth (higher quality)

The converter defaults to `--depth-backend auto`, which attempts MiDaS first and falls back to luma depth if unavailable.

To enable MiDaS explicitly, install:

```bash
pip install torch transformers
```

## Usage

### Basic conversion (2D -> full SBS)

```bash
python main.py /path/to/input.mp4 -o /path/to/output_sbs.mp4 --overwrite
```

### Convert and upscale to 4K

```bash
python main.py /path/to/input.mp4 \
  -o /path/to/output_4k_sbs.mp4 \
  --upscale --target 4k \
  --depth-backend auto \
  --device auto \
  --preset slow --crf 18 \
  --overwrite
```

### Force CPU-safe mode

```bash
python main.py /path/to/input.mp4 \
  --depth-backend luma \
  --device cpu \
  --overwrite
```

## Key CLI options

- `--sbs-mode {full,half}`: full keeps each eye at full width; half packs each eye at half width.
- `--upscale --target <value>`: enable upscaling (`2160p`, `4k`, `3840x2160`, etc.).
- `--depth-backend {auto,midas,luma}`: depth estimator selection.
- `--stereo-strength <float>`: disparity intensity (recommended range `0.4` to `1.2`).
- `--codec`, `--preset`, `--crf`: ffmpeg encode controls.

## Run tests

```bash
pytest
```
