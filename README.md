# VR SBS Converter (CLI + GUI)

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

### Launch the desktop GUI

```bash
python main.py --gui
```

GUI highlights:

- Dark theme with **Simple** and **Advanced** tabs.
- File pickers for input/output.
- Live progress + runtime status log.
- Preset-driven simple mode (including one-click 4K upscaling) and full-control advanced mode.

### GUI modes

- **Simple**: choose one overall profile preset and a strict compatibility toggle.
- **Advanced**: expose all conversion controls (SBS mode, upscale target, profile/performance, depth/stereo knobs, encoder/compatibility/audio fallback, temp handling).

The window auto-fits to the currently selected tab content by default and re-fits when switching between Simple and Advanced.

### Basic conversion (2D -> full SBS)

```bash
python main.py /path/to/input.mp4 -o /path/to/output_sbs.mp4 --overwrite
```

### HereSphere / Quest compatibility-focused conversion (full-SBS retained)

```bash
python main.py /path/to/input.mp4 \
  -o /path/to/output_compat_fullsbs.mp4 \
  --sbs-mode full \
  --compat-profile strict \
  --audio-fallback copy-aac \
  --overwrite
```

### Halo-safe quality mode (recommended default)

```bash
python main.py /path/to/input.mp4 \
  -o /path/to/output_halo_safe.mp4 \
  --profile halo-safe \
  --perf-mode quality \
  --depth-backend auto \
  --encoder auto \
  --overwrite
```

### GPU-balanced mode (GTX 2070 Super)

```bash
python main.py /path/to/input.mp4 \
  -o /path/to/output_gpu_balanced.mp4 \
  --profile halo-safe \
  --perf-mode gpu-balanced \
  --device cuda \
  --depth-backend auto \
  --encoder auto \
  --overwrite
```

### Fast mode (speed-priority with more artifact risk)

```bash
python main.py /path/to/input.mp4 \
  -o /path/to/output_fast.mp4 \
  --profile fast \
  --perf-mode max-speed \
  --device cuda \
  --depth-backend auto \
  --encoder auto \
  --overwrite
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
- `--profile {halo-safe,balanced,fast}`: halo control and disparity defaults.
- `--perf-mode {quality,gpu-balanced,max-speed}`: quality/speed profile.
- `--encoder {auto,libx264,h264_nvenc}`: auto tries NVENC and falls back safely.
- `--compat-profile {strict,off}`: `strict` forces player-friendly MP4/H.264 flags (`yuv420p`, high profile, `+faststart`, BT.709 tags).
- `--audio-fallback {copy-aac}`: copy source audio first, retry mux with AAC only when needed.
- `--max-disparity-px <int>`: explicit disparity cap to limit edge pull artifacts.
- `--depth-process-scale <float>`: depth inference resolution scale; lower values increase speed.
- `--edge-protect-strength <float>`: depth edge-preservation intensity.
- `--stereo-strength <float>`: disparity intensity (recommended range `0.4` to `1.2`).
- `--parallel-queue-size <int>`: per-stage queue depth in parallel mode (higher can improve throughput but uses more memory).
- `--codec`, `--preset`, `--crf`: ffmpeg encode controls.

At the end of conversion, the CLI prints a runtime summary containing selected profile/perf mode, encoder path, effective FPS, and average stage timings.

ETA is computed in the conversion pipeline and emitted to the GUI progress callback as the canonical ETA source. During warm-up and short clips, ETA can still fluctuate, but GUI/runtime progress now follows the same estimator inputs.

During conversion, the runtime also emits a live `Function timing top-5` status line every 2 seconds (frame-processing path only: decode/upscale/depth/stereo/compose/encode). Rankings are based on **average milliseconds per function call**.

At completion, the runtime emits a `Function timing summary` line with the final top averages.
In strict compatibility mode, the converter may print compatibility warnings after encoding if stream properties are likely to fail in stricter players.

> Note: keeping full-SBS at very large dimensions (for example 7680x2160) can still exceed decoder limits on some devices even with compatible codec/pixel format settings.

## Cancel/resume progress behavior

- Canceled runs persist progress in `progress/` at the project root.
- Resume is automatic when the next run matches the same input identity (path + file size + modified timestamp) and conversion settings.
- Resume checkpoints are stored as segmented video parts plus a manifest; final output is assembled when conversion completes.
- If input identity or conversion settings do not match, conversion starts fresh.

## Run tests

```bash
pytest
```
