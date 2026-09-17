# Swimming Performance Tracking — Proof of Concept

Takes a recorded video of a 25 m freestyle swim session, tracks each swimmer,
detects hand-entry events, and calculates stroke rate, distance per cycle,
clean-swim time, average speed, and cycle-time variation. Outputs a CSV plus
an annotated video. See [CLAUDE.md](CLAUDE.md) for the full specification.

This is a proof of concept, not the production system.

## Status

Build order (CLAUDE.md section 17) in progress. Completed so far:
1. Project skeleton, config, git, MLflow wiring — **done**
2. Video shortlist (`data/candidates.md`), sourced from the local
   University of Malta Swim2 dataset rather than the web — **done**. Found
   the camera pans/tracks the swimmer rather than staying fixed; recorded
   an approved amendment in CLAUDE.md section 3 to compensate for this in
   `detect_track.py` (not built yet).
3. Calibration tool (`src/calibrate.py`) — **done**, needs to be run by a
   person (it's click-based) to actually calibrate a chosen video.

Everything else (detection/tracking, pose, events, metrics, rendering,
manual review, web page) is not built yet.

## Requirements

- Windows with an NVIDIA GPU (CUDA)
- Python 3.12
- Git

## Setup

```
py -3.12 -m venv .venv --system-site-packages
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`--system-site-packages` reuses the machine's existing CUDA-enabled PyTorch
install instead of re-downloading it. Verify the GPU is visible:

```
python -c "import torch; print(torch.cuda.is_available())"
```

## Running

Nothing is wired into `run.py` yet — until the full pipeline exists, run
each finished stage directly.

### Calibration

```
.venv\Scripts\python.exe src\calibrate.py data\raw\<video-file>.MP4
```

Opens one frame in a window. Click through three steps:

1. **4 pool corners** (near-left, near-right, far-right, far-left of the
   visible water surface). You'll be asked in the terminal how much of the
   pool length (in metres) those corners span — press enter to accept the
   default (the whole pool), or type e.g. `5,20` if only a shorter, known
   segment is visible. Then how many lanes wide the corners are (default 1).
   A grid is drawn back over the frame immediately — press `y` to accept it
   or `r` to redo the clicks if the grid doesn't follow the pool edges.
2. **Water line per lane** — 2 clicks per lane, near edge to far edge.
3. **Lane boundaries** — 2 clicks per boundary line (lanes + 1 of them).

`u` undoes the last click at any point, `q` cancels. Output: a JSON file
and a `_preview.png` in `data/calibration/`, named after the video — look
at the preview image and confirm the grid, water lines, and lane boundaries
actually follow the pool before trusting the calibration.

## MLflow

Local file store at `./mlruns`.

```
mlflow ui
```

Then open `http://localhost:5000`.

Run name format: `{video}__{what_changed}__{MMDD-HHMM}` (e.g.
`freestyle_lane3_25m__baseline__0918-1430`).

Parameter and metric name prefixes (see CLAUDE.md section 10). Adding a new
prefix requires recording it here first:

- `video_*` — source video properties
- `detect_*` — detection stage
- `pose_*` — pose stage
- `calib_*` — calibration values
- `event_*` — event-detection stage
- `quality_*` — quality-gate stage
- `time_*` — per-stage timing
- `result_*` — final measurements
- `speed_fps` — processing speed (frames per second), fixed name

## Known limitations

Documented once real footage has been run through the pipeline (CLAUDE.md
section 16): frame rate, resolution, camera angle, and the placeholder
benchmark ranges all cap the accuracy ceiling of this POC.

## Deferred (explicitly out of scope, not built)

Nothing deferred yet — this list grows only when a specific future need
becomes obvious during the build (CLAUDE.md section 2).
