# Swimming Performance Tracking — Proof of Concept

Takes a recorded video of a 25 m freestyle swim session, tracks each swimmer,
detects hand-entry events, and calculates stroke rate, distance per cycle,
clean-swim time, average speed, and cycle-time variation. Outputs a CSV plus
an annotated video. See [CLAUDE.md](CLAUDE.md) for the full specification.

This is a proof of concept, not the production system.

## Status

Build order (CLAUDE.md section 17) in progress. Completed so far:
1. Project skeleton, config, git, MLflow wiring — **done**

Everything else (calibration, detection/tracking, pose, events, metrics,
rendering, manual review, web page) is not built yet.

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

Nothing is runnable end-to-end yet. Stages will be documented here as they
are completed, and wired into `run.py` — the single entry point the user
runs once the pipeline exists.

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
