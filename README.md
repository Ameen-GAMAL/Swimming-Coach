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
4. Detection and tracking (`src/detect_track.py`) — **done**, including
   camera-motion compensation and lane-based track ID merging. Verified to
   run correctly end-to-end on real footage; full accuracy validation is
   still pending a real calibration (see "Known limitations" below for two
   real findings from this stage).
5. Pose (`src/pose.py`) — **done**. Measured wrist visibility on real
   footage: **80.7%**, above CLAUDE.md's 70% gate (section 8.2), so the
   pretrained pose model is good enough to continue without fine-tuning.

Everything else (events, metrics, rendering, manual review, web page) is
not built yet.

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

### Detection and tracking

Requires a calibration for the video first.

```
.venv\Scripts\python.exe -m src.detect_track data\raw\<video-file>.MP4
```

Runs YOLO11 person detection and ByteTrack over every frame, corrects for
the camera's own motion, discards detections outside the calibrated pool
area, assigns each track to a lane, and merges track IDs that reappear in
the same lane shortly after a gap. Result is cached to `cache/detect_track/`
keyed on video, model, and confidence threshold — rerunning with the same
settings reuses the cache instantly.

### Pose

Requires detection and tracking to have run for the video first (runs it
automatically via the cache if not).

```
.venv\Scripts\python.exe -m src.pose data\raw\<video-file>.MP4
```

Runs YOLO11-pose on a crop around each tracked swimmer, applies the same
camera-motion correction, and reports `quality_wrist_visible_pct` — the
percentage of processed frames with a wrist visible above
`pose.conf_threshold`. Below 70%, stop: the pretrained model isn't reliable
enough and needs a decision from the user (CLAUDE.md section 8.2's
fallback is annotating 300–500 frames and fine-tuning). Measured on real
footage so far: **80.7%**.

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
- `track_*` — ByteTrack tracker settings (added in Phase 4, see CLAUDE.md
  section 3 amendments)
- `pose_*` — pose stage
- `calib_*` — calibration values
- `event_*` — event-detection stage
- `quality_*` — quality-gate stage
- `time_*` — per-stage timing
- `result_*` — final measurements
- `speed_fps` — processing speed (frames per second), fixed name

## Known limitations

CLAUDE.md section 16's four limitations (frame rate, resolution, camera
angle, placeholder benchmarks) apply once footage exists for accuracy
measurement. Two more, found while building detection/tracking (Phase 4):

5. **The pretrained YOLO11 "person" class is weak on swimmers in this
   footage.** Measured median top-of-frame confidence across a full clip
   was only ~0.13 — a horizontal, partly-submerged swimmer viewed from this
   close, low camera angle looks quite different from COCO's mostly-upright
   training data. Mitigated by running detection at a low confidence
   threshold (letting ByteTrack's own two-stage matching do the filtering
   it's designed for) and by merging track IDs that reappear in the same
   lane shortly after a gap (CLAUDE.md section 8.1). This keeps the
   swimmer's identity stable, but real accuracy on the actual measurements
   (stroke rate, distance per cycle) is only established once tested
   against a real calibration and ground truth — an argument for a
   fine-tuned or swimming-specific detector in the production system, not
   an excuse to skip measuring here.

6. **Matching every frame directly against one fixed reference frame
   degrades over a clip.** With a panning camera (see amendment above),
   background overlap between a frame and the reference frame shrinks as
   the pan accumulates — inlier matches dropped from ~90 to single digits
   over a 45s clip when tried this way. Fixed by chaining frame-to-frame
   matches instead (each frame matched against the previous one), which
   keeps overlap consistently high regardless of total pan.

## Deferred (explicitly out of scope, not built)

Nothing deferred yet — this list grows only when a specific future need
becomes obvious during the build (CLAUDE.md section 2).
