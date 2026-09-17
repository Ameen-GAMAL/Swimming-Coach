# Swimming Performance Tracking — Proof of Concept

Takes a recorded video of a 25 m freestyle swim session, tracks each swimmer,
detects hand-entry events, and calculates stroke rate, distance per cycle,
clean-swim time, average speed, and cycle-time variation. Outputs a CSV plus
an annotated video. See [CLAUDE.md](CLAUDE.md) for the full specification —
read it before changing anything under "Fixed decisions" (section 3).

This is a proof of concept, not the production system.

**Requires an NVIDIA GPU (CUDA).** Every stage runs on `device=0` and there
is no CPU fallback — this will not run on a machine without one.

## Getting the code and data running (for a new teammate)

```
git clone <this-repo-url>
cd swimming-poc
```

Then:
1. Follow **Setup** below to create the venv and install dependencies.
2. Get videos into `data/raw/` — **this folder is empty in git on purpose**
   (`.gitignore` excludes it; raw video isn't checked in). The clips this
   project has been built and tested against are the University of Malta
   Swim2 dataset's `Freestyle/Above/` folder — see
   [data/candidates.md](data/candidates.md) for the exact 12 filenames
   approved so far and where they come from. Ask whoever set up this
   project for access to that dataset, then either copy the approved files
   into `data/raw/` or, if you're on the same machine/drive, hard-link them
   (no extra disk space):
   ```
   New-Item -ItemType HardLink -Path data\raw\<file>.MP4 -Target <path-to-dataset>\<file>.MP4
   ```
   Any other freestyle, side-view swimming video will also work with the
   pipeline — the dataset above is just what's been validated so far.
3. `data/calibration/*.json` and `*_preview.png` for a couple of already-
   approved videos **are** checked in, so you can run detection/pose/etc.
   against those specific videos without recalibrating — but you still need
   the matching video file in `data/raw/` for any stage to run, since
   calibration alone isn't the video.
4. Run stages in the order listed under **Running** below.

## Status

Build order (CLAUDE.md section 17) in progress. Completed so far:
1. Project skeleton, config, git, MLflow wiring — **done**
2. Video shortlist (`data/candidates.md`), sourced from the local
   University of Malta Swim2 dataset rather than the web — **done**. Found
   the camera pans/tracks the swimmer rather than staying fixed; recorded
   an approved amendment in CLAUDE.md section 3 to compensate for this in
   `detect_track.py` (not built yet).
3. Calibration tool (`src/calibrate.py`) — **done, fully automatic** (no
   clicking — rebuilt this way after the manual version's length-scale
   input proved unreliable with no wall ever visible in this footage; see
   CLAUDE.md section 3 amendments). Just run it on a video.
4. Detection and tracking (`src/detect_track.py`) — **done**, including
   camera-motion compensation and lane-based track ID merging. Verified
   end-to-end on real footage with a real (automatic) calibration.
5. Pose (`src/pose.py`) — **done**. Measured wrist visibility on real
   footage: **80.7%**, above CLAUDE.md's 70% gate (section 8.2), so the
   pretrained pose model is good enough to continue without fine-tuning.

Everything else (events, metrics, rendering, manual review, web page) is
not built yet.

## Project layout

```
config.yaml            Every adjustable number lives here
data/
  candidates.md        Approved video shortlist (what's in data/raw/, and why)
  raw/                 Video files go here (gitignored — see above)
  calibration/         One JSON + preview PNG per calibrated video
src/
  calibrate.py         Automatic calibration (pool/lane/scale)
  detect_track.py      YOLO11 detection + ByteTrack + camera-motion compensation
  pose.py              YOLO11-pose on tracked swimmers, wrist visibility gate
  profiling.py         MLflow + timing helpers shared by the other stages
cache/                 Intermediate results per video/model/threshold (gitignored, auto-rebuilds)
```

See CLAUDE.md section 4 for the complete, fixed project structure — don't
add files/folders outside it without checking with the team first.

## Requirements

- Windows with an NVIDIA GPU (CUDA) — see the warning at the top
- Python 3.12 specifically (the venv command below needs the `py` launcher
  to have it installed; if `py -0p` doesn't list a 3.12 entry, install it
  from python.org first)
- Git

## Setup

```
py -3.12 -m venv .venv --system-site-packages
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`--system-site-packages` reuses the machine's existing CUDA-enabled PyTorch
install instead of re-downloading it — if your machine doesn't already have
a working CUDA PyTorch install globally, drop that flag and expect `pip
install` to pull one instead (larger download, ~2.5GB). Verify the GPU is
visible:

```
python -c "import torch; print(torch.cuda.is_available())"
```

### Troubleshooting first run

- **`ModuleNotFoundError: No module named 'lap'`** — Ultralytics' ByteTrack
  code needs this package but doesn't declare it, and its own auto-updater
  can install it into the wrong Python environment. Fix directly:
  `.venv\Scripts\python.exe -m pip install lap`
- **First run of any stage downloads a model** (`yolo11m.pt`,
  `yolo11m-pose.pt`, ~40MB each, from Ultralytics' GitHub releases) —
  needs internet access once; cached locally after that.
- **Run stages as a module** (`-m src.calibrate`, not
  `src\calibrate.py` directly) — the plain script path won't resolve the
  `src.profiling` import.

## Running

Nothing is wired into `run.py` yet — until the full pipeline exists, run
each finished stage directly.

### Calibration

Fully automatic — supply a video, nothing else:

```
.venv\Scripts\python.exe -m src.calibrate data\raw\<video-file>.MP4
```

What it does, with no interaction required:
1. Finds the pool by water colour, and finds the one lane rope this method
   can reliably detect (the boundary between open water and the rope
   floats).
2. Builds the 4 calibration corners from that rope; the lane's other edge
   and its water line/boundary are derived from it too (see CLAUDE.md
   section 7 for why a second rope isn't independently detected).
3. Estimates the length-axis scale from the swimmer's own body size via
   pose, since no wall or other known-distance landmark is ever visible in
   this footage — this is an approximate, provisional estimate, not a
   measurement (see "Known limitations" below).

Takes a few minutes (it runs detection and pose across 250 sampled frames
to get a robust body-size estimate). Output: a JSON file and a
`_preview.png` in `data/calibration/`, named after the video. Still worth
glancing at the preview — the grid should roughly follow the pool/rope,
even though the exact metre labels are approximate.

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

7. **No frame in this footage ever shows a pool wall**, confirmed across a
   full clip and 4 other candidate clips (Phase 3). The manual click-based
   calibration this originally called for couldn't work reliably as a
   result — any length-span number a person typed was a guess, amplified
   ~10x by extrapolation across the full 25m pool. Replaced with fully
   automatic calibration (see "Calibration" above), whose length scale is
   an approximate anthropometric estimate. Measured accuracy varies by
   clip: 27.7m vs the true 25m for one clip (11% over), 46.2m raw (clamped
   to 40m, 60% over) for another. Every calibration and everything computed
   from it carries a `scale_method: approximate_anthropometric` flag —
   distance and speed numbers from this POC should be read as
   directionally useful, not as precise measurements, until real footage
   with a visible wall (or another true landmark) is available.

## Deferred (explicitly out of scope, not built)

Nothing deferred yet — this list grows only when a specific future need
becomes obvious during the build (CLAUDE.md section 2).
