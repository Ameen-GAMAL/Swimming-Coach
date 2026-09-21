# Swimming Performance Tracking — Proof of Concept

Takes a recorded video of a 25 m freestyle swim session, tracks each swimmer,
detects hand-entry events, and calculates stroke rate, distance per cycle,
clean-swim time, average speed, and cycle-time variation. Outputs a CSV plus
an annotated video. See [CLAUDE.md](CLAUDE.md) for the full specification —
read it before changing anything under "Fixed decisions" (section 3).

This is a proof of concept, not the production system.

**Requires an NVIDIA GPU (CUDA).** Every stage runs on `device=0` and there
is no CPU fallback — this will not run on a machine without one.

## Quickstart (under 5 minutes)

Gets the app installed and running, so you can click through the UI —
**not** a full analysis, which takes far longer (see the honest timings
below). Assumes Python 3.12 and a working CUDA install are already on the
machine; if `--system-site-packages` below can't find a CUDA PyTorch,
drop that flag and expect `pip install` to pull one instead (~2.5GB,
takes longer than 5 minutes on its own).

```powershell
git clone https://github.com/Ameen-GAMAL/Swimming-Coach.git
cd Swimming-Coach
py -3.12 -m venv .venv --system-site-packages
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860** — you'll see the app running with its four
tabs (Run Analysis / Results / Diagnostics / About). That's the 5-minute
part.

**To actually click "Run analysis" and get a result**, you need a video
in `data/raw/` first — the repo ships with no video data (see
**Getting the code and data running** just below) — and then realistic
time budget, not 5 minutes:

| Stage | Typical time (one ~45-55s clip) |
|---|---|
| Calibration (automatic, first run only) | ~1-2 min |
| Detection & tracking | ~5-8 min |
| Pose | ~30-90 min (depends heavily on how much of the swimmer the detector finds — see "Known limitations") |
| Events / Metrics | a few seconds |
| Rendering the annotated video | ~2-4 min |

This is a research pipeline running several ML models per frame on full
4K source video, not a production-tuned service — the times above are
measured, not estimated. First run of any stage also downloads its model
checkpoints (`yolo11x.pt` ~110MB for detection, `yolo11m-pose.pt` ~40MB
for pose) from Ultralytics' GitHub releases — needs internet access once,
cached locally after.

## Getting the code and data running (for a new teammate)

```
git clone https://github.com/Ameen-GAMAL/Swimming-Coach.git
cd Swimming-Coach
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
6. Events (`src/events.py`) — **done**. Hand entry/exit detection with a
   state machine, plus the wrist-signal plot artifact.
7. Metrics and quality gate (`src/metrics.py`) — **done**. Turn/length
   detection, clean-zone filtering, the five client metrics, provisional
   benchmark comparison, and lap aggregation. Validated against a
   synthetic ideal-swimmer signal (every number came out exactly right);
   on real footage tested so far, correctly reports "nothing to report"
   rather than guessing — see "Known limitations" below for why.
8. Rendering (`src/render.py`) — **done**. Grid, boxes/IDs/lanes, wrist
   point, entry-flash markers, running counters, provisional watermark.
9. Manual review (`src/review.py`) — **done**, interactive by necessity
   (a person has to judge each detected event) so not fully testable by
   an agent; the non-interactive parts (clip extraction, cross-video
   ground-truth accumulation) are verified. Not one of CLAUDE.md's
   original 8 files — see section 13's amendment note.
10. Web page (`app.py`) — **done**, then rebuilt with a themed, tabbed
    layout (Run Analysis / Results / Diagnostics / About), a real
    per-stage progress checklist, styled provisional/rejected-result
    callouts, and a small accepted-metrics chart — see "Reliability pass"
    below.

**Reliability pass (post-Phase 11):** in response to real clips producing
empty reports, root-caused and fixed several issues:
- Replaced motion-based supplementary detection (investigated, found
  unreliable on this footage's close/wide-angle camera geometry, and
  disabled) with a colour-based signal instead — see CLAUDE.md section
  8.1's amendment.
- Fixed a real bug in `events.py`'s entry/exit state machine that silently
  discarded a track's first transition whenever it started already
  submerged (very likely on almost every track) — see section 8.3's
  amendment.
- Fixed the camera-motion background region being a fixed fraction of
  frame height, which worked for DIVE_P010 but captured almost no usable
  texture on DIVE_P023 (zero matched features in 63% of frames) — now
  sized per-clip from the calibration's own pool-top boundary. Measured
  effect on DIVE_P023: usable camera-motion frames went from 5% to 93%.
- Sped up `pose.py` by filtering out tracks too short for `events.py` to
  ever use, before paying for pose inference on them.

Remaining: accuracy measurement against real ground truth needs a person
to actually run `review.py` (see "Known limitations" and the final
summary below).

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
- **First run of any stage downloads a model** (`yolo11x.pt` ~110MB for
  detection, `yolo11m-pose.pt` ~40MB for pose, from Ultralytics' GitHub
  releases) — needs internet access once; cached locally after that.
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

### Events

Requires pose to have run for the video first (runs it automatically via
the cache if not).

```
.venv\Scripts\python.exe -m src.events data\raw\<video-file>.MP4
```

Detects hand-entry and hand-exit moments per tracked swimmer, tracking
whichever wrist is detected more confidently on average (the one nearer
the camera). Skips tracks too short to contain a full stroke cycle
(`events.min_track_frames_for_analysis`). Saves a wrist-height-over-time
plot per analysed track to `outputs/` — check this when a result looks
wrong, per CLAUDE.md section 8.3.

### Metrics

```
.venv\Scripts\python.exe -m src.metrics data\raw\<video-file>.MP4
```

Splits each swimmer's position into lengths (turns detected via a
minimum-swing zigzag, so tracking jitter isn't mistaken for a wall turn),
computes the five client metrics inside the clean zone, applies the
quality gate (never guesses — a rejected length's measurement columns are
blank with a `rejection_reason`), adds provisional benchmark comparisons,
and aggregates pairs of lengths into laps. Saves `outputs/<video>_metrics.csv`.

### Rendering

```
.venv\Scripts\python.exe -m src.render data\raw\<video-file>.MP4
```

Draws the grid, each tracked swimmer's box/ID/lane, their wrist, a flash
at each hand entry, running counters, and a provisional watermark onto
the video. Saves `outputs/<video>_annotated.mp4` (capped at 1920px wide
and 30fps — see `render.output_max_width`/`output_fps` in config.yaml —
since this machine's OpenCV has no H.264 encoder and the fallback codec
produces huge files at full source resolution/framerate).

### Coach report

```
.venv\Scripts\python.exe -m src.report data\raw\<video-file>.MP4
```

Builds `outputs/<video>_report.docx` — a Word document written for the
coach: stroke tempo, rhythm consistency, pacing across the swim, a tempo
chart, three frames from the annotated video, and observations generated
from the measured numbers. Deliberately leaves out diagnostics and every
distance/speed figure (see CLAUDE.md section 8.10). Opens in Word or
LibreOffice; export to PDF from there. The web page builds this at the end
of every run and offers it for download in the Results tab.

### Manual review

```
.venv\Scripts\python.exe -m src.review data\raw\<video-file>.MP4
```

Interactive — plays a short clip around each detected hand-entry, looping
until you press `y` (accept), `n` (reject), or `q` (quit early). Accepted
events accumulate across videos into `outputs/ground_truth_events.csv`.
Target: around 200 events across 3-4 videos (CLAUDE.md section 13).

### Web page

```
.venv\Scripts\python.exe app.py
```

Opens a Gradio page at `http://127.0.0.1:7860`. Upload a video or pick one
from `data/raw/`, click "Run analysis", and it runs the whole pipeline
(calibrating first if needed) then shows the annotated video, the results
table, and a download link for the coach report. Takes several minutes
depending on video length; a repeat run of the same video is near-instant
since every stage is cached.

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

8. **Detecting hand-entry and hand-exit crossings independently double-
   counted exits.** Signal jitter right at the water line, on a track long
   enough to matter, produced 31 detected exits against only 8 entries
   (Phase 6) — they should roughly pair up. Fixed with a state machine
   (the hand can only exit after entering, and vice versa), confirmed
   correct on real footage: 8 entries, 8 exits.

9. **Lane-based track ID merging (Phase 4) had merged tracks that overlap
   in time, not just tracks with a real gap between them** — found while
   building `metrics.py`: the "same swimmer" identity that produced the
   8-entry/8-exit result above turned out to contain 113 duplicate-frame
   rows with two conflicting wrist positions at the same instant, meaning
   two different, simultaneous detections had been merged into one track.
   Fixed (the merge now requires the new segment to start strictly after
   the previous one ends). After the fix, no single track in this
   specific clip currently has enough continuous, valid data to show a
   complete stroke cycle — the earlier 8/8 result was real code working
   correctly, but on corrupted input. This is a sobering, honest finding,
   not a regression: it reflects Phase 4's already-documented weak-detector
   problem more accurately than the buggy version did.

10. **OpenCV's `VideoWriter` has no working H.264 encoder on this machine**
    (the OpenH264 DLL isn't installed) and falls back to a much less
    efficient codec — writing the annotated video at full 4K source
    resolution and the source's ~120fps produced a 2.3GB file for a 45s
    clip. Fixed by writing at a capped resolution (1920px wide) and a
    capped output fps (30) — the "every frame" fixed decision governs
    *analysis*, not the review video, so this doesn't touch measurement
    accuracy. Result: ~365MB for the same clip (extrapolated from a 900-frame
    test render), a ~6x reduction.

## Deferred (explicitly out of scope, not built)

Nothing deferred yet — this list grows only when a specific future need
becomes obvious during the build (CLAUDE.md section 2).

## Final summary (CLAUDE.md section 17, step 11)

Every pipeline stage in the build order is built and runs end-to-end,
including through the actual Gradio UI. Reporting the real results
honestly, including the journey to get there, per section 15.

**Updated after a reliability pass (post-Phase 11):** the "nothing to
report" result below was the state before this pass. Four real, root-
caused fixes changed that:
1. Colour-based supplementary detection (replacing a motion-based
   approach that was tried and measured to fail — see CLAUDE.md section
   3's amendment).
2. A real bug fix in `events.py` (wrong initial state discarded a
   track's first transition on almost every track).
3. Per-clip camera-motion background region (fixed DIVE_P023's near-total
   tracking failure: 5% → 93% usable frames).
4. By explicit user request: the quality gate now reports speed/time over
   whatever partial clean-zone distance was actually tracked, clearly
   labelled, instead of requiring the full 15m span — see CLAUDE.md
   section 8.6's amendment for exactly what changed and why.

**Current real result, verified through the actual UI:** `DIVE_P023` now
produces **one accepted length** — 2.7m of the 15m clean zone, 1.33s,
2.05 m/s average speed (a physically plausible freestyle pace), correctly
flagged `partial_clean_zone_coverage` with no stroke-cycle data (too few
hand-entries tracked in a row) rather than blanked outright. `DIVE_P010`
still produces zero accepted lengths — an honest result: its longest
continuously-tracked segment never reaches the clean zone at all. Several
implausible "lengths" (position-jitter noise from the turn-detection
algorithm, one showing an implied speed of 196 m/s) are correctly
rejected by a new physical-plausibility check, not silently reported.

**What works, measured on real footage:**
- Wrist visibility: **80.6-80.7%**, clearing the 70% gate on both test
  clips — the pretrained pose model doesn't need fine-tuning.
- Camera-motion compensation: reliable after (a) chaining frame-to-frame
  matches instead of matching every frame against one fixed reference,
  and (b) sizing the background-matching region per clip instead of a
  fixed fraction.
- The full mechanical pipeline (calibrate → detect/track → pose → events
  → metrics → render) runs without crashing on real video, with a real
  (automatic) calibration, through the actual web page — verified live in
  a browser this session, not just from the CLI.
- `metrics.py`'s maths is verified correct: tested against a synthetic
  ideal-swimmer signal, every one of the five client metrics came out
  exactly right, and the quality gate correctly rejects/blanks a result
  when the data doesn't support it.

**What still doesn't work, and why — the honest part:**
- **No clip yet produces a full 3-cycle stroke-rate measurement.**
  Measured directly: on the longest, cleanest continuously-tracked
  segment found across both clips (728 frames, 98% pose success, a
  genuinely clean recovery→pull→recovery signal), the swimmer's second
  hand-entry never happens because the *video itself ends* one frame
  before it would. A cycle needs two consecutive entries; neither test
  clip's footage is long enough relative to this swimmer's stroke tempo
  to contain that in one continuous window. This is a data/footage
  limitation, not a code defect — confirmed by checking the raw frame
  numbers, not inferred.
- **Length-scale calibration accuracy varies by clip and is only
  approximate by construction**: no frame in the available footage ever
  shows a pool wall, so there's no way to make this exact with what's
  available.
- **Accuracy measurement against real ground truth has not been done** —
  `src/review.py`'s interactive review is a task for a person at the
  keyboard, not an agent.

**What this means for the production camera spec** (the actual point of
a POC, per CLAUDE.md's opening framing): the software-side fixes this
session closed most of the gap that detection/tracking/camera-motion
quality was responsible for. What's left pointing at hardware, not code:
longer continuous clips (or a camera framing that keeps the swimmer in
view for a full 25m, not just part of it) are needed to ever measure a
complete stroke cycle, and a wall-visible framing is needed for exact
distance calibration. Both were already CLAUDE.md section 16's
recommendation on separate (frame-rate/resolution) grounds; this session
adds a third, independent reason to the same conclusion.
