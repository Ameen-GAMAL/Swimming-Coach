# Swimming Performance Tracking — Proof of Concept

## 1. What we are building

A command-line tool plus a small web page that takes a recorded video of a
25 m swimming pool, finds the swimmers, follows each one through the water,
detects when their hands enter and leave the water, and from that calculates
how fast they swam, how many stroke cycles they took, and how far they
travelled per cycle.

The output is a CSV of measurements plus an annotated video that shows the
measurement grid, the tracked swimmers, and the detected events drawn on top
of the original footage.

This is a **proof of concept**. The goal is to show the client that the
approach works and to establish what the accuracy ceiling is with the video
we have. It is not the production system.

---

## 2. Scope

### In scope
- **Freestyle only.** No backstroke, breaststroke, or butterfly.
- **Multi-lane.** Several swimmers visible at once, each tracked separately.
- **25 m pool.**
- **Post-session processing.** The video is a file on disk. Nothing is live.
- Two measurements from the client spec: **stroke rate** and **distance per cycle**.
- Plus: clean-swim time, average speed, cycle-time variation.

### Out of scope
- Real-time processing
- Other strokes
- User accounts, authentication, databases
- Cloud deployment
- Comparing swimmers against a real professional dataset (we do not have one)

### Explicitly deferred
Do not build abstraction layers, plugin systems, or configuration hooks for
features that are out of scope. If a future need is obvious, write one line in
`README.md` about it. Do not write code for it.

---

## 3. Fixed decisions — do not change these without asking

| Decision | Value |
|---|---|
| Stroke | Freestyle |
| Pool length | 25 m |
| Lanes | Multiple, each swimmer assigned to a lane |
| Frame sampling | **Every frame.** No frame skipping. |
| Device | CUDA GPU on the local machine |
| Model family | Ultralytics YOLO11 (detection + pose) |
| Tracker | ByteTrack |
| Pixel-to-metre conversion | Homography, fully automatic (see amendment below -- no wall is visible in this footage, so width is exact via lane ropes and length is an approximate anthropometric estimate) |
| Lap definition | Swimmer goes to the far wall and returns |
| Experiment tracking | MLflow, local file store |
| Version control | Git, initialised inside the project |

**Amendment (approved during Phase 2, video sourcing):** the available
footage (University of Malta Swim2 dataset) uses a camera that pans/tracks
alongside the swimmer rather than staying fixed. A single static homography
would silently under-report distance and speed as the pan accumulates. The
pixel-to-metre conversion is still homography-from-4-points as specified,
but `detect_track.py` now also measures each frame's camera motion (via
background feature matching, away from the pool and swimmer) and corrects
tracked pixel positions back to the calibration reference frame before any
downstream stage uses them. No change to the calibration tool itself, and
no extra clicks for the user. See `data/candidates.md` for the measurements
that motivated this.

**Amendment detail (Phase 4, measured during build):** matching every frame
directly against the single calibration reference frame degrades as the pan
accumulates -- inlier feature matches dropped from ~90 near the reference
frame to single digits by the end of a 45s clip, since the visible
background eventually barely overlaps the reference frame's. The working
implementation instead matches each frame against the previous frame and
chains the transforms together (composing them frame by frame), which keeps
overlap consistently high regardless of total pan. Positions are still
finally expressed relative to the calibration's reference frame.

**Amendment (Phase 4, measured during build):** the pretrained YOLO11
"person" class is measurably weak on swimmers in this footage -- median
top-of-frame detection confidence was only ~0.13 across a full clip, likely
because a horizontal, partly-submerged swimmer viewed from this close, low
angle looks quite different from COCO's mostly-upright training data. Two
consequences, both implemented: (1) `detection.conf_threshold` in
`config.yaml` is set low (0.10) rather than high, so ByteTrack -- whose
design already expects and uses a range of detection confidences -- gets
the full signal to work with instead of having weak-but-real detections
filtered out before it ever sees them; (2) the "reuse the old ID" rule
already specified below (8.1) for a track reappearing in the same lane is
implemented as a real merge step, since it's relied on more heavily here
than a strong detector would need.

**Amendment (approved during Phase 3, calibration): pixel-to-metre
conversion is no longer manually clicked.** No frame in any checked clip
from this camera rig ever shows a pool wall or any other feature with a
precisely known real-world distance along the swim direction (confirmed
across a full clip's duration and 4 other candidate clips) -- lane rope
floats are a uniform pattern with no colour-coded distance markers. A
manually-typed length span was therefore always a guess, and guessing on a
short span gets amplified roughly 10x when extrapolated across the full 25m
pool, producing a badly wrong calibration in practice.

**Decision (approved):** replace manual clicking entirely with automatic
calibration in `src/calibrate.py`:
- The pool water is segmented by colour, and the boundary between open
  water and the lane rope floats is traced and fitted to a line -- this is
  the one lane rope this method can reliably find (colour segmentation
  proved far more robust than edge/line detection directly on the rippled
  water surface, which was tried first and produced unusable noise).
- Lane *width*'s real value is still exact (2.5m, FINA standard), but its
  *pixel placement* is now also a heuristic, not click-verified: the far
  edge of the lane is approximated as a fraction of the remaining image
  gap to the frame's bottom edge, since a second rope could not be reliably
  detected either way. `pool_area_margin_m` was widened (1.5 to 3.0) to
  absorb this.
- Lane *length* scale has no real landmark available, so it is estimated
  approximately from the swimmer's own detected body size (shoulder-to-hip
  torso length via pose against an assumed average adult value), sampled
  across many frames (250) and filtered hard: a pixel-length sanity floor
  turned out to matter more than keypoint confidence, since a handful of
  badly-placed keypoints scored high confidence anyway and would have
  thrown off the estimate by several times if not caught. Even filtered,
  the raw estimate is clamped to 0.7x-1.6x the pool's known true length
  (25m) as a prior, since the raw measurement alone was observed to be
  wrong by more than 3x in testing. Measured results on two real clips:
  P010 came out to 27.7m (11% over, no clamping needed); P023 came out to
  46.2m raw, clamped to 40m (60% over) -- the accuracy genuinely varies
  clip to clip, which is why every calibration and every downstream
  distance/speed number carries a `"scale_method":
  "approximate_anthropometric"` flag, in the same spirit as the
  provisional benchmark comparisons (section 8.7).
- The manual click-based tool is removed, not kept as a fallback -- the
  user's role is to supply a video, nothing else.

This changes the fixed decision in the table above and section 7 below.

**Amendment (investigated post-Phase 11, not adopted): motion-based
swimmer localization to fix the weak-detector root cause.** The Phase 4
amendment above already flags that pretrained YOLO11 confidence is weak
on swimmers (~0.13 median); the resulting track fragmentation and empty
metrics reports on real clips motivated trying a training-free fix:
background-subtract two camera-motion-compensated frames within the pool
area and use the one coherent moving region as a supplementary detection,
fused with YOLO's own boxes in `detect_track.py` (see `detect_motion_blob`,
`fuse_motion_detection`).

Four variants were implemented and validated with visual spot-checks on a
1200-frame slice of `DIVE_P010`, and all four failed for concrete,
measured reasons:
1. Diffing into a crop anchored to the calibration's *reference* frame:
   the camera tracks the swimmer down the pool's length, so within a few
   hundred frames the swimmer's real-world position is simply outside
   that fixed window even though the camera's own current view still
   frames them fine. Measured: the swimmer appeared in the crop in <1%
   of frames.
2. Diffing into a crop anchored to the *current* frame instead (fixes
   #1): works for alignment, but the largest resulting motion blob is
   consistently near-camera water ripple or specular glint, not the
   swimmer -- this rig mounts very close to and low over the water, so
   nearby ripples/glints displace by more pixels per frame than the
   swimmer (who is farther off, in the adjacent lane, and mostly
   submerged). Measured across threshold/kernel/resolution sweeps: even
   at the best-tuned settings, sampled detections were dominated by
   near-field noise, not the swimmer.
3. Restricting the crop to the calibrated pool area, reprojected into
   each current frame via the existing homography chain: this camera's
   field of view is close enough to the water that the reprojected pool
   area still covers nearly the whole frame, so it doesn't meaningfully
   exclude the near-field noise from #2.
4. A local search window anchored to each track's last known position
   (meant to bridge short YOLO gaps rather than search the whole frame):
   this removes the near-field-noise competition, but self-reinforces on
   drift -- if the seed position is itself a YOLO false positive (which,
   given ~0.13 median confidence, it often is), the window keeps
   "successfully" re-finding whatever coherent-looking static texture
   (e.g. the lane rope) sits at that wrong position, with no independent
   check against it being the real swimmer. Measured: seeded from a bad
   YOLO box, the window locked onto lane-rope texture for the following
   1000+ frames.

**Decision:** motion-detection fusion is implemented but disabled
(`camera_motion.motion_fusion_enabled: false` in `config.yaml`) rather
than removed -- injecting these detections would make tracking worse, not
better, since ByteTrack has no way to tell a synthesized false positive
from a real one. The camera-motion alignment code this relies on (warping
one frame into another's own current framing, not a stale fixed
reference) is a genuine, kept improvement over the bug described in
variant 1 above, and the plumbing (`motion_crop_rect`,
`motion_crop_transform`, `detect_motion_blob`, `fuse_motion_detection`)
is reusable for a future, better-verified localization signal -- most
promisingly a per-frame colour/appearance model (skin tone, cap colour,
or splash-white against the pool's turquoise water), which sidesteps
frame-to-frame alignment and near-field parallax entirely, since it
doesn't need a temporal diff at all. That wasn't attempted this session
for lack of time; see the build's final summary for current status and
recommended next steps.

---

## 4. Project structure

Create exactly this. Do not add folders or files beyond it without asking.

```
swimming-poc/
├── CLAUDE.md              This file
├── README.md              How to install and run, step by step
├── requirements.txt
├── config.yaml            Every adjustable number lives here
├── .gitignore
├── run.py                 The only file the user runs
├── app.py                 Small web page for the client demo
├── data/
│   ├── candidates.md      Suggested video links, for the user to approve
│   ├── raw/               Approved videos (user puts them here)
│   └── calibration/       One JSON per video, produced by calibrate.py
├── src/
│   ├── calibrate.py       Automatic setup tool (rewritten -- see section 3 amendments)
│   ├── detect_track.py    Find swimmers, give each a stable ID
│   ├── pose.py            Find body points on each swimmer
│   ├── events.py          Find hand entry and exit moments
│   ├── metrics.py         Turn events into measurements
│   ├── render.py          Draw the annotated output video
│   ├── report.py          Build the downloadable coach report (added -- see section 8.10)
│   ├── profiling.py       Time each stage, send results to MLflow
│   └── review.py          Manual review tool (added -- see section 13 amendment)
├── cache/                 Intermediate results, so reruns are fast
└── outputs/               Final CSV, annotated video, and the coach report
```

Eight source files, plus `review.py` (section 13 amendment) and `report.py` (section 8.10).
Each one does a single job. If a file starts doing two jobs, stop and ask
before splitting it.

---

## 5. Code style

### Comments
Write short, direct comments in plain language. Someone who does not write
code should be able to read a comment and understand what that part does.

Good:
```python
# Convert the swimmer's position from pixels to metres
# Only look at swimmers inside the pool, ignore people standing at the edge
# A cycle is one hand entering the water twice in a row
```

Bad:
```python
# Apply homography transformation matrix H to coordinate vector
# Instantiate the tracker with the configured hyperparameters
# TODO: refactor this later
```

Comment the **why** and the **what it means**, never restate the syntax.
One comment per logical block, not one per line.

### General
- Plain functions. No classes unless state genuinely needs to persist.
- No `utils.py`, no `helpers.py`, no dumping grounds.
- Every number that someone might want to change goes in `config.yaml`,
  never hardcoded in the middle of a function.
- Fail loudly with a clear message. If the calibration file is missing, say
  "No calibration found for this video. Run calibrate.py first." Do not
  silently fall back to a default.

---

## 6. Configuration

All tunable values live in `config.yaml`. Structure it like this:

```yaml
video:
  fps_override: null          # Leave null to read fps from the file

pool:
  length_m: 25
  lanes: 6
  segment_size_m: 2.5         # Size of each grid strip drawn on the video
  clean_zone_start_m: 5       # Skip the dive and the push-off
  clean_zone_end_m: 20        # Skip the approach to the wall

detection:
  model: yolo11m.pt
  conf_threshold: 0.35
  batch_size: 16
  half_precision: true

pose:
  model: yolo11m-pose.pt
  conf_threshold: 0.30
  batch_size: 16
  half_precision: true

events:
  wrist_conf_threshold: 0.40
  min_cycle_seconds: 0.6      # Anything faster than this is noise
  smoothing_window: 5         # Frames, to steady the wrist signal

quality:
  min_confidence: 0.90
  min_valid_cycles: 3

benchmarks:
  provisional: true           # These are placeholder numbers, not real data
  freestyle:
    stroke_rate_50m: [55, 70]   # cycles per minute
    stroke_rate_100m: [45, 60]
    distance_per_cycle: [1.8, 2.5]   # metres

mlflow:
  tracking_uri: "file:./mlruns"
  experiment_name: "swimming-poc"

cache:
  enabled: true
  directory: "./cache"
```

---

## 7. The calibration tool (`src/calibrate.py`)

**Rewritten per the Phase 3 amendment above: fully automatic, no clicking.**
This is still the most important piece for accuracy, which is exactly why
the length-scale limitation is measured and flagged rather than hidden.

### What it does
Given just a video, with no user interaction:

**Step 1 — Find the pool and the nearest lane rope.**
Segment the water region by colour (the pool's turquoise is distinct from
sky/stands/deck, once restricted to the lower part of the frame where the
pool always is in this camera's framing -- otherwise blue-painted building
trim can be confused for water). Edge/line detection directly on the water
surface was tried first and failed -- ripple texture produced too much
noise for Hough line detection to find a rope reliably. Instead, the
boundary between open water and the lane rope floats sitting on the
surface is traced (the topmost water pixel in every column) and fitted to
a line -- this is one real, reliably-detected lane rope.

**Step 2 — Corners and width.**
Two points sampled near and far along the fitted rope give one side of the
4 corners a manual click would have produced. A second rope could not be
reliably detected, so the lane's other edge is approximated as a fraction
of the remaining image gap to the frame's bottom edge (a heuristic, tuned
and sanity-checked visually against real footage). Lane width's real value
is still exact: 2.5m (FINA standard) -- only its pixel placement is
approximate.

**Step 3 — Water line and lane boundary.**
A rope floats on the water surface, so its own pixel position at any point
along its length already is that lane's water line -- no separate detection
needed. The same ropes double as the lane boundaries.

**Step 4 — Length scale (approximate).**
No wall or other known-distance landmark is visible in this footage (see
the Phase 3 amendment). Sample the swimmer's torso length (shoulder
midpoint to hip midpoint) via pose across 250 frames, keep only readings
above a pixel-length sanity floor (badly-placed keypoints can still score
high confidence), and clamp the result to 0.7x-1.6x the pool's known true
length as a prior against a noisy measurement. This remains a provisional
estimate, not a measurement -- every calibration is saved with
`"scale_method": "approximate_anthropometric"`, and every distance/speed
number computed from it should be presented as provisional, the same as
the benchmark comparisons in section 8.7.

### Output
Same as before: one JSON file per video in `data/calibration/`, named after
the video, containing the 4 corner points, the homography matrix, the
per-lane water lines, the lane boundaries, the frame number used, the pool
length, and now `scale_method` and `scale_estimate_clamped_to_prior` flags.

### Verification
Same preview image as before (grid, water lines, lane boundaries drawn over
the frame) -- still worth checking, since automatic lane-rope detection can
fail on a frame with unusual lighting or occlusion, and there is no click
step left to catch that by eye before it happens.

---

## 8. The processing pipeline

### 8.1 Detection and tracking (`src/detect_track.py`)
Run YOLO11 detection on every frame, person class only, in batches on the GPU.

Before anything else, correct for camera motion: match each frame's
background region (sky/stands, away from the pool and swimmer) against the
previous frame and chain those frame-to-frame transforms together, then
express the result relative to the calibration's reference frame. This
makes every downstream stage able to treat the camera as if it were fixed,
matching the calibration. (See the amendment notes in section 3 — the
Swim2 dataset's camera pans/tracks the swimmer rather than staying still,
and matching every frame directly against the reference frame instead of
chaining was tried first and measured to degrade over a clip.)

Discard any detection whose centre falls outside the pool area from the
calibration. Pass the surviving detections to ByteTrack so each swimmer keeps
a stable ID.

After tracking, assign each ID to a lane using the lane boundaries. If a track
disappears and a new one appears in the same lane shortly after, reuse the old
ID.

Cache the result keyed on video filename plus model plus thresholds.

**Amendment (reliability pass, post-Phase 11): colour-based swimmer
localization is now the active supplementary detection signal**, replacing
the motion-based approach investigated and rejected in section 3's
amendment above. Per frame (not temporal, so it doesn't depend on the
camera-motion chain at all): segment out the pool's own water colour using
the same HSV thresholds `calibrate.py` already trusts, then look for a
skin/dark-cap/white-splash-coloured blob near that water body. Fused with
YOLO's own boxes the same way the motion approach was designed to (trust
YOLO first; synthesize a detection only when nothing already covers the
blob) -- see `detect_color_blob`/`fuse_synthetic_detection`. Validated by
visual spot-check on real footage. See `config.yaml`'s `color_localize`
section for the full parameter rationale.

**Amendment (reliability pass, post-Phase 11): the background region used
for camera-motion matching is now sized per-clip, not a fixed fraction.**
Measured on real footage: `camera_motion.background_region_frac=0.35`
(tuned against DIVE_P010, where the stands fill that band) captured almost
nothing but plain sky on DIVE_P023, whose stands sit lower in frame --
ORB found zero matched features in 63% of frames there, not just "few".
Fixed by deriving the boundary from the calibration's own detected
pool-top corner position per clip instead (falling back to the old 0.35 as
a floor, never shrinking below what was already validated). Measured
effect on DIVE_P023: frames with a usable camera-motion estimate went from
5% to 93%, and tracks-found (after lane merge) dropped from 249 to 83.
See `dynamic_background_region_frac`.

### 8.2 Pose (`src/pose.py`)
Run YOLO11-pose on the tracked swimmers only. The pretrained model already
outputs the points we need: nose, shoulders, elbows, wrists, hips, knees,
ankles. Apply the same per-frame camera-motion correction from
`detect_track.py` to the keypoint coordinates, so every pixel coordinate
that leaves this stage is already in calibration-reference-frame space —
`events.py` can then compare a wrist position straight to a lane's water
line with no further correction.

**Measure before assuming.** The first thing this module reports is: in what
percentage of frames was a wrist detected with confidence above the threshold?
Log this as `quality_wrist_visible_pct`.

- 70% or above → the pretrained model is good enough, continue
- Below 70% → stop and report. Do not silently produce bad numbers. The
  fallback is to annotate 300–500 frames and fine-tune the pose model, but
  that is a separate decision the user must make.

Cache the result.

**Amendment (reliability pass, post-Phase 11, speed):** tracks shorter than
`events.min_track_frames_for_analysis` are filtered out *before* running
pose inference on them, not after -- `events.py` was going to discard them
anyway, so paying for pose inference on a weak detector's many short track
fragments (see section 3's Phase 4 amendment) was pure waste.

### 8.3 Events (`src/events.py`)
For each swimmer, track the wrist nearest to the camera. Record which side it
is (left or right) — with a side view the far arm is hidden by the body, so
tracking one consistent hand is the only reliable option.

Detect a **hand entry** using three signals together, not one:

1. The wrist crosses that lane's water line moving downward. This is the
   primary signal.
2. The wrist detection confidence drops. When the hand goes under, the model
   loses it — this confirms the entry.
3. The wrist path over time is periodic. Use this to reject spurious events
   that do not fit the rhythm.

Smooth the wrist position over `smoothing_window` frames before looking for
crossings. Reject any cycle shorter than `min_cycle_seconds`.

A **stroke cycle** runs from one hand entry to the next entry of the same hand.

Save the wrist-height-over-time signal as an artifact. When something looks
wrong, that plot is how the user diagnoses it.

**Amendment (Phase 6, found during build):** detecting entry and exit
crossings independently let jitter in the smoothed signal right at the
water line produce a burst of spurious exits for one real entry (measured:
31 exits vs 8 entries before this fix, when they should roughly pair up).
Fixed with a state machine: the hand can only exit after an entry, and
can't enter again until it has exited, so events strictly alternate.
Confirmed correct on real footage: 8 entries paired with 8 exits.

Also skips tracks shorter than `events.min_track_frames_for_analysis` (60
frames) before running detection on them -- given the tracking
fragmentation from Phase 4 (a weak pretrained detector produces many short
track fragments for one real swimmer), most fragments are too short to
contain even one full stroke cycle, and analysing/plotting all of them
produces mostly noise. On the one real, long-enough track, this correctly
found all 8 stroke cycles.

**Bug fix (found post-Phase 11, during the reliability pass):** the state
machine above always initialized `underwater = False` regardless of the
track's actual first sample -- but a track essentially never happens to
start exactly at a real entry/exit boundary, so it just as often starts
already submerged. When it does, the very first real transition
(submerged → surfaced) requires `underwater` to already be `True` to
register as a valid exit, and was silently discarded under the old
hardcoded start. Fixed to initialize from the observed first sample
(`underwater = not above[0]`) instead. This was very likely a contributor
to the zero-entries/zero-exits results seen on real clips throughout this
build, independent of any detection-quality issue -- confirmed by re-
running on cached pose data with no other change: 0 exits before the fix,
1 exit after, on the same track.

### 8.4 Lap splitting
The client's definition for this POC: a lap is complete when the swimmer
reaches the far wall and comes back.

Detect a turn as the moment the swimmer's direction of travel reverses, using
the smoothed centroid position in metres. Internally, compute metrics **per
length** (one 25 m traverse), then also report them aggregated per lap. The
CSV carries both.

### 8.5 Metrics (`src/metrics.py`)

Only count cycles whose entry and exit both fall inside the clean zone
(default 5 m to 20 m of each length).

```
clean_swim_time   = time from entering 5 m to exiting 20 m
average_speed     = clean distance / clean_swim_time          (m/s)
cycle_count       = number of complete cycles in the clean zone
stroke_rate       = cycle_count / clean_swim_time * 60        (cycles/min)
distance_per_cycle = clean distance / cycle_count             (m)
cycle_time_cv     = std(cycle times) / mean(cycle times) * 100
```

Position must be **continuous**, not quantised to grid strips. The strips are
for drawing and for the user to verify by eye. The maths uses the exact metre
position from the homography. Distance per cycle is around 1.8–2.5 m — if we
rounded position to 2.5 m strips, the number would be meaningless.

### 8.6 Quality gate
Do not report a score when any of these is true:
- Mean wrist confidence across the cycle is below `min_confidence`
- Fewer than `min_valid_cycles` complete cycles in the clean zone
- A hand entry or exit was occluded

When the gate blocks a result, the CSV row still exists but the measurement
columns are empty and a `rejection_reason` column explains why. Never guess.

**Amendment (explicit user request, reliability pass, post-Phase 11):**
measured that on both real clips tested, no continuously-tracked window
ever contains 2 consecutive hand-entries -- the clip ends, or tracking
drops, mid-cycle every time -- and separately, no window ever fully spans
the clean zone's 20m boundary before running out. Two changes, together,
to stop reporting a fully blank row when there's real, usable partial
data:
- `quality.min_valid_cycles` lowered from 3 to 1. Below this count,
  `stroke_rate_cpm`/`distance_per_cycle_m`/`cycle_time_cv_pct` are blanked
  (as before) with a `data_quality_note` column explaining why.
- **Speed and clean-swim-time are no longer gated on cycle count or
  confidence at all** -- they come from position tracking alone, an
  independent signal from hand-entry detection, so a missing stroke count
  shouldn't blank an otherwise-valid speed measurement. They ARE still
  gated on reaching the clean zone in the first place.
- A length that reaches only *part* of the clean zone (not both the 5m
  and 20m boundaries) is no longer a hard rejection: speed/time are
  computed over whatever sub-range was actually tracked, and the row is
  flagged `partial_clean_zone_coverage: true` with a
  `clean_zone_coverage_note` (e.g. "partial clean-zone coverage (10.2m of
  15.0m)") so it's never mistaken for the full 15m measurement. A length
  that never touches the clean zone at all is still a hard rejection —
  everything blanked, `rejection_reason` explains why.

### 8.7 Benchmarks and scoring

We do not have professional swimmer data. The numbers in `config.yaml` come
from the client's specification document and are explicitly marked as initial
placeholder ranges there.

So:
- Report all raw measurements.
- Compare them against the placeholder ranges and show whether they fall
  inside, below, or above.
- **Mark every comparison as provisional** in the CSV and on the web page.
- Do **not** produce the weighted 0–100 composite score. Two of its four
  components cannot be computed: the speed component needs percentiles from a
  matched professional group we do not have, and the technique component for
  freestyle is not defined in the client's specification at all.

Instead, output the components we *can* compute — stroke rate, distance per
cycle, speed, cycle-time variation — each with its provisional comparison.
Add a short note in the output explaining why the composite score is absent.
This is an honest, defensible position and it tells the client exactly what
data they need to supply next.

### 8.9 Rendering (`src/render.py`)
Produce an annotated video showing:
- The measurement grid drawn in perspective over the pool
- Each swimmer's bounding box, ID, and lane
- The tracked wrist point
- A marker flashing at each detected hand entry
- Running counters: cycle count, current stroke rate, distance covered
- A clear "PROVISIONAL" watermark on any benchmark comparison

The grid is what lets the client confirm the distance maths with their own
eyes. Make it clearly visible without hiding the swimmers.

### 8.10 The coach report (`src/report.py`) — added by user request

A downloadable Word document (`outputs/<video>_report.docx`) built at the
end of every run and offered in the app's Results tab. It is written for
the coach, not the engineer: no diagnostics, no system caveats, no
distance or speed figures.

What it contains, and why only this:
- **Stroke tempo** (cycles/min), **rhythm consistency** (cycle-time
  variation), and **pacing** (tempo change from the earliest to the latest
  tracked segment) — all derived from the video's timestamps alone, so
  they carry none of the distance-scale error.
- A tempo-through-the-swim chart, a per-segment table, and three frames
  from the annotated video.
- **Observations** generated from those numbers, phrased as what was
  measured and what is worth checking on the video — never as a
  prescription. The coach's own reading of technique takes precedence and
  the report says so.
- Reference tempo ranges come from `benchmarks.freestyle` and are labelled
  as general reference ranges, since those values are placeholders.

Deliberately excluded: the pull/recovery phase split. The events data
carries entry→exit pairs as close as 30–50 ms apart (jitter around the
shoulder line), and the minimum-cycle rule guards only entry-to-entry
spacing, so a within-cycle phase split would be built on noise. Add it
only once entry→exit spacing has its own guard.

`summarise_run()` in the same module derives every count from the cached
stage outputs; the app's summary line and Diagnostics tab read from it
too, which is what stopped cached runs from showing "n/a" everywhere.

---

## 9. Performance

The POC is not real-time, but speed still matters because the user will run
this many times while tuning.

Requirements:
- Batch frames through the GPU. Never process one frame at a time.
- Use `half=True` on CUDA.
- Run pose only on swimmers inside the pool area.
- Cache detection and pose output to `cache/`. Changing an event threshold
  must not require re-running the models. This is the single biggest time
  saver during tuning.
- Print a timing breakdown per stage at the end of every run.

Report processing speed as frames per second in the summary and in MLflow.

---

## 10. MLflow

Local file store at `./mlruns`. Start it with `mlflow ui` and view at
`localhost:5000`.

### Run names
Format: `{video}__{what_changed}__{MMDD-HHMM}`

```
freestyle_lane3_25m__baseline__0918-1430
freestyle_lane3_25m__wrist-thresh-0.5__0918-1455
```

Use `baseline` when nothing was deliberately changed. The name must let the
user tell runs apart in a list without opening them.

### Tags
`video`, `stroke`, `stage` (full / detect-only / events-only), `model`,
`git_commit`.

### Parameters
Every parameter name starts with the stage it belongs to.

```
video_name, video_fps, video_frames, video_duration_sec
detect_model, detect_conf_threshold, detect_batch_size, detect_half_precision
pose_model, pose_conf_threshold, pose_batch_size
calib_pool_length_m, calib_segment_size_m, calib_lanes_count
calib_clean_zone_start_m, calib_clean_zone_end_m
event_wrist_threshold, event_min_cycle_sec, event_smoothing_window
quality_min_confidence, quality_min_cycles
```

### Metrics
```
time_detect_sec, time_pose_sec, time_events_sec, time_total_sec, speed_fps
quality_wrist_visible_pct, quality_mean_confidence, quality_rejected_events
result_cycles_detected, result_stroke_rate_cpm, result_dpc_m, result_avg_speed_ms
```

### Artifacts
Annotated video, results CSV, calibration JSON, calibration preview image,
wrist-signal plot.

### Naming rule
Any new parameter or metric must use one of the prefixes above. Adding a new
prefix requires recording it in `README.md` first. Lowercase with
underscores, no spaces, no ambiguous abbreviations. This prevents `conf`,
`confidence`, and `conf_thresh` all existing side by side after fifty runs.

---

## 11. Git

Run `git init` inside the project. `.gitignore` must exclude:

```
data/raw/
outputs/
cache/
mlruns/
*.pt
__pycache__/
*.pyc
.venv/
```

Commit at meaningful milestones, not once at the end. Suggested sequence:
project skeleton → calibration tool → detection and tracking → pose →
events → metrics → rendering → MLflow → web page.

Commit messages in plain language: "Add calibration tool for pool corners and
lane water lines", not "feat(calib): impl homography".

---

## 12. Video sourcing (`data/candidates.md`)

Before writing pipeline code, search for suitable footage and write the
candidates into `data/candidates.md` as a table for the user to approve.

Required for each candidate:

| Column | Notes |
|---|---|
| Link | Direct URL |
| Source | Channel or site |
| Duration | |
| Camera angle | Describe it |
| Camera fixed? | Yes / No / Slight movement |
| Pool visible | How much of the length is in frame |
| Lanes visible | |
| Estimated fps | |
| Why it fits | One line |

What makes footage suitable:
- The camera does **not** move, pan, or zoom
- Most or all of the pool length is visible
- Any side angle is acceptable as long as swimmers are clearly visible
- Freestyle
- Swimmers large enough in frame that a wrist is more than a few pixels

What to avoid: broadcast competition coverage (the camera pans and the
swimmers are tiny), underwater-only footage, heavily edited clips with cuts.

Search terms that tend to work: "freestyle technique side view", "swim stroke
analysis fixed camera", "swimming training session side angle", "GoPro
swimming pool side view". Also check Roboflow Universe for existing swimmer
detection datasets.

Target **10 to 15 clips**, 20 to 60 seconds each.

**Do not download anything.** Write the list. The user reviews it, picks the
suitable ones, and places them in `data/raw/` themselves.

---

## 13. Manual review (`src/review.py`)

**Amendment (Phase 9, found during build):** this file isn't in section 4's
original list of 8 -- an apparent oversight there, since this section
describes its behaviour in detail as its own deliverable and the build
order (section 17, step 9) lists it as its own step. Added as `src/review.py`
rather than folded into an existing file, since it does a genuinely
different job (interactive human review) from any of the other seven.

The user wants to be involved in the labelling and verification. Build tools
that make their part fast — reviewing and correcting, never typing raw data.

Provide a small review script that:
- Plays back detected hand-entry moments one at a time
- Lets the user press a key to accept or reject each one
- Saves the accepted events as ground truth in a CSV

This ground truth is used to **measure** accuracy, not to train. Target around
200 events across 3 or 4 videos.

Ground truth accumulates across videos into one file
(`outputs/ground_truth_events.csv`) rather than one file per video, since
the target is a combined set spanning several clips.

---

## 14. The web page (`app.py`)

A single Gradio page. Deliberately minimal:
- Upload a video, or pick one already in `data/raw/`
- A button to run the analysis
- Shows the annotated video and the results table when finished
- Shows a provisional-data warning next to any benchmark comparison

No login, no database, no styling work. It exists so the client can try the
system themselves rather than watch a screen recording.

**Amendment (explicit user request, reliability pass): styling work is now
in scope, overriding "no styling work" above.** Rebuilt with a themed
layout and four tabs (Run Analysis / Results / Diagnostics / About), a
real per-stage progress checklist instead of one undifferentiated bar,
provisional/rejected-result callouts as a styled coloured panel rather
than a markdown line, a small chart of accepted-length metrics, and a
Diagnostics tab surfacing the calibration preview, wrist-signal plots, and
each stage's own diagnostics (frames processed, tracks found, wrist
visibility %, quality-gate detail). Still no login, no database -- those
weren't asked for.

---

## 15. Success criteria

The POC is accepted when, on at least 10 approved clips:
- Detected cycle count is within ±1 of a manual count
- Stroke rate is within ±5% of a manually calculated value
- Distance per cycle falls in a physically sensible range and is consistent
  across cycles by the same swimmer
- The quality gate correctly rejects clips that are genuinely too poor to
  measure, rather than producing confident nonsense

Report these numbers honestly in the final summary, including the failures.

---

## 16. Known limitations to state in the output

Write these into `README.md` and show them in the final summary. They are
arguments for the production camera specification, not excuses.

1. **Frame rate.** Internet footage is usually 25 or 30 fps. At 30 fps, timing
   resolution is 33 milliseconds, which introduces roughly 3% error in stroke
   rate. The client specification asks for 50–60 fps. This limit comes from
   the source video, not from the model.

2. **Resolution.** At 1080p covering 25 m, a hand is roughly 12–15 pixels
   across. 4K would roughly double that.

3. **Camera angle.** Footage found online will not match the production
   camera position. Accuracy on production footage should be better, not
   worse, but it must be re-measured once real footage exists.

4. **Benchmarks.** All comparisons use placeholder ranges. Real benchmarks
   require at least 20 professional swimmers per category (stroke, sex, age
   band, distance, pool length, session type).

---

## 17. Build order

Work through these in order. Stop and report at the end of each phase.

1. Project skeleton, `config.yaml`, git init, MLflow wiring, `README.md`
2. `data/candidates.md` — the video shortlist for the user to approve
3. `calibrate.py` with the verification preview image
4. `detect_track.py` with caching and lane assignment
5. `pose.py` with the wrist-visibility check — **stop here and report the
   percentage before continuing**
6. `events.py` with the wrist-signal plot
7. `metrics.py` and the quality gate
8. `render.py`
9. Manual review script
10. `app.py`
11. Accuracy measurement against ground truth, final summary

---

## 18. When to stop and ask

Ask the user before:
- Adding any file or folder not listed in section 4
- Changing anything in section 3
- Fine-tuning any model
- Downloading any video
- Adding a dependency not already in `requirements.txt`
- Producing a composite score, or any number presented as if the benchmarks
  were real

If wrist visibility comes back below 70%, stop and report. Do not work around
it silently.
