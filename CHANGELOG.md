# Changelog

Versions follow [semantic versioning](https://semver.org/) — `0.x` while this
is a proof of concept. MINOR goes up when what is measured, or how, changes;
PATCH for fixes that leave the method alone. Each version is a git tag
(`vX.Y.Z`), and the number lives in one place, `src/__init__.py`. It is
stamped on every output (the `pipeline_version` column of the metrics CSV, the
coach report's footer, the app's title and MLflow runs), so any result can be
traced to the code that produced it. The measured reasons behind each change
are in CLAUDE.md, in the section named.

## [0.3.0] — 2026-09-26

### Added
- **Position along the pool.** Every swim and 5-second split now reports where
  it is in the length (metres from the wall being swum away from), with an
  error estimate per split. It is read from the lane ropes' red markings — the
  5 m zones at each end and the mid-pool marker — and reported only where the
  error is within 2.5 m (5% of the length). Measured against hand-checked
  positions on DIVE_P012: 0.5 m and 1.4 m off; on the other clips, independent
  markings agree to within 0.1–0.8 m. (CLAUDE.md 8.11)
- The running panel on the annotated video shows the position in the length.
- The pipeline version on the metrics CSV, the coach report, the app and MLflow
  runs; this changelog.

### Changed
- **Distance where the rope's bands can't be counted.** In the solid-red last
  5 m at each end, and where splash or glare hides the bands, the rope's
  movement now comes from the pattern of individual float discs (agrees with
  the band count to within 2%). Directly read share of each swim: 66–95% →
  82–98% across eight clips.
- The rope search requires open water directly below a candidate rope, which
  stops it locking onto diagonals through the farther ropes on a steep camera
  angle (DIVE_P010: its distance went from 31.7 m to 36.1 m, confirmed by the
  red markings). The search is also ~40% faster.
- The app's Diagnostics tab reports the disc-pattern share and what placed the
  swimmer in the length.

### Fixed
- The filmed swimmer was lost on DIVE_P027 (352 of 5,749 frames attributed to
  them instead of 5,208): the size reference for picking out the swimmer is
  now the track that is the largest box on screen most often.
- Annotated videos rendered before 0.2.0 still showed the old metre grid, and
  one (DIVE_P030) was corrupt; the coach report embeds frames from them. All
  eight tested clips were re-rendered.

## [0.2.0] — 2026-09-25

### Added
- **Distance, speed and distance per cycle from the lane rope**
  (`src/rope_odometry.py`): the rope's navy/white bands are counted as they
  pass the swimmer's head. Band length 0.39 m, calibrated against the pool's
  red markings. (CLAUDE.md 8.11)
- `swim` and 5-second `split` rows in the metrics; a readable results table,
  speed chart and CSV download in the app; speed and distance per cycle in the
  coach report; `run.py` runs the whole pipeline.

### Changed
- Pose finds the swimmer far more often (its own person check has a separate,
  lower threshold); the quality gate measures stroke capture (share of strokes
  timed, 70% minimum) instead of wrist visibility. (CLAUDE.md 8.2)
- Near/far arm split by Otsu's method; arm timing reported only when the arm
  labels alternate at least 75% of the time. (CLAUDE.md 8.3)

### Removed
- Per-length, per-lap and clean-zone metrics built from the calibration and
  camera motion, which could not measure distance with a walking camera.

## [0.1.0] — 2026-09-25

First version whose stroke timing held up across clips: the tracker no longer
discards weak detections, strokes are timed from recovery peaks, nested
duplicate boxes are suppressed, the filmed swimmer is separated from other
people in the pool, and the coach report covers tempo, rhythm, pacing and
arm timing. (CLAUDE.md 8.1, 8.3, 8.10)

[0.3.0]: https://github.com/Ameen-GAMAL/Swimming-Coach/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Ameen-GAMAL/Swimming-Coach/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Ameen-GAMAL/Swimming-Coach/releases/tag/v0.1.0
