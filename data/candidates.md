# Video candidates

CLAUDE.md section 12 describes sourcing footage from the web. Instead, we
have a local research dataset already available and in scope: the
**University of Malta Swim2 dataset**, at

```
F:\Datasets\university_malta_swim2\extracted\Swim2 ONLY\Freestyle\Above\
```

This replaces the web search: every candidate below was inspected directly
(resolution/fps read from the file, camera angle and framing checked by
eye and measured with optical flow) rather than guessed from a thumbnail.
The `Under\` sibling folder (underwater camera) exists but is not used —
CLAUDE.md explicitly avoids underwater-only footage.

## Two things found during inspection

**1. The camera pans — it does not stay fixed.** This dataset uses a
deck-level GoPro-style camera that tracks alongside the swimmer (most
likely a poolside trolley or rail rig), rather than a stationary tripod.
Measured with optical flow on background features (stands/sky, away from
the pool and swimmer) on 4 clips across different dates:

| Clip | Cumulative background drift over the clip |
|---|---|
| P004 (37s in) | ~340 px |
| P010 (43s in) | ~550 px |
| P023 (53s in) | ~540 px |
| P030 (29s in) | ~320 px |

(out of a 3840 px frame width). This directly conflicts with CLAUDE.md's
"camera does not move, pan, or zoom" suitability criterion and its
single-static-homography pixel-to-metre method, which assumes a fixed
camera. **Decision (approved):** add automatic pan compensation —
detect_track.py will measure each frame's camera shift via background
optical flow and correct tracked positions back to the calibration
reference frame before anything downstream (lane checks, water-line
crossing, homography) touches them. No extra user clicks. This is a
recorded amendment to CLAUDE.md section 3/8.1.

**2. One swimmer per clip, not simultaneous multi-lane swimming.** Each
clip is a single-swimmer trial (participant IDs P004–P030). The pool
itself has multiple lanes and 2–3 lane ropes are usually visible near the
swimmer, so lane assignment and the multi-lane calibration steps are still
exercised, but no clip has several swimmers tracked at once. The pipeline
is still built to handle that case generically — this dataset just doesn't
provide a clip to test it with. Noted as a coverage gap, not a blocker.

## What's good about it

- **4K (3840×2160) at ~120 fps** — double the client's 50–60 fps target
  and well above the 1080p assumption in CLAUDE.md's known-limitations
  section. Two of the four "known limitations" (frame rate, resolution)
  are largely moot for this footage.
- Swimmer occupies a large portion of the frame (several hundred pixels
  across), so wrist detection should be well above the low-pixel-count
  concern called out in the spec.
- Real competition pool (Sport Malta national pool), clean water, good
  lighting, static-mount lane ropes give a natural perspective gradient —
  well suited to the per-lane water-line calibration step.

## Candidates (12 of 27 available, spanning the full collection period)

Dataset root for the `Link` column: `.../Freestyle/Above/`

| Link | Source | Duration | Camera angle | Camera fixed? | Pool visible | Lanes visible | Est. fps | Why it fits |
|---|---|---|---|---|---|---|---|---|
| DIVE_P004_FREE_ABOVE_20250612_001._SWIM2.MP4 | Univ. Malta Swim2 (local) | 38.9s | Deck-level side view, looking down the lane | No — pans with swimmer, auto-corrected | Most of the 25m length as camera tracks | 2–3 near swimmer's lane | 119.88 | 4K, largest date-range anchor, full stroke visible |
| DIVE_P007_FREE_ABOVE_20250619_001._SWIM2.MP4 | Univ. Malta Swim2 (local) | 35.9s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | Different session date, mid-length duration |
| DIVE_P010_FREE_ABOVE_20250621_001._SWIM2.MP4 | Univ. Malta Swim2 (local) | 44.9s | Same rig | No — auto-corrected (measured: ~550px drift) | Most of length | 2–3 | 119.88 | Directly verified via optical flow; good reference clip |
| DIVE_P013_FREE_ABOVE_20250621_024._SWIM2.MP4 | Univ. Malta Swim2 (local) | 46.0s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | Same date as P010, later trial number |
| DIVE_P016_FREE_ABOVE_20250626_001._SWIM2.MP4 | Univ. Malta Swim2 (local) | 38.0s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | New session date |
| DIVE_P017_FREE_ABOVE_20250627_001._SWIM2.MP4 | Univ. Malta Swim2 (local) | 39.1s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | New session date |
| DIVE_P019_FREE_ABOVE_20250701_001._SWIM2.MP4 | Univ. Malta Swim2 (local) | 31.9s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | Shorter clip, faster swimmer or shorter clean swim |
| DIVE_P021_FREE_ABOVE_20250701_017._SWIM2.MP4 | Univ. Malta Swim2 (local) | 34.9s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | Same date, later trial |
| DIVE_P023_FREE_ABOVE_20250702_009._SWIM2.MP4 | Univ. Malta Swim2 (local) | 54.1s | Same rig | No — auto-corrected (measured: ~540px drift) | Most of length | 2–3 | 119.88 | Longest clip available; directly verified |
| DIVE_P025_FREE_ABOVE_20250703_001._SWIM2.MP4 | Univ. Malta Swim2 (local) | 32.0s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | New session date |
| DIVE_P027_FREE_ABOVE_20250703_017._SWIM2.MP4 | Univ. Malta Swim2 (local) | 48.0s | Same rig | No — auto-corrected | Most of length | 2–3 | 119.88 | Same date, later trial, long duration |
| DIVE_P030_FREE_ABOVE_20250704_017_SWIM2.MP4 | Univ. Malta Swim2 (local) | 30.1s | Same rig | No — auto-corrected (measured: ~320px drift) | Most of length | 2–3 | 119.88 | Last session date; directly verified |

15 clips (out of 27) were not selected for this initial batch, purely to
keep to CLAUDE.md's "10 to 15 clips" target — they share the same rig and
technical properties, so more can be added later at no extra sourcing cost
if needed.

## Next step

Since these are already-approved local files (not downloads), review the
12 above and confirm the list. Once confirmed, approved videos will be
**hard-linked** (not copied) into `data/raw/` — same file, no duplicated
disk space, since both locations are on the F: drive.
