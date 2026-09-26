# Distance swum, measured against the lane rope.
#
# The footage is shot by a camera carried along the pool deck at water
# level, walking with the swimmer. Nothing fixed stays in view for long
# (no wall, and the stands are too far away to give a scale), so a
# calibration homography plus camera-motion compensation could never pin
# down how far the swimmer went -- the old distance/speed figures were
# built that way and were not trustworthy (CLAUDE.md section 8.11).
#
# The lane rope right behind the swimmer is a ruler lying in the water:
# navy and white bands of fixed length (pool.rope_band_m), alternating the
# whole length of the pool. Counting how many bands pass the swimmer's head
# gives the distance swum directly, whatever the camera is doing, because
# the rope and the head are measured in the same picture.
#
# Per sampled frame, on each side of the swimmer's box:
#   1. find the rope just behind the swimmer: at water level it passes at
#      about head height, so start there and follow it outward in short
#      chunks (it bends with the lens and the perspective);
#   2. read the navy/white pattern along it and locate each navy band;
#   3. between consecutive samples, match every navy band to where it was a
#      moment ago; its movement divided by the local navy-to-navy spacing is
#      the rope's movement in REPEATS (one navy + one white band) -- no pixel
#      scale is needed, and the perspective (bands shrink with distance)
#      cancels band by band.
# The swimmer's movement against the rope is then the head's own movement
# across the picture minus the rope's, both in repeats; one repeat is
# 2 x pool.rope_band_m.

import argparse
import hashlib
import json
import os

import cv2
import numpy as np
import pandas as pd

from src.detect_track import cache_path as detect_cache_path
from src.detect_track import run_detect_track_cached, subject_track_ids
from src.profiling import Timer, load_config

# Rope finding. Measured on DIVE_P012 / DIVE_P022 (3840x2160).
CHUNK_PX, STEP_PX, FIRST_CHUNK_PX = 320, 120, 420   # a chunk must span a navy AND a white band (~250-300 px near the swimmer)
FIRST_OFFSETS_PX = (0, 150, 300)                      # the extended arm and its splash often hide the rope at the box edge
FIRST_SLOPES = np.arange(-0.24, 0.241, 0.02)          # behind the swimmer on an oblique view the rope climbs up to ~0.15 px/px (DIVE_P012)
SEARCH_UP_PX, SEARCH_DOWN_PX = 170, 40                # around head height
FOLLOW_TOL_PX = 14                                    # rope row search either side of the predicted row, chunk to chunk
MAX_TRACE_PX = 1500
PROFILE_HALF_PX = 8
CONTINUITY_BLOCKS, CONTINUITY_MIN_OK = 6, 5                # open water below a starting rope in 5 of 6 pieces (_open_below)
TEXTURE_MAX_SHIFT_PX = 60                              # the rope moves ~5-15 px between readings at 30 per second
TEXTURE_REACH_PX = 900                                 # disc pattern compared over this much rope next to the swimmer
TEXTURE_MIN_OVERLAP_PX = 300
TEXTURE_MIN_QUALITY = 0.6                              # peak normalised correlation for a disc-pattern match to count
BELOW_MAX_FLOAT_PX, BELOW_DEPTH_PX = 40, 26            # floats end within 40 px below the line; the next 26 rows must be open water (_open_below)
HEAD_FROM_LEAD = 0.18                                 # the head sits ~18% of the box length behind its leading edge (pose, DIVE_P022: median 0.176)


def water_reference(img):
    """Median value, saturation and blueness of the water in this frame. The
    light differs a lot between clips (P022's navy floats read V~110 and
    saturated, P012's V~50 and nearly grey), so float colours are judged
    against it."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 60) & (hsv[..., 2] > 90) & (hsv[..., 0] >= 75) & (hsv[..., 0] <= 105)
    if m.sum() < 100:
        return 190.0, 130.0, 0.05
    f = img[m].astype(np.float32) + 1
    return (float(np.median(hsv[..., 2][m])), float(np.median(hsv[..., 1][m])),
            float(np.median((f[:, 0] - f[:, 1]) / f[:, 0])))


def float_masks(img, ref):
    """Navy, white and red float pixels, relative to the water. Navy = darker
    than the water AND bluer than it: shadows and the pool-floor line seen
    through the water are darker but keep the water's colour.
    White = pale and bright; kept tight because sun glare on the water is
    pale and bright too. Checked on hand-labelled pixels from DIVE_P022 and
    DIVE_P012: navy recall 0.39 / 0.67, water mistaken for navy 0.01 / 0.03,
    white recall 0.92 / 0.89, water mistaken for white 0.00 / 0.06."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.int16)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)
    f = img.astype(np.float32) + 1
    blue = (f[..., 0] - f[..., 1]) / f[..., 0]
    wv, ws, wb = ref
    red = ((H <= 10) | (H >= 150)) & (S > 100) & (V > 60)
    navy = (V < 0.78 * wv) & (blue > wb + 0.05) & ~red
    white = (S < 0.35 * ws) & (V > 0.85 * wv)
    return navy, white, red


def _smooth(a, n):
    return np.convolve(a, np.ones(n) / n, mode="same")


def _row_scores(nv, wt, rd, k):
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return (np.convolve(nv.mean(1), ker, "same"), np.convolve(wt.mean(1), ker, "same"),
            np.convolve(rd.mean(1), ker, "same"))


def _rope_like(n, w, r, allow_red=True):
    """Score per row and whether it looks like a rope: navy floats with white
    ones alongside -- the water is never navy, and sun glare alone reads only
    as white. A red stretch (the 5 m and 15 m markings) counts only while
    following a rope already found: the stands' red seats reflect in the
    water, so red on its own never starts a rope."""
    score = n + r + 0.3 * np.minimum(w, n)
    ok = (n >= 0.1) & (w >= 0.05)
    if allow_red:
        ok = ok | (r >= 0.4)
    return score, ok


def _open_below(band_from_line, coloured_from_line, blocks=CONTINUITY_BLOCKS, min_ok=CONTINUITY_MIN_OK,
                max_cover=0.10):
    """The rope we want is the one nearest the camera, so directly below it
    in the picture lies the swimmer's own lane: open water. A line cutting
    diagonally through the farther ropes -- which pack into a near-solid
    carpet of floats on a steep oblique view (DIVE_P010 at t=22/38/42 s) --
    scores as well as a real rope, but has floats below it too.
    `band_from_line` holds the float pixels (any colour) from the candidate
    line downward, `coloured_from_line` the navy and red ones only. The
    floats' own lower half is skipped first (it runs 10-25 px below the
    centre line depending on how close the rope is), then the next
    BELOW_DEPTH_PX rows are split into pieces, nearly all of which must be
    clear of navy and red floats. White is not counted there: the bow wave
    and splash ahead of the swimmer are white too (DIVE_P022's leading side
    failed this test in 6 of 14 sampled frames when it was)."""
    rows = band_from_line.mean(axis=1)
    end = next((k for k in range(len(rows)) if rows[k] < 0.3), None)
    if end is None or end > BELOW_MAX_FLOAT_PX:
        return False
    below = coloured_from_line[end + 4:end + 4 + BELOW_DEPTH_PX]
    if len(below) < BELOW_DEPTH_PX // 2:
        return True
    pieces = np.array_split(below.mean(axis=0), blocks)
    return sum(float(p.mean()) <= max_cover for p in pieces) >= min_ok


def _first_point(img, ref, x_edge, y_seed, direction):
    """The rope nearest the camera (lowest in the picture) near head height,
    beside the box edge, searched over a range of slopes. A real rope row
    along its own slope scores far higher than a line that cuts diagonally
    across two ropes, so only candidates scoring at least half the best one
    compete, and the lowest of those wins."""
    H, W = img.shape[:2]
    cands = []
    for off in FIRST_OFFSETS_PX:
        xa = int(x_edge + direction * off + (0 if direction > 0 else -FIRST_CHUNK_PX))
        xb = xa + FIRST_CHUNK_PX
        if xa < 0 or xb > W:
            continue
        pad = int(FIRST_SLOPES.max() * FIRST_CHUNK_PX / 2) + 2
        ya, yb = int(max(0, y_seed - SEARCH_UP_PX - pad)), int(min(H, y_seed + SEARCH_DOWN_PX + pad + BELOW_MAX_FLOAT_PX + 4 + BELOW_DEPTH_PX))
        if yb - ya < 2 * pad + 5:
            continue
        nv, wt, rd = float_masks(img[ya:yb, xa:xb], ref)
        # one array, one gather per slope: bit 1 navy, 2 white, 4 red
        packed = nv.astype(np.uint8) | (wt.astype(np.uint8) << 1) | (rd.astype(np.uint8) << 2)
        cols = np.arange(xb - xa) - (xb - xa) / 2
        rows = np.arange(yb - ya)
        cidx = np.broadcast_to(np.arange(xb - xa)[None, :], (len(rows), xb - xa))
        inside = (rows >= pad) & (rows < yb - ya - pad - BELOW_MAX_FLOAT_PX - 4 - BELOW_DEPTH_PX)
        reach = BELOW_MAX_FLOAT_PX + 4 + BELOW_DEPTH_PX
        for sl in FIRST_SLOPES:
            idx = np.clip(rows[:, None] + np.round(sl * cols).astype(int)[None, :], 0, yb - ya - 1)
            g = packed[idx, cidx]
            tot, ok = _rope_like(*_row_scores(g & 1, (g >> 1) & 1, (g >> 2) & 1, 4), allow_red=False)
            peak = np.zeros(len(tot), bool)
            peak[1:-1] = inside[1:-1] & ok[1:-1] & (tot[1:-1] >= tot[:-2]) & (tot[1:-1] >= tot[2:])
            for i in np.flatnonzero(peak):
                band = g[i:i + reach]
                if _open_below(band > 0, (band & 5) > 0):
                    cands.append(((xa + xb) / 2, ya + int(i), float(tot[i]), off, float(sl)))
    if not cands:
        return None
    top = max(c[2] for c in cands)
    cands = [c for c in cands if c[2] >= 0.5 * top]
    lowest = max(c[1] for c in cands)
    return max((c for c in cands if c[1] >= lowest - 8), key=lambda c: c[2])


def trace_rope(img, ref, x_edge, y_seed, direction, prior=None):
    """Follow the rope behind the swimmer outward from the box edge, chunk by
    chunk. With `prior` (the previous sample's curve on this side) each chunk
    is searched around that curve -- shifted by however far the camera has
    bobbed -- which keeps the choice of rope stable from one sample to the
    next; without it the starting point is searched for near head height.
    Returns (poly coefficients y(x), x_lo, x_hi) or None."""
    H, W = img.shape[:2]
    pts = []
    if prior is None:
        first = _first_point(img, ref, x_edge, y_seed, direction)
        if first is None:
            return None
        pts.append(first[:3])
        xc, slope = first[0], first[4]
    else:
        xc, slope = x_edge + direction * (CHUNK_PX / 2 - STEP_PX), 0.0
    bob, misses = 0.0, 0
    while True:
        xc += direction * STEP_PX
        if abs(xc - x_edge) > MAX_TRACE_PX:
            break
        xa, xb = int(xc - CHUNK_PX / 2), int(xc + CHUNK_PX / 2)
        if xa < 0 or xb > W:
            break
        good = [p for p in pts if p[2] > 0]
        if prior is not None:
            yp = float(np.polyval(prior, xc)) + bob
        elif good:
            yp = good[-1][1] + slope * (xc - good[-1][0])
        else:
            break
        ya, yb = int(max(0, yp - FOLLOW_TOL_PX)), int(min(H, yp + FOLLOW_TOL_PX + 1))
        if yb - ya < 5:
            break
        tot, ok = _rope_like(*_row_scores(*(m.astype(np.float32) for m in float_masks(img[ya:yb, xa:xb], ref)), 3))
        i = int(np.argmax(np.where(ok, tot, -1)))
        if not ok[i]:
            misses += 1
            if misses > 2:
                break
            continue
        misses = 0
        y = float(ya + i)
        pts.append((xc, y, float(tot[i])))
        if prior is not None:
            bob = float(np.median([p[1] - np.polyval(prior, p[0]) for p in pts[-4:]]))
        good = [p for p in pts if p[2] > 0]
        if len(good) >= 3:
            q = np.array(good[-5:])
            slope = float(np.polyfit(q[:, 0], q[:, 1], 1)[0])
    good = np.array([p for p in pts if p[2] > 0])
    if len(good) < 4:
        return None
    coef = np.polyfit(good[:, 0], good[:, 1], 2 if len(good) >= 7 else 1)
    lo, hi = int(max(0, good[:, 0].min() - CHUNK_PX / 2)), int(min(W, good[:, 0].max() + CHUNK_PX / 2))
    return coef, lo, hi


def rope_profile(img, coef, lo, hi, ref, half=PROFILE_HALF_PX):
    """Fraction of navy / white / red float pixels in a thin band along the rope, per column."""
    xs = np.arange(lo, hi)
    ys = np.polyval(coef, xs).round().astype(int)
    y0, y1 = max(0, ys.min() - half - 1), min(img.shape[0], ys.max() + half + 2)
    navy, white, red = float_masks(img[y0:y1, lo:hi], ref)
    cols = xs - lo
    nv = np.zeros(len(xs)); wt = np.zeros(len(xs)); rd = np.zeros(len(xs))
    for dy in range(-half, half + 1):
        yy = np.clip(ys + dy - y0, 0, navy.shape[0] - 1)
        nv += navy[yy, cols]; wt += white[yy, cols]; rd += red[yy, cols]
    k = 2 * half + 1
    return xs, nv / k, wt / k, rd / k


def rope_texture(img, coef, lo, hi, half=PROFILE_HALF_PX):
    """Brightness along the rope with its slow variation removed: the pattern
    of individual float discs. Unlike the navy/white bands it is there on the
    solid red end zones too, where there are no bands to count."""
    xs = np.arange(lo, hi)
    ys = np.polyval(coef, xs).round().astype(int)
    y0, y1 = max(0, ys.min() - half - 1), min(img.shape[0], ys.max() + half + 2)
    g = cv2.cvtColor(img[y0:y1, lo:hi], cv2.COLOR_BGR2GRAY).astype(np.float32)
    cols = xs - lo
    prof = np.zeros(len(xs), np.float32)
    for dy in range(-half, half + 1):
        prof += g[np.clip(ys + dy - y0, 0, g.shape[0] - 1), cols]
    prof /= 2 * half + 1
    return xs, prof - _smooth(prof, 41)


def texture_step(prev, cur, x_edge, direction, max_shift=TEXTURE_MAX_SHIFT_PX, reach=TEXTURE_REACH_PX):
    """Pixel shift d of the rope's disc pattern between two samples, such that
    cur(x) ~ prev(x - d), over the stretch of rope next to the swimmer; with
    the peak normalised correlation as its quality. (None, 0) if too little
    rope overlaps or the match is weak."""
    (xp, gp), (xc, gc) = prev, cur
    near_lo = x_edge - reach if direction < 0 else x_edge
    near_hi = x_edge if direction < 0 else x_edge + reach
    lo = max(xp[0], xc[0], near_lo) + max_shift
    hi = min(xp[-1], xc[-1], near_hi) - max_shift
    if hi - lo < TEXTURE_MIN_OVERLAP_PX:
        return None, 0.0
    c = gc[(xc >= lo) & (xc < hi)]
    c = c - c.mean()
    cn = float(np.sqrt((c * c).sum()))
    if cn == 0:
        return None, 0.0
    scores = np.full(2 * max_shift + 1, -1.0)
    for k, d in enumerate(range(-max_shift, max_shift + 1)):
        q = gp[(xp >= lo - d) & (xp < hi - d)]
        if len(q) != len(c):
            continue
        q = q - q.mean()
        den = cn * float(np.sqrt((q * q).sum()))
        if den > 0:
            scores[k] = float((c * q).sum()) / den
    k = int(np.argmax(scores))
    if scores[k] < TEXTURE_MIN_QUALITY or k in (0, len(scores) - 1):
        return None, float(max(scores[k], 0.0))
    a, b, cc = scores[k - 1], scores[k], scores[k + 1]
    den = a - 2 * b + cc
    return (k - max_shift) + (0.5 * (a - cc) / den if den else 0.0), float(b)


def _runs(mask, min_len, close_gap=0):
    m = mask.astype(np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    runs = [[a, b] for a, b in zip(e[::2], e[1::2])]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= close_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_len]


def navy_bands(xs, nv, wt, rd):
    """Centres of the navy bands along the rope, each with the local band
    period (centre-to-centre spacing) around it. The discs inside a band are
    merged first (gaps up to 14 px), so a band is never mistaken for several."""
    n, w, r = _smooth(nv, 9), _smooth(wt, 9), _smooth(rd, 9)
    runs = _runs((n > 0.3) & (n > w) & (n > r), 20, close_gap=14)
    runs = [(a, b) for a, b in runs if a > 0 and b < len(xs)]      # partial bands at the ends are unreliable
    if len(runs) < 2:
        return []
    c = np.array([(xs[a] + xs[b - 1]) / 2 for a, b in runs])
    width = np.array([b - a for a, b in runs], float)
    gaps = np.diff(c)
    out = []
    for j in range(len(c)):
        near = [g for g in (gaps[j - 1] if j > 0 else None, gaps[j] if j < len(gaps) else None) if g is not None]
        s = float(np.min(near))
        # a gap more than ~1.6x its neighbour is a missing band (splash, glare): not this band's period
        if len(near) == 2 and max(near) < 1.6 * min(near):
            s = float(np.mean(near))
        if s < 60 or not (0.2 * s <= width[j] <= 0.8 * s):
            continue
        out.append((float(c[j]), s))
    return out


def red_runs(xs, rd, min_px=40):
    return [(int(xs[a]), int(xs[b - 1]), a == 0, b == len(xs)) for a, b in _runs(_smooth(rd, 15) > 0.35, min_px)]


def ruler(bands, x_edge, direction):
    """Column -> rope repeats counted outward from the box edge, read off the
    navy bands on this side. Consecutive bands are one repeat apart, or more
    where one was missed (a gap ~2x the local spacing); outside the bands the
    nearest spacing extrapolates. Perspective is handled band by band, the
    same way the odometry handles it. None with fewer than two bands."""
    if len(bands) < 2:
        return None
    pts = sorted((direction * (c - x_edge), sp) for c, sp in bands)
    u = [q[0] for q in pts]
    sp = [q[1] for q in pts]
    ids = [u[0] / sp[0]]
    for j in range(1, len(u)):
        ids.append(ids[-1] + max(1, round((u[j] - u[j - 1]) / min(sp[j - 1], sp[j]))))

    def at(x):
        v = direction * (x - x_edge)
        if v <= u[0]:
            return ids[0] - (u[0] - v) / sp[0]
        if v >= u[-1]:
            return ids[-1] + (v - u[-1]) / sp[-1]
        return float(np.interp(v, u, ids))
    return at


def band_step(prev, cur):
    """Rope movement between two samples in repeats (navy+white; +ve = toward
    +x), as the median over navy bands of (movement / local navy-to-navy
    spacing). Returns (step, n_bands_matched)."""
    if len(prev) < 2 or len(cur) < 2:
        return None, 0
    pc = np.array([p[0] for p in prev])
    steps = []
    for c, s in cur:
        j = int(np.argmin(np.abs(pc - c)))
        d = c - pc[j]
        if abs(d) < 0.35 * s:
            steps.append(d / s)
    if len(steps) < 2:
        return None, len(steps)
    return float(np.median(steps)), len(steps)


def subject_boxes(tracks_df, config):
    """One box per frame for the filmed swimmer: the largest box among the
    subject's tracks (the subject can be split over several track ids)."""
    ids = subject_track_ids(tracks_df)
    d = tracks_df[tracks_df["track_id"].isin(ids)].copy()
    d["area"] = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
    return d.sort_values("area").groupby("frame").tail(1).set_index("frame").sort_index()


def run_rope_odometry(video_path, config, calibration_dir="data/calibration"):
    """Per sampled frame: rope movement on each side of the swimmer (repeats),
    the local band period next to the box, the box, and red rope sections."""
    cfg = config["rope_odometry"]
    tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
    boxes = subject_boxes(tracks_df, config)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / cfg["sample_hz"])))
    rows = []
    if boxes.empty:
        cap.release()
        return pd.DataFrame(rows), {"rope_samples": 0}
    f0, f1 = int(boxes.index.min()), int(boxes.index.max())
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    prev_bands = {"L": None, "R": None}
    prev_tex = {"L": None, "R": None}
    prior = {"L": None, "R": None}
    period_hist = {"L": [], "R": []}
    rejections = {"L": 0, "R": 0}
    for fr in range(f0, f1 + 1):
        if not cap.grab():
            break
        if (fr - f0) % stride:
            continue
        if fr not in boxes.index:
            prev_bands, prior = {"L": None, "R": None}, {"L": None, "R": None}
            prev_tex = {"L": None, "R": None}
            continue
        _, img = cap.retrieve()
        x1, y1, x2, y2 = (float(v) for v in boxes.loc[fr, ["x1", "y1", "x2", "y2"]])
        ref = water_reference(img[int(max(0, y2 - 380)):int(y2 + 10):4, ::8])
        y_seed = y2 - 0.4 * (y2 - y1)
        rec = {"frame": fr, "time_sec": fr / fps, "track_id": boxes.loc[fr, "track_id"],
               "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        for side, x_edge, direction in (("L", x1, -1), ("R", x2, 1)):
            tr = trace_rope(img, ref, x_edge, y_seed, direction, prior[side])
            if tr is None and prior[side] is not None:
                tr = trace_rope(img, ref, x_edge, y_seed, direction, None)
            if tr is None:
                prior[side], prev_bands[side], prev_tex[side] = None, None, None
                continue
            coef, lo, hi = tr
            xs, nv, wt, rd = rope_profile(img, coef, lo, hi, ref)
            # disc-pattern movement, in pixels: the fallback where bands can't be
            # matched (red end zones, bands hidden by splash or glare)
            tex = rope_texture(img, coef, lo, hi)
            if prev_tex[side] is not None:
                d_px, q = texture_step(prev_tex[side], tex, x_edge, direction)
                if d_px is not None:
                    rec[f"tex_px_{side}"], rec[f"tex_q_{side}"] = d_px, q
            prev_tex[side] = tex
            bands = navy_bands(xs, nv, wt, rd)
            if bands:
                nearest = min(bands, key=lambda b: abs(b[0] - x_edge))[1]
                # The band period next to the swimmer changes slowly (the camera's
                # distance and angle drift); a sudden jump means the trace has
                # wandered onto a farther rope (smaller bands) -- measured on
                # DIVE_P004 at t=14-16s: 110-145px against ~250px either side of
                # it. Drop the reading and search afresh next sample.
                # If the jumps persist, it was the history that was wrong (seeded
                # on a farther rope while the near one was hidden or red):
                # DIVE_P012's first 18 s were locked out that way. Start over.
                hist = period_hist[side]
                if len(hist) >= 5 and not (0.7 <= nearest / np.median(hist) <= 1.4):
                    rejections[side] += 1
                    if rejections[side] < 8:
                        prior[side], prev_bands[side], prev_tex[side] = None, None, None
                        rec.pop(f"tex_px_{side}", None)
                        rec.pop(f"tex_q_{side}", None)
                        rec[f"rejected_{side}"] = True
                        continue
                    hist = []
                rejections[side] = 0
                period_hist[side] = (hist + [nearest])[-15:]
                rec[f"period_px_{side}"] = nearest
                rec[f"bands_{side}"] = len(bands)
            prior[side] = coef
            if prev_bands[side] is not None:
                step, n = band_step(prev_bands[side], bands)
                rec[f"step_{side}"], rec[f"matched_{side}"] = step, n
            prev_bands[side] = bands
            reds = red_runs(xs, rd)
            if reds:
                rec[f"red_{side}"] = ";".join(f"{a}:{b}:{int(ca)}{int(cb)}" for a, b, ca, cb in reds)
                # Red sections of the rope are the pool's own distance markings
                # (pool_position): record each one's near and far ends in repeats
                # outward from the box edge, and whether either end ran off the
                # traced rope (then that end was not actually seen).
                r_at = ruler(bands, x_edge, direction)
                if r_at is not None:
                    marks = []
                    for a, b, clip_lo, clip_hi in reds:
                        near, far = (b, a) if direction < 0 else (a, b)
                        near_clip, far_clip = (clip_hi, clip_lo) if direction < 0 else (clip_lo, clip_hi)
                        marks.append(f"{r_at(near):.3f}:{r_at(far):.3f}:{int(near_clip)}{int(far_clip)}")
                    rec[f"marks_{side}"] = ";".join(marks)
        rows.append(rec)
    cap.release()
    df = pd.DataFrame(rows)
    diag = {"rope_samples": len(df),
            "rope_step_coverage_pct": float(100 * (df.filter(like="step_").notna().any(axis=1)).mean()) if len(df) else 0.0}
    return df, diag


def swimmer_progress(rope_df, config):
    """Integrate the rope measurements into the swimmer's progress along the
    pool. Adds: step_repeats (rope movement used, in repeats), head_x, progress_m (signed,
    +ve toward +x in the picture), measured (a real measurement at this
    sample rather than a bridged gap), and gap_before (continuity broken)."""
    cfg = config["rope_odometry"]
    if rope_df.empty:
        return rope_df
    df = rope_df.copy()
    for side in "LR":
        for col in (f"step_{side}", f"matched_{side}", f"period_px_{side}", f"tex_px_{side}", f"tex_q_{side}"):
            if col not in df:
                df[col] = np.nan
        # disc-pattern movement in repeats, with this side's recent band spacing
        # (it changes slowly, and carries over into a red zone from the bands
        # seen before it)
        side_period = df[f"period_px_{side}"].rolling(61, center=True, min_periods=1).median().ffill().bfill()
        df[f"tex_step_{side}"] = df[f"tex_px_{side}"] / side_period
    # a single reading far from its neighbours is a mismatch (splash over the
    # rope, a band lost to glare), not a change of speed: Hampel filter per side
    for side in "LR":
        col = df[f"step_{side}"]
        med = col.rolling(15, center=True, min_periods=5).median()
        bad = (col - med).abs() > np.maximum(0.015, 0.4 * med.abs())
        df.loc[bad, f"step_{side}"] = np.nan
    # both sides see the same rope moving: where both measured, they must agree
    both = df["step_L"].notna() & df["step_R"].notna()
    agree = (df["step_L"] - df["step_R"]).abs() <= cfg["side_agreement_repeats"]
    wl, wr = df["matched_L"].fillna(0), df["matched_R"].fillna(0)
    step = pd.Series(np.nan, index=df.index)
    step[both & agree] = ((df["step_L"] * wl + df["step_R"] * wr) / (wl + wr))[both & agree]
    only_l = df["step_L"].notna() & ~both
    only_r = df["step_R"].notna() & ~both
    step[only_l] = df.loc[only_l, "step_L"]
    step[only_r] = df.loc[only_r, "step_R"]
    step[step.abs() > cfg["max_step_repeats"]] = np.nan
    df["side_disagree"] = both & ~agree
    # where no band reading survived, fall back to the disc pattern (Hampel-
    # filtered the same way; both sides averaged when they agree)
    tex = pd.Series(np.nan, index=df.index)
    for side in "LR":
        col = df[f"tex_step_{side}"]
        med = col.rolling(15, center=True, min_periods=5).median()
        col = col.where((col - med).abs() <= np.maximum(0.015, 0.4 * med.abs()))
        df[f"tex_step_{side}"] = col
    tl, tr_ = df["tex_step_L"], df["tex_step_R"]
    tboth = tl.notna() & tr_.notna() & ((tl - tr_).abs() <= cfg["side_agreement_repeats"])
    tex[tboth] = ((tl + tr_) / 2)[tboth]
    tex[tl.notna() & tr_.isna()] = tl
    tex[tr_.notna() & tl.isna()] = tr_
    use_tex = step.isna() & tex.notna() & (tex.abs() <= cfg["max_step_repeats"])
    step[use_tex] = tex[use_tex]
    df["measured_texture"] = use_tex
    df["measured"] = step.notna()
    # heading from the rope's overall drift: rope moving +x means swimming toward -x
    heading = -1.0 if np.nanmedian(step) > 0 else 1.0
    w = df["x2"] - df["x1"]
    head = np.where(heading < 0, df["x1"] + HEAD_FROM_LEAD * w, df["x2"] - HEAD_FROM_LEAD * w)
    df["head_x"] = pd.Series(head, index=df.index).rolling(15, center=True, min_periods=1).median()
    period = df[["period_px_L", "period_px_R"]].mean(axis=1).rolling(31, center=True, min_periods=1).median()
    period = period.ffill().bfill()
    # Bridge short gaps at the neighbouring speed; break the measurement on long
    # ones. Two kinds: samples where the rope could not be read (the swimmer is
    # still in the picture), and stretches with no box at all (detection lost
    # the swimmer for a moment -- measured on DIVE_P022: one 0.13 s hole split
    # an otherwise continuous 30 s swim in two).
    dt = df["time_sec"].diff()
    max_gap = int(round(cfg["max_bridge_sec"] * cfg["sample_hz"]))
    missing = step.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    step_b = step.interpolate(limit_area="inside")
    step_b[missing & (run_len > max_gap)] = np.nan
    frame_gap = dt > 1.5 * dt.median()
    step_b[frame_gap] = np.nan          # the rope's movement across a hole in the detections is unknown
    df["period_px"] = period
    df["step_repeats"] = step_b
    move = df["head_x"].diff() / period - step_b          # repeats per sample
    speed = (move / dt).rolling(31, center=True, min_periods=5).median()
    bridge = move.isna() & frame_gap & (dt <= cfg["max_bridge_sec"])
    move[bridge] = (speed * dt)[bridge]
    df["gap_before"] = move.isna()
    df["progress_m"] = (move.fillna(0.0) * 2 * config["pool"]["rope_band_m"]).cumsum()
    df["heading"] = heading
    return df


def _marking_sightings(st, swum, cfg, rep_m, length, zone, ahead_side):
    """Every sighting of a red rope marking in one stretch, with the swimmer
    position it implies under each reading it allows. A long red run (or one
    running off the traced rope) can only be an end zone. A SHORT red run
    with both ends seen is ambiguous: the ~1 m mid-pool marker, or a partial
    glimpse of an end zone (broken up by splash or glare) -- measured at the
    start of DIVE_P012, where glimpses of the zone behind the swimmer read
    as "markers" and put the swimmer 20 m out. Both readings are kept here;
    _consensus picks. Returns rows (side, t, swum, reading, pos)."""
    out = []
    for i, r in st.iterrows():
        for side in "LR":
            marks = r.get(f"marks_{side}")
            if not isinstance(marks, str) or not marks:
                continue
            edge = r["x1"] if side == "L" else r["x2"]
            # repeats from the box edge back to the head. The rope is hidden
            # there, so the offset is converted with the band spacing just
            # outside the box -- which the wide-angle lens shrinks relative to
            # the middle of the picture. Measured over 2,207 sightings on eight
            # clips: that made this offset read 14% long, putting readings
            # behind the swimmer 0.1-1.0 m above readings ahead of it on every
            # clip. head_offset_scale corrects it.
            head_back = abs(r["head_x"] - edge) / r["period_px"] * cfg["head_offset_scale"]
            ahead = side == ahead_side
            zone_x = length - zone if ahead else zone
            for m in marks.split(";"):
                near, far, flags = m.split(":")
                near, far = float(near), float(far)
                if flags[0] == "1":
                    continue                         # its near end is hidden behind the swimmer
                span_m = (far - near) * rep_m
                readings = [(f"{zone:g} m zone {'ahead' if ahead else 'behind'}", zone_x, (near + head_back) * rep_m)]
                if flags[1] == "0" and span_m <= cfg["marker_max_len_m"]:
                    readings.append(("mid-pool marker", length / 2.0, ((near + far) / 2 + head_back) * rep_m))
                for name, x_mark, d in readings:
                    if d <= cfg["max_landmark_distance_m"]:
                        out.append((side, float(r["time_sec"]), float(swum[i]), name,
                                    x_mark - d if ahead else x_mark + d))
    return pd.DataFrame(out, columns=["side", "t", "swum", "reading", "pos"])


def _episodes(obs, cfg):
    """Group sightings into episodes (same side, contiguous in time) and,
    per reading, keep the episodes that behave like a marking FIXED on the
    rope: the start position each sighting implies (position minus distance
    swum) must stay constant. Something that travels with the camera
    instead -- red floats on a farther rope, a reflection -- drifts:
    measured on DIVE_P004, a red run that approached at 0.4 repeats/s while
    the swimmer covered 15 m. Returns rows (episode, reading, origin, n)."""
    out = []
    for side, g in obs.groupby("side"):
        g = g.sort_values("t")
        times = g["t"].drop_duplicates()
        ep_of_t = (times.diff() > cfg["episode_gap_sec"]).cumsum()
        g = g.assign(ep=g["t"].map(dict(zip(times, ep_of_t))))
        for (ep, reading), e in g.groupby(["ep", "reading"]):
            if len(e) < cfg["min_landmark_samples"]:
                continue
            origin = e["pos"] - e["swum"]
            if float(e["swum"].max() - e["swum"].min()) >= 1.0:
                if abs(float(np.polyfit(e["swum"], origin, 1)[0])) > cfg["max_episode_drift"]:
                    continue
            med = float(origin.median())
            if float((origin - med).abs().median()) > cfg["max_episode_scatter_m"]:
                continue
            out.append((f"{side}{ep}", reading, med, len(e), float(e["swum"].median())))
    return pd.DataFrame(out, columns=["episode", "reading", "origin", "n", "swum"])


def _consensus(eps, tol):
    """The start position most sightings agree on: each episode may be read
    one of its allowed ways; pick the value that the most sightings support
    (within tol) with one reading per episode, and each episode's reading
    closest to it."""
    best, best_support = None, -1
    for o in eps["origin"]:
        near = eps[(eps["origin"] - o).abs() <= tol]
        support = near.groupby("episode")["n"].max().sum()
        if support > best_support:
            best, best_support = o, support
    chosen = eps.assign(dev=(eps["origin"] - best).abs())
    chosen = chosen[chosen["dev"] <= tol].sort_values("dev").groupby("episode").head(1)
    return chosen


def pool_position(progress, config):
    """Where the swimmer is along the pool: metres into the length, measured
    from the wall they are swimming away from.

    The ropes carry the pool's own distance markings: red for the last
    `pool.rope_red_zone_m` at each end, and a short red marker at mid-pool.
    Each time one is in view on the traced rope, counting the bands between
    it and the swimmer's head places the swimmer exactly at that moment:
      - an end zone BEHIND the swimmer: its boundary is `zone` m into the length;
      - an end zone AHEAD: its boundary is `length - zone` m in;
      - the mid-pool marker: its centre is `length / 2` m in.
    The rope distance carries that position through the rest of the stretch.
    Consistent sighting episodes (_episodes) each give an estimate of where
    the stretch started, under each reading a red run allows; the value most
    sightings agree on wins (_consensus), and the agreeing episodes are
    averaged, weighted by sightings.

    The error estimate (`pool_position_err_m`, per moment) is honest about
    what was checked: the largest deviation of an agreeing substantial
    episode from the result (floor `base_uncertainty_m`), plus `drift_pct`
    of the distance swum outside the stretch the markings cover -- the rope
    distance is good to ~2-4%, so position error grows away from them.
    Checked on DIVE_P012's hand-measured positions: 0.5 m off where it
    claims +-0.9, 1.4 m off where it claims +-1.4. Positions are reported
    only where the error is within `max_uncertainty_m`.

    The scale is NOT refitted from the markings. Tried: two markings ~19 m
    apart (DIVE_P012: the zone behind and the mid-pool marker), each placed
    to +-0.5 m, pin the scale only to +-5% -- the fit came out +4.6% and
    moved P012's hand-checked end position from +1.0 m to +1.9 m off.

    Returns (progress with `pool_position_m`, list of per-stretch fixes)."""
    cfg = config["pool_position"]
    rep_m = 2 * config["pool"]["rope_band_m"]
    length, zone = config["pool"]["length_m"], config["pool"]["rope_red_zone_m"]
    df = progress.copy()
    df["pool_position_m"] = np.nan
    df["pool_position_err_m"] = np.nan
    if df.empty:
        return df, []
    heading = float(df["heading"].iloc[0])
    ahead_side = "L" if heading < 0 else "R"
    swum = heading * df["progress_m"]            # distance swum, increasing along the stretch
    stretch = df["gap_before"].cumsum()
    fixes = []
    for sid, st in df.groupby(stretch):
        obs = _marking_sightings(st, swum, cfg, rep_m, length, zone, ahead_side)
        eps = _episodes(obs, cfg) if len(obs) else pd.DataFrame()
        if eps.empty:
            continue
        chosen = _consensus(eps, cfg["consensus_tol_m"])
        w = chosen["n"].to_numpy(float)
        origin = float(np.average(chosen["origin"], weights=w))
        s0, s1 = float(swum[st.index].min()), float(swum[st.index].max())
        # error at each moment: how well the substantial episodes agree (a
        # handful of sightings is too noisy to set it), plus drift for the
        # distance swum outside the stretch they cover
        big = chosen[chosen["n"] >= cfg["substantial_episode_n"]]
        big = big if len(big) else chosen.nlargest(1, "n")
        agree = max(cfg["base_uncertainty_m"], float((big["origin"] - origin).abs().max()))
        lo_cov, hi_cov = float(big["swum"].min()), float(big["swum"].max())
        sw = swum[st.index]
        err = agree + cfg["drift_pct"] / 100.0 * np.maximum(lo_cov - sw, 0).clip(lower=0)             + cfg["drift_pct"] / 100.0 * np.maximum(sw - hi_cov, 0).clip(lower=0)
        pos = origin + sw
        # a position past a wall is impossible: its excess is error (e.g. the
        # rope distance wobbling backwards while the swimmer is still on the
        # wall before a push-off, DIVE_P027) and it is clipped to the wall
        excess = np.maximum(np.maximum(-pos, pos - length), 0.0)
        err = err + excess
        pos = pos.clip(0.0, float(length))
        ok = err <= cfg["max_uncertainty_m"]
        kinds = chosen.groupby("reading")["n"].sum()
        fixes.append({"stretch": int(sid), "origin_m": origin, "uncertainty_m": float(err[ok].max()) if ok.any() else float(err.min()),
                      "placed_pct": 100.0 * float(ok.mean()),
                      "markings": ", ".join(f"{k} (x{n})" for k, n in kinds.items()),
                      "episodes_used": len(chosen), "episodes_seen": int(eps["episode"].nunique())})
        df.loc[st.index[ok.to_numpy()], "pool_position_m"] = pos[ok]
        df.loc[st.index[ok.to_numpy()], "pool_position_err_m"] = err[ok]
    return df, fixes


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or os.path.join(config["cache"]["directory"], "rope")
    os.makedirs(cache_dir, exist_ok=True)
    upstream = os.path.splitext(os.path.basename(detect_cache_path(video_path, config)))[0]
    key = {"rope": config["rope_odometry"], "subject_min_frames": config["events"]["min_track_frames_for_analysis"]}
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"{upstream}__rope{h}.csv")


def run_rope_odometry_cached(video_path, config, calibration_dir="data/calibration"):
    path = cache_path(video_path, config)
    if config["cache"]["enabled"] and os.path.exists(path):
        print(f"Using cached rope odometry: {path}")
        return pd.read_csv(path), {"cached": True}
    df, diag = run_rope_odometry(video_path, config, calibration_dir)
    if config["cache"]["enabled"]:
        df.to_csv(path, index=False)
        print(f"Cached rope odometry: {path}")
    return df, diag


def main():
    parser = argparse.ArgumentParser(description="Measure the swimmer's progress against the lane rope.")
    parser.add_argument("video")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    timers = {}
    with Timer(timers, "time_rope_sec"):
        df, diag = run_rope_odometry_cached(args.video, config)
    prog = swimmer_progress(df, config)
    print(diag, f"{timers['time_rope_sec']:.1f}s")
    if not prog.empty:
        span = prog["time_sec"].iloc[-1] - prog["time_sec"].iloc[0]
        dist = abs(prog["progress_m"].iloc[-1] - prog["progress_m"].iloc[0])
        print(f"measured {100 * prog['measured'].mean():.0f}% of samples; {dist:.1f} m in {span:.1f} s "
              f"= {dist / span:.2f} m/s; sides disagreed at {int(prog['side_disagree'].sum())} samples")


if __name__ == "__main__":
    main()
