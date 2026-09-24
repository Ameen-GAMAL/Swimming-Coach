# Turns detected hand-entry/exit events and tracked position into the
# measurements the client asked for: clean swim time, average speed,
# stroke rate, distance per cycle, and cycle-time variation -- computed
# per length (one traverse of the pool) and aggregated per lap (there and
# back), only counting cycles inside the clean zone. Applies the quality
# gate: never reports a number the data doesn't support.

import argparse
import os

import numpy as np
import pandas as pd

from src.detect_track import run_detect_track_cached
from src.events import pick_tracked_hand, run_events_cached
from src.pose import run_pose_cached
from src.profiling import Timer, load_config

MEASUREMENT_COLUMNS = [
    "clean_swim_time_sec", "average_speed_ms", "cycle_count",
    "stroke_rate_cpm", "distance_per_cycle_m", "cycle_time_cv_pct",
    "mean_wrist_confidence",
]


def detect_lengths(position_df, min_swing_m):
    """Splits one swimmer's continuous position into lengths (one traverse
    each) using a minimum-swing zigzag: a turn is only confirmed once the
    smoothed position has moved back at least min_swing_m from its most
    recent extreme, so ordinary tracking noise isn't mistaken for a wall
    turn. Returns (df, [(start_idx, end_idx), ...]) with df row-index
    ranges into the returned, position-only, frame-sorted dataframe."""
    df = position_df.dropna(subset=["y_m"]).sort_values("frame").reset_index(drop=True)
    if len(df) < 2:
        return df, []

    y = df["y_m"].rolling(9, center=True, min_periods=1).mean()
    df = df.assign(y_m_smoothed=y)

    turn_indices = [0]
    direction = None
    extremum_idx, extremum_val = 0, y.iloc[0]

    for i in range(1, len(y)):
        val = y.iloc[i]
        if direction is None:
            if val > extremum_val:
                direction = "up"
            elif val < extremum_val:
                direction = "down"
            extremum_val, extremum_idx = val, i
            continue

        if direction == "up":
            if val >= extremum_val:
                extremum_val, extremum_idx = val, i
            elif extremum_val - val >= min_swing_m:
                turn_indices.append(extremum_idx)
                direction, extremum_val, extremum_idx = "down", val, i
        else:
            if val <= extremum_val:
                extremum_val, extremum_idx = val, i
            elif val - extremum_val >= min_swing_m:
                turn_indices.append(extremum_idx)
                direction, extremum_val, extremum_idx = "up", val, i

    turn_indices.append(len(y) - 1)
    lengths = [(a, b) for a, b in zip(turn_indices[:-1], turn_indices[1:]) if b > a]
    return df, lengths


def crossing_time(df, a, b, target_y):
    # Interpolates the first time the (smoothed) position crosses target_y
    # within row range [a, b] -- used to find exactly when the swimmer
    # entered/left the clean zone, without quantising to a frame.
    y = df["y_m_smoothed"]
    for i in range(a, b):
        y0, y1 = y.iloc[i], y.iloc[i + 1]
        if (y0 - target_y) * (y1 - target_y) <= 0 and y0 != y1:
            t0, t1 = df["time_sec"].iloc[i], df["time_sec"].iloc[i + 1]
            frac = (target_y - y0) / (y1 - y0)
            return t0 + frac * (t1 - t0)
    return None


def cycle_mean_confidence(pose_track_df, hand, start_frame, end_frame):
    window = pose_track_df[(pose_track_df["frame"] >= start_frame) & (pose_track_df["frame"] <= end_frame)]
    if window.empty:
        return None
    return float(window[f"{hand}_wrist_conf"].mean())


def compute_length_metrics(df, a, b, events_df, pose_track_df, hand, config):
    zone_start_m = config["pool"]["clean_zone_start_m"]
    zone_end_m = config["pool"]["clean_zone_end_m"]
    lo, hi = min(zone_start_m, zone_end_m), max(zone_start_m, zone_end_m)

    start_time, end_time = df["time_sec"].iloc[a], df["time_sec"].iloc[b]
    y_a, y_b = df["y_m_smoothed"].iloc[a], df["y_m_smoothed"].iloc[b]
    direction = "forward" if y_b > y_a else "backward"
    result = {"start_time_sec": start_time, "end_time_sec": end_time, "direction": direction}

    # The full clean zone (5-20m) exists to skip the dive/push-off and the
    # wall-approach distortion -- but on real footage a continuously
    # tracked segment doesn't always span the whole thing (a short clip
    # can end, or tracking can drop, before the swimmer reaches the far
    # boundary). Rather than reject that outright, measure over whatever
    # portion of the clean zone this segment actually covers, and flag it
    # clearly rather than silently reporting a partial distance as if it
    # were the full 15m (see CLAUDE.md section 8.5 amendment).
    seg_lo, seg_hi = min(y_a, y_b), max(y_a, y_b)
    overlap_lo, overlap_hi = max(lo, seg_lo), min(hi, seg_hi)
    if overlap_hi <= overlap_lo:
        return {**result, "rejection_reason": "this length's tracked position never reaches the clean zone (5-20m)"}

    def boundary_time(target_y):
        t = crossing_time(df, a, b, target_y)
        if t is not None:
            return t
        # target_y sits at (or past) one end of the segment's own range --
        # no sign change to interpolate, so use whichever endpoint is closer.
        return start_time if abs(y_a - target_y) <= abs(y_b - target_y) else end_time

    t1, t2 = boundary_time(overlap_lo), boundary_time(overlap_hi)
    zone_enter_time, zone_exit_time = min(t1, t2), max(t1, t2)
    clean_swim_time = zone_exit_time - zone_enter_time
    clean_distance_m = overlap_hi - overlap_lo

    # A genuine partial-coverage window can still be short, but a human
    # swimmer physically cannot cover real pool distance in a few dozen
    # milliseconds -- that combination means the "length" is turn-detection
    # noise (a spurious zigzag from position jitter/a tracking
    # discontinuity), not a real partial observation. The full-crossing
    # requirement this replaces used to filter these out implicitly, since
    # noise essentially never spans the full 15m; the partial-coverage path
    # needs its own explicit sanity check instead (measured on real
    # footage: without it, noise segments produced "speeds" of 180-200 m/s).
    implied_speed = clean_distance_m / clean_swim_time if clean_swim_time > 0 else float("inf")
    max_speed = config["quality"]["max_plausible_speed_ms"]
    if clean_swim_time <= 0 or implied_speed > max_speed:
        return {**result, "rejection_reason": f"implied speed ({implied_speed:.1f} m/s) exceeds what's "
                f"physically plausible ({max_speed} m/s) -- likely a tracking discontinuity, not a real length"}

    partial_coverage = overlap_lo > lo or overlap_hi < hi
    result["partial_clean_zone_coverage"] = partial_coverage
    if partial_coverage:
        result["clean_zone_coverage_note"] = f"partial clean-zone coverage ({clean_distance_m:.1f}m of {hi - lo:.1f}m)"

    events_in_length = events_df[
        (events_df["time_sec"] >= start_time) & (events_df["time_sec"] <= end_time)
    ].sort_values("time_sec")
    entries = events_in_length[events_in_length["event_type"] == "entry"].reset_index(drop=True)
    exits = events_in_length[events_in_length["event_type"] == "exit"].reset_index(drop=True)

    cycle_times, cycle_confidences, occluded = [], [], False
    for i in range(len(entries) - 1):
        entry_i, entry_next = entries.iloc[i], entries.iloc[i + 1]
        exit_between = exits[(exits["time_sec"] > entry_i["time_sec"]) & (exits["time_sec"] < entry_next["time_sec"])]
        if exit_between.empty:
            occluded = True  # no matching exit before the next entry -- occluded, or the stroke straddles this length's boundary
            continue
        exit_i = exit_between.iloc[0]

        if not (overlap_lo <= entry_i["y_m"] <= overlap_hi) or not (overlap_lo <= exit_i["y_m"] <= overlap_hi):
            continue
        if not (zone_enter_time <= entry_i["time_sec"] <= zone_exit_time):
            continue

        conf = cycle_mean_confidence(pose_track_df, hand, entry_i["frame"], entry_next["frame"])
        cycle_times.append(entry_next["time_sec"] - entry_i["time_sec"])
        if conf is not None:
            cycle_confidences.append(conf)

    cycle_count = len(cycle_times)
    result.update({
        "clean_swim_time_sec": clean_swim_time,
        "average_speed_ms": clean_distance_m / clean_swim_time if clean_swim_time > 0 else None,
        "cycle_count": cycle_count,
        "stroke_rate_cpm": (cycle_count / clean_swim_time * 60) if clean_swim_time > 0 and cycle_count else None,
        "distance_per_cycle_m": (clean_distance_m / cycle_count) if cycle_count else None,
        "cycle_time_cv_pct": (float(np.std(cycle_times) / np.mean(cycle_times) * 100) if cycle_count >= 2 else None),
        "mean_wrist_confidence": float(np.mean(cycle_confidences)) if cycle_confidences else None,
        "occluded": occluded,
    })
    return result


def apply_quality_gate(result, config):
    # A hard rejection from compute_length_metrics (the tracked position
    # never reached the clean zone at all) blanks everything -- there's
    # nothing usable to report.
    if result.get("rejection_reason"):
        blanked = {col: None for col in MEASUREMENT_COLUMNS}
        return {**result, **blanked}

    # Past that point, speed and clean-swim-time come from position
    # tracking alone and don't need any stroke-cycle data to be valid --
    # gating them on cycle count/confidence would blank a perfectly good
    # measurement just because a *different* signal (hand-entry detection)
    # was insufficient. Only the cycle-derived numbers (stroke rate,
    # distance per cycle, their variability) need their own gate.
    q = config["quality"]
    notes = []
    cycle_count = result.get("cycle_count") or 0
    cycle_ok = cycle_count >= q["min_valid_cycles"]
    conf = result.get("mean_wrist_confidence")
    conf_ok = conf is not None and conf >= q["min_confidence"]

    out = dict(result)
    if not cycle_ok or not conf_ok:
        for col in ["stroke_rate_cpm", "distance_per_cycle_m", "cycle_time_cv_pct", "mean_wrist_confidence"]:
            out[col] = None
        if not cycle_ok:
            notes.append(f"fewer than {q['min_valid_cycles']} complete cycle(s) measured in the tracked window "
                         f"(found {cycle_count}) -- stroke-cycle metrics not reported")
        if not conf_ok:
            notes.append(f"mean wrist confidence {conf:.2f} below {q['min_confidence']}" if conf is not None
                         else "no confidence data for any counted cycle")

    if result.get("occluded"):
        notes.append("a hand entry was not followed by a matching exit before the next entry -- some cycles may be missing")
    if result.get("partial_clean_zone_coverage"):
        notes.append(result.get("clean_zone_coverage_note", "partial clean-zone coverage"))

    out["data_quality_note"] = "; ".join(notes)
    out["rejection_reason"] = ""
    return out


def compare_to_benchmark(value, bounds):
    if value is None or bounds is None:
        return None
    lo, hi = bounds
    if value < lo:
        return "below"
    if value > hi:
        return "above"
    return "inside"


def add_benchmark_comparisons(result, config):
    b = config["benchmarks"]["freestyle"]
    result["provisional_benchmarks"] = config["benchmarks"]["provisional"]
    result["stroke_rate_vs_50m_benchmark"] = compare_to_benchmark(result.get("stroke_rate_cpm"), b["stroke_rate_50m"])
    result["stroke_rate_vs_100m_benchmark"] = compare_to_benchmark(result.get("stroke_rate_cpm"), b["stroke_rate_100m"])
    result["distance_per_cycle_vs_benchmark"] = compare_to_benchmark(result.get("distance_per_cycle_m"), b["distance_per_cycle"])
    return result


def aggregate_laps(length_rows):
    # A lap is one length there and one back (CLAUDE.md section 8.4).
    #
    # Pairs each usable length with the next usable one going the OTHER way,
    # rather than pairing raw list positions. Position pairing was wrong on
    # real data: a rejected length still consumed a slot, so every later
    # pairing shifted by one and two unrelated lengths (sometimes both
    # running the same direction) could be summed into a single "lap".
    # An unpaired trailing length has no lap total.
    laps = []
    pending = None
    for i, row in enumerate(length_rows):
        if row["rejection_reason"] or row.get("average_speed_ms") is None:
            continue
        if pending is None:
            pending = (i, row)
            continue
        prev_i, first = pending
        if first["direction"] == row["direction"]:
            pending = (i, row)  # same way -- can't close a lap; start again from this one
            continue
        second = row
        pending = None
        total_time = first["clean_swim_time_sec"] + second["clean_swim_time_sec"]
        total_cycles = first["cycle_count"] + second["cycle_count"]
        # Each length may cover a different partial slice of the clean zone
        # (see the compute_length_metrics amendment) -- sum them separately
        # rather than assuming they match, so a lap made of two differently-
        # partial lengths still gets the right total distance.
        total_distance = config_clean_distance(first) + config_clean_distance(second)
        notes = [n for n in (first.get("data_quality_note"), second.get("data_quality_note")) if n]
        laps.append({
            "length_indices": (prev_i, i),
            "clean_swim_time_sec": total_time,
            "average_speed_ms": total_distance / total_time if total_time > 0 else None,
            "cycle_count": total_cycles,
            "stroke_rate_cpm": (total_cycles / total_time * 60) if total_time > 0 and total_cycles else None,
            "distance_per_cycle_m": (total_distance / total_cycles) if total_cycles else None,
            "data_quality_note": "; ".join(notes),
            "rejection_reason": "",
        })
    return laps


def config_clean_distance(length_row):
    # Recovers the clean-zone distance from a length's own numbers, so
    # aggregate_laps doesn't need config passed through separately.
    return length_row["average_speed_ms"] * length_row["clean_swim_time_sec"]


def stroke_records(track_events, config):
    """One record per detected stroke (recovery peak) for a track, in time
    order, with whatever timing could be read around it.

    Each peak is one ARM stroke. Consecutive peaks are alternate arms when
    both are detected -- but the far arm's smaller recovery is sometimes
    missed for a whole stretch, and then consecutive peaks are the SAME
    arm. Pairing by index (k -> k+2) is wrong in that regime, so the same
    arm's next stroke is found by TIME instead, using a physical prior: a
    freestyle half-cycle (one arm to the other) runs ~0.75-1.3s and a full
    arm cycle ~1.8-2.6s, so the first later peak at least
    `arm_cycle_min_seconds` away is this arm's next stroke whether or not
    the other arm was seen in between. Measured on DIVE_P004: with index
    pairing, a track whose far arm was never detected reported 29
    strokes/min (half the truth) and a 9% recovery share against 0.5s /
    1.8s phase figures that plainly say 21%.

    Per stroke:
      t_peak, arm ('near' / 'far' / 'unclassified'),
      interval_sec   -- to the next detected peak, either arm; None if last,
      half_cycle_sec -- interval_sec normalised to a half-cycle: halved when
                        it exceeds arm_cycle_min_seconds (a missed stroke),
      missed_after   -- True when interval_sec implies a missed stroke,
      arm_cycle_sec  -- to this arm's next stroke (time-based); None if none
                        within arm_cycle_max_seconds,
      recovery_sec   -- exit -> entry around this peak (hand above the
                        shoulder line, the well-observed phase),
      pull_sec       -- entry after this peak -> exit before this arm's
                        next stroke, i.e. the underwater phase,
      recovery_ok / pull_ok -- both boundaries observed, neither phase
                        implausibly short, no long unobserved gap."""
    if track_events.empty or "peak" not in set(track_events["event_type"]):
        return []
    ev = track_events.sort_values("time_sec").reset_index(drop=True)
    peaks = ev[ev["event_type"] == "peak"].reset_index(drop=True)
    cfg = config["events"]
    max_gap, min_phase = cfg["max_phase_gap_seconds"], cfg["min_phase_seconds"]
    cyc_min, cyc_max = cfg["arm_cycle_min_seconds"], cfg["arm_cycle_max_seconds"]

    def observed(row):
        return row is not None and float(row.get("observed_gap_sec", 0.0)) <= max_gap

    def exit_before(t, after=-1.0):
        cand = ev[(ev["event_type"] == "exit") & (ev["time_sec"] < t) & (ev["time_sec"] > after)]
        return cand.iloc[-1] if len(cand) else None

    def entry_after(t, before):
        cand = ev[(ev["event_type"] == "entry") & (ev["time_sec"] > t) & (ev["time_sec"] < before)]
        return cand.iloc[0] if len(cand) else None

    times = peaks["time_sec"].to_numpy(dtype=float)
    out = []
    for k in range(len(peaks)):
        pk = peaks.iloc[k]
        t = times[k]
        t_next = times[k + 1] if k + 1 < len(peaks) else None
        arm = pk.get("arm") if isinstance(pk.get("arm"), str) else "unclassified"
        interval = (t_next - t) if t_next is not None else None
        missed_after = interval is not None and interval >= cyc_min
        half = (interval / 2 if missed_after else interval) if interval is not None else None

        # this arm's next stroke, by time
        later = [j for j in range(k + 1, len(peaks)) if times[j] - t >= cyc_min]
        j_same = later[0] if later and times[later[0]] - t <= cyc_max else None
        arm_cycle = (times[j_same] - t) if j_same is not None else None

        prev_peak_t = times[k - 1] if k > 0 else -1.0
        ex = exit_before(t, after=prev_peak_t)
        en = entry_after(t, t_next if t_next is not None else float("inf"))
        recovery = (float(en["time_sec"]) - float(ex["time_sec"])) if (ex is not None and en is not None) else None
        recovery_ok = (recovery is not None and recovery >= min_phase and observed(ex) and observed(en))

        pull = None
        pull_ok = False
        if en is not None and j_same is not None:
            prev_of_same = times[j_same - 1] if j_same - 1 > k else float(en["time_sec"])
            ex2 = exit_before(times[j_same], after=max(prev_of_same, float(en["time_sec"])))
            if ex2 is not None:
                pull = float(ex2["time_sec"]) - float(en["time_sec"])
                pull_ok = pull >= min_phase and observed(en) and observed(ex2)
        out.append({
            "t_peak": t, "frame_peak": int(pk["frame"]), "arm": arm,
            "amplitude_px": float(pk["amplitude_px"]) if pd.notna(pk.get("amplitude_px")) else None,
            "interval_sec": interval, "half_cycle_sec": half, "missed_after": bool(missed_after),
            "arm_cycle_sec": arm_cycle,
            "recovery_sec": recovery, "recovery_ok": bool(recovery_ok),
            "pull_sec": pull, "pull_ok": bool(pull_ok),
        })
    return out


def _cv(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    if len(values) < 2 or np.mean(values) <= 0:
        return None
    return float(np.std(values) / np.mean(values) * 100)


def summarise_track_cycles(track_events, pose_track_df, hand, config):
    """Stroke timing over a whole track, all from timestamps: stroke count
    and rate, rhythm consistency, the recovery / pull split, and the
    near-side / far-side arm comparison. Carries none of the distance
    scale's error."""
    strokes = stroke_records(track_events, config)
    if len(strokes) < 2:
        return None
    halves = [s["half_cycle_sec"] for s in strokes if s["half_cycle_sec"] is not None]
    window = float(strokes[-1]["t_peak"] - strokes[0]["t_peak"])
    n_strokes = len(strokes)
    # Rate from the MEDIAN half-cycle interval, with any interval that
    # implies a missed stroke halved first: robust to a few misses (a
    # strokes-over-window figure was dragged from 55 to 45 by two gaps on
    # one 12-stroke track), and correct even when only one arm was seen.
    med_half = float(np.median(halves)) if halves else None
    spm = (60.0 / med_half) if med_half else None
    clean_halves = [h for h in halves if med_half and 0.6 * med_half <= h <= 1.5 * med_half]
    cycles = [s["arm_cycle_sec"] for s in strokes if s["arm_cycle_sec"] is not None]
    confs = []
    for a, b in zip(strokes[:-1], strokes[1:]):
        c = cycle_mean_confidence(pose_track_df, hand, a["frame_peak"], b["frame_peak"])
        if c is not None:
            confs.append(c)

    recs = [s["recovery_sec"] for s in strokes if s["recovery_ok"]]
    pulls = [s["pull_sec"] for s in strokes if s["pull_ok"]]
    rec_frac = [s["recovery_sec"] / s["arm_cycle_sec"] for s in strokes
                if s["recovery_ok"] and s["arm_cycle_sec"]]

    def arm_stats(name):
        sub = [s for s in strokes if s["arm"] == name]
        r = [s["recovery_sec"] for s in sub if s["recovery_ok"]]
        c = [s["arm_cycle_sec"] for s in sub if s["arm_cycle_sec"] is not None]
        return {
            f"{name}_strokes": len(sub),
            f"{name}_recovery_sec": float(np.mean(r)) if r else None,
            f"{name}_arm_cycle_sec": float(np.mean(c)) if c else None,
        }

    row = {
        "stroke_count": n_strokes,
        "clean_swim_time_sec": window,
        "stroke_rate_spm": spm,
        "cycle_rate_cpm": (spm / 2) if spm else None,
        "stroke_interval_cv_pct": _cv(clean_halves),
        "arm_cycle_cv_pct": _cv(cycles) if len(cycles) >= 3 else None,
        "strokes_missed_estimate": int(sum(1 for s in strokes if s["missed_after"])),
        "phase_strokes_measured": len(recs),
        "recovery_sec_mean": float(np.mean(recs)) if recs else None,
        "recovery_sec_cv_pct": _cv(recs),
        "pull_sec_mean": float(np.mean(pulls)) if pulls else None,
        "arm_cycle_sec_mean": float(np.mean(cycles)) if cycles else None,
        "recovery_fraction_mean": float(np.mean(rec_frac)) if rec_frac else None,
        "mean_wrist_confidence": float(np.mean(confs)) if confs else None,
        "rejection_reason": "",
        "data_quality_note": "stroke timing over the whole tracked window (time-based, "
                              "independent of the distance scale); distance-derived figures omitted",
    }
    row.update(arm_stats("near"))
    row.update(arm_stats("far"))
    return row


def run_metrics(video_path, config, calibration_dir="data/calibration"):
    tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
    events_df, _ = run_events_cached(video_path, config, calibration_dir)
    pose_df, _ = run_pose_cached(video_path, config, calibration_dir)

    min_swing_m = config["pool"]["min_turn_swing_m"]
    rows = []
    for track_id, track_pos in tracks_df.groupby("track_id"):
        # Hand events are OPTIONAL for a length to be worth reporting.
        # Speed and clean-swim-time come from position tracking alone, and
        # apply_quality_gate already blanks only the cycle-derived numbers
        # when there aren't enough cycles. Skipping the whole track when it
        # had no events (the previous behaviour) threw away a perfectly
        # good speed measurement and directly contradicted that gate's
        # stated design -- measured: it was the sole reason a clip whose
        # entry detection found nothing reported zero lengths rather than
        # position-only ones.
        no_events = events_df.iloc[0:0]
        pose_track_df = pose_df[pose_df["track_id"] == track_id] if not pose_df.empty else pose_df
        hand = pick_tracked_hand(pose_track_df) if not pose_track_df.empty else "left"
        # events.py now emits both arms; cycles are counted from the
        # better-tracked one only, so a stroke is never counted twice.
        track_events = (events_df[(events_df["track_id"] == track_id) & (events_df["hand"] == hand)]
                        if not events_df.empty else no_events)

        df, lengths = detect_lengths(track_pos, min_swing_m)
        if not lengths:
            continue

        length_rows = []
        for idx, (a, b) in enumerate(lengths):
            result = compute_length_metrics(df, a, b, track_events, pose_track_df, hand, config)
            result = apply_quality_gate(result, config)
            if not result["rejection_reason"]:
                result = add_benchmark_comparisons(result, config)
            result.update({"video": os.path.basename(video_path), "track_id": track_id,
                            "row_type": "length", "length_index": idx})
            length_rows.append(result)

        laps = aggregate_laps(length_rows)
        for lap in laps:
            if not lap["rejection_reason"]:
                lap = add_benchmark_comparisons(lap, config)
            lap.update({"video": os.path.basename(video_path), "track_id": track_id, "row_type": "lap"})
            length_rows.append(lap)

        # Per-track stroke timing, measured over the whole tracked window
        # instead of inside a clean-zone length.
        #
        # Stroke rate and cycle-time variability need only TIME, which the
        # 120fps timestamps give exactly -- unlike every distance-derived
        # number here, which inherits the unresolved scale problem. Gating
        # them on the clean zone throws away good data for a bad reason:
        # measured on DIVE_P023, 41 detected entries collapsed to a single
        # counted cycle purely because the spurious "lengths" each covered
        # 0.1-1.8m of the 15m zone.
        track_row = summarise_track_cycles(track_events, pose_track_df, hand, config)
        if track_row:
            track_row.update({"video": os.path.basename(video_path), "track_id": track_id,
                               "row_type": "track"})
            length_rows.append(track_row)

        rows.extend(length_rows)

    df_out = pd.DataFrame(rows)
    diagnostics = {
        "result_lengths_computed": int((df_out["row_type"] == "length").sum()) if not df_out.empty else 0,
        "result_lengths_accepted": int(((df_out["row_type"] == "length") & (df_out["rejection_reason"] == "")).sum()) if not df_out.empty else 0,
        "result_laps_computed": int((df_out["row_type"] == "lap").sum()) if not df_out.empty else 0,
    }
    return df_out, diagnostics


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or "outputs"
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(cache_dir, f"{stem}_metrics.csv")


def run_metrics_cached(video_path, config, calibration_dir="data/calibration"):
    path = cache_path(video_path, config)
    df, diagnostics = run_metrics(video_path, config, calibration_dir)
    df.to_csv(path, index=False)
    print(f"Saved metrics: {path}")
    return df, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Compute swim metrics from tracked position and hand events.")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    timers = {}
    with Timer(timers, "time_metrics_sec"):
        df, diagnostics = run_metrics_cached(args.video, config)

    print(f"Processed in {timers['time_metrics_sec']:.1f}s")
    print(diagnostics)
    if not df.empty:
        pd.set_option("display.width", 160)
        print(df[["row_type", "track_id", "length_index", "direction", "cycle_count",
                   "stroke_rate_cpm", "distance_per_cycle_m", "rejection_reason"]].to_string(index=False))
    else:
        print("No lengths detected -- nothing to report.")


if __name__ == "__main__":
    main()
