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
        track_events = (events_df[events_df["track_id"] == track_id]
                        if not events_df.empty else no_events)
        pose_track_df = pose_df[pose_df["track_id"] == track_id] if not pose_df.empty else pose_df
        hand = pick_tracked_hand(pose_track_df) if not pose_track_df.empty else "left"

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
