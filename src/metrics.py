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
    direction = "forward" if df["y_m_smoothed"].iloc[b] > df["y_m_smoothed"].iloc[a] else "backward"

    t1 = crossing_time(df, a, b, lo)
    t2 = crossing_time(df, a, b, hi)
    result = {"start_time_sec": start_time, "end_time_sec": end_time, "direction": direction}
    if t1 is None or t2 is None:
        return {**result, "rejection_reason": "did not cross both clean-zone boundaries"}

    zone_enter_time, zone_exit_time = min(t1, t2), max(t1, t2)
    clean_swim_time = zone_exit_time - zone_enter_time
    clean_distance_m = hi - lo

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

        if not (lo <= entry_i["y_m"] <= hi) or not (lo <= exit_i["y_m"] <= hi):
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
    q = config["quality"]
    reasons = []
    if result.get("rejection_reason"):
        reasons.append(result["rejection_reason"])
    if (result.get("cycle_count") or 0) < q["min_valid_cycles"]:
        reasons.append(f"fewer than {q['min_valid_cycles']} complete cycles in the clean zone "
                        f"(found {result.get('cycle_count', 0)})")
    if result.get("mean_wrist_confidence") is None or result["mean_wrist_confidence"] < q["min_confidence"]:
        conf = result.get("mean_wrist_confidence")
        reasons.append(f"mean wrist confidence {conf:.2f} below {q['min_confidence']}" if conf is not None
                        else "no confidence data for any counted cycle")
    if result.get("occluded"):
        reasons.append("a hand entry was not followed by a matching exit before the next entry (occluded)")

    if reasons:
        blanked = {col: None for col in MEASUREMENT_COLUMNS}
        return {**result, **blanked, "rejection_reason": "; ".join(reasons)}
    return {**result, "rejection_reason": ""}


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
    # A lap is one length there and one back (CLAUDE.md section 8.4). Pairs
    # consecutive lengths; an unpaired trailing length has no lap total.
    laps = []
    for i in range(0, len(length_rows) - 1, 2):
        first, second = length_rows[i], length_rows[i + 1]
        if first["rejection_reason"] or second["rejection_reason"]:
            laps.append({"length_indices": (i, i + 1), "rejection_reason": "one or both lengths in this lap were rejected"})
            continue
        total_time = first["clean_swim_time_sec"] + second["clean_swim_time_sec"]
        total_cycles = first["cycle_count"] + second["cycle_count"]
        total_distance = 2 * (config_clean_distance(first))
        laps.append({
            "length_indices": (i, i + 1),
            "clean_swim_time_sec": total_time,
            "average_speed_ms": total_distance / total_time if total_time > 0 else None,
            "cycle_count": total_cycles,
            "stroke_rate_cpm": (total_cycles / total_time * 60) if total_time > 0 and total_cycles else None,
            "distance_per_cycle_m": (total_distance / total_cycles) if total_cycles else None,
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

    min_swing_m = config["pool"]["lane_width_m"]  # a real turn moves further than a lane's width; jitter shouldn't
    rows = []
    for track_id, track_pos in tracks_df.groupby("track_id"):
        if events_df.empty or track_id not in events_df["track_id"].unique():
            continue
        track_events = events_df[events_df["track_id"] == track_id]
        pose_track_df = pose_df[pose_df["track_id"] == track_id]
        if pose_track_df.empty:
            continue
        hand = pick_tracked_hand(pose_track_df)

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
