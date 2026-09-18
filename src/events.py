# Finds hand-entry and hand-exit moments for each tracked swimmer.
#
# Tracks one wrist per swimmer -- whichever one is detected more
# confidently on average, since with a side view the far arm is
# periodically hidden behind the body and only one hand can be tracked
# reliably. Detects a crossing using three signals together, not one:
# the wrist crossing the lane's water line, a confidence drop (entry) or
# rise (exit) right after -- confirming the hand actually went under or
# came back up -- and a minimum spacing between entries so noise isn't
# counted as a stroke.

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detect_track import project_point
from src.pose import run_pose_cached
from src.profiling import Timer, load_config


def load_calibration(video_path, calibration_dir="data/calibration"):
    import json
    stem = os.path.splitext(os.path.basename(video_path))[0]
    calib_path = os.path.join(calibration_dir, f"{stem}.json")
    if not os.path.exists(calib_path):
        raise RuntimeError(f"No calibration found for this video. Run calibrate.py first. Expected: {calib_path}")
    with open(calib_path) as f:
        return json.load(f)


def water_line_y_at_x(water_line_points, x):
    # Linear interpolation/extrapolation along the lane's water line --
    # perspective means the water surface isn't a flat horizontal line, so
    # the reference height depends on where along the pool the wrist is.
    (x1, y1), (x2, y2) = water_line_points
    if x2 == x1:
        return y1
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x - x1)


def pick_tracked_hand(track_df):
    left_conf = track_df["left_wrist_conf"].mean()
    right_conf = track_df["right_wrist_conf"].mean()
    return "left" if left_conf >= right_conf else "right"


def detect_events_for_track(track_df, hand, water_line_points, config, fps, H):
    ev_cfg = config["events"]
    df = track_df.sort_values("frame").reset_index(drop=True)

    ref_y_raw = df[f"{hand}_wrist_ref_y"]
    ref_y = ref_y_raw.rolling(ev_cfg["smoothing_window"], center=True, min_periods=1).mean()
    ref_x = df[f"{hand}_wrist_ref_x"]
    conf = df[f"{hand}_wrist_conf"]

    water_y = ref_x.apply(lambda x: water_line_y_at_x(water_line_points, x))
    # Smaller y = higher in the image = above the water surface (hand in the air)
    above = (ref_y < water_y).to_numpy()

    min_cycle_frames = ev_cfg["min_cycle_seconds"] * fps
    drop_window = ev_cfg["confidence_drop_window_frames"]
    wrist_conf_threshold = ev_cfg["wrist_conf_threshold"]

    def confirmed(i, want_drop):
        window = conf.iloc[i: min(len(df), i + drop_window)]
        return (window.min() < wrist_conf_threshold) if want_drop else (window.max() >= wrist_conf_threshold)

    # A state machine, not independent per-crossing detection: the hand can
    # only exit after it has entered, and can't enter again until it has
    # exited. Without this, jitter in the smoothed signal right at the
    # water line produces a burst of spurious exits for one real entry
    # (measured: 31 exits vs 8 entries before this fix, when they should
    # roughly pair up).
    # Initialized from the observed first sample, not hardcoded to "starts
    # surfaced" -- a track almost never happens to begin exactly at a real
    # entry/exit boundary, so it can just as easily start already
    # submerged. Measured: with a hardcoded False start, a track beginning
    # mid-underwater had its first (and sometimes only) real transition
    # silently discarded, since an "exit" can only register once
    # `underwater` is True.
    accepted = []
    underwater = not above[0]
    last_entry_frame = None
    for i in range(1, len(df)):
        frame_num = df["frame"].iloc[i]
        if not underwater and above[i - 1] and not above[i]:
            if last_entry_frame is not None and (frame_num - last_entry_frame) < min_cycle_frames:
                continue  # periodicity signal: too soon to be a real new cycle
            if confirmed(i, want_drop=True):
                accepted.append(("entry", i))
                underwater = True
                last_entry_frame = frame_num
        elif underwater and not above[i - 1] and above[i]:
            if confirmed(i, want_drop=False):
                accepted.append(("exit", i))
                underwater = False

    events = []
    for kind, i in accepted:
        x_m, y_m = project_point(H, ref_x.iloc[i], ref_y.iloc[i])
        events.append({
            "event_type": kind,
            "frame": int(df["frame"].iloc[i]),
            "time_sec": float(df["time_sec"].iloc[i]),
            "ref_x": float(ref_x.iloc[i]),
            "ref_y": float(ref_y.iloc[i]),
            "x_m": float(x_m),
            "y_m": float(y_m),
            "confidence": float(conf.iloc[i]),
        })

    signal = pd.DataFrame({
        "frame": df["frame"], "time_sec": df["time_sec"],
        "wrist_ref_y_smoothed": ref_y, "water_line_y": water_y, "wrist_conf": conf,
    })
    return events, signal


def save_wrist_signal_plot(signal, events, video_path, track_id, hand, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}__track{track_id}_wrist_signal.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(signal["time_sec"], signal["wrist_ref_y_smoothed"], label=f"{hand} wrist height (smoothed)", color="tab:blue")
    ax.plot(signal["time_sec"], signal["water_line_y"], label="water line", color="tab:cyan", linestyle="--")
    ax.invert_yaxis()  # smaller y is higher in the frame -- flip so "up" reads as up

    for ev in events:
        color = "tab:red" if ev["event_type"] == "entry" else "tab:green"
        ax.axvline(ev["time_sec"], color=color, alpha=0.4, linewidth=1)
    ax.plot([], [], color="tab:red", label="hand entry")
    ax.plot([], [], color="tab:green", label="hand exit")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pixel height in reference frame (inverted)")
    ax.set_title(f"{os.path.basename(video_path)} -- track {track_id} ({hand} wrist)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def run_events(video_path, config, calibration_dir="data/calibration"):
    pose_df, _ = run_pose_cached(video_path, config, calibration_dir)
    if pose_df.empty:
        raise RuntimeError("No pose data found -- nothing to detect events on.")

    calibration = load_calibration(video_path, calibration_dir)
    water_lines_by_lane = {wl["lane"]: wl["points"] for wl in calibration["water_lines"]}
    H = np.array(calibration["homography"])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    min_frames = config["events"]["min_track_frames_for_analysis"]
    all_events = []
    plot_paths = []
    skipped_short_tracks = 0
    for track_id, track_df in pose_df.groupby("track_id"):
        if len(track_df) < min_frames:
            skipped_short_tracks += 1
            continue

        lane = track_df["lane"].mode().iloc[0] if track_df["lane"].notna().any() else None
        water_line_points = water_lines_by_lane.get(lane, next(iter(water_lines_by_lane.values())))

        hand = pick_tracked_hand(track_df)
        events, signal = detect_events_for_track(track_df, hand, water_line_points, config, fps, H)
        for ev in events:
            ev["track_id"] = track_id
            ev["lane"] = lane
            ev["hand"] = hand
        all_events.extend(events)

        plot_path = save_wrist_signal_plot(signal, events, video_path, track_id, hand)
        plot_paths.append(plot_path)

    # Explicit columns even when empty (zero events is a valid, honest
    # result -- e.g. no track had a long enough, clean crossing) so the
    # cached CSV still has a header row and can be read back rather than
    # round-tripping to a genuinely empty, unparseable file.
    event_columns = ["event_type", "frame", "time_sec", "ref_x", "ref_y", "x_m", "y_m",
                      "confidence", "track_id", "lane", "hand"]
    df = pd.DataFrame(all_events, columns=event_columns)
    entry_count = int((df["event_type"] == "entry").sum()) if not df.empty else 0
    diagnostics = {
        "event_entries_detected": entry_count,
        "event_exits_detected": int((df["event_type"] == "exit").sum()) if not df.empty else 0,
        "event_tracks_processed": pose_df["track_id"].nunique() - skipped_short_tracks,
        "event_tracks_skipped_too_short": skipped_short_tracks,
        "event_wrist_signal_plots": plot_paths,
    }
    return df, diagnostics


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or os.path.join(config["cache"]["directory"], "events")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(cache_dir, f"{stem}.csv")


def run_events_cached(video_path, config, calibration_dir="data/calibration"):
    path = cache_path(video_path, config)
    if config["cache"]["enabled"] and os.path.exists(path):
        print(f"Using cached events result: {path}")
        return pd.read_csv(path), {"cached": True}

    df, diagnostics = run_events(video_path, config, calibration_dir)
    if config["cache"]["enabled"]:
        df.to_csv(path, index=False)
        print(f"Cached events result: {path}")
    return df, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Detect hand-entry/exit events for tracked swimmers.")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    timers = {}
    with Timer(timers, "time_events_sec"):
        df, diagnostics = run_events_cached(args.video, config)

    print(f"Processed in {timers['time_events_sec']:.1f}s")
    print(f"Entries detected: {diagnostics.get('event_entries_detected', 'n/a (cached)')}")
    print(f"Exits detected: {diagnostics.get('event_exits_detected', 'n/a (cached)')}")
    if "event_wrist_signal_plots" in diagnostics:
        for p in diagnostics["event_wrist_signal_plots"]:
            print(f"  wrist signal plot: {p}")


if __name__ == "__main__":
    main()
