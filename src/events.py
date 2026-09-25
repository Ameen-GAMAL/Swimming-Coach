# Finds each tracked swimmer's strokes: one "peak" event per arm recovery
# (the hand at its highest above the swimmer's own shoulder line), plus
# the "exit" and "entry" crossings of that line on either side of it where
# they were actually observed. Uses the better-tracked wrist channel only
# -- the pose model's left/right labels follow whichever arm is visible,
# so the two channels carry the same strokes. See detect_events_for_track
# for why peaks rather than line crossings, and CLAUDE.md section 8.3.

import argparse
import hashlib
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detect_track import project_point
from src.pose import cache_path as pose_cache_path
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
    """Picks whichever wrist is tracked better across the WHOLE track.
    Scores on summed confidence rather than the mean: pandas' mean skips
    missing values, so a wrist seen in a handful of frames at high
    confidence used to outrank one tracked steadily across hundreds."""
    left = float(track_df["left_wrist_conf"].fillna(0).sum())
    right = float(track_df["right_wrist_conf"].fillna(0).sum())
    return "left" if left >= right else "right"


def detect_events_for_track(track_df, hand, water_line_points, config, fps, H):
    """Strokes for one track, anchored on the RECOVERY PEAK -- the moment the
    hand is highest above the swimmer's own shoulder line.

    Why peaks, and why the shoulder line: on this footage the hand in the
    air is the one thing the pose model tracks well (high confidence, a
    large excursion above the shoulder), while the underwater pull is where
    the wrist is lost or guessed. Line-crossing detection therefore kept
    firing on the underwater guesses -- measured: 23 of 30 entry-to-exit
    intervals under 0.2s, and a "pull" shorter than the "recovery", the
    reverse of a real stroke -- and every fix that anchored on crossings
    traded false strokes for missed ones. A recovery peak is unambiguous:
    it is found with a prominence test on confident samples only, and the
    line crossings on either side of it are then read off as that stroke's
    exit (hand rising through the line) and entry (hand dropping through
    it), each only when actually observed.

    The calibrated water line is not used at all: calibrate.py derives it
    from the far lane rope, but the camera sits at water level so nearer
    water projects lower in the image -- measured on DIVE_P023 every wrist
    sample sat 15-281px below that line, giving zero crossings.

    One more thing the peaks reveal, measured on real footage: consecutive
    peaks alternate large / small at exactly half the arm-cycle period, and
    the model's "left" and "right" wrist channels show the same pattern at
    the same times. The model labels whichever arm is visible, not the
    anatomical side -- so consecutive peaks are ALTERNATE arms, each peak
    is one arm stroke, and the camera-side arm is the one that rises
    higher in the image. Peaks are classified near/far by height when the
    two groups separate clearly; a left/right split by label would be
    fiction.
    """
    from scipy.signal import find_peaks

    ev_cfg = config["events"]
    df = track_df.sort_values("frame").reset_index(drop=True)
    win = ev_cfg["smoothing_window"]

    ref_y = df[f"{hand}_wrist_ref_y"].rolling(win, center=True, min_periods=1).mean()
    ref_x = df[f"{hand}_wrist_ref_x"]
    conf = df[f"{hand}_wrist_conf"]
    shoulder_y = df[["left_shoulder_ref_y", "right_shoulder_ref_y"]].mean(axis=1)
    shoulder_y = shoulder_y.rolling(win, center=True, min_periods=1).mean()
    signal = ref_y - shoulder_y              # negative = wrist above the shoulder line
    # A sample counts only when the wrist was read with some confidence:
    # a low-confidence wrist is the model guessing at a hand that has gone
    # under (measured: confidence 0.1-0.3 right after entry, position
    # wobbling +/-20px around the line).
    known = (ref_y.notna() & shoulder_y.notna() & (conf >= ev_cfg["wrist_conf_threshold"])).to_numpy()

    plot = pd.DataFrame({"frame": df["frame"], "time_sec": df["time_sec"],
                         "wrist_ref_y_smoothed": ref_y, "water_line_y": shoulder_y, "wrist_conf": conf})
    idx = np.flatnonzero(known)
    if idx.size < 6:
        return [], plot
    t = df["time_sec"].to_numpy(dtype=float)[idx]
    s = signal.to_numpy(dtype=float)[idx]
    frames = df["frame"].to_numpy()[idx]
    height = -s                              # recovery peaks are maxima of this

    # Scale from the signal's own swing so it follows the swimmer through
    # perspective. band is the minimum height a recovery must reach above
    # the line, and the minimum prominence a peak needs.
    swing = float(np.percentile(s, 95) - np.percentile(s, 5))
    band = ev_cfg["crossing_hysteresis_frac"] * swing
    if swing <= 0:
        return [], plot

    # Peaks on the confident-sample sequence, then a minimum time separation
    # enforced greedily by prominence (samples are irregular in time, so
    # find_peaks' own distance argument -- in samples -- would be wrong).
    pk, props = find_peaks(height, height=band, prominence=band)
    order = np.argsort(-props["prominences"]) if pk.size else np.array([], dtype=int)
    kept = []
    min_sep = ev_cfg["min_stroke_seconds"]
    for k in order:
        if all(abs(t[pk[k]] - t[j]) >= min_sep for j in kept):
            kept.append(pk[k])
    kept.sort()
    if not kept:
        return [], plot

    max_gap = ev_cfg["max_phase_gap_seconds"]
    lookback = ev_cfg["max_crossing_seconds"]

    def crossing_before(p):
        """Last confident upward crossing of the line before peak p, if it
        was observed (both samples confident and close in time)."""
        for j in range(p, 0, -1):
            if t[p] - t[j] > lookback:
                return None
            if s[j - 1] >= 0 > s[j]:
                return j if (t[j] - t[j - 1]) <= max_gap else None
        return None

    def crossing_after(p):
        """First confident downward crossing after peak p, if observed."""
        for j in range(p + 1, len(s)):
            if t[j] - t[p] > lookback:
                return None
            if s[j - 1] < 0 <= s[j]:
                return j if (t[j] - t[j - 1]) <= max_gap else None
        return None

    # Near / far arm by peak height: two clear groups mean alternate arms.
    # The split is where it best separates two groups (Otsu: most variance
    # between them), not at the single largest gap between sorted heights --
    # a few in-between heights close that gap even when the alternation is
    # plain: measured on DIVE_P023 once pose found more of the swimmer, a
    # track reading 142, 78, 136, 73, 136, 70 ... with a handful of 90-110s
    # went wholly unclassified under the largest-gap rule.
    amp = np.array([height[p] for p in kept])
    arm = ["unclassified"] * len(kept)
    if len(kept) >= 4:
        sorted_amp = np.sort(amp)
        best_cut, best_var = None, -1.0
        for cut in range(2, len(sorted_amp) - 1):
            lo_, hi_ = sorted_amp[:cut], sorted_amp[cut:]
            var = len(lo_) * len(hi_) * (hi_.mean() - lo_.mean()) ** 2
            if var > best_var:
                best_cut, best_var = cut, var
        if best_cut is not None:
            low, high = sorted_amp[:best_cut], sorted_amp[best_cut:]
            if np.median(high) >= (1 + ev_cfg["arm_split_min_ratio"]) * np.median(low):
                thresh = (low.max() + high.min()) / 2
                arm = ["near" if a >= thresh else "far" for a in amp]

    def record(kind, j, extra=None):
        i = idx[j]
        x_m, y_m = project_point(H, ref_x.iloc[i], ref_y.iloc[i])
        rec = {
            "event_type": kind,
            "frame": int(df["frame"].iloc[i]),
            "time_sec": float(df["time_sec"].iloc[i]),
            "ref_x": float(ref_x.iloc[i]),
            "ref_y": float(ref_y.iloc[i]),
            "x_m": float(x_m),
            "y_m": float(y_m),
            "confidence": float(conf.iloc[i]),
            "observed_gap_sec": float(t[j] - t[j - 1]) if j > 0 else 0.0,
            "amplitude_px": None,
            "arm": None,
        }
        if extra:
            rec.update(extra)
        return rec

    events = []
    for k, p in enumerate(kept):
        ex = crossing_before(p)
        en = crossing_after(p)
        if ex is not None:
            events.append(record("exit", ex, {"arm": arm[k]}))
        events.append(record("peak", p, {"amplitude_px": float(height[p]), "arm": arm[k]}))
        if en is not None:
            events.append(record("entry", en, {"arm": arm[k]}))
    events.sort(key=lambda e: e["time_sec"])
    return events, plot


def save_wrist_signal_plot(signal, events, video_path, track_id, hand, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}__track{track_id}_wrist_signal.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(signal["time_sec"], signal["wrist_ref_y_smoothed"], label=f"{hand} wrist height (smoothed)", color="tab:blue")
    ax.plot(signal["time_sec"], signal["water_line_y"], label="shoulder line (reference)", color="tab:cyan", linestyle="--")
    ax.invert_yaxis()  # smaller y is higher in the frame -- flip so "up" reads as up

    for ev in events:
        color = {"entry": "tab:red", "exit": "tab:green"}.get(ev["event_type"], "tab:blue")
        ax.axvline(ev["time_sec"], color=color, alpha=0.4, linewidth=1)
    ax.plot([], [], color="tab:blue", label="recovery peak (stroke)")
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

        # One wrist channel only. The model's "left" and "right" wrist
        # outputs were measured to carry the same peaks at the same times --
        # it labels whichever arm is visible, not the anatomical side -- so
        # a second channel would only duplicate every stroke. The
        # better-tracked channel is used; near/far arm comes from peak
        # height (see detect_events_for_track), not from the label.
        hand = pick_tracked_hand(track_df)
        events, signal = detect_events_for_track(track_df, hand, water_line_points, config, fps, H)
        for ev in events:
            ev["track_id"] = track_id
            ev["lane"] = lane
            ev["hand"] = hand
        all_events.extend(events)
        plot_paths.append(save_wrist_signal_plot(signal, events, video_path, track_id, hand))

    # Explicit columns even when empty (zero events is a valid, honest
    # result -- e.g. no track had a long enough, clean crossing) so the
    # cached CSV still has a header row and can be read back rather than
    # round-tripping to a genuinely empty, unparseable file.
    event_columns = ["event_type", "frame", "time_sec", "ref_x", "ref_y", "x_m", "y_m",
                      "confidence", "observed_gap_sec", "amplitude_px", "arm", "track_id", "lane", "hand"]
    df = pd.DataFrame(all_events, columns=event_columns)
    entry_count = int((df["event_type"] == "entry").sum()) if not df.empty else 0
    diagnostics = {
        "event_entries_detected": entry_count,
        "event_strokes_detected": int((df["event_type"] == "peak").sum()) if not df.empty else 0,
        "event_exits_detected": int((df["event_type"] == "exit").sum()) if not df.empty else 0,
        "event_tracks_processed": pose_df["track_id"].nunique() - skipped_short_tracks,
        "event_tracks_skipped_too_short": skipped_short_tracks,
        "event_wrist_signal_plots": plot_paths,
    }
    return df, diagnostics


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or os.path.join(config["cache"]["directory"], "events")
    os.makedirs(cache_dir, exist_ok=True)
    # Previously keyed on the video stem ALONE, so every detection model,
    # pose model and events-threshold variation shared one cached result and
    # a re-run after tuning silently returned the old events.
    upstream = os.path.splitext(os.path.basename(pose_cache_path(video_path, config)))[0]
    ev_hash = hashlib.sha1(json.dumps(config["events"], sort_keys=True).encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"{upstream}__ev{ev_hash}.csv")


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
