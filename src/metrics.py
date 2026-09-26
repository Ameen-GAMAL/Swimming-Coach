# Turns the stage outputs into the measurements the client asked for.
#
# Distance comes from the lane rope (src/rope_odometry.py): how many rope
# bands passed the swimmer. From it, per continuous stretch of the filmed
# swimmer -- and per short split for pacing -- the time, distance, average
# speed, stroke tempo and distance per cycle. Stroke timing comes from the
# recovery peaks (src/events.py). Per-track stroke-timing rows are kept for
# every tracked person, as before.
#
# This replaces per-length figures built from a calibration homography plus
# camera-motion compensation, which could not measure distance with a camera
# walking alongside the swimmer: every length came back rejected or covering
# a fraction of a metre (CLAUDE.md section 8.11).

import argparse
import os

import numpy as np
import pandas as pd

from src import __version__
from src.detect_track import run_detect_track_cached, subject_track_ids
from src.events import pick_tracked_hand, run_events_cached
from src.pose import run_pose_cached
from src.profiling import Timer, load_config
from src.rope_odometry import pool_position, run_rope_odometry_cached, swimmer_progress


def cycle_mean_confidence(pose_track_df, hand, start_frame, end_frame):
    window = pose_track_df[(pose_track_df["frame"] >= start_frame) & (pose_track_df["frame"] <= end_frame)]
    if window.empty:
        return None
    return float(window[f"{hand}_wrist_conf"].mean())


def compare_to_benchmark(value, bounds):
    if value is None or bounds is None:
        return None
    lo, hi = bounds
    if value < lo:
        return "below"
    if value > hi:
        return "above"
    return "inside"


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


def subject_strokes(events_df, pose_df, subject_ids, config):
    """Stroke records for the filmed swimmer across all of their tracks, in
    time order. Where two of the subject's tracks overlap in time (a brief
    double-track at an ID handoff) the same stroke can appear twice; a stroke
    closer than min_stroke_seconds to one already kept from another track is
    that duplicate."""
    rows = []
    if events_df.empty:
        return pd.DataFrame(rows)
    for tid, g in events_df[events_df["track_id"].isin(subject_ids)].groupby("track_id"):
        pose_track = pose_df[pose_df["track_id"] == tid] if not pose_df.empty else pose_df
        hand = pick_tracked_hand(pose_track) if not pose_track.empty else None
        ev = g[g["hand"] == hand] if hand is not None and "hand" in g else g
        for rec in stroke_records(ev, config):
            rec["track_id"] = tid
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("t_peak").reset_index(drop=True)
    keep, last_t, last_tid = [], None, None
    for r in df.itertuples():
        dup = (last_t is not None and r.track_id != last_tid
               and r.t_peak - last_t < config["events"]["min_stroke_seconds"])
        keep.append(not dup)
        if not dup:
            last_t, last_tid = r.t_peak, r.track_id
    return df[keep].reset_index(drop=True)


def tempo_in_window(strokes, t0, t1):
    """(strokes in [t0, t1], cycles/min) from the median half-cycle -- the
    same robust estimate as the per-track rows: an interval that implies a
    missed stroke is halved first."""
    if strokes.empty:
        return 0, None
    w = strokes[(strokes["t_peak"] >= t0) & (strokes["t_peak"] <= t1)]
    halves = w["half_cycle_sec"].dropna()
    if len(halves) < 2:
        return len(w), None
    return len(w), 60.0 / (2.0 * float(np.median(halves)))


def stroke_capture(n_timed, duration_sec, cpm):
    """Share of the strokes swum that were timed: timed recovery peaks over
    time x tempo x 2 (one stroke per arm per cycle). This is what the pose
    quality gate was standing in for (CLAUDE.md section 8.2 amendment)."""
    if not cpm or duration_sec <= 0:
        return None
    return min(100.0, 100.0 * n_timed / (duration_sec * cpm / 60.0 * 2))


def _position_at(st, t):
    p = st["pool_position_m"]
    if p.isna().all():
        return None
    ok = p.notna()
    return float(np.interp(t, st.loc[ok, "time_sec"], p[ok]))


def _position_err(st, t0, t1):
    e = st.loc[(st["time_sec"] >= t0) & (st["time_sec"] <= t1), "pool_position_err_m"].dropna()
    return float(e.max()) if len(e) else None


def swim_rows(progress, strokes, config, fixes=()):
    """One `swim` row per continuous rope-measured stretch of the filmed
    swimmer, and `split` rows every metrics.split_sec inside it. Where the
    red rope markings placed the stretch (pool_position), each row also
    carries where in the length it starts and ends."""
    fix_by_stretch = {f["stretch"]: f for f in fixes}
    cfg = config["metrics"]
    bench = config["benchmarks"]["freestyle"]
    rows = []
    if progress.empty:
        return rows
    stretch_id = progress["gap_before"].cumsum()
    n_stretch = 0
    for sid, st in progress.groupby(stretch_id):
        fix = fix_by_stretch.get(int(sid))
        if len(st) > 1:
            st = st.iloc[1:] if st["gap_before"].iloc[0] else st
        if len(st) < 2:
            continue
        t0, t1 = float(st["time_sec"].iloc[0]), float(st["time_sec"].iloc[-1])
        dur = t1 - t0
        if dur < cfg["min_stretch_sec"]:
            continue
        dist = abs(float(st["progress_m"].iloc[-1] - st["progress_m"].iloc[0]))
        speed = dist / dur
        measured = 100.0 * float(st["measured"].mean())
        n_str, cpm = tempo_in_window(strokes, t0, t1)
        dpc = speed * 60.0 / cpm if cpm else None
        reason = ""
        if measured < cfg["min_rope_measured_pct"]:
            reason = (f"the rope was read in only {measured:.0f}% of this stretch "
                      f"(minimum {cfg['min_rope_measured_pct']}%) -- too much of the distance would be bridged")
        row = {
            "row_type": "swim", "segment_index": n_stretch,
            "start_sec": t0, "end_sec": t1, "duration_sec": dur,
            "distance_m": None if reason else dist,
            "average_speed_ms": None if reason else speed,
            "stroke_count": n_str, "cycle_rate_cpm": cpm,
            "distance_per_cycle_m": None if reason else dpc,
            "distance_per_cycle_vs_benchmark": None if reason else compare_to_benchmark(dpc, bench["distance_per_cycle"]),
            "rope_measured_pct": measured,
            "stroke_capture_pct": stroke_capture(n_str, dur, cpm),
            "start_position_m": _position_at(st, t0), "end_position_m": _position_at(st, t1),
            "position_uncertainty_m": _position_err(st, t0, t1),
            "position_markings": fix["markings"] if fix else "",
            "rejection_reason": reason,
        }
        rows.append(row)
        n_stretch += 1
        if reason:
            continue
        # splits for pacing; a short remainder joins the last split
        edges = list(np.arange(t0, t1, cfg["split_sec"])) + [t1]
        if len(edges) > 2 and edges[-1] - edges[-2] < cfg["split_sec"] / 2:
            edges.pop(-2)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            sp = st[(st["time_sec"] >= a) & (st["time_sec"] <= b)]
            if len(sp) < 2:
                continue
            d = abs(float(sp["progress_m"].iloc[-1] - sp["progress_m"].iloc[0]))
            dt = float(sp["time_sec"].iloc[-1] - sp["time_sec"].iloc[0])
            n_s, c = tempo_in_window(strokes, a, b)
            if n_s < cfg["min_split_strokes"]:
                c = None
            v = d / dt if dt > 0 else None
            rows.append({
                "row_type": "split", "segment_index": row["segment_index"], "split_index": k,
                "start_sec": float(a), "end_sec": float(b), "duration_sec": dt,
                "distance_m": d, "average_speed_ms": v,
                "stroke_count": n_s, "cycle_rate_cpm": c,
                "distance_per_cycle_m": (v * 60.0 / c) if (v and c) else None,
                "rope_measured_pct": 100.0 * float(sp["measured"].mean()),
                "start_position_m": _position_at(st, float(a)), "end_position_m": _position_at(st, float(b)),
                "position_uncertainty_m": _position_err(st, float(a), float(b)),
                "rejection_reason": "",
            })
    return rows


def run_metrics(video_path, config, calibration_dir="data/calibration"):
    tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
    events_df, _ = run_events_cached(video_path, config, calibration_dir)
    pose_df, _ = run_pose_cached(video_path, config, calibration_dir)
    rope_df, _ = run_rope_odometry_cached(video_path, config, calibration_dir)
    video = os.path.basename(video_path)

    subject_ids = subject_track_ids(tracks_df)
    strokes = subject_strokes(events_df, pose_df, subject_ids, config)
    progress, fixes = pool_position(swimmer_progress(rope_df, config), config)
    rows = [dict(r, video=video) for r in swim_rows(progress, strokes, config, fixes)]

    # Per-track stroke timing for everyone tracked (time-based only).
    for track_id, _ in tracks_df.groupby("track_id"):
        pose_track_df = pose_df[pose_df["track_id"] == track_id] if not pose_df.empty else pose_df
        hand = pick_tracked_hand(pose_track_df) if not pose_track_df.empty else "left"
        track_events = (events_df[(events_df["track_id"] == track_id) & (events_df["hand"] == hand)]
                        if not events_df.empty else events_df.iloc[0:0])
        track_row = summarise_track_cycles(track_events, pose_track_df, hand, config)
        if track_row:
            track_row.update({"video": video, "track_id": track_id, "row_type": "track",
                              "subject": track_id in subject_ids})
            rows.append(track_row)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["pipeline_version"] = __version__
    swims = df_out[df_out["row_type"] == "swim"] if not df_out.empty else df_out
    diagnostics = {
        "result_swims_computed": int(len(swims)),
        "result_swims_accepted": int((swims["rejection_reason"] == "").sum()) if len(swims) else 0,
        "rope_measured_pct": float(100 * progress["measured"].mean()) if not progress.empty else 0.0,
        "rope_texture_pct": float(100 * progress["measured_texture"].mean()) if not progress.empty else 0.0,
        "rope_side_disagreements": int(progress["side_disagree"].sum()) if not progress.empty else 0,
        "position_fixes": fixes,
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
    swims = df[df["row_type"].isin(["swim", "split"])] if not df.empty else df
    if not swims.empty:
        pd.set_option("display.width", 160)
        print(swims[["row_type", "segment_index", "start_sec", "end_sec", "distance_m", "average_speed_ms",
                     "stroke_count", "cycle_rate_cpm", "distance_per_cycle_m", "rope_measured_pct",
                     "rejection_reason"]].round(2).to_string(index=False))
    else:
        print("No rope-measured swim in this clip -- nothing to report.")


if __name__ == "__main__":
    main()
