# Builds the downloadable swim report: a Word document written for the
# coach, not the engineer. It collates what the other stages already
# produced -- nothing here computes a new measurement: distance, speed and
# distance per cycle (measured against the lane rope, src/rope_odometry.py),
# and the figures from the timing of the swimmer's strokes (tempo, rhythm,
# pacing, and the timing between the two arms). The duration of individual
# stroke phases is deliberately absent -- see CLAUDE.md section 8.10.
#
# All counts are derived from the cached stage outputs rather than the
# stages' live diagnostics dicts, so a report built from a cached run
# carries the same numbers as one built from a fresh run; the app's own
# summary uses summarise_run() for the same reason.
#
# Units: a STROKE is one arm's recovery (each recovery peak in the wrist
# signal). Consecutive strokes are alternate arms, so a full CYCLE (both
# arms) is two strokes, and cycles/min is half of strokes/min.

import argparse
import os
import re
import tempfile
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from src.detect_track import run_detect_track_cached, subject_track_ids
from src.events import load_calibration, run_events_cached
from src.metrics import run_metrics_cached, subject_strokes
from src.pose import run_pose_cached
from src.profiling import Timer, load_config

ANNOTATED_FRAME_FRACS = (0.2, 0.5, 0.8)
MIN_STROKES_STEADY = 3        # a segment needs this many strokes before its rhythm is rated
STEADY_CV_PCT = 12.0          # arm-cycle time variation at or below this reads as a steady rhythm
UNEVEN_CV_PCT = 20.0          # above this, worth a look at the video
PACING_CHANGE_PCT = 10.0      # tempo change between early and late segments worth remarking on
ARM_TIMING_DIFF_PCT = 10.0    # difference between the two half-cycles worth remarking on (also must exceed 2x its standard error)
MIN_ARM_PAIRS = 3             # arm-to-arm intervals needed in EACH direction before arm timing is compared
MIN_ALTERNATION_PCT = 75.0    # consecutive strokes are alternate arms; below this share of classified pairs alternating, the near/far labels are closer to a coin toss than to a measurement (DIVE_P012: 36-50%) and arm timing is not reported
MIN_STROKES_FOR_PICTURE = 8
SPEED_CHANGE_PCT = 8.0        # speed / distance-per-cycle change between the first and second half of the swim worth remarking on
PLAUSIBLE_SPM_BAND = (0.65, 1.5)  # a segment's tempo this far from the swimmer's own median is a detection artefact (double peaks, or every other stroke missed), not a real change of pace


def stem_of(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def session_info(video_path):
    """Swimmer id, stroke and session date, read from the clip's filename
    where it follows the dataset's DIVE_P0xx_FREE_ABOVE_YYYYMMDD pattern."""
    name = stem_of(video_path)
    swimmer = re.search(r"P(\d{3})", name)
    date = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    stroke = {"FREE": "Freestyle", "BACK": "Backstroke", "BREAST": "Breaststroke", "FLY": "Butterfly"}
    stroke_name = next((v for k, v in stroke.items() if f"_{k}_" in name), "Swim")
    return {
        "swimmer": f"Swimmer P{swimmer.group(1)}" if swimmer else name,
        "date": (datetime(int(date.group(1)), int(date.group(2)), int(date.group(3))).strftime("%d %b %Y")
                 if date else "date not in filename"),
        "stroke": stroke_name,
    }


def summarise_run(video_path, config, calibration_dir="data/calibration"):
    """Every number the report (and the app's summary) needs, derived from
    the stage outputs themselves so cached and fresh runs read identically."""
    tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
    pose_df, pose_diag = run_pose_cached(video_path, config, calibration_dir)
    events_df, _ = run_events_cached(video_path, config, calibration_dir)
    metrics_df, metrics_diag = run_metrics_cached(video_path, config, calibration_dir)

    cap = cv2.VideoCapture(video_path)
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames_detected = int(tracks_df["frame"].nunique()) if not tracks_df.empty else 0
    subject_ids = subject_track_ids(tracks_df, config["events"]["min_track_frames_for_analysis"])
    subject_df = tracks_df[tracks_df["track_id"].isin(subject_ids)] if not tracks_df.empty else tracks_df
    subject_frames = int(subject_df["frame"].nunique()) if not subject_df.empty else 0
    track_sizes = subject_df["track_id"].value_counts() if not subject_df.empty else pd.Series(dtype=int)
    longest_frames = int(track_sizes.max()) if len(track_sizes) else 0

    track_rows = (metrics_df[(metrics_df["row_type"] == "track") & metrics_df["track_id"].isin(subject_ids)].copy()
                  if not metrics_df.empty and "row_type" in metrics_df else pd.DataFrame())
    if not track_rows.empty:
        span = tracks_df.groupby("track_id")["frame"].agg(["min", "max"]) / fps
        track_rows["t_start"] = track_rows["track_id"].map(span["min"])
        track_rows["t_end"] = track_rows["track_id"].map(span["max"])
        track_rows = track_rows.sort_values("t_start").reset_index(drop=True)

    # the filmed swimmer's strokes, with a stroke seen by two overlapping
    # tracks of the same swimmer counted once (the metrics use the same list)
    strokes_df = subject_strokes(events_df, pose_df, subject_ids, config)

    def med(col, frame):
        return float(frame[col].dropna().median()) if len(frame) and frame[col].notna().any() else None

    multi = track_rows[track_rows["stroke_count"] >= MIN_STROKES_STEADY] if not track_rows.empty else pd.DataFrame()
    spm_all = track_rows["stroke_rate_spm"].dropna() if not track_rows.empty else pd.Series(dtype=float)
    # A segment whose tempo sits far outside the swimmer's own median is a
    # detection artefact, not pacing: double peaks read as ~100 strokes/min,
    # a stretch with every other stroke missed as ~half. Only plausible
    # segments feed the pacing and range statements; the rest are shown in
    # the table flagged "check".
    ref_spm = med("stroke_rate_spm", multi)
    if not track_rows.empty:
        lo, hi = PLAUSIBLE_SPM_BAND
        track_rows["plausible"] = (track_rows["stroke_rate_spm"].notna() & (ref_spm is not None)
                                   & (track_rows["stroke_rate_spm"] >= lo * (ref_spm or 0))
                                   & (track_rows["stroke_rate_spm"] <= hi * (ref_spm or 0)))
        multi = track_rows[(track_rows["stroke_count"] >= MIN_STROKES_STEADY) & track_rows["plausible"]]

    swims = (metrics_df[(metrics_df["row_type"] == "swim") & (metrics_df["rejection_reason"].fillna("") == "")]
             if not metrics_df.empty and "row_type" in metrics_df else pd.DataFrame())
    splits = (metrics_df[metrics_df["row_type"] == "split"].copy()
              if not metrics_df.empty and "row_type" in metrics_df else pd.DataFrame())
    swim_dist = float(swims["distance_m"].sum()) if len(swims) else None
    swim_time = float(swims["duration_sec"].sum()) if len(swims) else None
    swim_speed = swim_dist / swim_time if swim_dist and swim_time else None
    cpm_swim = (float(np.average(swims["cycle_rate_cpm"].dropna(),
                                 weights=swims.loc[swims["cycle_rate_cpm"].notna(), "duration_sec"]))
                if len(swims) and swims["cycle_rate_cpm"].notna().any() else None)
    calib = load_calibration(video_path, calibration_dir)
    return {
        "video": os.path.basename(video_path),
        "frames_total": frames_total, "fps": fps, "duration_sec": frames_total / fps,
        "resolution": f"{width}x{height}",
        "frames_detected": frames_detected,
        "coverage_pct": 100.0 * frames_detected / frames_total if frames_total else 0.0,
        "subject_coverage_pct": 100.0 * subject_frames / frames_total if frames_total else 0.0,
        "subject_tracks": len(subject_ids),
        "subject_ids": subject_ids,
        "other_people_tracks": int(tracks_df["track_id"].nunique() - len(subject_ids)) if not tracks_df.empty else 0,
        "detections": int(len(tracks_df)),
        "tracks": int(tracks_df["track_id"].nunique()) if not tracks_df.empty else 0,
        "longest_track_frames": longest_frames, "longest_track_sec": longest_frames / fps,
        "pose_rows": int(len(pose_df)),
        "wrist_visible_pct": float(pose_diag.get("quality_wrist_visible_pct", 0.0)),
        "entries": int(((events_df["event_type"] == "entry") & events_df["track_id"].isin(subject_ids)).sum())
                   if not events_df.empty else 0,
        "exits": int(((events_df["event_type"] == "exit") & events_df["track_id"].isin(subject_ids)).sum())
                 if not events_df.empty else 0,
        "strokes_total": int(len(strokes_df)),
        "tracks_with_strokes": int(len(track_rows)),
        "tracks_multi": int(len(multi)),
        "median_spm": float(spm_all.median()) if len(spm_all) else None,
        "median_spm_multi": med("stroke_rate_spm", multi),
        "median_cv_multi": med("arm_cycle_cv_pct", multi) if len(multi) and multi["arm_cycle_cv_pct"].notna().any()
                           else med("stroke_interval_cv_pct", multi),
        "phase_strokes": int(track_rows["phase_strokes_measured"].sum()) if not track_rows.empty else 0,
        "swims_computed": int((metrics_df["row_type"] == "swim").sum()) if not metrics_df.empty else 0,
        "swims_accepted": int(len(swims)),
        "swim_distance_m": swim_dist, "swim_time_sec": swim_time, "swim_speed_ms": swim_speed,
        "swim_cycle_rate_cpm": cpm_swim,
        "swim_dpc_m": (swim_speed * 60.0 / cpm_swim) if swim_speed and cpm_swim else None,
        # ONE tempo figure for the headline everywhere: over the measured swim
        # when there is one (the same window its distance per cycle uses),
        # else the median over tracked segments. Showing both side by side
        # read as a contradiction (27 vs 28 cycles/min on DIVE_P022).
        "headline_cpm": cpm_swim if cpm_swim is not None else (
            med("stroke_rate_spm", multi) / 2 if med("stroke_rate_spm", multi) is not None else None),
        "rope_measured_pct": float(metrics_diag.get("rope_measured_pct", 0.0)),
        "stroke_capture_pct": (float(np.average(swims["stroke_capture_pct"].dropna(),
                                                weights=swims.loc[swims["stroke_capture_pct"].notna(), "duration_sec"]))
                               if len(swims) and "stroke_capture_pct" in swims and swims["stroke_capture_pct"].notna().any()
                               else None),
        "person_found_pct": float(pose_diag.get("quality_person_found_pct", 0.0)),
        "rope_side_disagreements": int(metrics_diag.get("rope_side_disagreements", 0)),
        "splits_df": splits,
        "scale_method": calib.get("scale_method", "unknown"),
        "visible_span_m": float(calib.get("corner_span_m", {}).get("end_m", 0.0)),
        "scale_clamped": bool(calib.get("scale_estimate_clamped_to_prior", False)),
        "metrics_df": metrics_df, "track_rows": track_rows, "strokes_df": strokes_df,
    }


def rated_segments(rows):
    """Segments with enough strokes and a tempo inside the swimmer's own
    range -- the only ones pacing, range and rhythm statements draw on."""
    if rows.empty:
        return rows
    ok = rows["stroke_count"] >= MIN_STROKES_STEADY
    if "plausible" in rows:
        ok &= rows["plausible"]
    return rows[ok]


def rhythm_label(cv, strokes):
    if strokes < MIN_STROKES_STEADY or cv is None or (isinstance(cv, float) and np.isnan(cv)):
        return "too few strokes"
    if cv <= STEADY_CV_PCT:
        return "steady"
    if cv <= UNEVEN_CV_PCT:
        return "mostly steady"
    return "uneven"


def segment_cv(row):
    cv = row.get("arm_cycle_cv_pct")
    if cv is None or (isinstance(cv, float) and np.isnan(cv)):
        cv = row.get("stroke_interval_cv_pct")
    return cv


def pacing_trend(multi):
    """Early-vs-late tempo across the tracked swim, from segments with
    enough strokes. Returns (first_spm, last_spm, pct_change) or None."""
    if len(multi) < 2:
        return None
    first, last = multi.iloc[0], multi.iloc[-1]
    if first["t_start"] == last["t_start"] or not first["stroke_rate_spm"]:
        return None
    change = 100.0 * (last["stroke_rate_spm"] - first["stroke_rate_spm"]) / first["stroke_rate_spm"]
    return float(first["stroke_rate_spm"]), float(last["stroke_rate_spm"]), float(change)


def speed_trend(splits):
    """First half of the swim against the second: speed, tempo and distance
    per cycle, each averaged over the splits that have it. None if there are
    fewer than two splits."""
    if splits is None or len(splits) < 2:
        return None
    sp = splits.sort_values("start_sec")
    half = len(sp) // 2
    a, b = sp.iloc[:half], sp.iloc[half:]

    def mean(df, col):
        v = df[col].dropna()
        return float(v.mean()) if len(v) else None

    out = {}
    for col in ("average_speed_ms", "cycle_rate_cpm", "distance_per_cycle_m"):
        x, y = mean(a, col), mean(b, col)
        out[col] = (x, y, 100.0 * (y - x) / x if x and y else None)
    return out


def arm_timing(strokes_df):
    """Timing between the two arms: the interval from a camera-side stroke
    to the next far-side stroke, against the interval from a far-side
    stroke back to the camera-side one. In an even stroke the two
    half-cycles match; a consistently longer one is the classic "gallop",
    often tied to breathing to one side.

    Built from stroke TIMES only, which is what this pipeline measures
    reliably -- the same peaks gave tempo that agreed to within 2 cycles/min
    across three different swimmers. Measured on DIVE_P022 / P023: each
    direction's intervals scattered by 8-25% over 5-24 pairs, precise
    enough to tell even from uneven. Returns None unless both directions
    have MIN_ARM_PAIRS intervals (on DIVE_P004 the arms mostly could not be
    told apart, so it reports nothing rather than guess)."""
    if strokes_df.empty or "arm" not in strokes_df:
        return None
    nf, fn, same = [], [], 0
    for _, g in strokes_df.groupby("track_id"):
        g = g.sort_values("t_peak")
        for a, b in zip(g.itertuples(), list(g.itertuples())[1:]):
            if a.interval_sec is None or np.isnan(a.interval_sec) or a.missed_after:
                continue
            if a.arm == "near" and b.arm == "far":
                nf.append(a.interval_sec)
            elif a.arm == "far" and b.arm == "near":
                fn.append(a.interval_sec)
            elif a.arm == b.arm and a.arm in ("near", "far"):
                same += 1
    classified_pairs = len(nf) + len(fn) + same
    if classified_pairs == 0 or 100.0 * (len(nf) + len(fn)) / classified_pairs < MIN_ALTERNATION_PCT:
        return None
    if len(nf) < MIN_ARM_PAIRS or len(fn) < MIN_ARM_PAIRS:
        return None
    nf, fn = np.array(nf), np.array(fn)
    near_to_far, far_to_near = float(np.median(nf)), float(np.median(fn))
    diff = near_to_far - far_to_near
    pct = 100.0 * diff / ((near_to_far + far_to_near) / 2)
    se = float(np.sqrt(np.var(nf) / len(nf) + np.var(fn) / len(fn)))
    return {"near_to_far": near_to_far, "far_to_near": far_to_near, "n_near_to_far": len(nf),
            "n_far_to_near": len(fn), "diff_pct": pct,
            "uneven": abs(pct) >= ARM_TIMING_DIFF_PCT and abs(diff) > 2 * se}


def observations(s, config):
    """Coach-facing observations generated from the measured tempo, rhythm,
    pacing and arm timing. Factual first, then what is worth checking on
    the video; the coach's own judgement of technique takes precedence."""
    rows = s["track_rows"]
    multi = rated_segments(rows) if not rows.empty else pd.DataFrame()
    bench = config["benchmarks"]["freestyle"]
    out = []

    # distance and speed
    if s["swim_speed_ms"] is not None:
        line = (f"The swimmer covered {s['swim_distance_m']:.1f} m in {s['swim_time_sec']:.1f} s of continuous "
                f"swimming, an average of {s['swim_speed_ms']:.2f} m/s")
        if s["swim_dpc_m"] is not None:
            line += (f", travelling {s['swim_dpc_m']:.2f} m with each full stroke cycle at "
                     f"{s['swim_cycle_rate_cpm']:.0f} cycles per minute")
        out.append(line + ".")
        tr = speed_trend(s["splits_df"])
        if tr and tr["average_speed_ms"][2] is not None:
            v0, v1, dv = tr["average_speed_ms"]
            d0, d1, dd = tr["distance_per_cycle_m"]
            c0, c1, dc = tr["cycle_rate_cpm"]
            if abs(dv) < SPEED_CHANGE_PCT:
                out.append(f"Speed held steady: {v0:.2f} m/s over the first half and {v1:.2f} m/s over the second "
                           f"({dv:+.0f}%).")
            elif dv < 0:
                why = ""
                if dd is not None and dd <= -SPEED_CHANGE_PCT and (dc is None or abs(dc) < SPEED_CHANGE_PCT):
                    why = (f" The tempo held ({c0:.0f} to {c1:.0f} cycles/min) while each cycle carried less "
                           f"distance ({d0:.2f} to {d1:.2f} m) -- the stroke itself got shorter, the usual sign "
                           "of a tiring catch or a shortened finish.")
                elif dc is not None and dc <= -SPEED_CHANGE_PCT:
                    why = f" The tempo dropped too ({c0:.0f} to {c1:.0f} cycles/min)."
                out.append(f"Speed faded from {v0:.2f} to {v1:.2f} m/s between the first and second half "
                           f"({dv:+.0f}%).{why}")
            else:
                out.append(f"Speed built from {v0:.2f} to {v1:.2f} m/s between the first and second half "
                           f"({dv:+.0f}%).")

    if s["strokes_total"] == 0:
        return out + ["No stroke could be timed in this clip, so there is no tempo picture to report. A longer "
                      "clip, or one where the swimmer stays in view for more of the length, would give one."]

    # tempo -- benchmarks.freestyle is in CYCLES/min (both arms), so the
    # comparison converts our strokes/min (one arm) down by half rather
    # than comparing spm against a cpm range directly. Measured bug: a
    # 56 strokes/min segment (28 cycles/min, clearly distance-style) was
    # being matched against the 55-70 "sprint" range and reported as one.
    if s["headline_cpm"] is not None and len(multi):
        cpm = s["headline_cpm"]
        spm = 2 * cpm
        lo, hi = multi["stroke_rate_spm"].min() / 2, multi["stroke_rate_spm"].max() / 2
        r50, r100 = bench["stroke_rate_50m"], bench["stroke_rate_100m"]
        if r50[0] <= cpm <= r50[1]:
            context = "a sprint-range tempo"
        elif r100[0] <= cpm <= r100[1]:
            context = "in the range usually seen over 100 m"
        elif cpm < r100[0]:
            context = "a distance-style tempo, longer and slower than a sprint stroke"
        else:
            context = "a very high tempo"
        out.append(f"Stroke tempo averaged {cpm:.0f} full cycles per minute ({spm:.0f} strokes per minute, "
                   f"counting each arm); tracked segments with {MIN_STROKES_STEADY}+ strokes ran "
                   f"{lo:.0f}–{hi:.0f} cycles/min -- {context}.")
    elif s["median_spm"] is not None:
        out.append(f"Only short segments were captured, at around {s['median_spm']:.0f} strokes per minute. "
                   "Treat this as a snapshot of tempo rather than a measurement of the whole swim.")

    # rhythm
    if len(multi):
        cvs = multi.apply(segment_cv, axis=1)
        rated = multi[cvs.notna()]
        steady = rated[cvs[rated.index] <= STEADY_CV_PCT]
        uneven = rated[cvs[rated.index] > UNEVEN_CV_PCT]
        if len(rated):
            uneven_part = ("none were uneven" if not len(uneven)
                           else f"{len(uneven)} {'was' if len(uneven) == 1 else 'were'} uneven")
            out.append(f"Rhythm was steady in {len(steady)} of the {len(rated)} rated segments (arm-cycle time "
                       f"varying by {STEADY_CV_PCT:.0f}% or less from stroke to stroke); {uneven_part}.")
        if len(uneven):
            spans = ", ".join(f"{r.t_start:.0f}–{r.t_end:.0f} s ({cvs[i]:.0f}%)" for i, r in zip(uneven.index, uneven.itertuples()))
            out.append(f"The uneven stretch{'es' if len(uneven) > 1 else ''}: {spans} -- one or more strokes "
                       "there ran noticeably longer or shorter than the swimmer's own average. Worth reviewing "
                       "on the video: a breath, a broken catch or a hurried recovery usually shows up here.")

    # pacing
    trend = pacing_trend(multi)
    if trend:
        first, last, change = trend
        if change <= -PACING_CHANGE_PCT:
            out.append(f"Tempo eased from about {first/2:.0f} to {last/2:.0f} cycles per minute between the "
                       f"earliest and latest tracked segments ({change:+.0f}%). A drop like this late in a swim "
                       "usually points either to fatigue or to a deliberately longer stroke -- the coach will "
                       "know which was intended.")
        elif change >= PACING_CHANGE_PCT:
            out.append(f"Tempo built from about {first/2:.0f} to {last/2:.0f} cycles per minute across the tracked "
                       f"swim ({change:+.0f}%), consistent with a finishing push or a stroke that shortened as "
                       "the swimmer tired -- worth checking which on the video.")
        else:
            out.append(f"Pacing was even: tempo held within {max(1, round(abs(change)))}% from the earliest to the "
                       f"latest tracked segment ({first/2:.0f} to {last/2:.0f} cycles per minute).")

    # arm timing
    at = arm_timing(s["strokes_df"])
    if at:
        if at["uneven"]:
            longer = "camera-side to far-side" if at["diff_pct"] > 0 else "far-side to camera-side"
            out.append(f"The two arms are not evenly spaced: the half-cycle from {longer} stroke runs "
                       f"{abs(at['diff_pct']):.0f}% longer ({max(at['near_to_far'], at['far_to_near']):.2f} s "
                       f"against {min(at['near_to_far'], at['far_to_near']):.2f} s). An uneven spacing like this "
                       "-- a slight gallop -- often goes with breathing to one side or a pause in one arm's catch; "
                       "worth checking on the video which arm is waiting.")
        else:
            out.append(f"The two arms are evenly spaced: camera-side to far-side and back take "
                       f"{at['near_to_far']:.2f} s and {at['far_to_near']:.2f} s -- no gallop in the timing.")

    if s["strokes_total"] < MIN_STROKES_FOR_PICTURE:
        out.append(f"{s['strokes_total']} strokes were timed cleanly in this clip, which is enough for a "
                   "snapshot but not a full-swim picture; a longer or steadier clip would give a firmer read.")
    return out


def tempo_chart(track_rows, config, out_path):
    """Stroke tempo along the swim: one bar per tracked segment, placed at
    its time in the clip and as wide as the segment lasted; lighter where
    fewer than MIN_STROKES_STEADY strokes were captured."""
    rows = track_rows.dropna(subset=["stroke_rate_spm"])
    if rows.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    # benchmarks.freestyle is in cycles/min (both arms); this chart is in
    # strokes/min (one arm), so the band is doubled to plot on the same axis.
    r100_cpm = config["benchmarks"]["freestyle"]["stroke_rate_100m"]
    r100 = (r100_cpm[0] * 2, r100_cpm[1] * 2)
    ax.axhspan(r100[0], r100[1], color="#e6eef5", zorder=0)
    ax.text(0.2, r100[1] - 1.2, "reference range (100 m freestyle, doubled to strokes/min)",
            fontsize=7.5, color="#6b7a83", va="top")
    for r in rows.itertuples():
        width = max(0.6, r.t_end - r.t_start)
        strong = r.stroke_count >= MIN_STROKES_STEADY and getattr(r, "plausible", True)
        ax.bar(r.t_start, r.stroke_rate_spm, width=width, align="edge",
               color="#2a78d6" if strong else "#b9d3f2", edgecolor="white", linewidth=0.8, zorder=2)
        ax.text(r.t_start + width / 2, r.stroke_rate_spm + 1.2, f"{r.stroke_rate_spm:.0f}",
                ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xlabel("time in clip (s)", fontsize=9)
    ax.set_ylabel("stroke tempo\n(strokes / min)", fontsize=9)
    ax.set_ylim(0, max(80, rows["stroke_rate_spm"].max() * 1.18))
    ax.set_xlim(0, max(rows["t_end"].max() + 1, 5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    ax.set_title(f"Stroke tempo through the swim  (lighter: fewer than {MIN_STROKES_STEADY} strokes, or outside the swimmer's own range)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def speed_chart(splits, out_path):
    """Speed through the swim, one bar per split, labelled with distance per
    cycle where the split had enough strokes to time it."""
    rows = splits.dropna(subset=["average_speed_ms"]).sort_values("start_sec")
    if rows.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    for r in rows.itertuples():
        width = max(0.6, r.end_sec - r.start_sec)
        ax.bar(r.start_sec, r.average_speed_ms, width=width, align="edge", color="#2a78d6",
               edgecolor="white", linewidth=0.8, zorder=2)
        label = f"{r.average_speed_ms:.2f}"
        if isinstance(r.distance_per_cycle_m, float) and not np.isnan(r.distance_per_cycle_m):
            label += f"\n{r.distance_per_cycle_m:.2f} m/cyc"
        ax.text(r.start_sec + width / 2, r.average_speed_ms + 0.02, label, ha="center", va="bottom",
                fontsize=7.5, color="#333333")
    ax.set_xlabel("time in clip (s)", fontsize=9)
    ax.set_ylabel("speed (m/s)", fontsize=9)
    ax.set_ylim(0, rows["average_speed_ms"].max() * 1.3)
    ax.set_xlim(0, rows["end_sec"].max() + 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    ax.set_title("Speed through the swim  (label: speed, and distance per stroke cycle where timed)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def annotated_frames(video_path, tmp_dir, output_dir="outputs"):
    """A few frames from the rendered annotated video, if it exists."""
    annotated = os.path.join(output_dir, f"{stem_of(video_path)}_annotated.mp4")
    if not os.path.exists(annotated):
        return []
    cap = cv2.VideoCapture(annotated)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    out = []
    if n <= 0:
        cap.release()
        return out
    for frac in ANNOTATED_FRAME_FRACS:
        idx = min(n - 1, int(n * frac))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > 1400:
            frame = cv2.resize(frame, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
        path = os.path.join(tmp_dir, f"frame_{idx}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        out.append((path, idx / fps))
    cap.release()
    return out


def fmt(v, digits=1, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}{suffix}"


def add_table(doc, header, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(header))
    try:
        table.style = "Light Grid Accent 1"
    except KeyError:
        table.style = "Table Grid"
    for i, h in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(9)
    for r in rows:
        cells = table.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9.5)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    return table


def add_glance(doc, items, cols=3):
    """Headline figures in a grid of shaded cells."""
    n = len(items)
    rows_n = (n + cols - 1) // cols
    table = doc.add_table(rows=rows_n, cols=cols)
    table.style = "Table Grid"
    for k, (label, value, sub) in enumerate(items):
        cell = table.rows[k // cols].cells[k % cols]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(label.upper())
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x6B, 0x7A, 0x83)
        p2 = cell.add_paragraph()
        r2 = p2.add_run(value)
        r2.bold = True
        r2.font.size = Pt(17)
        p3 = cell.add_paragraph(sub)
        for rr in p3.runs:
            rr.font.size = Pt(8)
            rr.font.color.rgb = RGBColor(0x4A, 0x5A, 0x63)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "F1F5F8")
        tcPr.append(shd)
    doc.add_paragraph()


def caption(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in p.runs:
        r.font.size = Pt(8.5)
        r.italic = True
        r.font.color.rgb = RGBColor(0x4A, 0x5A, 0x63)
    return p


def build_report(video_path, config, calibration_dir="data/calibration", output_dir="outputs"):
    s = summarise_run(video_path, config, calibration_dir)
    info = session_info(video_path)
    rows = s["track_rows"]
    multi = rated_segments(rows) if not rows.empty else pd.DataFrame()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{stem_of(video_path)}_report.docx")

    doc = Document()
    base = doc.styles["Normal"]
    base.font.name = "Calibri"
    base.font.size = Pt(10.5)
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(0.9)
        section.top_margin = section.bottom_margin = Inches(0.8)

    # --- title ---
    doc.add_heading(f"{info['stroke']} swim report", level=0)
    p = doc.add_paragraph()
    p.add_run(info["swimmer"]).bold = True
    p.add_run(f"   ·   session {info['date']}   ·   clip {s['duration_sec']:.0f} s"
              f"   ·   report {datetime.now():%d %b %Y}")
    for r in p.runs:
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x4A, 0x5A, 0x63)

    # --- at a glance ---
    if s["headline_cpm"] is not None:
        tempo_value = f"{s['headline_cpm']:.0f} /min"
        tempo_sub = f"full stroke cycles per minute (≈ {2 * s['headline_cpm']:.0f} strokes/min, counting each arm)"
    elif s["median_spm"] is not None:
        tempo_value = f"{s['median_spm']:.0f} /min"
        tempo_sub = "strokes per minute, from short segments only -- indicative"
    else:
        tempo_value, tempo_sub = "—", "no strokes timed"
    cv = s["median_cv_multi"]
    rhythm_value = rhythm_label(cv, MIN_STROKES_STEADY).capitalize() if cv is not None else "—"
    rhythm_sub = f"arm-cycle times vary by {cv:.0f}% (median across segments)" if cv is not None else "needs segments with 3+ strokes"
    trend = pacing_trend(multi)
    if trend:
        _, _, change = trend
        if abs(change) >= PACING_CHANGE_PCT:
            trend_value, trend_sub = f"{change:+.0f}%", "tempo change, earliest to latest segment"
        else:
            trend_value, trend_sub = "Even", f"tempo within {max(1, round(abs(change)))}% start to finish"
    else:
        trend_value, trend_sub = "—", "needs two or more segments with 3+ strokes"
    at = arm_timing(s["strokes_df"])
    if at is None:
        arm_value, arm_sub = "—", "the two arms could not be told apart often enough to compare"
    elif at["uneven"]:
        arm_value, arm_sub = f"{abs(at['diff_pct']):.0f}% uneven", "one half-cycle between the arms runs longer"
    else:
        arm_value, arm_sub = "Even", "the two arms are evenly spaced -- no gallop"
    tempo_sub += f" · {s['strokes_total']} strokes timed"
    if s["swim_speed_ms"] is not None:
        speed_value = f"{s['swim_speed_ms']:.2f} m/s"
        speed_sub = f"{s['swim_distance_m']:.1f} m in {s['swim_time_sec']:.1f} s of continuous swimming"
    else:
        speed_value, speed_sub = "—", "the swimmer could not be followed along the rope for long enough"
    if s["swim_dpc_m"] is not None:
        dpc_value = f"{s['swim_dpc_m']:.2f} m"
        dpc_sub = "travelled per full stroke cycle: speed divided by cycle rate"
    else:
        dpc_value, dpc_sub = "—", "needs both a speed and a stroke tempo"
    add_glance(doc, [
        ("Speed", speed_value, speed_sub),
        ("Distance per cycle", dpc_value, dpc_sub),
        ("Stroke tempo", tempo_value, tempo_sub),
        ("Rhythm", rhythm_value, rhythm_sub),
        ("Pacing", trend_value, trend_sub),
        ("Arm timing", arm_value, arm_sub),
    ], cols=2)

    # --- speed through the swim ---
    splits = s["splits_df"]
    if s["swim_speed_ms"] is not None and len(splits):
        doc.add_heading("Speed through the swim", level=1)
        with tempfile.TemporaryDirectory() as tmp:
            chart = speed_chart(splits, os.path.join(tmp, "speed.png"))
            if chart:
                doc.add_picture(chart, width=Inches(6.4))
        split_rows = []
        for r in splits.sort_values("start_sec").itertuples():
            split_rows.append([f"{r.start_sec:.1f} – {r.end_sec:.1f} s", fmt(r.distance_m, 1),
                               fmt(r.average_speed_ms, 2), fmt(r.cycle_rate_cpm, 0), fmt(r.distance_per_cycle_m, 2)])
        add_table(doc, ["Split", "Distance (m)", "Speed (m/s)", "Tempo (cycles/min)", "Distance per cycle (m)"],
                  split_rows, col_widths=[1.4, 1.1, 1.1, 1.4, 1.7])
        caption(doc, "Distance is measured against the lane rope's coloured bands as they pass the swimmer. "
                     "Distance per cycle is how far one full two-arm stroke cycle carries the swimmer: speed "
                     "divided by cycle rate. A split needs 3 strokes before its tempo is timed.")

    # --- tempo through the swim ---
    doc.add_heading("Tempo through the swim", level=1)
    if rows.empty:
        doc.add_paragraph("No strokes were timed in this clip.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            chart = tempo_chart(rows, config, os.path.join(tmp, "tempo.png"))
            if chart:
                doc.add_picture(chart, width=Inches(6.4))
        caption(doc, "Each bar is one stretch of the swim where the swimmer was tracked continuously, placed at "
                     "its time in the clip. Bar height is stroke tempo in strokes per minute -- one stroke is one "
                     "arm's recovery, so a full two-arm cycle is two strokes.")
        table_rows = []
        for r in rows.itertuples():
            cv_r = segment_cv(r._asdict())
            label = rhythm_label(cv_r, r.stroke_count).capitalize()
            if r.stroke_count >= MIN_STROKES_STEADY and cv_r is not None and not np.isnan(cv_r):
                label += f"  ({cv_r:.0f}%)"
            tempo_txt = fmt(r.stroke_rate_spm, 0)
            if not getattr(r, "plausible", True):
                tempo_txt += "  (check)"
            table_rows.append([f"{r.t_start:.1f} – {r.t_end:.1f} s", int(r.stroke_count), tempo_txt, label])
        add_table(doc, ["Segment", "Strokes", "Tempo (strokes/min)", "Rhythm (arm-cycle variation)"],
                  table_rows, col_widths=[1.5, 0.8, 1.5, 2.9])
        caption(doc, f"Rhythm reads the spread of each arm's cycle time within a segment: steady is "
                     f"{STEADY_CV_PCT:.0f}% or less, uneven is above {UNEVEN_CV_PCT:.0f}%. A segment needs "
                     f"{MIN_STROKES_STEADY} strokes before it is rated. A tempo marked (check) sits far outside "
                     "the swimmer's own range and is more likely a tracking artefact than a real change of pace; "
                     "it is left out of the pacing and range figures.")

    # --- arm timing ---
    doc.add_heading("Timing between the arms", level=1)
    doc.add_paragraph(
        "Consecutive strokes are alternate arms. The arm nearer the camera rises higher in the picture, which is "
        "how the two are told apart -- so 'camera-side' is the swimmer's left or right depending on the direction "
        "of swim, and the coach will know which. In an even stroke the time from one arm's stroke to the other's "
        "is the same in both directions; a consistently longer gap one way is a gallop.")
    if at:
        add_table(doc, ["Half-cycle", "Intervals timed", "Median time (s)"], [
            ["Camera-side stroke to far-side stroke", at["n_near_to_far"], fmt(at["near_to_far"], 2)],
            ["Far-side stroke to camera-side stroke", at["n_far_to_near"], fmt(at["far_to_near"], 2)],
        ], col_widths=[3.2, 1.3, 1.5])
    else:
        doc.add_paragraph(f"The two arms could not be told apart often enough in this clip (each direction needs "
                          f"at least {MIN_ARM_PAIRS} timed intervals), so no comparison is reported.")

    # --- observations ---
    doc.add_heading("Observations", level=1)
    for text in observations(s, config):
        doc.add_paragraph(text, style="List Bullet")
    caption(doc, "Generated from the measured speed, distance per cycle, tempo, rhythm, pacing and arm timing. "
                 "The coach's own reading of technique on the video takes precedence over any of these.")

    # --- from the video ---
    with tempfile.TemporaryDirectory() as tmp:
        frames = annotated_frames(video_path, tmp, output_dir)
        if frames:
            doc.add_heading("From the video", level=1)
            for path, t in frames:
                doc.add_picture(path, width=Inches(6.4))
                caption(doc, f"{t:.0f} s into the clip. The box marks the tracked swimmer, with the distance "
                             "covered and speed so far; the small marker follows the tracked hand.")

    # --- scope ---
    doc.add_paragraph()
    caption(doc, f"Produced automatically from {s['video']}. Timing comes from the video's own timestamps at "
                 f"{s['fps']:.0f} frames per second, distance from the lane rope's coloured bands; the swimmer "
                 f"was in view and tracked for {s['subject_coverage_pct']:.0f}% of the clip.")

    doc.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build the downloadable swim report for an analysed video.")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    timers = {}
    with Timer(timers, "time_report_sec"):
        path = build_report(args.video, config)
    print(f"Built in {timers['time_report_sec']:.1f}s")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
