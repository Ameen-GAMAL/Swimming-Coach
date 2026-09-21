# Builds the downloadable swim report: a Word document written for the
# coach, not the engineer. It collates what the other stages already
# produced -- nothing here computes a new measurement -- and keeps to the
# figures that come from the video's own timestamps (stroke tempo, rhythm
# consistency, pacing across the swim), because those are the ones the
# pipeline measures reliably. Distance and speed are deliberately absent.
#
# All counts are derived from the cached stage outputs rather than the
# stages' live diagnostics dicts, so a report built from a cached run
# carries the same numbers as one built from a fresh run; the app's own
# summary uses summarise_run() for the same reason.

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

from src.detect_track import run_detect_track_cached
from src.events import load_calibration, run_events_cached
from src.metrics import run_metrics_cached
from src.pose import run_pose_cached
from src.profiling import Timer, load_config

ANNOTATED_FRAME_FRACS = (0.2, 0.5, 0.8)
STEADY_CV_PCT = 12.0      # cycle-time variation at or below this reads as a steady rhythm
UNEVEN_CV_PCT = 20.0      # above this, worth a look at the video
PACING_CHANGE_PCT = 10.0  # tempo change between early and late segments worth remarking on
MIN_CYCLES_FOR_PICTURE = 6


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
    track_sizes = tracks_df["track_id"].value_counts() if not tracks_df.empty else pd.Series(dtype=int)
    longest_frames = int(track_sizes.max()) if len(track_sizes) else 0

    track_rows = (metrics_df[metrics_df["row_type"] == "track"].copy()
                  if not metrics_df.empty and "row_type" in metrics_df else pd.DataFrame())
    if not track_rows.empty:
        span = tracks_df.groupby("track_id")["frame"].agg(["min", "max"]) / fps
        track_rows["t_start"] = track_rows["track_id"].map(span["min"])
        track_rows["t_end"] = track_rows["track_id"].map(span["max"])
        track_rows = track_rows.sort_values("t_start").reset_index(drop=True)
    cpm = track_rows["stroke_rate_cpm"].dropna() if not track_rows.empty else pd.Series(dtype=float)
    multi = track_rows[track_rows["cycle_count"] >= 2] if not track_rows.empty else pd.DataFrame()

    calib = load_calibration(video_path, calibration_dir)

    return {
        "video": os.path.basename(video_path),
        "frames_total": frames_total,
        "fps": fps,
        "duration_sec": frames_total / fps,
        "resolution": f"{width}x{height}",
        "frames_detected": frames_detected,
        "coverage_pct": 100.0 * frames_detected / frames_total if frames_total else 0.0,
        "detections": int(len(tracks_df)),
        "tracks": int(tracks_df["track_id"].nunique()) if not tracks_df.empty else 0,
        "longest_track_frames": longest_frames,
        "longest_track_sec": longest_frames / fps,
        "pose_rows": int(len(pose_df)),
        "wrist_visible_pct": float(pose_diag.get("quality_wrist_visible_pct", 0.0)),
        "entries": int((events_df["event_type"] == "entry").sum()) if not events_df.empty else 0,
        "exits": int((events_df["event_type"] == "exit").sum()) if not events_df.empty else 0,
        "cycles_total": int(track_rows["cycle_count"].sum()) if not track_rows.empty else 0,
        "tracks_with_cycles": int(len(track_rows)),
        "tracks_with_two_plus_cycles": int(len(multi)),
        "median_cpm": float(cpm.median()) if len(cpm) else None,
        "median_cpm_multi": float(multi["stroke_rate_cpm"].median()) if len(multi) else None,
        "median_cv_multi": (float(multi["cycle_time_cv_pct"].dropna().median())
                            if len(multi) and multi["cycle_time_cv_pct"].notna().any() else None),
        "lengths_computed": int(metrics_diag.get("result_lengths_computed", 0)),
        "lengths_accepted": int(metrics_diag.get("result_lengths_accepted", 0)),
        "laps_computed": int(metrics_diag.get("result_laps_computed", 0)),
        "scale_method": calib.get("scale_method", "unknown"),
        "visible_span_m": float(calib.get("corner_span_m", {}).get("end_m", 0.0)),
        "scale_clamped": bool(calib.get("scale_estimate_clamped_to_prior", False)),
        "metrics_df": metrics_df,
        "track_rows": track_rows,
    }


def rhythm_label(cv, cycles):
    if cycles < 2 or cv is None or (isinstance(cv, float) and np.isnan(cv)):
        return "single cycle"
    if cv <= STEADY_CV_PCT:
        return "steady"
    if cv <= UNEVEN_CV_PCT:
        return "mostly steady"
    return "uneven"


def pacing_trend(multi):
    """Early-vs-late tempo across the tracked swim, from segments with two
    or more cycles. Returns (first_cpm, last_cpm, pct_change) or None."""
    if len(multi) < 2:
        return None
    first, last = multi.iloc[0], multi.iloc[-1]
    if first["t_start"] == last["t_start"]:
        return None
    change = 100.0 * (last["stroke_rate_cpm"] - first["stroke_rate_cpm"]) / first["stroke_rate_cpm"]
    return float(first["stroke_rate_cpm"]), float(last["stroke_rate_cpm"]), float(change)


def observations(s, config):
    """Coach-facing observations generated from the measured tempo and
    rhythm. Factual first, then what is worth checking on the video; the
    coach's own judgement of technique takes precedence over any of these."""
    rows = s["track_rows"]
    multi = rows[rows["cycle_count"] >= 2] if not rows.empty else pd.DataFrame()
    bench = config["benchmarks"]["freestyle"]
    out = []

    if s["cycles_total"] == 0:
        return ["No complete stroke cycles could be timed in this clip, so there is no tempo picture to report. "
                "A longer clip, or one where the swimmer stays in view for more of the length, would give one."]

    # tempo
    if s["median_cpm_multi"] is not None:
        cpm = s["median_cpm_multi"]
        lo, hi = multi["stroke_rate_cpm"].min(), multi["stroke_rate_cpm"].max()
        r50, r100 = bench["stroke_rate_50m"], bench["stroke_rate_100m"]
        if r50[0] <= cpm <= r50[1]:
            context = "a sprint-range tempo"
        elif r100[0] <= cpm <= r100[1]:
            context = "in the range usually seen over 100 m"
        elif cpm < r100[0]:
            context = "a distance-style tempo, longer and slower than a sprint stroke"
        else:
            context = "a very high tempo"
        out.append(f"Stroke tempo averaged {cpm:.0f} cycles per minute across the segments with two or more "
                   f"cycles (individual segments ran {lo:.0f}–{hi:.0f}) -- {context}.")
    else:
        out.append(f"Only single-cycle segments were captured, at around {s['median_cpm']:.0f} cycles per minute. "
                   "Treat this as a snapshot of tempo rather than a measurement of the whole swim.")

    # rhythm
    if len(multi):
        steady = multi[multi["cycle_time_cv_pct"] <= STEADY_CV_PCT]
        uneven = multi[multi["cycle_time_cv_pct"] > UNEVEN_CV_PCT]
        if len(steady):
            spans = ", ".join(f"{r.t_start:.0f}–{r.t_end:.0f} s" for r in steady.itertuples())
            out.append(f"Rhythm was steady at {spans}: cycle times varied by "
                       f"{steady['cycle_time_cv_pct'].max():.0f}% or less from stroke to stroke.")
        if len(uneven):
            spans = ", ".join(f"{r.t_start:.0f}–{r.t_end:.0f} s ({r.cycle_time_cv_pct:.0f}%)" for r in uneven.itertuples())
            out.append(f"Rhythm was uneven at {spans} -- one or more strokes there ran noticeably longer or "
                       "shorter than the swimmer's own average. Worth reviewing that stretch of the video: "
                       "a breath, a broken catch or a hurried recovery usually shows up here.")

    # pacing
    trend = pacing_trend(multi)
    if trend:
        first, last, change = trend
        if change <= -PACING_CHANGE_PCT:
            out.append(f"Tempo eased from about {first:.0f} to {last:.0f} cycles per minute between the earliest "
                       f"and latest tracked segments ({change:+.0f}%). A drop like this late in a swim usually "
                       "points either to fatigue or to a deliberately longer stroke -- the coach will know which "
                       "was intended.")
        elif change >= PACING_CHANGE_PCT:
            out.append(f"Tempo built from about {first:.0f} to {last:.0f} cycles per minute across the tracked "
                       f"swim ({change:+.0f}%), consistent with a finishing push or a stroke that shortened as "
                       "the swimmer tired -- worth checking which on the video.")
        else:
            out.append(f"Tempo held within {abs(change):.0f}% from the earliest to the latest tracked segment "
                       f"({first:.0f} to {last:.0f} cycles per minute) -- pacing was even.")

    # picture completeness, stated as scope rather than fault
    if s["cycles_total"] < MIN_CYCLES_FOR_PICTURE:
        out.append(f"{s['cycles_total']} cycles were timed cleanly in this clip, which is enough for a snapshot "
                   "but not a full-swim picture; a longer or steadier clip would give a firmer read.")
    return out


def tempo_chart(track_rows, config, out_path):
    """Stroke tempo along the swim: one bar per tracked segment, placed at
    its time in the clip and as wide as the segment lasted; lighter where
    only one cycle was captured. The shaded band is the reference range."""
    rows = track_rows.dropna(subset=["stroke_rate_cpm"])
    if rows.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    r100 = config["benchmarks"]["freestyle"]["stroke_rate_100m"]
    ax.axhspan(r100[0], r100[1], color="#e6eef5", zorder=0)
    ax.text(0.2, r100[1] - 1.2, "reference range (100 m freestyle)", fontsize=7.5, color="#6b7a83", va="top")
    for r in rows.itertuples():
        width = max(0.6, r.t_end - r.t_start)
        ax.bar(r.t_start, r.stroke_rate_cpm, width=width, align="edge",
               color="#2a78d6" if r.cycle_count >= 2 else "#b9d3f2", edgecolor="white", linewidth=0.8, zorder=2)
        ax.text(r.t_start + width / 2, r.stroke_rate_cpm + 1.2,
                f"{r.stroke_rate_cpm:.0f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xlabel("time in clip (s)", fontsize=9)
    ax.set_ylabel("stroke tempo\n(cycles / min)", fontsize=9)
    ax.set_ylim(0, max(75, rows["stroke_rate_cpm"].max() * 1.18))
    ax.set_xlim(0, max(rows["t_end"].max() + 1, 5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    ax.set_title("Stroke tempo through the swim  (lighter bars: one cycle only)", fontsize=10, loc="left")
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


def add_glance(doc, items):
    """Four headline figures in a 2x2 grid of shaded cells."""
    table = doc.add_table(rows=2, cols=2)
    table.style = "Table Grid"
    for k, (label, value, sub) in enumerate(items[:4]):
        cell = table.rows[k // 2].cells[k % 2]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(label.upper())
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x6B, 0x7A, 0x83)
        p2 = cell.add_paragraph()
        r2 = p2.add_run(value)
        r2.bold = True
        r2.font.size = Pt(18)
        p3 = cell.add_paragraph(sub)
        for rr in p3.runs:
            rr.font.size = Pt(8.5)
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
    multi = rows[rows["cycle_count"] >= 2] if not rows.empty else pd.DataFrame()
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
    trend = pacing_trend(multi)
    if trend:
        _, _, change = trend
        trend_value = f"{change:+.0f}%"
        trend_sub = ("tempo change, earliest to latest segment" if abs(change) >= PACING_CHANGE_PCT
                     else "tempo held even across the swim")
    else:
        trend_value, trend_sub = "—", "needs two or more multi-cycle segments"
    if s["median_cpm_multi"] is not None:
        tempo_value = f"{s['median_cpm_multi']:.0f} /min"
        tempo_sub = f"median over {s['tracks_with_two_plus_cycles']} segments with 2+ cycles"
    elif s["median_cpm"] is not None:
        tempo_value = f"{s['median_cpm']:.0f} /min"
        tempo_sub = "from single-cycle segments only -- indicative"
    else:
        tempo_value, tempo_sub = "—", "no complete cycles timed"
    cv = s["median_cv_multi"]
    rhythm_value = rhythm_label(cv, 2) if cv is not None else "—"
    rhythm_sub = (f"cycle times vary by {cv:.0f}% (median across segments)" if cv is not None
                  else "needs segments with 2+ cycles")
    add_glance(doc, [
        ("Stroke tempo", tempo_value, tempo_sub),
        ("Rhythm", rhythm_value.capitalize(), rhythm_sub),
        ("Pacing", trend_value, trend_sub),
        ("Cycles timed", f"{s['cycles_total']}", f"across {s['tracks_with_cycles']} tracked segments, "
                                                 f"{s['coverage_pct']:.0f}% of the clip in view"),
    ])

    # --- tempo through the swim ---
    doc.add_heading("Tempo through the swim", level=1)
    if rows.empty:
        doc.add_paragraph("No complete stroke cycles were timed in this clip.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            chart = tempo_chart(rows, config, os.path.join(tmp, "tempo.png"))
            if chart:
                doc.add_picture(chart, width=Inches(6.4))
        caption(doc, "Each bar is one stretch of the swim where the swimmer was tracked continuously, placed "
                     "at its time in the clip. Bar height is stroke tempo in cycles per minute; a cycle is one "
                     "full arm cycle, timed from the hand's entry to its next entry.")
        table_rows = []
        for r in rows.itertuples():
            table_rows.append([
                f"{r.t_start:.1f} – {r.t_end:.1f} s",
                int(r.cycle_count),
                fmt(r.stroke_rate_cpm, 0),
                rhythm_label(getattr(r, "cycle_time_cv_pct", None), r.cycle_count).capitalize()
                + (f"  ({r.cycle_time_cv_pct:.0f}%)" if r.cycle_count >= 2 and not np.isnan(r.cycle_time_cv_pct) else ""),
            ])
        add_table(doc, ["Segment", "Cycles", "Tempo (cycles/min)", "Rhythm (cycle-time variation)"],
                  table_rows, col_widths=[1.5, 0.8, 1.5, 2.9])
        caption(doc, "Rhythm reads the spread of cycle times within a segment: steady is 12% or less, "
                     "uneven is above 20%. A segment with a single cycle cannot be rated.")

    # --- observations ---
    doc.add_heading("Observations", level=1)
    for text in observations(s, config):
        doc.add_paragraph(text, style="List Bullet")
    caption(doc, "Generated from the measured tempo and rhythm. The coach's own reading of technique on the "
                 "video takes precedence over any of these.")

    # --- from the video ---
    with tempfile.TemporaryDirectory() as tmp:
        frames = annotated_frames(video_path, tmp, output_dir)
        if frames:
            doc.add_heading("From the video", level=1)
            for path, t in frames:
                doc.add_picture(path, width=Inches(6.4))
                caption(doc, f"{t:.0f} s into the clip. The box marks the tracked swimmer; the small marker "
                             "follows the tracked hand.")

    # --- scope ---
    doc.add_paragraph()
    caption(doc, f"Produced automatically from {s['video']}. Stroke timing is measured from the video's own "
                 f"timestamps at {s['fps']:.0f} frames per second; the swimmer was in view and tracked for "
                 f"{s['coverage_pct']:.0f}% of the clip. This report covers tempo, rhythm and pacing only; "
                 "distance and speed are not part of this version.")

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
