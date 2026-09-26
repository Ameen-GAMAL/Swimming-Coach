# Web page for the client demo: upload a video (or pick one already
# approved in data/raw/), run the analysis, see the annotated video and
# results. Deliberately goes beyond CLAUDE.md section 14's original "no
# styling work" -- explicit, user-requested departure, recorded in
# CLAUDE.md's section 3 amendments -- to make the tool something a client
# can actually sit down with, not just a bare functional demo.

import os
import shutil

import gradio as gr
import pandas as pd

from src import __version__
from src.calibrate import calibrate_video
from src.detect_track import run_detect_track_cached
from src.events import run_events_cached
from src.metrics import cache_path as metrics_csv_path
from src.metrics import run_metrics_cached
from src.pose import run_pose_cached
from src.profiling import load_config
from src.render import render_video
from src.report import build_report, summarise_run
from src.rope_odometry import run_rope_odometry_cached

CONFIG = load_config("config.yaml")

STAGES = [
    "Calibrating (automatic -- no clicking needed)",
    "Detecting and tracking swimmers",
    "Measuring distance against the lane rope",
    "Finding body keypoints",
    "Timing the strokes",
    "Computing metrics",
    "Rendering the annotated video",
    "Building the coach report",
]


def list_raw_videos():
    raw_dir = "data/raw"
    if not os.path.isdir(raw_dir):
        return []
    return sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".mp4"))


def refresh_picker(selected=None):
    # The list is read when the page loads and again after every run, so a
    # video uploaded in this session shows up without restarting the app
    # (it used to be built once at startup and never change).
    return gr.update(choices=list_raw_videos(), value=selected)


def calibration_path(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("data", "calibration", f"{stem}.json")


def preview_path(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("data", "calibration", f"{stem}_preview.png")


def render_stage_status(current_idx, total, error_msg=None):
    # A real per-stage checklist, not one undifferentiated progress bar --
    # the client can see exactly which step is running (or where it failed)
    # instead of staring at a spinner for several minutes.
    lines = []
    for i, name in enumerate(STAGES):
        if error_msg is not None and i == current_idx:
            lines.append(f"- ❌ **{name}** -- {error_msg}")
        elif i < current_idx or (error_msg is None and current_idx >= total):
            lines.append(f"- ✅ {name}")
        elif i == current_idx:
            lines.append(f"- ⏳ **{name}...**")
        else:
            lines.append(f"- ◯ {name}")
    return "\n".join(lines)


def styled_callout(kind, title, body):
    # A distinct coloured panel instead of a line of markdown. Plain text
    # only (no nested <code>/<strong>) -- the app's dark theme was found to
    # override those tags' text colour to match the (dark) page background,
    # making them silently invisible inside this light-background box.
    colors = {
        "warning": ("#fff7ed", "#c2410c", "#fdba74"),
        "info": ("#eff6ff", "#1d4ed8", "#93c5fd"),
        "success": ("#f0fdf4", "#15803d", "#86efac"),
    }
    bg, fg, border = colors[kind]
    return (
        f'<div style="background:{bg}; color:{fg}; border:1px solid {border}; '
        f'border-radius:10px; padding:14px 18px; margin:8px 0; font-size:14px; line-height:1.5;">'
        f'<span style="font-size:15px; font-weight:600; color:{fg};">{title}</span>'
        f'<div style="color:{fg};">{body}</div></div>'
    )


def _fmt(v, digits):
    return "—" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def results_summary_md(s):
    # The headline, before the detail table. Reads from summarise_run(),
    # which derives every figure from the cached stage outputs, so a cached
    # run shows the same numbers as a fresh one.
    tempo = (f"{s['headline_cpm']:.0f} cycles/min ({2 * s['headline_cpm']:.0f} strokes/min)"
             if s["headline_cpm"] is not None
             else (f"{s['median_spm'] / 2:.0f} cycles/min (short segments only)" if s["median_spm"] is not None
                   else "not measurable"))
    if s["swim_speed_ms"] is not None:
        swim = (f"Distance: **{s['swim_distance_m']:.1f} m** in {s['swim_time_sec']:.1f} s · "
                f"Speed: **{s['swim_speed_ms']:.2f} m/s** · "
                f"Distance per cycle: **{_fmt(s['swim_dpc_m'], 2)} m** · ")
    else:
        swim = "Distance/speed: **not measurable in this clip** · "
    return (f"**Summary** -- {swim}Stroke tempo: **{tempo}** · Strokes timed: **{s['strokes_total']}** · "
            f"Swimmer in view: **{s['subject_coverage_pct']:.0f}%** of the clip")


def position_text(r):
    # metres into the length, from the wall being swum away from
    a, b = getattr(r, "start_position_m", None), getattr(r, "end_position_m", None)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "—"
    txt = f"{a:.1f} → {b:.1f}"
    u = getattr(r, "position_uncertainty_m", None)
    return txt + (f"  (±{u:.1f})" if u is not None and not pd.isna(u) else "")


def results_table(metrics_df):
    """The filmed swimmer's results, one row per continuous swim and one per
    5-second split, with readable headers and rounded values. The raw
    metrics CSV -- every column, every tracked person -- is the download
    next to it; showing it here left most cells blank, because each row type
    fills a different subset of its ~40 columns."""
    if metrics_df.empty or "row_type" not in metrics_df:
        return pd.DataFrame([{"Note": "Nothing was measured in this clip -- see the Diagnostics tab."}])
    rows = metrics_df[metrics_df["row_type"].isin(["swim", "split"])]
    if rows.empty:
        return pd.DataFrame([{"Note": "The swimmer could not be followed along the lane rope for long "
                                      "enough to measure a swim -- see the Diagnostics tab."}])
    out = []
    for r in rows.itertuples():
        if r.row_type == "swim":
            part = f"Swim {int(r.segment_index) + 1} (whole)"
        else:
            part = f"  split {int(r.split_index) + 1}"
        out.append({
            "Part": part,
            "From (s)": _fmt(r.start_sec, 1), "To (s)": _fmt(r.end_sec, 1),
            "Pool position (m)": position_text(r),
            "Distance (m)": _fmt(r.distance_m, 1),
            "Speed (m/s)": _fmt(r.average_speed_ms, 2),
            # cycles swum = time x tempo: the stroke count itself only covers
            # the strokes the pose model could time (70-97% of them on the
            # tested clips), so as a count of the swim it always reads short
            "Cycles swum": _fmt(r.duration_sec * r.cycle_rate_cpm / 60.0, 1) if pd.notna(r.cycle_rate_cpm) else "—",
            "Tempo (cycles/min)": _fmt(r.cycle_rate_cpm, 0),
            "Distance per cycle (m)": _fmt(r.distance_per_cycle_m, 2),
            "Rope read (%)": _fmt(r.rope_measured_pct, 0),
            "Note": r.rejection_reason if isinstance(r.rejection_reason, str) else "",
        })
    return pd.DataFrame(out)


def capture_line(s):
    gate = CONFIG["metrics"]["min_stroke_capture_pct"]
    if s["stroke_capture_pct"] is None:
        return "not measurable (needs a measured swim with a tempo)"
    verdict = "passes" if s["stroke_capture_pct"] >= gate else "BELOW"
    return (f"{s['stroke_capture_pct']:.0f}% of the strokes swum (timed recovery peaks against time x tempo) "
            f"-- {verdict} the {gate}% gate (CLAUDE.md section 8.2)")


def position_line(s):
    fixes = s.get("position_fixes") or []
    if not fixes:
        return "not placed -- no red rope marking (5 m zone or mid-pool marker) was seen close enough to the swimmer"
    return "; ".join(f"placed from {f['markings']}, to within ±{f['uncertainty_m']:.1f} m"
                     + ("" if f["uncertainty_m"] <= CONFIG["pool_position"]["max_uncertainty_m"] else " -- too uncertain, not reported")
                     for f in fixes)


def diagnostics_summary_md(s):
    return f"""
### Detection & tracking
- Frames in clip: {s['frames_total']:,} at {s['fps']:.2f} fps
- Frames with a detection: {s['frames_detected']:,} ({s['coverage_pct']:.1f}%)
- Tracks found (after merge): {s['tracks']} -- {s['subject_tracks']} on the filmed swimmer, {s['other_people_tracks']} on other people in the pool
- Filmed swimmer in view: {s['subject_coverage_pct']:.1f}% of frames -- longest continuous track {s['longest_track_frames']} frames ({s['longest_track_sec']:.1f}s)

### Distance (lane rope)
- Rope readings used directly: {s['rope_measured_pct']:.0f}% of samples -- {s['rope_texture_pct']:.0f}% of them from the float-disc pattern where the navy/white bands couldn't be counted (red end zones, splash); the rest bridged at the neighbouring speed, up to {CONFIG['rope_odometry']['max_bridge_sec']:.0f} s at a time
- Samples where the two sides of the swimmer disagreed (discarded): {s['rope_side_disagreements']}
- Swims measured: {s['swims_accepted']} of {s['swims_computed']} continuous stretches (a stretch needs {CONFIG['metrics']['min_rope_measured_pct']}% of its rope readings)
- Band length used: {CONFIG['pool']['rope_band_m']:.2f} m per navy or white band, calibrated against the pool's 5 m and 25 m markings
- Position along the pool: {position_line(s)}

### Pose and stroke capture
- Strokes timed: {capture_line(s)}
- Pose found the swimmer in {s['person_found_pct']:.0f}% of the tracked frames; wrist visible in {s['wrist_visible_pct']:.0f}% of them overall (the old 70% gate measured this -- a stand-in for stroke capture, which is now measured directly above)
- Pose readings produced: {s['pose_rows']:,}

### Stroke events
- Strokes timed (recovery peaks): {s['strokes_total']} across {s['tracks_with_strokes']} tracks ({s['tracks_multi']} with 3+ strokes)
- Hand entries / exits observed at the shoulder line: {s['entries']} / {s['exits']}
"""


def wrist_plots(video_path, subject_ids):
    # The wrist-height signal plots events.py saves per track -- the filmed
    # swimmer's only (other people's tracks made the gallery unreadable).
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if not os.path.isdir("outputs"):
        return []
    wanted = {f"__track{float(t)}_wrist_signal.png" for t in subject_ids}
    return sorted(os.path.join("outputs", f) for f in os.listdir("outputs")
                  if f.startswith(stem) and any(f.endswith(w) for w in wanted))


def build_speed_chart(metrics_df):
    # Speed per 5-second split of the filmed swimmer -- the pacing picture.
    if metrics_df.empty or "row_type" not in metrics_df:
        return None
    sp = metrics_df[(metrics_df["row_type"] == "split") & metrics_df["average_speed_ms"].notna()]
    if sp.empty:
        return None
    return pd.DataFrame({"split": [f"{a:.0f}-{b:.0f}s" for a, b in zip(sp["start_sec"], sp["end_sec"])],
                         "speed (m/s)": sp["average_speed_ms"].round(2)})


def run_pipeline(uploaded_file, picked_video, progress=gr.Progress()):
    n = len(STAGES)
    empty_table = pd.DataFrame([{"Note": "Run an analysis to see results here."}])
    # Output order, matched by run_button.click below:
    #   0 stage_status  1 summary_md  2 callout  3 video  4 table  5 chart
    #   6 calibration preview  7 diagnostics md  8 report file  9 metrics csv  10 picker  11 wrist gallery
    NUM_OUTPUTS = 12
    failed_tail = (None, empty_table, None, None, "", None, None, gr.skip(), None)

    if uploaded_file is not None:
        video_path = os.path.join("data", "raw", os.path.basename(uploaded_file))
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        shutil.copy(uploaded_file, video_path)
    elif picked_video:
        video_path = os.path.join("data", "raw", picked_video)
    else:
        yield (render_stage_status(0, n, "Upload a video or pick one from data/raw/ first."),
               "", styled_callout("warning", "No video selected", "Upload a file or choose one from the dropdown.")
               ) + failed_tail
        return

    def step(i, frac):
        progress(frac)
        return (render_stage_status(i, n),) + (gr.skip(),) * (NUM_OUTPUTS - 1)

    try:
        yield step(0, 0.02)
        if not os.path.exists(calibration_path(video_path)):
            calibrate_video(video_path, CONFIG)
        calib_preview = preview_path(video_path) if os.path.exists(preview_path(video_path)) else None

        yield (render_stage_status(1, n),) + (gr.skip(),) * 5 + (calib_preview,) + (gr.skip(),) * 5
        run_detect_track_cached(video_path, CONFIG)
        yield step(2, 0.35)
        run_rope_odometry_cached(video_path, CONFIG)
        yield step(3, 0.45)
        run_pose_cached(video_path, CONFIG)
        yield step(4, 0.65)
        run_events_cached(video_path, CONFIG)
        yield step(5, 0.8)
        metrics_df, _ = run_metrics_cached(video_path, CONFIG)
        yield step(6, 0.85)
        annotated_path = render_video(video_path, CONFIG)
        yield step(7, 0.97)
        report_path = build_report(video_path, CONFIG)
        summary = summarise_run(video_path, CONFIG)
    except Exception as exc:
        yield (render_stage_status(n, n, str(exc)),
               "", styled_callout("warning", "Analysis failed", str(exc))) + failed_tail
        return

    if summary["swim_speed_ms"] is None:
        callout = styled_callout(
            "warning", "Distance and speed could not be measured in this clip",
            "Distance is read from the lane rope just behind the swimmer, and here it could not be followed "
            "for long enough (see the Diagnostics tab). Stroke tempo, rhythm and arm timing are unaffected.")
    else:
        callout = styled_callout(
            "info", "How distance is measured",
            "The lane rope behind the swimmer is used as a ruler: its navy and white bands are counted as they "
            f"pass the swimmer ({CONFIG['pool']['rope_band_m']:.2f} m per band, checked against the pool's 5 m "
            "and 25 m markings on two clips to within about 2%).")

    display_df = results_table(metrics_df)
    diag_md = diagnostics_summary_md(summary)
    chart_df = build_speed_chart(metrics_df)
    csv_path = metrics_csv_path(video_path, CONFIG)

    yield (render_stage_status(n, n), results_summary_md(summary), callout, annotated_path, display_df,
           chart_df, calib_preview, diag_md, report_path, csv_path,
           refresh_picker(os.path.basename(video_path)), wrist_plots(video_path, summary["subject_ids"]))


THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="blue")

CUSTOM_CSS = """
.stage-status { font-size: 15px; line-height: 1.9; }
"""

with gr.Blocks(title="Swimming Performance Tracking (POC)") as demo:
    gr.Markdown(f"# 🏊 Swimming Performance Tracking — Proof of Concept  ·  v{__version__}")
    gr.Markdown(
        "Upload a freestyle swim video (side view, filmed along the pool at water level), or pick one already "
        "in `data/raw/`, then run the analysis. Processing takes several minutes depending on video length -- "
        "this is a proof of concept, not the production system. See the **About** tab for scope, or "
        "[CLAUDE.md](CLAUDE.md) for the full specification."
    )

    with gr.Tabs():
        with gr.Tab("Run Analysis"):
            with gr.Row():
                upload = gr.File(label="Upload a video", file_types=[".mp4"])
                picker = gr.Dropdown(choices=list_raw_videos(), label="...or pick one already in data/raw/")
            run_button = gr.Button("Run analysis", variant="primary", size="lg")
            stage_status = gr.Markdown(render_stage_status(-1, len(STAGES)), elem_classes=["stage-status"])

        with gr.Tab("Results"):
            summary_md_out = gr.Markdown()
            callout_box = gr.HTML()
            with gr.Row():
                report_out = gr.File(label="Coach report (.docx) -- speed, distance per cycle, tempo, rhythm, "
                                           "pacing and observations", interactive=False)
                csv_out = gr.File(label="All measurements (.csv) -- every row and column, for analysis",
                                  interactive=False)
            with gr.Row():
                video_out = gr.Video(label="Annotated video")
                chart_out = gr.BarPlot(x="split", y="speed (m/s)", title="Speed through the swim (per 5-second split)",
                                       x_title="Split", y_title="Speed (m/s)", height=300)
            table_out = gr.Dataframe(label="The filmed swimmer: each continuous swim, then its 5-second splits",
                                     value=pd.DataFrame([{"Note": "Run an analysis to see results here."}]))

        with gr.Tab("Diagnostics"):
            gr.Markdown("Internal detail for checking *why* a result came out the way it did.")
            with gr.Row():
                calib_preview_out = gr.Image(label="Calibration preview (used to find the pool area)")
                wrist_gallery_out = gr.Gallery(label="Wrist-height signal, filmed swimmer's tracks", columns=2, height=400)
            diag_md_out = gr.Markdown()

        with gr.Tab("About"):
            gr.Markdown("""
## What this is

A proof-of-concept pipeline that takes a freestyle swim video and measures
distance, speed, distance per stroke cycle, stroke tempo and rhythm for the
filmed swimmer, using a fine-tuned swimmer detector and an off-the-shelf pose
model.

## How the numbers are measured

- **Distance and speed** come from the lane rope just behind the swimmer. Its
  navy and white bands have a fixed length, so counting the bands that pass
  the swimmer's head gives the distance swum -- whatever the camera is doing.
  The band length was calibrated against the pool's own markings (red rope at
  5 m from each wall, and at mid-pool), and two clips checked that way land
  within about 2% of the known distances.
- **Stroke tempo and rhythm** come from the timing of each arm's recovery,
  read from the wrist in the video's own 120 fps timestamps.
- **Distance per cycle** is speed divided by cycle rate.

## Limits

- The swimmer must be filmed side-on with the lane rope behind them in view;
  if the rope can't be followed for long enough, distance and speed are left
  blank rather than guessed.
- Benchmark ranges in the configuration are placeholders, not real
  professional-swimmer data.
""")

    run_button.click(
        run_pipeline, inputs=[upload, picker],
        outputs=[stage_status, summary_md_out, callout_box, video_out, table_out, chart_out,
                 calib_preview_out, diag_md_out, report_out, csv_out, picker, wrist_gallery_out])
    demo.load(refresh_picker, inputs=None, outputs=picker)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
