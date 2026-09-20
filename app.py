# Web page for the client demo: upload a video (or pick one already
# approved in data/raw/), run the analysis, see the annotated video and
# results. Deliberately goes beyond CLAUDE.md section 14's original "no
# styling work" -- explicit, user-requested departure, recorded in
# CLAUDE.md's section 3 amendments -- to make the tool something a client
# can actually sit down with, not just a bare functional demo.

import json
import os
import shutil

import gradio as gr
import pandas as pd

from src.calibrate import calibrate_video
from src.detect_track import run_detect_track_cached
from src.events import run_events_cached
from src.metrics import run_metrics_cached
from src.pose import run_pose_cached
from src.profiling import load_config
from src.render import render_video

CONFIG = load_config("config.yaml")

STAGES = [
    "Calibrating (automatic -- no clicking needed)",
    "Detecting and tracking swimmers",
    "Finding body keypoints",
    "Detecting hand-entry events",
    "Computing metrics",
    "Rendering the annotated video",
]


def list_raw_videos():
    raw_dir = "data/raw"
    if not os.path.isdir(raw_dir):
        return []
    return sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".mp4"))


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
    # A distinct coloured panel instead of a line of markdown -- provisional
    # or rejected data should be impossible to scroll past without noticing.
    # Plain text only (no nested <code>/<strong>) -- the app's own dark
    # theme was found to override those tags' text colour to match the
    # (dark) page background, making them silently invisible inside this
    # light-background box.
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


def results_summary_md(detect_diag, events_diag, metrics_diag):
    # The honest headline, before the detail table -- how much of the
    # video is actually usable data, not just what got accepted.
    def get(d, key, default=None):
        return d.get(key, default) if d else default

    longest_sec = get(detect_diag, "detect_longest_track_sec")
    longest_frames = get(detect_diag, "detect_longest_track_frames")
    entries = get(events_diag, "event_entries_detected")
    lengths_ok = get(metrics_diag, "result_lengths_accepted", 0)
    lengths_total = get(metrics_diag, "result_lengths_computed", 0)

    longest_line = (f"Longest continuously-tracked segment: **{longest_sec:.1f}s** ({longest_frames} frames)"
                    if longest_sec is not None else "Longest continuously-tracked segment: n/a (cached result)")
    entries_line = (f"Real hand-entries detected: **{entries}**" if entries is not None
                    else "Real hand-entries detected: n/a (cached result)")

    return (f"**Summary** -- {longest_line} · {entries_line} · "
            f"Lengths accepted: **{lengths_ok} / {lengths_total}**")


def diagnostics_summary_md(detect_diag, pose_diag, events_diag, metrics_diag):
    def get(d, key, default="n/a (cached result -- see CSV for full data)"):
        return d.get(key, default) if d else default

    return f"""
### Detection & tracking
- Frames processed: {get(detect_diag, 'detect_frames_total')}
- Tracks found (after lane merge): {get(detect_diag, 'detect_tracks_found')}
- Colour-based detections added: {get(detect_diag, 'color_blob_synthesized')}

### Pose
- Wrist visibility: {pose_diag.get('quality_wrist_visible_pct', 0):.1f}% (gate: ≥70%)
- Frames with a valid pose reading: {get(pose_diag, 'pose_frames_processed')} / {get(pose_diag, 'pose_frames_available')}

### Events
- Entries detected: {get(events_diag, 'event_entries_detected')}
- Exits detected: {get(events_diag, 'event_exits_detected')}
- Tracks too short to analyse: {get(events_diag, 'event_tracks_skipped_too_short')}

### Metrics / quality gate
- Lengths computed: {get(metrics_diag, 'result_lengths_computed')}
- Lengths accepted (passed the quality gate): {get(metrics_diag, 'result_lengths_accepted')}
- Laps computed: {get(metrics_diag, 'result_laps_computed')}
"""


def build_cycle_chart(metrics_df):
    # Charts the per-track stroke timing rather than per-length figures.
    # Stroke rate is derived from timestamps alone, so it survives the
    # unresolved distance-scale problem that leaves every per-length
    # stroke_rate_cpm/distance_per_cycle_m blank.
    if metrics_df.empty or "row_type" not in metrics_df:
        return None
    tracks = metrics_df[(metrics_df["row_type"] == "track") & metrics_df["stroke_rate_cpm"].notna()]
    if tracks.empty:
        return None
    chart_df = tracks[["track_id", "stroke_rate_cpm", "cycle_count"]].melt(
        id_vars="track_id", var_name="metric", value_name="value")
    return chart_df


def run_pipeline(uploaded_file, picked_video, progress=gr.Progress()):
    n = len(STAGES)
    empty_table = pd.DataFrame([{"note": "Run an analysis to see results here."}])
    NUM_OUTPUTS = 9

    if uploaded_file is not None:
        video_path = os.path.join("data", "raw", os.path.basename(uploaded_file))
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        shutil.copy(uploaded_file, video_path)
    elif picked_video:
        video_path = os.path.join("data", "raw", picked_video)
    else:
        yield (render_stage_status(0, n, "Upload a video or pick one from data/raw/ first."),
               "", styled_callout("warning", "No video selected", "Upload a file or choose one from the dropdown."),
               None, empty_table, None, None, None, "")
        return

    try:
        progress(0.02)
        yield (render_stage_status(0, n),) + (gr.skip(),) * (NUM_OUTPUTS - 1)
        if not os.path.exists(calibration_path(video_path)):
            calibrate_video(video_path, CONFIG)
        calib_preview = preview_path(video_path) if os.path.exists(preview_path(video_path)) else None

        progress(0.15)
        yield (render_stage_status(1, n),) + (gr.skip(),) * 6 + (calib_preview,) + (gr.skip(),) * 1
        _, detect_diag = run_detect_track_cached(video_path, CONFIG)

        progress(0.45)
        yield (render_stage_status(2, n),) + (gr.skip(),) * (NUM_OUTPUTS - 1)
        _, pose_diag = run_pose_cached(video_path, CONFIG)

        progress(0.7)
        yield (render_stage_status(3, n),) + (gr.skip(),) * (NUM_OUTPUTS - 1)
        _, events_diag = run_events_cached(video_path, CONFIG)

        progress(0.85)
        yield (render_stage_status(4, n),) + (gr.skip(),) * (NUM_OUTPUTS - 1)
        metrics_df, metrics_diag = run_metrics_cached(video_path, CONFIG)

        progress(0.95)
        yield (render_stage_status(5, n),) + (gr.skip(),) * (NUM_OUTPUTS - 1)
        annotated_path = render_video(video_path, CONFIG)

    except Exception as exc:
        yield (render_stage_status(n, n, str(exc)),
               "", styled_callout("warning", "Analysis failed", str(exc)),
               None, empty_table, None, None, None, "")
        return

    accepted_any = not metrics_df.empty and (metrics_df["rejection_reason"] == "").any()
    if metrics_df.empty or not accepted_any:
        callout = styled_callout(
            "warning", "No lengths met the quality gate",
            "The tracked position never reached the clean zone (5-20m), or the implied speed was "
            "physically implausible (a tracking artefact, not a real length). See the Diagnostics tab "
            "and the table's rejection_reason column for exactly why -- this is a real, honest result, "
            "not a bug.")
    else:
        callout = styled_callout(
            "info", "Read the numbers below as approximate.",
            "No pool wall is visible in this footage, so the distance scale is estimated from the "
            "swimmer's own body size, not measured directly. Rows may cover only part of the clean "
            "zone or have no stroke-cycle data -- check each row's data_quality_note before quoting "
            "a number. Benchmark comparisons use placeholder ranges, not real professional-swimmer data.")

    summary_md = results_summary_md(detect_diag, events_diag, metrics_diag)
    display_df = metrics_df if not metrics_df.empty else pd.DataFrame(
        [{"note": "No lengths were detected for this video -- see the Diagnostics tab."}])

    wrist_plots = sorted(
        os.path.join("outputs", f) for f in os.listdir("outputs")
        if f.startswith(os.path.splitext(os.path.basename(video_path))[0]) and f.endswith("_wrist_signal.png")
    ) if os.path.isdir("outputs") else []

    diag_md = diagnostics_summary_md(detect_diag, pose_diag, events_diag, metrics_diag)
    chart_df = build_cycle_chart(metrics_df)

    yield (render_stage_status(n, n), summary_md, callout, annotated_path, display_df,
           chart_df, wrist_plots, calib_preview, diag_md)


THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="blue")

CUSTOM_CSS = """
.stage-status { font-size: 15px; line-height: 1.9; }
"""

with gr.Blocks(title="Swimming Performance Tracking (POC)") as demo:
    gr.Markdown("# 🏊 Swimming Performance Tracking — Proof of Concept")
    gr.Markdown(
        "Upload a freestyle swim video (side view, one or more lanes), or pick one already "
        "approved in `data/raw/`, then run the analysis. Processing takes several minutes "
        "depending on video length -- this is a proof of concept, not the production system. "
        "See the **About** tab for scope and known limitations, or "
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
                video_out = gr.Video(label="Annotated video")
                with gr.Column():
                    chart_out = gr.BarPlot(
                        x="track_id", y="value", color="metric",
                        title="Stroke rate (cycles/min) and cycle count, per tracked swimmer",
                        x_title="Track", y_title="Value", height=280)
            table_out = gr.Dataframe(label="Results (per length, per lap, and per track)",
                                      value=pd.DataFrame([{"note": "Run an analysis to see results here."}]))

        with gr.Tab("Diagnostics"):
            gr.Markdown("Internal detail for checking *why* a result came out the way it did.")
            with gr.Row():
                calib_preview_out = gr.Image(label="Calibration preview (grid should follow the pool/lane edges)")
                wrist_gallery_out = gr.Gallery(label="Wrist-height signal per analysed track", columns=2, height=400)
            diag_md_out = gr.Markdown()

        with gr.Tab("About"):
            gr.Markdown("""
## What this is

A proof-of-concept pipeline that takes a freestyle swim video and estimates
stroke rate, distance per stroke cycle, and average speed, using off-the-shelf
detection/tracking/pose models -- no custom training. Built to test whether
this approach is viable before investing in production camera hardware or a
fine-tuned model.

## What to trust, and what to treat as provisional

- **Detection & tracking**: the pretrained person detector is measurably weak
  on swimmers in this kind of footage (median confidence ~0.13) -- horizontal,
  partly submerged, viewed at a close, low angle. Track-merging and a
  colour-based supplementary signal (skin/cap/splash against the pool's
  water colour) help, but gaps still happen.
- **Distance/speed numbers are approximate.** No frame in the available
  footage shows a pool wall or other fixed landmark, so the pixel-to-metre
  scale is estimated from the swimmer's own body size, not measured
  directly. Every such result is flagged `approximate_anthropometric`.
- **Benchmark comparisons use placeholder ranges**, not real professional-
  swimmer data.
- **A result of "no lengths met the quality gate" is a valid, honest
  outcome** -- the gate exists specifically so this tool never reports a
  number the underlying data doesn't actually support.

## Production recommendations

Higher frame rate (50-60fps vs. typical 25-30fps internet footage), higher
resolution, a camera position chosen for this specific use case (not
whatever was available), and eventually a fine-tuned detector once labelled
swimming footage exists, would each directly improve on the limitations
above.
""")

    run_button.click(
        run_pipeline, inputs=[upload, picker],
        outputs=[stage_status, summary_md_out, callout_box, video_out, table_out, chart_out,
                 wrist_gallery_out, calib_preview_out, diag_md_out])


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
