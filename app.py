# Small web page for the client demo: upload a video (or pick one already
# approved in data/raw/), run the analysis, see the annotated video and
# results table. No login, no database, no styling work -- it exists so
# the client can try this themselves rather than watch a screen recording.

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


def list_raw_videos():
    raw_dir = "data/raw"
    if not os.path.isdir(raw_dir):
        return []
    return sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".mp4"))


def calibration_path(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join("data", "calibration", f"{stem}.json")


def run_pipeline(uploaded_file, picked_video, progress=gr.Progress()):
    if uploaded_file is not None:
        video_path = os.path.join("data", "raw", os.path.basename(uploaded_file))
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        shutil.copy(uploaded_file, video_path)
    elif picked_video:
        video_path = os.path.join("data", "raw", picked_video)
    else:
        raise gr.Error("Upload a video or pick one from data/raw/ first.")

    progress(0.05, desc="Calibrating automatically (no clicking needed)...")
    if not os.path.exists(calibration_path(video_path)):
        calibrate_video(video_path, CONFIG)

    progress(0.2, desc="Detecting and tracking swimmers (this is the slow part)...")
    run_detect_track_cached(video_path, CONFIG)

    progress(0.5, desc="Finding body keypoints...")
    _, pose_diag = run_pose_cached(video_path, CONFIG)

    progress(0.75, desc="Detecting hand-entry events...")
    run_events_cached(video_path, CONFIG)

    progress(0.85, desc="Computing metrics...")
    metrics_df, _ = run_metrics_cached(video_path, CONFIG)

    progress(0.95, desc="Rendering the annotated video...")
    annotated_path = render_video(video_path, CONFIG)

    if metrics_df.empty or metrics_df["rejection_reason"].replace("", None).notna().all():
        warning = (
            "**No lengths for this video met the quality gate** -- either no complete stroke "
            "cycles were found in the clean zone, or wrist-tracking confidence was too low. "
            "See the table's `rejection_reason` column and CLAUDE.md's known limitations."
        )
    else:
        warning = (
            "**PROVISIONAL** -- this video's length-scale calibration is an approximate estimate "
            "(no pool wall is visible in this footage to measure against directly), and the "
            "benchmark comparison columns use placeholder ranges, not real professional-swimmer "
            "data. Read distance/speed numbers as directionally useful, not as precise measurements."
        )

    display_df = metrics_df if not metrics_df.empty else pd.DataFrame(
        [{"note": "No lengths were detected for this video -- see CLAUDE.md's known limitations."}])
    return annotated_path, display_df, warning


with gr.Blocks(title="Swimming Performance Tracking (POC)") as demo:
    gr.Markdown("# Swimming Performance Tracking — Proof of Concept")
    gr.Markdown(
        "Upload a freestyle swim video (side view, one or more lanes), or pick one already "
        "approved in `data/raw/`, then run the analysis. Processing takes several minutes "
        "depending on video length -- this is a proof of concept, not the production system. "
        "See [CLAUDE.md](CLAUDE.md) for the full specification and known limitations."
    )

    with gr.Row():
        upload = gr.File(label="Upload a video", file_types=[".mp4"])
        picker = gr.Dropdown(choices=list_raw_videos(), label="...or pick one already in data/raw/")

    run_button = gr.Button("Run analysis", variant="primary")

    warning_box = gr.Markdown()
    with gr.Row():
        video_out = gr.Video(label="Annotated video")
        table_out = gr.Dataframe(label="Results (per length and per lap)")

    run_button.click(run_pipeline, inputs=[upload, picker], outputs=[video_out, table_out, warning_box])


if __name__ == "__main__":
    demo.launch()
