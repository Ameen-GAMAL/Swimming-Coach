# The single command-line entry point: runs the whole pipeline on one video,
# the same stages in the same order as the web page (app.py). Every stage
# caches its result, so a repeat run only redoes what changed.

import argparse
import os

from src import __version__
from src.calibrate import calibrate_video
from src.detect_track import run_detect_track_cached
from src.events import run_events_cached
from src.metrics import run_metrics_cached
from src.pose import run_pose_cached
from src.profiling import Timer, load_config
from src.render import render_video
from src.report import build_report
from src.rope_odometry import run_rope_odometry_cached


def main():
    parser = argparse.ArgumentParser(description="Swimming performance tracking POC: analyse one video end to end.")
    parser.add_argument("video", help="Path to a video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--no-render", action="store_true", help="Skip the annotated video (the slowest optional step)")
    args = parser.parse_args()

    config = load_config(args.config)  # fails loudly if config.yaml is missing or malformed
    print(f"Swimming POC pipeline v{__version__}")
    stem = os.path.splitext(os.path.basename(args.video))[0]
    timers = {}
    if not os.path.exists(os.path.join("data", "calibration", f"{stem}.json")):
        with Timer(timers, "calibrate"):
            calibrate_video(args.video, config)
    with Timer(timers, "detect_track"):
        run_detect_track_cached(args.video, config)
    with Timer(timers, "rope_odometry"):
        run_rope_odometry_cached(args.video, config)
    with Timer(timers, "pose"):
        run_pose_cached(args.video, config)
    with Timer(timers, "events"):
        run_events_cached(args.video, config)
    with Timer(timers, "metrics"):
        metrics_df, _ = run_metrics_cached(args.video, config)
    if not args.no_render:
        with Timer(timers, "render"):
            render_video(args.video, config)
    with Timer(timers, "report"):
        report = build_report(args.video, config)

    swims = metrics_df[metrics_df["row_type"] == "swim"] if not metrics_df.empty else metrics_df
    for r in swims.itertuples():
        if r.rejection_reason:
            print(f"swim {int(r.segment_index) + 1}: {r.start_sec:.1f}-{r.end_sec:.1f}s -- {r.rejection_reason}")
            continue
        dpc = "" if r.distance_per_cycle_m != r.distance_per_cycle_m else f", {r.distance_per_cycle_m:.2f} m/cycle"
        print(f"swim {int(r.segment_index) + 1}: {r.distance_m:.1f} m in {r.duration_sec:.1f}s = "
              f"{r.average_speed_ms:.2f} m/s{dpc}")
    print("stage times: " + ", ".join(f"{k} {v:.0f}s" for k, v in timers.items()))
    print(f"report: {report}")


if __name__ == "__main__":
    main()
