# Draws the measurement grid, each swimmer's box/ID/lane, their tracked
# wrist, a flash at each detected hand entry, and running counters onto
# the original video -- this is what lets the client visually check the
# distance maths against the real footage.

import argparse
import json
import os

import cv2
import numpy as np

from src.calibrate import draw_length_grid, draw_lines
from src.detect_track import run_detect_track_cached
from src.events import pick_tracked_hand, run_events_cached
from src.pose import run_pose_cached
from src.profiling import Timer, load_config


def load_calibration(video_path, calibration_dir="data/calibration"):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    calib_path = os.path.join(calibration_dir, f"{stem}.json")
    if not os.path.exists(calib_path):
        raise RuntimeError(f"No calibration found for this video. Run calibrate.py first. Expected: {calib_path}")
    with open(calib_path) as f:
        return json.load(f)


def cumulative_distance_m(track_pos_sorted):
    # Real distance travelled so far (not net displacement) -- sum of the
    # absolute position change frame to frame, along the pool's length axis.
    y = track_pos_sorted["y_m"]
    return y.diff().abs().fillna(0).cumsum()


def render_video(video_path, config, calibration_dir="data/calibration", output_dir="outputs", _max_frames=None):
    calibration = load_calibration(video_path, calibration_dir)
    H = np.array(calibration["homography"])
    width_m = calibration["corner_span_m"]["width_m"]
    provisional = calibration.get("scale_method") == "approximate_anthropometric"

    tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
    pose_df, _ = run_pose_cached(video_path, config, calibration_dir)
    events_df, _ = run_events_cached(video_path, config, calibration_dir)

    hand_by_track = {}
    for track_id, track_pose in pose_df.groupby("track_id"):
        hand_by_track[track_id] = pick_tracked_hand(track_pose)

    tracks_df = tracks_df.sort_values(["track_id", "frame"])
    tracks_df["distance_so_far_m"] = tracks_df.groupby("track_id", group_keys=False).apply(
        lambda g: cumulative_distance_m(g), include_groups=False)

    pose_lookup = {(int(r.frame), r.track_id): r for r in pose_df.itertuples()}
    tracks_by_frame = {f: g for f, g in tracks_df.groupby("frame")}
    entries_df = events_df[events_df["event_type"] == "entry"].sort_values("frame") if not events_df.empty else events_df

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Draw at full source resolution (matches the calibration/tracking data's
    # own pixel space) but write at a downscaled size -- see config comment:
    # this OpenCV build has no working H.264 encoder, and its fallback codec
    # produces huge files at 4K.
    out_max_w = config["render"]["output_max_width"]
    out_scale = min(1.0, out_max_w / w)
    out_w, out_h = int(w * out_scale), int(h * out_scale)

    # Skip frames to hit the target output fps, rather than writing every
    # source frame at the source's own high fps -- keeps real-time playback
    # speed correct while cutting file size proportionally.
    output_fps = min(fps, config["render"]["output_fps"])
    keep_every = max(1, round(fps / output_fps))

    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{stem}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / keep_every, (out_w, out_h))

    flash_frames = int(0.3 * fps)
    pool_length_m = config["pool"]["length_m"]
    segment_size_m = config["pool"]["segment_size_m"]

    frame_idx = 0
    while True:
        if _max_frames is not None and frame_idx >= _max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        frame = draw_length_grid(frame, H, pool_length_m, segment_size_m, width_m)

        rows = tracks_by_frame.get(frame_idx)
        if rows is not None:
            for counter_line, row in enumerate(rows.itertuples()):
                x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                lane_label = "?" if row.lane is None or (isinstance(row.lane, float) and np.isnan(row.lane)) else int(row.lane)
                cv2.putText(frame, f"ID {row.track_id} lane {lane_label}", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                hand = hand_by_track.get(row.track_id)
                pose_row = pose_lookup.get((frame_idx, row.track_id))
                if hand and pose_row is not None:
                    wx, wy = getattr(pose_row, f"{hand}_wrist_x"), getattr(pose_row, f"{hand}_wrist_y")
                    if not (np.isnan(wx) or np.isnan(wy)):
                        cv2.circle(frame, (int(wx), int(wy)), 8, (255, 0, 255), -1)

                if not entries_df.empty:
                    recent = entries_df[(entries_df["track_id"] == row.track_id) &
                                         (entries_df["frame"] <= frame_idx) &
                                         (entries_df["frame"] >= frame_idx - flash_frames)]
                    if not recent.empty:
                        cv2.circle(frame, (x1, y1), 24, (0, 0, 255), 4)

                cycle_count_so_far = 0
                if not entries_df.empty:
                    cycle_count_so_far = int(((entries_df["track_id"] == row.track_id) &
                                               (entries_df["frame"] <= frame_idx)).sum())
                distance_so_far = row.distance_so_far_m if not np.isnan(row.distance_so_far_m) else 0.0
                counter_text = f"ID {row.track_id}: {cycle_count_so_far} cycles, {distance_so_far:.1f}m covered"
                cv2.putText(frame, counter_text, (10, 40 + 30 * counter_line),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if provisional:
            cv2.putText(frame, "PROVISIONAL: approximate length-scale calibration (no wall visible in source footage)",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if frame_idx % keep_every == 0:
            if out_scale < 1.0:
                frame = cv2.resize(frame, (out_w, out_h))
            writer.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  ...{frame_idx}/{frame_count} frames rendered", flush=True)

    cap.release()
    writer.release()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render an annotated video with the grid, tracked swimmers, and events.")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    timers = {}
    with Timer(timers, "time_render_sec"):
        out_path = render_video(args.video, config)

    print(f"Rendered in {timers['time_render_sec']:.1f}s")
    print(f"Annotated video: {out_path}")


if __name__ == "__main__":
    main()
