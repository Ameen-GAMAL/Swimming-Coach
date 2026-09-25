# Draws the analysis onto the original video so the client can check it
# against the footage: the filmed swimmer's box with a running panel of
# distance covered, current speed and strokes counted (distance from the lane
# rope, src/rope_odometry.py), the tracked wrist, a flash at each timed
# stroke, and other people in the pool in a muted colour. The panel shows the
# running tempo rather than a stroke count: only the strokes the pose model
# could time would be counted (70-97% of them on the tested clips), so a
# count would drift visibly behind what anyone watching counts.
#
# It used to draw the calibration's metre grid and per-track "m covered"
# counters built from calibrated positions. With the camera walking alongside
# the swimmer those were wrong -- counters could run backwards or show metres
# the swimmer never swam -- so both are gone; distance now comes only from
# the rope, the same number the metrics and the report use.

import argparse
import os

import cv2
import numpy as np

from src.detect_track import run_detect_track_cached, subject_track_ids
from src.events import pick_tracked_hand, run_events_cached
from src.metrics import subject_strokes
from src.pose import run_pose_cached
from src.profiling import Timer, load_config
from src.rope_odometry import run_rope_odometry_cached, swimmer_progress

SUBJECT_COLOUR = (0, 220, 255)
OTHER_COLOUR = (150, 150, 150)
SPEED_WINDOW_SEC = 3.0
TEMPO_STROKES = 6          # running tempo over the last this-many timed strokes


def progress_lookup(progress):
    """Per-frame distance covered so far and current speed for the filmed
    swimmer, from the rope measurement (None outside the measured span)."""
    if progress.empty:
        return None
    t = progress["time_sec"].to_numpy()
    p = progress["progress_m"].to_numpy()
    p = np.abs(p - p[0])

    def at(time_sec):
        if time_sec < t[0] or time_sec > t[-1]:
            return None
        dist = float(np.interp(time_sec, t, p))
        t0 = max(t[0], time_sec - SPEED_WINDOW_SEC)
        speed = (dist - float(np.interp(t0, t, p))) / (time_sec - t0) if time_sec - t0 > 0.5 else None
        return dist, speed
    return at


def render_video(video_path, config, calibration_dir="data/calibration", output_dir="outputs", _max_frames=None):
    tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
    pose_df, _ = run_pose_cached(video_path, config, calibration_dir)
    events_df, _ = run_events_cached(video_path, config, calibration_dir)
    rope_df, _ = run_rope_odometry_cached(video_path, config, calibration_dir)

    subject_ids = subject_track_ids(tracks_df, config["events"]["min_track_frames_for_analysis"])
    strokes = subject_strokes(events_df, pose_df, subject_ids, config)
    stroke_frames = np.sort(strokes["frame_peak"].to_numpy()) if not strokes.empty else np.array([], dtype=int)
    halves = strokes.sort_values("frame_peak")["half_cycle_sec"].to_numpy() if not strokes.empty else np.array([])
    progress_at = progress_lookup(swimmer_progress(rope_df, config))

    hand_by_track = {tid: pick_tracked_hand(g) for tid, g in pose_df.groupby("track_id")}
    pose_lookup = {(int(r.frame), r.track_id): r for r in pose_df.itertuples()}
    boxes = tracks_df.assign(area=(tracks_df["x2"] - tracks_df["x1"]) * (tracks_df["y2"] - tracks_df["y1"]))
    by_frame = {f: g for f, g in boxes.groupby("frame")}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Draw at full source resolution but write downscaled: this OpenCV build
    # has no working H.264 encoder and its fallback codec is huge at 4K.
    out_scale = min(1.0, config["render"]["output_max_width"] / w)
    out_w, out_h = int(w * out_scale), int(h * out_scale)
    # Skip frames to hit the target output fps, keeping real-time playback speed.
    keep_every = max(1, round(fps / min(fps, config["render"]["output_fps"])))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{stem}_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / keep_every, (out_w, out_h))

    flash_frames = int(0.3 * fps)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, h / 1080.0
    frame_idx = 0
    while True:
        if _max_frames is not None and frame_idx >= _max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % keep_every:
            frame_idx += 1
            continue

        rows = by_frame.get(frame_idx)
        subject_row = None
        if rows is not None:
            subj = rows[rows["track_id"].isin(subject_ids)]
            subject_row = subj.loc[subj["area"].idxmax()] if len(subj) else None
            for r in rows.itertuples():
                is_subject = subject_row is not None and r.Index == subject_row.name
                colour = SUBJECT_COLOUR if is_subject else OTHER_COLOUR
                cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), colour, max(2, int(3 * scale)))
                if not is_subject:
                    continue
                hand = hand_by_track.get(r.track_id)
                pr = pose_lookup.get((frame_idx, r.track_id))
                if hand and pr is not None:
                    wx, wy = getattr(pr, f"{hand}_wrist_x"), getattr(pr, f"{hand}_wrist_y")
                    if not (np.isnan(wx) or np.isnan(wy)):
                        cv2.circle(frame, (int(wx), int(wy)), int(10 * scale), (255, 0, 255), -1)
                k = np.searchsorted(stroke_frames, frame_idx, side="right")
                if k and frame_idx - stroke_frames[k - 1] <= flash_frames:
                    cv2.circle(frame, (int(r.x1), int(r.y1)), int(28 * scale), (0, 0, 255), max(3, int(4 * scale)))

        # running panel for the filmed swimmer
        t = frame_idx / fps
        k = int(np.searchsorted(stroke_frames, frame_idx, side="right"))
        recent = halves[max(0, k - TEMPO_STROKES):k]
        recent = recent[~np.isnan(recent)] if len(recent) else recent
        tempo = 60.0 / (2.0 * float(np.median(recent))) if len(recent) >= 3 else None
        prog = progress_at(t) if progress_at else None
        lines = [f"Tempo: {tempo:.0f} cycles/min" if tempo else "Tempo: --"]
        if prog is not None:
            dist, speed = prog
            lines.insert(0, f"Distance: {dist:.1f} m")
            lines.insert(1, f"Speed: {speed:.2f} m/s" if speed is not None else "Speed: --")
        pad = int(14 * scale)
        line_h = int(46 * scale)
        panel_w, panel_h = int(430 * scale), pad * 2 + line_h * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (2 * pad, pad + line_h * (i + 1) - int(8 * scale)), font, 1.2 * scale,
                        (255, 255, 255), max(2, int(2 * scale)), cv2.LINE_AA)

        if out_scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 1000 < keep_every:
            print(f"  ...{frame_idx}/{frame_count} frames rendered", flush=True)

    cap.release()
    writer.release()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render an annotated video of the analysis.")
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
