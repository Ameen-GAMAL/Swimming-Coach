# Finds body keypoints -- especially the wrists -- for each tracked swimmer.
#
# Runs on crops of the already-tracked bounding boxes, not full frames.
# Applies the same per-frame camera-motion correction detect_track.py
# computed (stored alongside its cached tracks), so every keypoint that
# leaves this stage is already in the calibration's reference-frame pixel
# space -- events.py can compare a wrist straight to a lane's water line
# with no further correction.
#
# Measures before assuming: the first thing reported is what percentage of
# frames had a wrist visible above the confidence threshold. Below 70%, the
# pretrained model isn't good enough and this stops rather than silently
# producing bad numbers (CLAUDE.md section 8.2).

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.detect_track import run_detect_track_cached
from src.profiling import Timer, load_config

# COCO keypoint order, used by YOLO11-pose
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

WRIST_VISIBILITY_THRESHOLD_PCT = 70  # CLAUDE.md section 8.2's fixed gate


def project_point(M, x, y):
    point = np.array([x, y, 1.0])
    mapped = M @ point
    mapped /= mapped[2]
    return mapped[0], mapped[1]


def crop_box(frame, x1, y1, x2, y2, pad_frac=0.25):
    # Pad around the tracked box so limbs extended past it (an arm mid-stroke)
    # aren't cut off
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    return frame[cy1:cy2, cx1:cx2], cx1, cy1


def frame_transform(row):
    # None if this frame's camera motion wasn't reliable (see detect_track.py)
    if np.isnan(row["m00"]):
        return None
    return np.array([
        [row["m00"], row["m01"], row["m02"]],
        [row["m10"], row["m11"], row["m12"]],
        [row["m20"], row["m21"], row["m22"]],
    ])


def run_pose(video_path, config, calibration_dir="data/calibration"):
    tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
    if tracks_df.empty:
        raise RuntimeError("No tracked swimmers found -- nothing to run pose on.")

    model = YOLO(config["pose"]["model"])
    conf_threshold = config["pose"]["conf_threshold"]
    batch_size = config["pose"]["batch_size"]
    half = config["pose"]["half_precision"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    records = []
    frame_numbers = sorted(tracks_df["frame"].unique())

    for batch_start in range(0, len(frame_numbers), batch_size):
        batch_frame_numbers = frame_numbers[batch_start: batch_start + batch_size]
        crops, offsets, meta = [], [], []
        for frame_num in batch_frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
            ok, frame = cap.read()
            if not ok:
                continue
            for _, row in tracks_df[tracks_df["frame"] == frame_num].iterrows():
                crop, ox, oy = crop_box(frame, row["x1"], row["y1"], row["x2"], row["y2"])
                if crop.size == 0:
                    continue
                crops.append(crop)
                offsets.append((ox, oy))
                meta.append(row)

        if not crops:
            continue

        results = model.predict(crops, conf=conf_threshold, device=0, half=half, verbose=False)

        for row, (ox, oy), result in zip(meta, offsets, results):
            kpts = result.keypoints.cpu().numpy() if result.keypoints is not None else None
            if kpts is None or kpts.xy.shape[0] == 0:
                continue  # the crop didn't yield a person -- occlusion, splash, etc.

            # A tight crop should hold exactly the tracked swimmer, but if the
            # model finds more than one, keep the most confident detection.
            best = 0
            if result.boxes is not None and len(result.boxes) > 1:
                best = int(np.argmax(result.boxes.conf.cpu().numpy()))

            xy = kpts.xy[best]
            confs = kpts.conf[best] if kpts.conf is not None else np.zeros(len(xy))
            M = frame_transform(row)

            record = {
                "frame": row["frame"], "time_sec": row["time_sec"],
                "track_id": row["track_id"], "lane": row["lane"],
            }
            for i, name in enumerate(KEYPOINT_NAMES):
                px, py = xy[i, 0] + ox, xy[i, 1] + oy  # back to full-frame pixel space
                if M is not None:
                    ref_x, ref_y = project_point(M, px, py)
                else:
                    ref_x, ref_y = np.nan, np.nan
                record[f"{name}_x"] = px
                record[f"{name}_y"] = py
                record[f"{name}_ref_x"] = ref_x
                record[f"{name}_ref_y"] = ref_y
                record[f"{name}_conf"] = float(confs[i])
            records.append(record)

    cap.release()

    df = pd.DataFrame.from_records(records)
    if df.empty:
        wrist_visible_pct = 0.0
    else:
        wrist_visible = (df["left_wrist_conf"] >= conf_threshold) | (df["right_wrist_conf"] >= conf_threshold)
        wrist_visible_pct = 100.0 * wrist_visible.mean()

    diagnostics = {
        "quality_wrist_visible_pct": wrist_visible_pct,
        "pose_frames_processed": len(df),
        "pose_frames_available": len(frame_numbers),
    }
    return df, diagnostics


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or os.path.join(config["cache"]["directory"], "pose")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    model = os.path.splitext(config["pose"]["model"])[0]
    conf = config["pose"]["conf_threshold"]
    return os.path.join(cache_dir, f"{stem}__{model}__conf{conf}.csv")


def run_pose_cached(video_path, config, calibration_dir="data/calibration"):
    path = cache_path(video_path, config)
    if config["cache"]["enabled"] and os.path.exists(path):
        print(f"Using cached pose result: {path}")
        df = pd.read_csv(path)
        wrist_visible = (df["left_wrist_conf"] >= config["pose"]["conf_threshold"]) | \
                         (df["right_wrist_conf"] >= config["pose"]["conf_threshold"])
        wrist_visible_pct = 100.0 * wrist_visible.mean() if len(df) else 0.0
        return df, {"cached": True, "quality_wrist_visible_pct": wrist_visible_pct}

    df, diagnostics = run_pose(video_path, config, calibration_dir)
    if config["cache"]["enabled"]:
        df.to_csv(path, index=False)
        print(f"Cached pose result: {path}")
    return df, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Find body keypoints for tracked swimmers in a video.")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    timers = {}
    with Timer(timers, "time_pose_sec"):
        df, diagnostics = run_pose_cached(args.video, config)

    pct = diagnostics["quality_wrist_visible_pct"]
    print(f"Processed in {timers['time_pose_sec']:.1f}s")
    print(diagnostics)
    print(f"\nquality_wrist_visible_pct = {pct:.1f}%")
    if pct >= WRIST_VISIBILITY_THRESHOLD_PCT:
        print(f"{pct:.1f}% >= {WRIST_VISIBILITY_THRESHOLD_PCT}% -- the pretrained pose model is good enough, continue.")
    else:
        print(f"STOP: {pct:.1f}% < {WRIST_VISIBILITY_THRESHOLD_PCT}%. Do not continue past this without a decision "
              f"from the user -- the fallback is annotating 300-500 frames and fine-tuning the pose model "
              f"(CLAUDE.md section 8.2).")


if __name__ == "__main__":
    main()
