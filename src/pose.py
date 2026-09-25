# Finds body keypoints -- especially the wrists -- for each tracked swimmer.
#
# Runs on crops of the already-tracked bounding boxes, not full frames.
# Applies the same per-frame camera-motion correction detect_track.py
# computed (stored alongside its cached tracks), so every keypoint that
# leaves this stage is already in the calibration's reference-frame pixel
# space -- events.py can compare a wrist straight to a lane's water line
# with no further correction.
#
# Reports how often it found the swimmer and how often a wrist was visible.
# The quality gate those numbers used to feed (70% wrist visibility) now
# measures what it stood in for -- the share of strokes actually timed --
# in src/metrics.py (CLAUDE.md section 8.2 amendment).

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.detect_track import cache_path as detect_cache_path
from src.detect_track import run_detect_track_cached
from src.profiling import Timer, load_config, model_path

# COCO keypoint order, used by YOLO11-pose
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]



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

    # events.py discards any track shorter than min_track_frames_for_analysis
    # before it ever looks at pose data, so paying for pose inference on
    # those frames is pure waste -- on real footage the vast majority of
    # tracks are short fragments (a few frames each) from a detector that's
    # weak on this footage (see CLAUDE.md section 3 amendment), so this cuts
    # total pose work substantially without changing any result.
    min_frames = config["events"]["min_track_frames_for_analysis"]
    track_lengths = tracks_df["track_id"].value_counts()
    long_enough_ids = track_lengths[track_lengths >= min_frames].index
    tracks_df = tracks_df[tracks_df["track_id"].isin(long_enough_ids)]
    if tracks_df.empty:
        raise RuntimeError(
            f"No track has at least {min_frames} frames -- nothing long enough for pose/event analysis.")

    model = YOLO(model_path(config["pose"]["model"]))
    conf_threshold = config["pose"]["conf_threshold"]
    person_conf = config["pose"]["person_conf_threshold"]
    batch_size = config["pose"]["batch_size"]
    half = config["pose"]["half_precision"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    records = []
    frame_numbers = sorted(tracks_df["frame"].unique())
    # Grouped once up front: the per-frame `tracks_df[tracks_df["frame"] ==
    # n]` scan this replaces was a full table scan per frame, i.e. quadratic
    # in the number of tracked detections.
    rows_by_frame = {n: g for n, g in tracks_df.groupby("frame")}
    needed = set(frame_numbers)

    def consume_result(row, ox, oy, result):
        kpts = result.keypoints.cpu().numpy() if result.keypoints is not None else None
        if kpts is None or kpts.xy.shape[0] == 0:
            return  # the crop didn't yield a person -- occlusion, splash, etc.

        # The crop is cut around a box the tracker already says is the
        # swimmer, so when the pose model returns several people (more likely
        # at the low person threshold, and the padding can take in a
        # neighbouring lane) keep the one that overlaps that box best, not
        # simply the most confident.
        best = 0
        if result.boxes is not None and len(result.boxes) > 1:
            b = result.boxes.xyxy.cpu().numpy()
            tx1, ty1, tx2, ty2 = row["x1"] - ox, row["y1"] - oy, row["x2"] - ox, row["y2"] - oy
            iw = np.clip(np.minimum(b[:, 2], tx2) - np.maximum(b[:, 0], tx1), 0, None)
            ih = np.clip(np.minimum(b[:, 3], ty2) - np.maximum(b[:, 1], ty1), 0, None)
            inter = iw * ih
            union = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (tx2 - tx1) * (ty2 - ty1) - inter
            best = int(np.argmax(inter / np.maximum(union, 1e-6)))
        person_score = float(result.boxes.conf[best]) if result.boxes is not None and len(result.boxes) else np.nan

        xy = kpts.xy[best]
        confs = kpts.conf[best] if kpts.conf is not None else np.zeros(len(xy))
        M = frame_transform(row)

        record = {
            "frame": row["frame"], "time_sec": row["time_sec"],
            "track_id": row["track_id"], "lane": row["lane"], "person_conf": person_score,
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

    def run_batch(crops, offsets, meta):
        if not crops:
            return
        # The person threshold, not the keypoint one: the pose model runs its
        # own person detector on the crop first, and a swimmer scores low with
        # stock COCO weights -- measured on DIVE_P004/P012/P022/P023, at 0.30
        # it found nobody in 43-60% of crops that the tracker had already
        # confirmed held the swimmer. Keypoints are still judged against
        # conf_threshold downstream.
        results = model.predict(crops, conf=person_conf, device=0, half=half, verbose=False)
        for row, (ox, oy), result in zip(meta, offsets, results):
            consume_result(row, ox, oy, result)

    # Decoded sequentially rather than seeking to each wanted frame. Random
    # seeking in long-GOP 4K video forces the decoder to fall back to the
    # preceding keyframe and roll forward every time, which dominated this
    # stage's runtime; the frames wanted here are dense enough (whole tracks,
    # never isolated frames) that one linear pass is far cheaper.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    crops, offsets, meta = [], [], []
    while frame_idx <= frame_numbers[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in needed:
            for _, row in rows_by_frame[frame_idx].iterrows():
                crop, ox, oy = crop_box(frame, row["x1"], row["y1"], row["x2"], row["y2"])
                if crop.size == 0:
                    continue
                crops.append(crop)
                offsets.append((ox, oy))
                meta.append(row)
            if len(crops) >= batch_size:
                run_batch(crops, offsets, meta)
                crops, offsets, meta = [], [], []
        frame_idx += 1
    run_batch(crops, offsets, meta)

    cap.release()

    df = pd.DataFrame.from_records(records)
    # The denominator is every tracked detection pose was ATTEMPTED on, not
    # just the ones where it succeeded. Measuring over successful rows alone
    # excluded exactly the failures this gate exists to catch -- measured on
    # one real track, the filtered figure read 78.6% (passing CLAUDE.md
    # section 8.2's 70% gate) where true coverage was 41.8%.
    attempted = len(tracks_df)
    if df.empty or attempted == 0:
        wrist_visible_n, wrist_visible_pct = 0, 0.0
    else:
        wrist_visible = (df["left_wrist_conf"] >= conf_threshold) | (df["right_wrist_conf"] >= conf_threshold)
        wrist_visible_n = int(wrist_visible.sum())
        wrist_visible_pct = 100.0 * wrist_visible_n / attempted

    diagnostics = {
        "quality_wrist_visible_pct": wrist_visible_pct,
        "quality_wrist_visible_pct_of_successful_crops": (100.0 * wrist_visible_n / len(df)) if len(df) else 0.0,
        "quality_person_found_pct": (100.0 * len(df) / attempted) if attempted else 0.0,
        "pose_rows_produced": len(df),
        "pose_detections_attempted": attempted,
        "pose_frames_available": len(frame_numbers),
    }
    return df, diagnostics


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or os.path.join(config["cache"]["directory"], "pose")
    os.makedirs(cache_dir, exist_ok=True)
    # Built on top of detect_track's own cache identity. The previous key was
    # video + pose model + pose threshold only, so two runs using different
    # DETECTION models (or different camera-motion tuning) silently shared a
    # single pose cache and the second run read the first one's boxes back.
    upstream = os.path.splitext(os.path.basename(detect_cache_path(video_path, config)))[0]
    model = os.path.splitext(os.path.basename(config["pose"]["model"]))[0]
    conf = config["pose"]["conf_threshold"]
    person = config["pose"]["person_conf_threshold"]
    return os.path.join(cache_dir, f"{upstream}__pose_{model}__conf{conf}__person{person}.csv")


def run_pose_cached(video_path, config, calibration_dir="data/calibration"):
    path = cache_path(video_path, config)
    if config["cache"]["enabled"] and os.path.exists(path):
        print(f"Using cached pose result: {path}")
        df = pd.read_csv(path)
        conf_threshold = config["pose"]["conf_threshold"]
        # Same honest denominator as run_detect_track: every detection pose
        # was attempted on, not just the rows where it produced keypoints.
        tracks_df, _ = run_detect_track_cached(video_path, config, calibration_dir)
        min_frames = config["events"]["min_track_frames_for_analysis"]
        track_lengths = tracks_df["track_id"].value_counts()
        attempted = int(tracks_df["track_id"].isin(
            track_lengths[track_lengths >= min_frames].index).sum())
        if len(df) and attempted:
            wrist_visible = (df["left_wrist_conf"] >= conf_threshold) | \
                             (df["right_wrist_conf"] >= conf_threshold)
            wrist_visible_pct = 100.0 * int(wrist_visible.sum()) / attempted
        else:
            wrist_visible_pct = 0.0
        return df, {"cached": True, "quality_wrist_visible_pct": wrist_visible_pct,
                     "quality_person_found_pct": (100.0 * len(df) / attempted) if attempted else 0.0,
                     "pose_detections_attempted": attempted, "pose_rows_produced": len(df)}

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

    print(f"Processed in {timers['time_pose_sec']:.1f}s")
    print(diagnostics)
    print(f"\nperson found in {diagnostics.get('quality_person_found_pct', 0):.1f}% of tracked crops; "
          f"wrist visible in {diagnostics['quality_wrist_visible_pct']:.1f}% of them. The quality gate is now "
          f"stroke capture (CLAUDE.md section 8.2 amendment), reported by src.metrics.")


if __name__ == "__main__":
    main()
