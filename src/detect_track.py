# Finds swimmers in every frame and gives each one a stable ID.
#
# The available footage pans/tracks the swimmer instead of staying fixed
# (see CLAUDE.md section 3 amendment), so before anything else, each frame's
# own camera motion is measured -- against the previous frame, then chained
# together frame by frame -- and used to correct positions back into the
# calibration's reference frame. This is what lets a single, one-time
# calibration stay valid for the whole video, even as the camera keeps moving.

import argparse
import json
import os
import types

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from src.profiling import Timer, load_config


def load_calibration(video_path, calibration_dir="data/calibration"):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    calib_path = os.path.join(calibration_dir, f"{stem}.json")
    if not os.path.exists(calib_path):
        raise RuntimeError(f"No calibration found for this video. Run calibrate.py first. Expected: {calib_path}")
    with open(calib_path) as f:
        return json.load(f)


def project_point(H, x, y):
    # Applies a 3x3 homography to a single (x, y) point
    point = np.array([x, y, 1.0])
    mapped = H @ point
    mapped /= mapped[2]
    return mapped[0], mapped[1]


def lane_boundary_real_x(calibration, H):
    # Each boundary was clicked as a straight line on the water surface, so
    # its real-world x position should be near-constant along its length --
    # average the two endpoints to get one x value per boundary.
    xs = []
    for boundary in sorted(calibration["lane_boundaries"], key=lambda b: b["boundary_index"]):
        pts_m = [project_point(H, px, py) for px, py in boundary["points"]]
        xs.append(np.mean([p[0] for p in pts_m]))
    return sorted(xs)


def assign_lane(x_m, boundary_xs):
    # A point between boundary[i] and boundary[i+1] is lane i+1 (1-indexed).
    # Outside every boundary means it's not in a calibrated lane.
    for i in range(len(boundary_xs) - 1):
        if boundary_xs[i] <= x_m <= boundary_xs[i + 1]:
            return i + 1
    return None


def merge_lane_tracks(df, config, fps):
    # A swimmer never changes lane, so if ByteTrack loses and re-acquires
    # them (a real risk here -- the pretrained person detector is weak on
    # swimmers in this footage), a new track starting in the same lane
    # shortly after an old one ends is almost certainly the same swimmer.
    # Merge those by relabelling to the earlier track's ID.
    if df.empty:
        return df
    max_gap_frames = config["tracking"]["id_reuse_max_gap_sec"] * fps

    id_map = {}
    for lane, group in df.dropna(subset=["lane"]).groupby("lane"):
        segments = sorted(
            (seg["frame"].min(), seg["frame"].max(), track_id)
            for track_id, seg in group.groupby("track_id")
        )
        current_id, chain_end = None, None
        for start, end, track_id in segments:
            if current_id is not None and start - chain_end <= max_gap_frames:
                id_map[track_id] = current_id
                chain_end = max(chain_end, end)
            else:
                current_id = track_id
                chain_end = end

    df = df.copy()
    df["track_id"] = df["track_id"].replace(id_map)
    return df


def background_crop(frame, background_region_frac):
    h = frame.shape[0]
    return frame[0:int(h * background_region_frac), :]


def detect_orb_features(orb, frame_bg_bgr, resize_width):
    # Shrinking the background crop before ORB runs is the single biggest
    # speed lever here (feature detection cost scales with pixel count, and
    # the background is large simple structures that survive downscaling
    # fine). Keypoint coordinates are scaled back up immediately so every
    # caller keeps working in the crop's original-resolution pixel space.
    gray = cv2.cvtColor(frame_bg_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, resize_width / w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    kp, des = orb.detectAndCompute(gray, None)
    if scale < 1.0:
        for k in kp:
            k.pt = (k.pt[0] / scale, k.pt[1] / scale)
    return kp, des


def estimate_step_homography(matcher, anchor_kp, anchor_des, cur_kp, cur_des, config):
    """Matches the current frame's background against the last successfully
    matched frame (the "anchor"), not the fixed calibration reference frame.
    Consecutive frames always have high background overlap regardless of how
    far the camera has panned overall, whereas matching everything against
    one fixed reference frame degrades as the pan accumulates -- measured on
    real footage, inlier matches dropped from ~90 near the reference frame to
    single digits by the end of a 45s clip. Returns a homography mapping the
    CURRENT frame's pixels to the ANCHOR frame's pixels, plus the inlier
    count (so a step with too little texture to trust can be flagged)."""
    if cur_des is None or anchor_des is None or len(cur_des) < 4 or len(anchor_des) < 4:
        return None, 0

    raw_matches = matcher.knnMatch(cur_des, anchor_des, k=2)
    ratio = config["camera_motion"]["ratio_test_threshold"]
    good = [m for pair in raw_matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 4:
        return None, 0

    src_pts = np.float32([cur_kp[m.queryIdx].pt for m in good])
    dst_pts = np.float32([anchor_kp[m.trainIdx].pt for m in good])

    M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    good_matches = int(inliers.sum()) if inliers is not None else 0
    if M is None or good_matches < config["camera_motion"]["min_good_features"]:
        return None, good_matches
    return M, good_matches


def build_tracker_args(config):
    t = config["tracking"]
    return types.SimpleNamespace(
        track_high_thresh=t["track_high_thresh"],
        track_low_thresh=t["track_low_thresh"],
        new_track_thresh=t["new_track_thresh"],
        track_buffer=t["track_buffer"],
        match_thresh=t["match_thresh"],
        fuse_score=t["fuse_score"],
    )


def run_detect_track(video_path, config, calibration_dir="data/calibration", _max_frames=None):
    calibration = load_calibration(video_path, calibration_dir)
    H = np.array(calibration["homography"])
    boundary_xs = lane_boundary_real_x(calibration, H)

    margin = config["pool_area_margin_m"]
    width_m = calibration["corner_span_m"]["width_m"]
    pool_length_m = config["pool"]["length_m"]
    x_bounds = (-margin, width_m + margin)
    y_bounds = (-margin, pool_length_m + margin)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if _max_frames is not None:
        frame_count = min(frame_count, _max_frames)

    bg_frac = config["camera_motion"]["background_region_frac"]
    orb = cv2.ORB_create(nfeatures=config["camera_motion"]["orb_features"])
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Camera-motion chain state. cumulative_M maps the current frame's pixels
    # to frame 0's pixel space, built by composing each frame-to-previous-
    # frame step as we go. Once the loop reaches the calibration's reference
    # frame, its cumulative_M is captured as ref_to_frame0 -- after that,
    # every frame's position relative to the reference frame is
    # inv(ref_to_frame0) @ cumulative_M. Frames before the reference frame
    # (the dive/entry, before calibration's chosen frame) are left
    # uncorrected rather than guessed.
    chain_kp, chain_des = None, None
    cumulative_M = np.eye(3)
    ref_to_frame0 = None

    model = YOLO(config["detection"]["model"])
    tracker = BYTETracker(build_tracker_args(config), frame_rate=int(round(fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    batch_size = config["detection"]["batch_size"]
    conf_threshold = config["detection"]["conf_threshold"]
    half = config["detection"]["half_precision"]

    records = []
    frames_low_motion_confidence = 0
    frame_idx = 0

    while frame_idx < frame_count:
        batch_frames = []
        for _ in range(batch_size):
            ok, frame = cap.read()
            if not ok:
                break
            batch_frames.append(frame)
        if not batch_frames:
            break

        results = model.predict(batch_frames, classes=[0], conf=conf_threshold, device=0, half=half, verbose=False)

        for frame, result in zip(batch_frames, results):
            kp, des = detect_orb_features(orb, background_crop(frame, bg_frac), config["camera_motion"]["match_resize_width"])

            if chain_kp is None:
                step_ok, good_features = True, 0  # first frame of the video: nothing to chain from yet
            else:
                step_M, good_features = estimate_step_homography(matcher, chain_kp, chain_des, kp, des, config)
                step_ok = step_M is not None
                if step_ok:
                    cumulative_M = cumulative_M @ step_M
            if step_ok:
                chain_kp, chain_des = kp, des  # only advance the anchor on a successful match

            if frame_idx == calibration["frame_number"]:
                ref_to_frame0 = cumulative_M.copy()

            motion_ok = step_ok and ref_to_frame0 is not None
            if motion_ok:
                M = np.linalg.inv(ref_to_frame0) @ cumulative_M
            if not motion_ok:
                frames_low_motion_confidence += 1

            boxes = result.boxes.cpu().numpy()
            # Default to keeping every detection: if we can't verify a
            # detection's real-world position this frame (motion estimate
            # failed), dropping it would fragment the track for no reason --
            # tracking association only needs frame-to-frame IoU, which
            # doesn't require the camera to be motion-corrected at all. Only
            # detections we CAN place get checked against the pool area.
            keep = np.ones(len(boxes), dtype=bool)
            ref_positions = {}
            for i in range(len(boxes)):
                cx, cy = boxes.xywh[i, 0], boxes.xywh[i, 1]
                if motion_ok:
                    ref_x, ref_y = project_point(M, cx, cy)
                    x_m, y_m = project_point(H, ref_x, ref_y)
                    inside = x_bounds[0] <= x_m <= x_bounds[1] and y_bounds[0] <= y_m <= y_bounds[1]
                    keep[i] = inside
                    ref_positions[i] = (ref_x, ref_y, x_m, y_m)
                else:
                    ref_positions[i] = (np.nan, np.nan, np.nan, np.nan)

            filtered = boxes[keep]
            tracks = tracker.update(filtered, img=frame)

            # Flattened so pose.py can rebuild this frame's raw-pixel ->
            # reference-frame transform for its own keypoints, without
            # redoing the background feature matching itself.
            m_flat = M.flatten() if motion_ok else [np.nan] * 9

            for row in tracks:
                x1, y1, x2, y2, track_id, score, cls, idx = row
                kept_indices = np.where(keep)[0]
                orig_i = kept_indices[int(idx)]
                ref_x, ref_y, x_m, y_m = ref_positions[orig_i]
                lane = assign_lane(x_m, boundary_xs)
                records.append({
                    "frame": frame_idx,
                    "time_sec": frame_idx / fps,
                    "track_id": int(track_id),
                    "lane": lane,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "ref_cx": ref_x, "ref_cy": ref_y,
                    "x_m": x_m, "y_m": y_m,
                    "score": float(score),
                    "camera_motion_good_features": good_features,
                    "m00": m_flat[0], "m01": m_flat[1], "m02": m_flat[2],
                    "m10": m_flat[3], "m11": m_flat[4], "m12": m_flat[5],
                    "m20": m_flat[6], "m21": m_flat[7], "m22": m_flat[8],
                })
            frame_idx += 1

        if frame_idx % (batch_size * 10) == 0 or frame_idx >= frame_count:
            print(f"  ...{frame_idx}/{frame_count} frames", flush=True)

    cap.release()

    df = pd.DataFrame.from_records(records)
    tracks_before_merge = df["track_id"].nunique() if not df.empty else 0
    df = merge_lane_tracks(df, config, fps)
    diagnostics = {
        "detect_frames_total": frame_count,
        "detect_frames_low_motion_confidence": frames_low_motion_confidence,
        "detect_tracks_before_merge": tracks_before_merge,
        "detect_tracks_found": df["track_id"].nunique() if not df.empty else 0,
    }
    return df, diagnostics


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or os.path.join(config["cache"]["directory"], "detect_track")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    model = os.path.splitext(config["detection"]["model"])[0]
    conf = config["detection"]["conf_threshold"]
    return os.path.join(cache_dir, f"{stem}__{model}__conf{conf}.csv")


def run_detect_track_cached(video_path, config, calibration_dir="data/calibration"):
    path = cache_path(video_path, config)
    if config["cache"]["enabled"] and os.path.exists(path):
        print(f"Using cached detect_track result: {path}")
        return pd.read_csv(path), {"cached": True}

    df, diagnostics = run_detect_track(video_path, config, calibration_dir)
    if config["cache"]["enabled"]:
        df.to_csv(path, index=False)
        print(f"Cached detect_track result: {path}")
    return df, diagnostics


def main():
    parser = argparse.ArgumentParser(description="Detect and track swimmers in a calibrated video.")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    timers = {}
    with Timer(timers, "time_detect_sec"):
        df, diagnostics = run_detect_track_cached(args.video, config)

    print(f"Processed {diagnostics.get('detect_frames_total', len(df))} frames in {timers['time_detect_sec']:.1f}s")
    print(f"Tracks found: {df['track_id'].nunique() if not df.empty else 0}")
    print(diagnostics)


if __name__ == "__main__":
    main()
