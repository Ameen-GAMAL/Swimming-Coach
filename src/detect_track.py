# Finds swimmers in every frame and gives each one a stable ID.
#
# The available footage pans/tracks the swimmer instead of staying fixed
# (see CLAUDE.md section 3 amendment), so before anything else, each frame's
# own camera motion is measured -- against the previous frame, then chained
# together frame by frame -- and used to correct positions back into the
# calibration's reference frame. This is what lets a single, one-time
# calibration stay valid for the whole video, even as the camera keeps moving.

import argparse
import hashlib
import json
import os
import types

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
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


def merge_continuous_tracks(df, config, fps):
    # Lane-only merging (the original approach) silently gives up whenever
    # a segment's lane is unknown (motion_ok was False for it) -- and lane
    # is DERIVED from x_m, so "lane known" and "position known" are exactly
    # the same condition; it was never adding information beyond position.
    # Measured on real footage (DIVE_P023): YOLO+colour combined detect the
    # swimmer in only ~51% of frames, and 74% of the resulting track-ID
    # segments last exactly ONE frame before ByteTrack's own ID resets.
    #
    # Things tried and rejected before this version:
    # 1. A per-segment velocity-plausibility check between segment
    #    ENDPOINTS: a single detection's position has enough noise on its
    #    own (camera-motion/homography error) that comparing two 1-2 frame
    #    segments' endpoints produced implied speeds of 10-35 m/s even
    #    between segments that are almost certainly the same swimmer a
    #    frame apart -- too noisy to gate on.
    # 2. Pure time-gap merging (any two substantial segments within a
    #    gap): only partially fixed things. Two substantial segments often
    #    OVERLAP in time by a few dozen frames -- measured: this is
    #    ByteTrack briefly maintaining two IDs for the same swimmer during
    #    a handoff (their MEAN positions during the overlap were within
    #    centimetres of each other, not two different objects), but a
    #    blanket "any overlap = concurrent object, don't merge" rule
    #    treated every one of these as a hard chain break.
    #
    # Fix: only substantial segments (long enough to be real tracking, not
    # a one-frame noise blip) anchor the merge chain, AND a modest time
    # overlap is allowed to merge too, but only when confirmed by mean
    # position agreement -- a large or position-disagreeing overlap is
    # still treated as a genuinely different, concurrent object.
    if df.empty:
        return df
    t_cfg = config["tracking"]
    max_gap_frames = t_cfg["id_reuse_max_gap_sec"] * fps
    max_overlap_frames = t_cfg["max_overlap_frames_for_merge"]
    min_segment_frames = t_cfg["min_segment_frames_for_merge"]
    max_merge_dist_m = t_cfg["max_merge_distance_m"]

    segments = []
    for track_id, g in df.groupby("track_id"):
        if len(g) < min_segment_frames:
            continue
        segments.append({
            "track_id": track_id, "start": g["frame"].min(), "end": g["frame"].max(),
            "mean_x": g["x_m"].mean(), "mean_y": g["y_m"].mean(),
        })
    segments.sort(key=lambda s: s["start"])

    id_map = {}
    current_id, chain = None, None
    for s in segments:
        if chain is None:
            current_id, chain = s["track_id"], s
            continue

        gap = s["start"] - chain["end"]
        have_pos = not (pd.isna(chain["mean_x"]) or pd.isna(chain["mean_y"])
                         or pd.isna(s["mean_x"]) or pd.isna(s["mean_y"]))
        dist_m = (((s["mean_x"] - chain["mean_x"]) ** 2 + (s["mean_y"] - chain["mean_y"]) ** 2) ** 0.5
                  if have_pos else None)

        if gap > 0:
            can_merge = gap <= max_gap_frames
        else:
            # Overlapping in time: only a handoff artifact, not a real
            # concurrent object, if the overlap is brief AND position
            # confirms it's the same place -- without position to check,
            # don't risk merging two objects that both happened to be
            # detected during the same stretch.
            can_merge = gap >= -max_overlap_frames and have_pos and dist_m <= max_merge_dist_m

        if can_merge:
            id_map[s["track_id"]] = current_id
            chain = {"track_id": current_id, "start": chain["start"], "end": max(chain["end"], s["end"]),
                     "mean_x": s["mean_x"], "mean_y": s["mean_y"]}
        else:
            current_id, chain = s["track_id"], s

    df = df.copy()
    df["track_id"] = df["track_id"].replace(id_map)
    return df


def background_crop(frame, background_region_frac):
    h = frame.shape[0]
    return frame[0:int(h * background_region_frac), :]


def dynamic_background_region_frac(calibration, frame_h, default_frac, safety_margin_frac=0.10):
    # A fixed fraction doesn't generalize across clips: measured on real
    # footage, DIVE_P023's stands sit much lower in frame than DIVE_P010's,
    # so the same default_frac (0.35) captured almost nothing but plain
    # sky there -- ORB found zero matched features in 63% of frames,
    # not just "few" (see CLAUDE.md section 3 amendment). The calibration's
    # own corners already mark where the water starts (never used for
    # background matching, by construction), so use that as a per-clip
    # boundary instead, falling back to default_frac only if it would be
    # smaller (i.e. never shrink below what was already validated on
    # DIVE_P010).
    corners = np.array(calibration["corners_px"], dtype=np.float64)
    pool_top_frac = corners[:, 1].min() / frame_h
    return max(default_frac, pool_top_frac - safety_margin_frac)


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


def motion_crop_rect(frame_w, frame_h, background_region_frac):
    # The current frame's own pool/swimmer region -- everything below the
    # background strip used for camera-motion ORB matching (sky/stands are
    # never the swimmer). Deliberately expressed in the CURRENT frame's own
    # pixel space, not the calibration reference frame's: on this footage
    # the camera tracks the swimmer down the pool's length, so a window
    # fixed to the reference frame's real-world framing stops containing
    # the swimmer at all within a few seconds, even though the camera's
    # own current view still frames them fine (see CLAUDE.md section 3
    # amendment -- measured: the reference-anchored crop showed zero
    # swimmer motion in >99% of a 1200-frame test slice).
    y0 = frame_h * background_region_frac
    return 0.0, y0, float(frame_w), float(frame_h)


def motion_crop_transform(crop_rect, target_width):
    # Scale+translate that maps crop_rect down to a small canvas for the
    # diff, for speed.
    x0, y0, x1, y1 = crop_rect
    crop_w, crop_h = x1 - x0, y1 - y0
    scale = min(1.0, target_width / crop_w) if crop_w > 0 else 1.0
    out_w, out_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    return S @ T, (out_w, out_h)


def detect_motion_blob(cur_small, prev_small, config):
    """cur_small/prev_small: grayscale, reference-aligned crops of the pool
    area at two different times. After alignment the water background
    nearly cancels out in the difference; the swimmer -- the one large,
    coherent region of real change -- doesn't. Returns (x1,y1,x2,y2) in
    the SAME small-crop coordinate space, or None if nothing swimmer-
    sized was found."""
    cam_cfg = config["camera_motion"]
    diff = cv2.absdiff(cur_small, prev_small)
    _, thresh = cv2.threshold(diff, cam_cfg["motion_diff_threshold"], 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    areas = [cv2.contourArea(c) for c in contours]
    best = int(np.argmax(areas))
    total_area = cur_small.shape[0] * cur_small.shape[1]
    min_area, max_area = cam_cfg["motion_min_area_frac"] * total_area, cam_cfg["motion_max_area_frac"] * total_area
    if not (min_area <= areas[best] <= max_area):
        return None  # too small to be more than noise, or too large to be anything but a lighting/alignment glitch

    x, y, w, h = cv2.boundingRect(contours[best])
    return float(x), float(y), float(x + w), float(y + h)


def small_bbox_to_raw(bbox_small, M_final):
    # Maps a bounding box found in the small motion-detection crop back to
    # the current frame's own raw pixel space, via the same transform that
    # produced that crop.
    M_inv = np.linalg.inv(M_final)
    x1, y1, x2, y2 = bbox_small
    corners = [project_point(M_inv, x, y) for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
    xs, ys = [p[0] for p in corners], [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def boxes_iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def fuse_synthetic_detection(boxes, synthetic_box_raw, overlap_thresh, synthetic_conf):
    # YOLO's own detections are always trusted first -- a synthesized
    # (motion- or colour-based) box only fills in when nothing YOLO found
    # already covers it, which is most of the time on this footage (median
    # YOLO confidence ~0.13 on swimmers here, see CLAUDE.md section 3
    # amendment).
    # Returns (boxes, status) where status is one of "no_blob" (nothing to
    # fuse), "confirmed" (blob agreed with an existing YOLO box), or
    # "synthesized" (blob added as a new detection) -- used only for the
    # funnel diagnostics below, not for any downstream logic.
    if synthetic_box_raw is None:
        return boxes, "no_blob"
    for i in range(len(boxes)):
        if boxes_iou(tuple(boxes.xyxy[i]), synthetic_box_raw) >= overlap_thresh:
            return boxes, "confirmed"

    synthetic_row = np.array([[*synthetic_box_raw, synthetic_conf, 0.0]], dtype=np.float32)
    combined = np.concatenate([boxes.data, synthetic_row], axis=0) if len(boxes) else synthetic_row
    return Boxes(combined, boxes.orig_shape), "synthesized"


def detect_color_blob(frame, config):
    """Per-frame (not temporal) swimmer localization: the swimmer's skin,
    dark cap, or splash foam differ in hue from the pool's turquoise water,
    which sidesteps the frame-to-frame alignment problems that broke
    motion-based detection (see CLAUDE.md section 3 amendment). Returns a
    bounding box in this frame's own raw pixel space, or None."""
    cl_cfg = config["color_localize"]
    cal_cfg = config["calibration"]
    frame_h, frame_w = frame.shape[:2]

    scale = min(1.0, cl_cfg["downscale_width"] / frame_w)
    small = cv2.resize(frame, (max(1, int(frame_w * scale)), max(1, int(frame_h * scale)))) if scale < 1.0 else frame
    small_h, small_w = small.shape[:2]
    y0 = int(small_h * cl_cfg["search_top_frac"])
    y1 = int(small_h * (1 - cl_cfg["search_bottom_margin_frac"]))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    roi = hsv[y0:y1, :]
    h, s, v = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]

    water_mask = ((h >= cal_cfg["water_hue_min"]) & (h <= cal_cfg["water_hue_max"]) &
                  (s >= cal_cfg["water_sat_min"]) & (v >= cal_cfg["water_value_min"])).astype(np.uint8) * 255

    # Candidates must be near the pool's own water -- otherwise a warm/tan
    # background object (e.g. a building facade) that happens to be
    # skin-hue too would also qualify just for being below the top margin.
    # Dilating the WHOLE water mask, not just its single largest connected
    # component: measured on real footage, cloud reflections on the water
    # surface fragment the mask into hundreds of small disconnected pieces
    # on an overcast day (DIVE_P007: 870 separate contours, vs ~1 on a
    # clear day), so isolating "the largest contour" left most of the real
    # pool outside the allowed region and the whole colour signal barely
    # fired (2 detections in an entire clip, vs hundreds on clear-sky clips).
    margin = cl_cfg["near_water_margin_px"]
    near_water = cv2.dilate(water_mask, np.ones((margin, margin), np.uint8))

    skin_or_cap = (h <= cl_cfg["skin_hue_max"]) & (v >= 40)
    white_splash = (s <= cl_cfg["white_splash_sat_max"]) & (v >= cl_cfg["white_splash_value_min"])
    candidate = (~water_mask.astype(bool)) & (skin_or_cap | white_splash) & (near_water > 0)

    kernel = np.ones((cl_cfg["morph_kernel"], cl_cfg["morph_kernel"]), np.uint8)
    mask = (candidate * 255).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    total_area = small_w * (y1 - y0)
    areas = [cv2.contourArea(c) for c in contours]
    best = int(np.argmax(areas))
    frac = areas[best] / total_area
    if not (cl_cfg["min_area_frac"] <= frac <= cl_cfg["max_area_frac"]):
        return None

    x, y, w, hh = cv2.boundingRect(contours[best])
    x1r, y1r, x2r, y2r = x, y + y0, x + w, y + hh + y0
    if scale < 1.0:
        x1r, y1r, x2r, y2r = x1r / scale, y1r / scale, x2r / scale, y2r / scale

    # The colour blob usually only catches skin/splash -- a strip much
    # shorter than a full swimmer body (measured on real footage: median
    # blob height ~36px vs YOLO's own median swimmer box height ~89px).
    # Pad it toward that proportion so it (a) has enough IoU with
    # ByteTrack's Kalman-predicted full-body box to extend an existing
    # track instead of spawning a new one, and (b) gives pose.py's crop
    # enough of the body to find keypoints in. Centered, so the position
    # used for distance/speed (the box centre) doesn't shift.
    cx, cy = (x1r + x2r) / 2, (y1r + y2r) / 2
    half_h = (y2r - y1r) / 2 * cl_cfg["body_height_pad_multiplier"]
    y1r, y2r = cy - half_h, cy + half_h
    return float(x1r), float(y1r), float(x2r), float(y2r)


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
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if _max_frames is not None:
        frame_count = min(frame_count, _max_frames)

    bg_frac = dynamic_background_region_frac(
        calibration, frame_h, config["camera_motion"]["background_region_frac"])
    motion_fusion_enabled = config["camera_motion"]["motion_fusion_enabled"]
    motion_detect_width = config["camera_motion"]["motion_detect_width"]
    motion_crop = motion_crop_rect(frame_w, frame_h, bg_frac)
    crop_transform, small_size = motion_crop_transform(motion_crop, motion_detect_width)

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
    anchor_gray = None  # raw grayscale frame matching chain_kp/chain_des, for the motion-blob diff below
    cumulative_M = np.eye(3)
    ref_to_frame0 = None

    model = YOLO(config["detection"]["model"])
    tracker = BYTETracker(build_tracker_args(config), frame_rate=int(round(fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    batch_size = config["detection"]["batch_size"]
    conf_threshold = config["detection"]["conf_threshold"]
    half = config["detection"]["half_precision"]

    color_localize_enabled = config["color_localize"]["enabled"]

    records = []
    frames_low_motion_confidence = 0
    motion_blob_found = 0
    motion_blob_confirmed = 0
    motion_blob_synthesized = 0
    color_blob_found = 0
    color_blob_confirmed = 0
    color_blob_synthesized = 0
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
            full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = detect_orb_features(orb, background_crop(frame, bg_frac), config["camera_motion"]["match_resize_width"])

            # Motion-based swimmer localization: align the last successfully-
            # matched frame ("anchor") into the CURRENT frame's own pixel
            # space (not a fixed reference window -- see motion_crop_rect)
            # and diff against the current frame there. Because the anchor
            # is warped into the frame that's actually being looked at right
            # now, this keeps working even while the camera tracks the
            # swimmer down the pool's length. The swimmer is the one
            # coherent region that doesn't cancel out. Confirms/supplements
            # YOLO; see fuse_motion_detection.
            motion_box_raw = None
            if chain_kp is None:
                step_ok, good_features = True, 0  # first frame of the video: nothing to chain from yet
            else:
                step_M, good_features = estimate_step_homography(matcher, chain_kp, chain_des, kp, des, config)
                step_ok = step_M is not None
                if step_ok:
                    cumulative_M = cumulative_M @ step_M
                    if motion_fusion_enabled:
                        align_transform = crop_transform @ np.linalg.inv(step_M)
                        prev_small_aligned = cv2.warpPerspective(anchor_gray, align_transform, small_size)
                        cur_small = cv2.warpPerspective(full_gray, crop_transform, small_size)
                        bbox_small = detect_motion_blob(cur_small, prev_small_aligned, config)
                        if bbox_small is not None:
                            motion_box_raw = small_bbox_to_raw(bbox_small, crop_transform)
            if step_ok:
                chain_kp, chain_des, anchor_gray = kp, des, full_gray  # only advance the anchor on a successful match

            if frame_idx == calibration["frame_number"]:
                ref_to_frame0 = cumulative_M.copy()

            motion_ok = step_ok and ref_to_frame0 is not None
            if motion_ok:
                M = np.linalg.inv(ref_to_frame0) @ cumulative_M
            if not motion_ok:
                frames_low_motion_confidence += 1

            boxes = result.boxes.cpu().numpy()
            boxes, motion_status = fuse_synthetic_detection(
                boxes, motion_box_raw,
                config["camera_motion"]["motion_yolo_iou_match"],
                config["camera_motion"]["motion_synthetic_confidence"])
            if motion_status != "no_blob":
                motion_blob_found += 1
                if motion_status == "confirmed":
                    motion_blob_confirmed += 1
                elif motion_status == "synthesized":
                    motion_blob_synthesized += 1

            color_box_raw = detect_color_blob(frame, config) if color_localize_enabled else None
            boxes, color_status = fuse_synthetic_detection(
                boxes, color_box_raw,
                config["color_localize"]["yolo_iou_match"],
                config["color_localize"]["synthetic_confidence"])
            if color_status != "no_blob":
                color_blob_found += 1
                if color_status == "confirmed":
                    color_blob_confirmed += 1
                elif color_status == "synthesized":
                    color_blob_synthesized += 1
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
    df = merge_continuous_tracks(df, config, fps)
    longest_track_frames = int(df["track_id"].value_counts().max()) if not df.empty else 0
    diagnostics = {
        "detect_frames_total": frame_count,
        "detect_frames_low_motion_confidence": frames_low_motion_confidence,
        "detect_tracks_before_merge": tracks_before_merge,
        "detect_tracks_found": df["track_id"].nunique() if not df.empty else 0,
        "detect_longest_track_frames": longest_track_frames,
        "detect_longest_track_sec": longest_track_frames / fps if fps else 0,
        "motion_blob_found": motion_blob_found,
        "motion_blob_confirmed": motion_blob_confirmed,
        "motion_blob_synthesized": motion_blob_synthesized,
        "color_blob_found": color_blob_found,
        "color_blob_confirmed": color_blob_confirmed,
        "color_blob_synthesized": color_blob_synthesized,
    }
    return df, diagnostics


def cache_path(video_path, config, cache_dir=None):
    cache_dir = cache_dir or os.path.join(config["cache"]["directory"], "detect_track")
    os.makedirs(cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    model = os.path.splitext(config["detection"]["model"])[0]
    conf = config["detection"]["conf_threshold"]
    # Tuning camera_motion/color_localize and rerunning should never silently
    # reuse a cached result from before the tuning change.
    fusion_key = json.dumps(
        {"camera_motion": config["camera_motion"], "color_localize": config["color_localize"]},
        sort_keys=True)
    fusion_hash = hashlib.sha1(fusion_key.encode()).hexdigest()[:10]
    return os.path.join(cache_dir, f"{stem}__{model}__conf{conf}__{fusion_hash}.csv")


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
