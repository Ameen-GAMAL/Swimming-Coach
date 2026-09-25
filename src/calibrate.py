# Automatically calibrates a video: finds the pool water, the lane rope
# nearest the swimmer, the water line, and the lane boundary -- all from the
# footage itself, no clicking. Produces the same JSON schema the manual tool
# used to, plus a preview image, so detect_track.py and pose.py don't need
# to change.
#
# No frame in this camera's footage ever shows a pool wall or another
# landmark with a precisely known real-world distance along the swim
# direction (see CLAUDE.md section 3 amendment). Lane WIDTH is still exact
# -- 2.5m, FINA standard -- but lane LENGTH scale has no real landmark
# available, so it is estimated approximately from the swimmer's own
# detected torso length via pose, against an assumed average adult value.
# Every calibration this produces is marked "scale_method":
# "approximate_anthropometric" for that reason -- it is a provisional
# estimate, not a measurement.

import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

from src.profiling import load_config, model_path

ASSUMED_TORSO_LENGTH_M = 0.48  # average adult shoulder-midpoint-to-hip-midpoint length
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12


def read_calibration_frame(video_path):
    # A few seconds in, so the dive splash has settled
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = min(int(fps * 3), frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")
    return frame, frame_number


def detect_water_mask(frame, config):
    # Pool water is a strongly saturated turquoise, distinct in hue from sky
    # and in brightness from the blue-painted building elements above it.
    # Restricted to the lower part of the frame -- where the pool always is
    # in this camera's framing -- so blue building trim never gets confused
    # for water even when its hue overlaps.
    c = config["calibration"]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([c["water_hue_min"], c["water_sat_min"], c["water_value_min"]])
    upper = np.array([c["water_hue_max"], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask[: int(frame.shape[0] * c["water_search_top_frac"]), :] = 0

    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def fit_rope_line(water_mask):
    """Finds the topmost water pixel in every column -- tracing the
    boundary between open water and the lane rope floats sitting on the
    surface, i.e. the nearest lane rope itself -- and fits a straight line
    to it. Returns (slope, intercept, min_x, max_x)."""
    h, w = water_mask.shape
    boundary_y = np.full(w, -1, dtype=np.int32)
    for x in range(w):
        col = np.nonzero(water_mask[:, x])[0]
        if len(col) > 0:
            boundary_y[x] = col[0]

    valid_x = np.nonzero(boundary_y >= 0)[0]
    if len(valid_x) < w * 0.5:
        raise RuntimeError(
            "Could not find the pool water in this frame (water mask too small/empty). "
            "The colour thresholds in config.yaml's calibration section may need adjusting for this footage."
        )

    ys = boundary_y[valid_x]
    A = np.vstack([valid_x, np.ones_like(valid_x)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(slope), float(intercept), int(valid_x.min()), int(valid_x.max())


def rope_corners(slope, intercept, min_x, max_x, frame_shape, config):
    """Builds the 4 calibration corners from the one detected rope line.
    near/far come from sampling the rope near its two ends (5th/95th
    percentile of where it was confidently detected, to avoid edge noise).
    The lane's far side (2.5m across, no second rope detected) is
    approximated as a fraction of the remaining gap to the bottom of the
    frame, which shrinks with distance the same way real perspective
    foreshortening would -- see the module docstring for why this is a
    heuristic, not a measurement."""
    span = max_x - min_x
    near_x = min_x + span * 0.05
    far_x = min_x + span * 0.95
    near_y = slope * near_x + intercept
    far_y = slope * far_x + intercept

    frac = config["calibration"]["lane_far_edge_gap_fraction"]
    frame_h = frame_shape[0]
    near_y2 = near_y + frac * (frame_h - near_y)
    far_y2 = far_y + frac * (frame_h - far_y)

    # Order: near-left, near-right, far-right, far-left
    return [(near_x, near_y), (near_x, near_y2), (far_x, far_y2), (far_x, far_y)]


def compute_homography(corners_px, start_m, end_m, width_m):
    # Corner order: near-left, near-right, far-right, far-left.
    # Real-world axes: x = across the lane, y = along the pool.
    real_points = np.array([
        [0.0, start_m],
        [width_m, start_m],
        [width_m, end_m],
        [0.0, end_m],
    ], dtype=np.float32)
    px_points = np.array(corners_px, dtype=np.float32)
    H, _ = cv2.findHomography(px_points, real_points)
    if H is None:
        raise RuntimeError("Could not compute a homography from the detected corners -- they may be collinear.")
    return H


def project_point(H_inv, x_m, y_m):
    point = np.array([x_m, y_m, 1.0])
    mapped = H_inv @ point
    mapped /= mapped[2]
    return mapped[0], mapped[1]


def sample_swimmer_boxes(video_path, model, config):
    """Runs detection on frames spread across the video to see where the
    swimmer actually is, and to collect crops for the pose-based scale
    estimate. Samples many frames because the pretrained detector and pose
    model are both unreliable on this footage (see the Phase 4/5 findings)
    -- most samples get discarded by estimate_length_scale's sanity checks,
    so a small sample count leaves too few, too noisy survivors."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    conf = config["detection"]["conf_threshold"]
    num_samples = config["calibration"]["scale_estimate_num_samples"]

    samples = []
    for frame_num in np.linspace(0, frame_count - 1, num_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ok, frame = cap.read()
        if not ok:
            continue
        result = model.predict(frame, classes=[0], conf=conf, device=0, half=config["detection"]["half_precision"],
                                verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        if len(boxes) == 0:
            continue
        best = int(np.argmax(boxes.conf))
        samples.append((int(frame_num), frame, boxes.xyxy[best]))
    cap.release()
    return samples


def estimate_length_scale(samples, pose_model, config, H_placeholder):
    """Estimates the correction factor that turns the placeholder length
    scale used to build H_placeholder into an approximate real one, from
    the swimmer's DETECTION BOX length along the pool axis.

    Measured from the box, not from pose keypoints (shoulder midpoint to
    hip midpoint), because on this footage the keypoints collapse: a prone,
    side-on, partly-submerged swimmer whose detection box spans ~1040px
    produced a shoulder-to-hip reading of only ~60px, where the real torso
    should be ~230px. That understated torso is what drove the whole
    calibration ~4x long even after the clamp bug was removed. The box is
    a far steadier signal -- the detector finds the swimmer in ~99% of
    frames and the box visibly sits on the body.

    The trade-off is honest and worth stating: a stroking swimmer's box
    covers the body plus the forward arm reach, which varies through the
    stroke cycle, so this is good to roughly +/-15-20%, not better. A
    trimmed median (middle 60%) over many samples keeps stroke-phase
    variation and the occasional bad box from dominating.
    """
    cal_cfg = config["calibration"]
    min_box_px = cal_cfg["scale_estimate_min_box_px"]
    assumed_box_m = cal_cfg["assumed_swimmer_box_length_m"]
    ratios = []

    for frame_num, frame, box in samples:
        x1, y1, x2, y2 = [float(v) for v in box]
        if (x2 - x1) < min_box_px:
            continue  # too small to be a real side-on swimmer at this range
        y_mid = (y1 + y2) / 2
        a = project_point(H_placeholder, x1, y_mid)
        b = project_point(H_placeholder, x2, y_mid)
        box_placeholder_units = np.hypot(b[0] - a[0], b[1] - a[1])
        if box_placeholder_units > 1e-6:
            ratios.append(assumed_box_m / box_placeholder_units)

    if len(ratios) < cal_cfg["scale_estimate_min_samples"]:
        return None, len(ratios)
    ratios.sort()
    trim = int(len(ratios) * 0.2)
    trimmed = ratios[trim: len(ratios) - trim] if trim else ratios
    return float(np.median(trimmed)), len(ratios)


def _estimate_length_scale_from_torso(samples, pose_model, config, H_placeholder):
    """Superseded by the box-based estimate above; kept only because it
    documents the keypoint failure mode described there."""
    cal_cfg = config["calibration"]
    conf_threshold = cal_cfg["scale_estimate_min_keypoint_conf"]
    min_torso_px = cal_cfg["scale_estimate_min_torso_px"]
    ratios = []

    for frame_num, frame, box in samples:
        x1, y1, x2, y2 = box
        pad_x, pad_y = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        h, w = frame.shape[:2]
        cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        cx2, cy2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        result = pose_model.predict(crop, conf=config["pose"]["conf_threshold"], device=0,
                                     half=config["pose"]["half_precision"], verbose=False)[0]
        kpts = result.keypoints.cpu().numpy() if result.keypoints is not None else None
        if kpts is None or kpts.xy.shape[0] == 0:
            continue
        xy, confs = kpts.xy[0], kpts.conf[0] if kpts.conf is not None else np.zeros(17)
        needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
        if any(confs[i] < conf_threshold for i in needed):
            continue

        shoulder_mid = ((xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2) + [cx1, cy1]
        hip_mid = ((xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2) + [cx1, cy1]
        torso_px = np.hypot(*(shoulder_mid - hip_mid))
        if torso_px < min_torso_px:
            continue  # implausibly short for a real torso -- a bad keypoint read

        s_placeholder = project_point(H_placeholder, *shoulder_mid)
        h_placeholder = project_point(H_placeholder, *hip_mid)
        torso_placeholder_units = np.hypot(s_placeholder[0] - h_placeholder[0], s_placeholder[1] - h_placeholder[1])
        if torso_placeholder_units > 1e-6:
            ratios.append(ASSUMED_TORSO_LENGTH_M / torso_placeholder_units)

    if len(ratios) < cal_cfg["scale_estimate_min_samples"]:
        return None, len(ratios)

    ratios.sort()
    trim = int(len(ratios) * 0.2)
    trimmed = ratios[trim: len(ratios) - trim] if trim else ratios
    return float(np.median(trimmed)), len(ratios)


def draw_length_grid(canvas, H, pool_length_m, segment_size_m, width_m, scale=1.0):
    # Draws a line every segment_size_m along the pool's length, across the
    # FULL pool_length_m even though only a short stretch was ever detected
    # -- a homography is a valid mapping for the whole plane, so this
    # doubles as a sanity check: if these lines drift off the real pool
    # edges near the far end, the calibration was wrong.
    H_inv = np.linalg.inv(H)
    for length_m in np.arange(0, pool_length_m + 1e-6, segment_size_m):
        x1, y1 = project_point(H_inv, 0.0, length_m)
        x2, y2 = project_point(H_inv, width_m, length_m)
        pt1 = (int(x1 * scale), int(y1 * scale))
        pt2 = (int(x2 * scale), int(y2 * scale))
        cv2.line(canvas, pt1, pt2, (0, 255, 0), max(1, int(2 * scale)))
        cv2.putText(canvas, f"{length_m:.1f}m", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(scale, 0.6), (0, 255, 0), 2)
    return canvas


def draw_lines(canvas, lines, color, label_prefix, scale=1.0):
    for entry in lines:
        (x1, y1), (x2, y2) = entry["points"]
        pt1, pt2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.line(canvas, pt1, pt2, color, max(1, int(2 * scale)))
        label = f"{label_prefix} {entry.get('lane', entry.get('boundary_index'))}"
        cv2.putText(canvas, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(scale, 0.6), color, 2)
    return canvas


def save_calibration(video_path, frame_number, corners, H, span, water_lines, boundaries, config,
                      scale_samples, scale_was_clamped, out_dir="data/calibration"):
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.json")
    start_m, end_m, width_m = span
    data = {
        "video": os.path.basename(video_path),
        "frame_number": frame_number,
        "pool_length_m": config["pool"]["length_m"],
        "corners_px": corners,
        "corner_span_m": {"start_m": start_m, "end_m": end_m, "width_m": width_m},
        "homography": H.tolist(),
        "water_lines": water_lines,
        "lane_boundaries": boundaries,
        "scale_method": "approximate_anthropometric",
        "scale_estimate_frames_used": scale_samples,
        "scale_estimate_clamped_to_prior": scale_was_clamped,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def save_preview_image(frame, H, config, span, water_lines, boundaries, video_path, out_dir="data/calibration"):
    _, _, width_m = span
    canvas = draw_length_grid(frame.copy(), H, config["pool"]["length_m"], config["pool"]["segment_size_m"], width_m)
    canvas = draw_lines(canvas, water_lines, (255, 0, 0), "lane")
    canvas = draw_lines(canvas, boundaries, (0, 0, 255), "boundary")
    cv2.putText(canvas, "PROVISIONAL LENGTH SCALE (approximate, no wall visible in this footage)",
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_preview.png")
    cv2.imwrite(out_path, canvas)
    return out_path


def calibrate_video(video_path, config, calibration_dir="data/calibration"):
    frame, frame_number = read_calibration_frame(video_path)
    pool_length_m = config["pool"]["length_m"]
    lane_width_m = config["pool"]["lane_width_m"]

    water_mask = detect_water_mask(frame, config)
    slope, intercept, min_x, max_x = fit_rope_line(water_mask)
    corners = rope_corners(slope, intercept, min_x, max_x, frame.shape, config)

    # Placeholder length scale (0 to pool_length_m over the detected rope's
    # own pixel span) just to get the homography's correct PERSPECTIVE
    # SHAPE -- the length axis gets corrected below using the swimmer's own
    # body size, since no real landmark is visible to calibrate it directly.
    H_placeholder = compute_homography(corners, 0.0, pool_length_m, lane_width_m)

    detect_model = YOLO(model_path(config["detection"]["model"]))
    pose_model = YOLO(model_path(config["pose"]["model"]))
    samples = sample_swimmer_boxes(video_path, detect_model, config)
    correction, scale_samples = estimate_length_scale(samples, pose_model, config, H_placeholder)
    if correction is None:
        raise RuntimeError(
            f"Could not estimate a length scale -- only {scale_samples} frame(s) had a confident enough, "
            f"plausible-looking torso reading (need at least {config['calibration']['scale_estimate_min_samples']}). "
            f"Try a video where the swimmer is more clearly visible, or lower calibration.scale_estimate_min_keypoint_conf "
            f"in config.yaml."
        )

    # The estimate is bounded by how much pool the camera can actually SEE,
    # not by a ratio around the pool's full length.
    #
    # The old clamp (a ratio of pool_length_m, default [0.7, 1.6]) encoded
    # the assumption that the detected rope span IS roughly the whole pool.
    # That is false for this footage: the camera sits at water level a few
    # metres from the near lane and sees only a short stretch. Measured on
    # DIVE_P023, the clamp forced the visible span to 40m when the swimmer's
    # own body implies well under 10m -- so the homography claimed each
    # swimmer was 8.5-11.4m long, every distance was ~5.7x too large, and
    # every speed ~5.7x too slow. The correct value sat a factor of 2.5
    # outside the clamp, i.e. unreachable no matter what the estimator
    # measured.
    span_min_m, span_max_m = config["calibration"]["visible_span_bounds_m"]
    raw_span_m = pool_length_m * correction
    end_m = float(np.clip(raw_span_m, span_min_m, span_max_m))
    was_clamped = abs(end_m - raw_span_m) > 1e-9
    if was_clamped:
        print(f"WARNING: estimated visible span {raw_span_m:.1f}m fell outside the plausible "
              f"range [{span_min_m}, {span_max_m}]m and was clamped to {end_m:.1f}m -- treat "
              f"this clip's distances as unreliable.")

    start_m = 0.0
    H = compute_homography(corners, start_m, end_m, lane_width_m)

    # Sanity check against something physical, reported rather than hidden:
    # a horizontal swimmer's detection box should come out around 2-3m.
    box_lengths = []
    for _, _, box in samples:
        bx1, by1, bx2, by2 = [float(v) for v in box]
        if (bx2 - bx1) < config["calibration"]["scale_estimate_min_box_px"]:
            continue
        y_mid = (by1 + by2) / 2
        a = project_point(H, bx1, y_mid)
        b = project_point(H, bx2, y_mid)
        box_lengths.append(float(np.hypot(b[0] - a[0], b[1] - a[1])))
    if box_lengths:
        print(f"Visible pool span: {end_m:.1f}m over {len(box_lengths)} swimmer samples. "
              f"Median box spans {np.median(box_lengths):.2f}m "
              f"(target {config['calibration']['assumed_swimmer_box_length_m']}m).")

    water_lines = [{"lane": 1, "points": [corners[0], corners[3]]}]
    boundaries = [
        {"boundary_index": 0, "points": [corners[0], corners[3]]},
        {"boundary_index": 1, "points": [corners[1], corners[2]]},
    ]
    span = (start_m, end_m, lane_width_m)

    json_path = save_calibration(video_path, frame_number, corners, H, span, water_lines, boundaries, config,
                                  scale_samples, was_clamped, calibration_dir)
    preview_path = save_preview_image(frame, H, config, span, water_lines, boundaries, video_path, calibration_dir)
    return json_path, preview_path, scale_samples


def main():
    parser = argparse.ArgumentParser(description="Automatically calibrate a video (no clicking required).")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Calibrating {args.video} automatically...")
    json_path, preview_path, scale_samples = calibrate_video(args.video, config)

    print(f"Saved calibration: {json_path}")
    print(f"Saved preview image: {preview_path}")
    print(f"Length scale estimated from {scale_samples} frames' worth of swimmer body size (approximate -- "
          f"see CLAUDE.md section 3 amendment).")
    print("Look at the preview image -- if the grid does not roughly follow the pool edges, the water-colour "
          "thresholds in config.yaml's calibration section may need adjusting for this footage.")


if __name__ == "__main__":
    main()
