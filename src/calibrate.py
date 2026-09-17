# Click-based calibration tool. Produces one JSON file per video describing
# the pool corners (as a homography), each lane's water line, and the lane
# boundaries -- plus one preview image so the result can be checked by eye.

import argparse
import json
import os

import cv2
import numpy as np

from src.profiling import load_config


def read_calibration_frame(video_path):
    # A few seconds in, so the dive splash has settled and the swimmer is
    # visible mid-stroke -- useful visual context for clicking, even though
    # only static pool features (never the swimmer) get clicked.
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


def collect_points(window_name, base_frame, display_scale, num_points, instructions):
    """Lets the user click num_points on base_frame (already scaled for
    display). Returns the points in ORIGINAL, full-resolution pixel
    coordinates. 'u' undoes the last click, enter confirms once all points
    are placed, 'q' cancels the whole tool."""
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x / display_scale, y / display_scale))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            canvas = base_frame.copy()
            cv2.putText(canvas, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(canvas, f"{len(points)}/{num_points} points  |  u = undo, enter = confirm, q = quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            for i, (px, py) in enumerate(points):
                dx, dy = int(px * display_scale), int(py * display_scale)
                cv2.circle(canvas, (dx, dy), 6, (0, 0, 255), -1)
                cv2.putText(canvas, str(i + 1), (dx + 8, dy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('u') and points:
                points.pop()
            elif key == ord('q'):
                raise SystemExit("Calibration cancelled by user.")
            elif key in (13, 10) and len(points) == num_points:
                return points
    finally:
        cv2.destroyWindow(window_name)


def compute_homography(corners_px, start_m, end_m, width_m):
    # Corner click order is fixed: near-left, near-right, far-right, far-left.
    # Real-world axes: x = across the lanes (0 at the left edge clicked),
    # y = along the pool (start_m at the near edge clicked, end_m at the far edge).
    real_points = np.array([
        [0.0, start_m],
        [width_m, start_m],
        [width_m, end_m],
        [0.0, end_m],
    ], dtype=np.float32)
    px_points = np.array(corners_px, dtype=np.float32)
    H, _ = cv2.findHomography(px_points, real_points)
    if H is None:
        raise RuntimeError("Could not compute a homography from these 4 points -- they may be collinear.")
    return H


def project_point(H_inv, x_m, y_m):
    # Maps a real-world metre position back to a pixel position, for drawing
    point = np.array([x_m, y_m, 1.0])
    mapped = H_inv @ point
    mapped /= mapped[2]
    return mapped[0], mapped[1]


def draw_length_grid(canvas, H, pool_length_m, segment_size_m, width_m, scale=1.0):
    # Draws a line every segment_size_m along the pool's length. Extends
    # across the FULL pool_length_m even if only part of it was clicked --
    # a homography is a valid mapping for the whole plane, not just the
    # clicked rectangle, so this doubles as a sanity check: if these lines
    # drift away from the real pool edges near the far end, the calibration
    # or the length span entered was wrong.
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


def prompt_span(pool_length_m):
    # Retries on bad input instead of crashing, so a typo doesn't throw away
    # the 4 corner clicks the user already made.
    while True:
        span = input(
            f"\nHow much of the pool length do these 4 corners span, in metres?\n"
            f"  - If they start at the near wall, just enter the length (e.g. 7)\n"
            f"  - If they cover a shorter stretch that does NOT start at the wall, enter start,end (e.g. 5,20)\n"
            f"  - Press enter for the default: the whole pool (0,{pool_length_m})\n> "
        ).strip()
        if not span:
            return 0.0, float(pool_length_m)
        try:
            if "," in span:
                start_m, end_m = (float(v) for v in span.split(","))
            else:
                start_m, end_m = 0.0, float(span)
        except ValueError:
            print(f"Could not read '{span}' as a number or a 'start,end' pair -- try again.")
            continue
        if end_m <= start_m:
            print(f"End ({end_m}) must be greater than start ({start_m}) -- try again.")
            continue
        return start_m, end_m


def prompt_int(question, default):
    while True:
        text = input(f"{question} [default {default}]: ").strip()
        if not text:
            return default
        try:
            value = int(text)
        except ValueError:
            print(f"Could not read '{text}' as a whole number -- try again.")
            continue
        if value < 1:
            print("Must be at least 1 -- try again.")
            continue
        return value


def calibrate_corners(display_frame, display_scale, config):
    pool_length_m = config["pool"]["length_m"]
    lane_width_m = config["pool"]["lane_width_m"]
    window = "Step 1: click 4 corners (near-left, near-right, far-right, far-left)"

    while True:
        corners = collect_points(
            window, display_frame, display_scale, 4,
            "Click the water surface corners: near-left, near-right, far-right, far-left",
        )

        start_m, end_m = prompt_span(pool_length_m)
        lanes_spanned = prompt_int("How many lanes wide are these corners?", 1)
        width_m = lanes_spanned * lane_width_m

        H = compute_homography(corners, start_m, end_m, width_m)

        preview = draw_length_grid(display_frame.copy(), H, pool_length_m, config["pool"]["segment_size_m"], width_m, display_scale)
        cv2.putText(preview, "Does the grid follow the pool edges? y = accept, r = redo", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window)
        if key == ord('y'):
            return corners, H, (start_m, end_m, width_m)
        # anything else (default 'r') -- loop back and redo the clicks


def calibrate_water_lines(base_frame, display_scale, lanes_visible):
    lines = []
    for lane_num in range(1, lanes_visible + 1):
        pts = collect_points(
            f"Step 2: lane {lane_num} water line ({lane_num}/{lanes_visible})",
            base_frame, display_scale, 2,
            f"Click 2 points along lane {lane_num}'s water line, near edge to far edge",
        )
        lines.append({"lane": lane_num, "points": pts})
    return lines


def calibrate_lane_boundaries(base_frame, display_scale, lanes_visible):
    boundaries = []
    total = lanes_visible + 1
    for b in range(total):
        pts = collect_points(
            f"Step 3: boundary {b + 1} of {total}",
            base_frame, display_scale, 2,
            f"Click 2 points along lane boundary {b + 1} (left edge of the visible lanes if b=1)",
        )
        boundaries.append({"boundary_index": b, "points": pts})
    return boundaries


def save_calibration(video_path, frame_number, corners, H, span, water_lines, boundaries, config, out_dir="data/calibration"):
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
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def save_preview_image(frame, H, config, span, water_lines, boundaries, video_path, out_dir="data/calibration"):
    _, _, width_m = span
    canvas = draw_length_grid(frame.copy(), H, config["pool"]["length_m"], config["pool"]["segment_size_m"], width_m)
    canvas = draw_lines(canvas, water_lines, (255, 0, 0), "lane")
    canvas = draw_lines(canvas, boundaries, (0, 0, 255), "boundary")

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_preview.png")
    cv2.imwrite(out_path, canvas)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Calibrate a video: pool corners, water lines, lane boundaries.")
    parser.add_argument("video", help="Path to the video to calibrate, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    frame, frame_number = read_calibration_frame(args.video)

    h, w = frame.shape[:2]
    display_scale = min(1.0, config["calibration"]["display_max_width"] / w)
    display_frame = cv2.resize(frame, (int(w * display_scale), int(h * display_scale)))

    print(f"Calibrating {args.video} using frame {frame_number} ({w}x{h})")
    corners, H, span = calibrate_corners(display_frame, display_scale, config)

    lanes_visible = prompt_int("How many lanes are visible in this frame?", config["pool"]["lanes"])

    grid_frame = draw_length_grid(display_frame.copy(), H, config["pool"]["length_m"], config["pool"]["segment_size_m"], span[2], display_scale)
    water_lines = calibrate_water_lines(grid_frame, display_scale, lanes_visible)

    lines_frame = draw_lines(grid_frame.copy(), water_lines, (255, 0, 0), "lane", display_scale)
    boundaries = calibrate_lane_boundaries(lines_frame, display_scale, lanes_visible)

    cv2.destroyAllWindows()

    json_path = save_calibration(args.video, frame_number, corners, H, span, water_lines, boundaries, config)
    preview_path = save_preview_image(frame, H, config, span, water_lines, boundaries, args.video)

    print(f"Saved calibration: {json_path}")
    print(f"Saved preview image: {preview_path}")
    print("Check the preview image -- if the grid does not follow the pool edges, rerun this tool.")


if __name__ == "__main__":
    main()
