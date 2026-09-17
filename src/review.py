# Lets a person quickly accept or reject each detected hand-entry moment,
# by watching a short clip around it and pressing one key. Never asks for
# raw data entry -- just a fast yes/no per event. Saves accepted events as
# ground truth, used later to measure accuracy (not to train anything).
#
# Not one of the originally-listed 8 source files (CLAUDE.md section 4),
# but the build order (section 17, step 9) explicitly calls for this as
# its own script -- likely an oversight in the file list rather than a
# deliberate omission, since section 13 describes its behaviour in detail.
# Flagged as an amendment in CLAUDE.md rather than silently added.

import argparse
import os

import cv2
import pandas as pd

from src.events import run_events_cached
from src.profiling import load_config


def play_and_decide(window_name, clip_frames, fps, label):
    # Loops the clip until the user presses a decision key, rather than
    # showing it once -- a hand-entry moment is easy to miss on a single pass.
    idx = 0
    while True:
        frame = clip_frames[idx % len(clip_frames)]
        display = cv2.resize(frame, (960, 540))
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, "y = accept, n = reject, q = quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key in (ord("y"), ord("n"), ord("q")):
            return chr(key)
        idx += 1


def load_clip(cap, start_frame, end_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def append_ground_truth(accepted_rows, video_path, ground_truth_path):
    if not accepted_rows:
        return 0, 0
    new_df = pd.DataFrame(accepted_rows)
    new_df["video"] = os.path.basename(video_path)
    if os.path.exists(ground_truth_path):
        existing = pd.read_csv(ground_truth_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    os.makedirs(os.path.dirname(ground_truth_path), exist_ok=True)
    combined.to_csv(ground_truth_path, index=False)
    return len(new_df), len(combined)


def review_video(video_path, config, calibration_dir="data/calibration",
                  ground_truth_path="outputs/ground_truth_events.csv"):
    events_df, _ = run_events_cached(video_path, config, calibration_dir)
    entries = events_df[events_df["event_type"] == "entry"].sort_values("time_sec").reset_index(drop=True)
    if entries.empty:
        print("No hand-entry events found for this video -- nothing to review.")
        return 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    window_frames = int(0.5 * fps)

    window_name = "Review hand-entry events"
    cv2.namedWindow(window_name)

    accepted_rows = []
    try:
        for i, event in entries.iterrows():
            start_frame = max(0, int(event["frame"] - window_frames))
            end_frame = int(event["frame"] + window_frames)
            clip_frames = load_clip(cap, start_frame, end_frame)
            if not clip_frames:
                continue

            label = f"Event {i + 1}/{len(entries)}  t={event['time_sec']:.2f}s  lane={event.get('lane')}"
            decision = play_and_decide(window_name, clip_frames, fps, label)
            if decision == "q":
                print("Stopping early -- saving what's been reviewed so far.")
                break
            if decision == "y":
                accepted_rows.append(event.to_dict())
    finally:
        cv2.destroyAllWindows()
        cap.release()

    new_count, total_count = append_ground_truth(accepted_rows, video_path, ground_truth_path)
    print(f"Saved {new_count} accepted events to {ground_truth_path} (total across all reviewed videos: {total_count})")
    return new_count, total_count


def main():
    parser = argparse.ArgumentParser(description="Manually review detected hand-entry events (accept/reject).")
    parser.add_argument("video", help="Path to the video, e.g. data/raw/xyz.MP4")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    review_video(args.video, config)


if __name__ == "__main__":
    main()
