# Samples frames from a set of training videos and pre-labels each one
# using the CURRENT pipeline (yolo11x detection + colour-fusion), so the
# review pass is accept/correct/reject instead of drawing boxes from
# scratch. Deliberately NOT one of CLAUDE.md's fixed files -- this is
# one-off tooling for the fine-tuning experiment, not part of the pipeline
# itself. Held-out validation videos (P010/P023/P007/P027) are
# intentionally excluded here so the later before/after comparison is fair.

import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detect_track import detect_color_blob, boxes_iou
from src.profiling import load_config

TRAINING_VIDEOS = [
    "DIVE_P004_FREE_ABOVE_20250612_001._SWIM2.MP4",
    "DIVE_P013_FREE_ABOVE_20250621_024._SWIM2.MP4",
    "DIVE_P016_FREE_ABOVE_20250626_001._SWIM2.MP4",
    "DIVE_P017_FREE_ABOVE_20250627_001._SWIM2.MP4",
    "DIVE_P019_FREE_ABOVE_20250701_001._SWIM2.MP4",
    "DIVE_P021_FREE_ABOVE_20250701_017._SWIM2.MP4",
    "DIVE_P025_FREE_ABOVE_20250703_001._SWIM2.MP4",
    "DIVE_P030_FREE_ABOVE_20250704_017_SWIM2.MP4",
]
FRAMES_PER_VIDEO = 30
OUT_DIR = "finetune_data"


def sample_frame_indices(frame_count, n):
    # Evenly spaced, skipping the first/last 5% (dive/wall approach --
    # rarely useful training signal, matches the clean-zone reasoning
    # used throughout the rest of the project).
    lo, hi = int(frame_count * 0.05), int(frame_count * 0.95)
    return np.linspace(lo, hi, n, dtype=int).tolist()


def propose_box(frame, model, config):
    conf_threshold = config["detection"]["conf_threshold"]
    half = config["detection"]["half_precision"]
    result = model.predict([frame], classes=[0], conf=conf_threshold, device=0, half=half, verbose=False)[0]
    boxes = result.boxes.cpu().numpy()

    best_yolo = None
    if len(boxes):
        best_i = int(np.argmax(boxes.conf))
        best_yolo = tuple(float(v) for v in boxes.xyxy[best_i])

    color_box = detect_color_blob(frame, config)

    if best_yolo and color_box and boxes_iou(best_yolo, color_box) >= 0.1:
        return best_yolo, "yolo+colour agree"
    if best_yolo:
        return best_yolo, "yolo only"
    if color_box:
        return color_box, "colour only"
    return None, "no proposal"


def main():
    parser = argparse.ArgumentParser(description="Sample and pre-label frames for the fine-tuning experiment.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model = YOLO(config["detection"]["model"])

    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    manifest = []

    for video_name in TRAINING_VIDEOS:
        video_path = os.path.join(args.raw_dir, video_name)
        if not os.path.exists(video_path):
            print(f"SKIP (not found): {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stem = os.path.splitext(video_name)[0]

        for frame_idx in sample_frame_indices(frame_count, FRAMES_PER_VIDEO):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            box, source = propose_box(frame, model, config)
            image_name = f"{stem}__f{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(images_dir, image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "image": image_name, "video": video_name, "frame": frame_idx,
                "proposed_box": box, "source": source, "status": "pending",
            })
        cap.release()
        print(f"{video_name}: sampled {FRAMES_PER_VIDEO} frames")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    n_with_proposal = sum(1 for m in manifest if m["proposed_box"] is not None)
    print(f"\nTotal: {len(manifest)} frames sampled, {n_with_proposal} have a starting proposal.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
