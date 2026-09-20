# The actual test: compares the fine-tuned model against today's yolo11x
# baseline on the 4 videos held out from fine-tuning entirely (P010/P023/
# P007/P027), sampling frames across each clip. One-off tooling, not part
# of the pipeline.

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.profiling import load_config

VALIDATION_VIDEOS = [
    "DIVE_P010_FREE_ABOVE_20250621_001._SWIM2.MP4",
    "DIVE_P023_FREE_ABOVE_20250702_009._SWIM2.MP4",
    "DIVE_P007_FREE_ABOVE_20250619_001._SWIM2.MP4",
    "DIVE_P027_FREE_ABOVE_20250703_017._SWIM2.MP4",
]
FRAMES_PER_VIDEO = 40


def sample_frame_indices(frame_count, n):
    lo, hi = int(frame_count * 0.05), int(frame_count * 0.95)
    return np.linspace(lo, hi, n, dtype=int).tolist()


def evaluate(model, frames, config):
    conf_threshold = config["detection"]["conf_threshold"]
    half = config["detection"]["half_precision"]
    confidences, hit = [], 0
    for frame in frames:
        result = model.predict([frame], classes=[0], conf=conf_threshold, device=0, half=half, verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        if len(boxes):
            confidences.append(float(boxes.conf.max()))
            hit += 1
        else:
            confidences.append(0.0)
    return confidences, hit


def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuned vs baseline detector on held-out videos.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--finetuned", default="finetune_data/runs/swimmer_finetune_v1/weights/best.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    baseline_model = YOLO(config["detection"]["model"])
    finetuned_model = YOLO(args.finetuned)

    print(f"{'video':45s} {'metric':10s} {'baseline':>10s} {'finetuned':>10s}")
    grand_base_conf, grand_ft_conf = [], []
    grand_base_hits, grand_ft_hits, grand_total = 0, 0, 0

    for video_name in VALIDATION_VIDEOS:
        video_path = os.path.join(args.raw_dir, video_name)
        if not os.path.exists(video_path):
            print(f"SKIP (not found): {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for frame_idx in sample_frame_indices(frame_count, FRAMES_PER_VIDEO):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()

        base_conf, base_hits = evaluate(baseline_model, frames, config)
        ft_conf, ft_hits = evaluate(finetuned_model, frames, config)

        stem = video_name[:45]
        print(f"{stem:45s} {'mean_conf':10s} {np.mean(base_conf):10.3f} {np.mean(ft_conf):10.3f}")
        print(f"{'':45s} {'hit_rate':10s} {base_hits/len(frames):10.1%} {ft_hits/len(frames):10.1%}")

        grand_base_conf.extend(base_conf)
        grand_ft_conf.extend(ft_conf)
        grand_base_hits += base_hits
        grand_ft_hits += ft_hits
        grand_total += len(frames)

    print()
    print("=== OVERALL (all 4 held-out videos combined) ===")
    print(f"Mean confidence -- baseline: {np.mean(grand_base_conf):.3f}   finetuned: {np.mean(grand_ft_conf):.3f}")
    print(f"Hit rate        -- baseline: {grand_base_hits/grand_total:.1%}   finetuned: {grand_ft_hits/grand_total:.1%}")


if __name__ == "__main__":
    main()
