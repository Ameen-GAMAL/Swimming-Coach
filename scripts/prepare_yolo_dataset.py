# Converts finetune_data/manifest.json's accepted/corrected boxes into
# Ultralytics' expected training layout (images/labels split into
# train/val, plus a data.yaml). One-off tooling for the fine-tuning
# experiment, not part of the pipeline.

import argparse
import json
import os
import random
import shutil

import cv2

VAL_FRACTION = 0.15
SEED = 42


def to_yolo_label(box, img_w, img_h):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
    w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Prepare the YOLO training layout from reviewed labels.")
    parser.add_argument("--dir", default="finetune_data")
    parser.add_argument("--out-dir", default="finetune_data/yolo_dataset")
    args = parser.parse_args()

    with open(os.path.join(args.dir, "manifest.json")) as f:
        manifest = json.load(f)

    usable = [m for m in manifest if m["status"] in ("accepted", "accepted_corrected")]
    print(f"{len(usable)} usable labelled frames")

    random.seed(SEED)
    random.shuffle(usable)
    n_val = max(1, int(len(usable) * VAL_FRACTION))
    val_items, train_items = usable[:n_val], usable[n_val:]
    print(f"train={len(train_items)} val={len(val_items)}")

    images_src = os.path.join(args.dir, "images")
    for split, items in [("train", train_items), ("val", val_items)]:
        img_dir = os.path.join(args.out_dir, "images", split)
        lbl_dir = os.path.join(args.out_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for item in items:
            src_img = os.path.join(images_src, item["image"])
            frame = cv2.imread(src_img)
            h, w = frame.shape[:2]
            box = item.get("final_box") or item.get("proposed_box")

            stem = os.path.splitext(item["image"])[0]
            shutil.copy(src_img, os.path.join(img_dir, item["image"]))
            with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
                f.write(to_yolo_label(box, w, h) + "\n")

    data_yaml = os.path.join(args.out_dir, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"""path: {os.path.abspath(args.out_dir)}
train: images/train
val: images/val
names:
  0: swimmer
""")
    print(f"Dataset ready: {args.out_dir}")
    print(f"data.yaml: {data_yaml}")


if __name__ == "__main__":
    main()
