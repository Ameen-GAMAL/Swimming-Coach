# Fine-tunes the current detection model on the small hand-reviewed
# swimmer dataset, starting from the SAME checkpoint the pipeline uses
# (yolo11x.pt) so the before/after comparison isolates the effect of
# fine-tuning, not a different base model. One-off tooling for the
# fine-tuning experiment, not part of the pipeline.

import os

# Ultralytics auto-logs training to MLflow, which (same issue already fixed in
# src/profiling.py) blocks the plain filesystem backend on MLflow 3.x unless
# this is set.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from ultralytics import YOLO

BASE_MODEL = "yolo11x.pt"
DATA_YAML = "finetune_data/yolo_dataset/data.yaml"
EPOCHS = 60
RUN_NAME = "swimmer_finetune_v1"


def main():
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=640,          # reduced from 960 -- yolo11x (57M params) training OOM'd this 8GB GPU at 960/batch=8; inference (this session's earlier work) is far lighter than training's backprop memory needs
        batch=4,
        patience=15,        # stop early if val loss stalls -- guards against overfitting a 167-image training set
        workers=2,          # default (8) spawns that many parallel data-loading processes -- unnecessary for a 167-image dataset and a likely contributor to the system-RAM OOM hit at the default setting
        project="finetune_data/runs",
        name=RUN_NAME,
        device=0,
        half=True,
        pretrained=True,
        verbose=True,
    )
    print("Done. Best weights at finetune_data/runs/" + RUN_NAME + "/weights/best.pt")


if __name__ == "__main__":
    main()
