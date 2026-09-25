# Times each pipeline stage and sends parameters, metrics, and artifacts to MLflow.

import os
import re
import subprocess
import time

# MLflow 3.x blocks the plain filesystem backend ("file:./mlruns") unless this
# is set. CLAUDE.md section 3 fixes "MLflow, local file store" as the decision,
# so this unlocks that exact setup instead of switching to a database backend.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import yaml


def load_config(config_path="config.yaml"):
    # Every tunable number lives in one file, never hardcoded in a function
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


STOCK_WEIGHTS = re.compile(r"^yolo(v\d+|11)[nslmx](-pose|-seg|-cls|-obb)?\.pt$")


def model_path(path):
    """The weights file a stage should load. Stock Ultralytics names
    (yolo11x.pt, yolo11m-pose.pt, ...) download themselves on first use;
    anything else is a local file, and *.pt is gitignored -- so on a fresh
    clone the fine-tuned detector is simply missing, and Ultralytics then
    fails with an unhelpful download error. Say what to do instead."""
    if os.path.exists(path) or STOCK_WEIGHTS.match(os.path.basename(path)):
        return path
    raise FileNotFoundError(
        f"Model weights '{path}' are not here. Weights are not stored in git (*.pt is ignored). "
        f"Either copy the file into the project root, rebuild it with scripts/finetune_train.py "
        f"(then copy its best.pt to '{path}'), or set detection.model in config.yaml to yolo11x.pt "
        f"to run on the stock detector.")


def git_commit_hash():
    # Tags every run with the exact code version that produced it
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_run_name(video_name, what_changed="baseline"):
    # Format: {video}__{what_changed}__{MMDD-HHMM}, so runs are readable in a list
    timestamp = time.strftime("%m%d-%H%M")
    return f"{video_name}__{what_changed}__{timestamp}"


def start_run(config, run_name, tags):
    # Point MLflow at the local file store and open one run for this pipeline execution
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags({**tags, "git_commit": git_commit_hash()})
    return run


class Timer:
    """Times one pipeline stage. A plain function can't hold a start time
    across the two ends of a `with` block, so this is the one place here
    that needs a class instead of a function."""

    def __init__(self, store, name):
        self.store = store
        self.name = name

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.store[self.name] = time.perf_counter() - self._start
