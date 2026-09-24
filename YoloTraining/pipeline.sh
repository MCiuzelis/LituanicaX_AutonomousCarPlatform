#!/usr/bin/env bash
# Full labeling + two-round training pipeline (see README).
set -euo pipefail
cd "$(dirname "$0")"
uv run scripts/02_autolabel.py --preview --redo
uv run scripts/03_build_dataset.py
uv run scripts/04_train.py --name apple_yolov8s
uv run scripts/07_refine_labels.py --preview
uv run scripts/03_build_dataset.py --labels data/labels_r2
uv run scripts/04_train.py --name apple_yolov8s_r2
echo PIPELINE_DONE
