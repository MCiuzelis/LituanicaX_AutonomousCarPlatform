"""Round-2 labels: add confident detections from the trained YOLO model that Grounding DINO missed.

Grounding DINO under-labels apples cut off by the frame edge. After a first training run the
student model often finds them; this merges its confident boxes (no IoU overlap with any
existing DINO box) into a new label set, then retrain on it:

    uv run scripts/07_refine_labels.py --preview
    uv run scripts/03_build_dataset.py --labels data/labels_r2
    uv run scripts/04_train.py --name apple_yolov8s_r2
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def read_yolo(path: Path, W: int, H: int) -> np.ndarray:
    rows = [list(map(float, l.split()[1:])) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4))
    a = np.array(rows)
    return np.stack([(a[:, 0] - a[:, 2] / 2) * W, (a[:, 1] - a[:, 3] / 2) * H,
                     (a[:, 0] + a[:, 2] / 2) * W, (a[:, 1] + a[:, 3] / 2) * H], 1)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/apple_yolov8s/weights/best.pt")
    ap.add_argument("--frames", type=Path, default=ROOT / "data/frames")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels")
    ap.add_argument("--out", type=Path, default=ROOT / "data/labels_r2")
    ap.add_argument("--conf", type=float, default=0.6, help="only add student boxes at least this confident")
    ap.add_argument("--iou", type=float, default=0.2, help="treat as already labeled above this IoU")
    ap.add_argument("--min-bottom", type=float, default=0.25, help="same ground-plane rule as 02_autolabel")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    prev = ROOT / "data/preview_r2"
    if args.preview:
        shutil.rmtree(prev, ignore_errors=True)
        prev.mkdir()

    model = YOLO(args.weights)
    imgs = sorted(args.frames.glob("*.jpg"))
    added = touched = 0
    for p in imgs:
        r = model.predict(str(p), conf=args.conf, imgsz=960, verbose=False)[0]
        H, W = r.orig_shape
        src = args.labels / f"{p.stem}.txt"
        old_lines = [l for l in src.read_text().splitlines() if l.strip()] if src.exists() else []
        old = read_yolo(src, W, H) if src.exists() else np.zeros((0, 4))
        new = r.boxes.xyxy.cpu().numpy()
        new = new[new[:, 3] >= args.min_bottom * H]
        if len(new) and len(old):
            new = new[iou(new, old).max(1) < args.iou]
        lines = old_lines + [
            f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
            for x1, y1, x2, y2 in new
        ]
        (args.out / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        added += len(new)
        touched += len(new) > 0

        if args.preview and len(new):
            im = r.orig_img.copy()
            for x1, y1, x2, y2 in old.astype(int):
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x1, y1, x2, y2 in new.astype(int):
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = added in round 2
            cv2.imwrite(str(prev / p.name), im, [cv2.IMWRITE_JPEG_QUALITY, 80])

    print(f"added {added} boxes across {touched}/{len(imgs)} frames -> {args.out}")


if __name__ == "__main__":
    main()
