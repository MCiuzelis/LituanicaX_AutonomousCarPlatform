"""Compare the compiled HEF (run on the car's Hailo) against the PyTorch model on the val split.

Both are scored with the same matcher: greedy IoU>=0.5 matching per image, AP50 from the
precision/recall curve, plus P/R/F1 at a fixed confidence. Coordinates are normalised 0..1.

    scp -r data/yolo/images/val rp5:LituanicaX_InteligentRobotCompetition/Pi5/vision/val_images
    ssh rp5 "cd LituanicaX_InteligentRobotCompetition/Pi5/vision && python3 eval_hef.py --images val_images --out preds_hef.json && rm -rf val_images"
    scp rp5:LituanicaX_InteligentRobotCompetition/Pi5/vision/preds_hef.json runs/hef_eval/
    uv run scripts/08_compare_hef.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def load_gt(label_dir):
    gt = {}
    for f in sorted(label_dir.glob("*.txt")):
        rows = [list(map(float, l.split()[1:])) for l in f.read_text().splitlines() if l.strip()]
        a = np.array(rows).reshape(-1, 4)
        gt[f.stem] = np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2,
                               a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], 1)
    return gt


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def score(preds, gt, conf):
    """preds: {stem: [[x1,y1,x2,y2,score], ...]} -> dict of metrics."""
    tps, scores, n_gt = [], [], 0
    for stem, g in gt.items():
        n_gt += len(g)
        p = np.array(preds.get(stem, []), dtype=float).reshape(-1, 5)
        p = p[np.argsort(-p[:, 4])]
        used = np.zeros(len(g), bool)
        ious = iou(p[:, :4], g)
        for i in range(len(p)):
            j = int(np.argmax(ious[i])) if len(g) else -1
            hit = j >= 0 and ious[i, j] >= 0.5 and not used[j]
            if hit:
                used[j] = True
            tps.append(hit)
            scores.append(p[i, 4])
    order = np.argsort(-np.array(scores))
    tp = np.array(tps)[order]
    s = np.array(scores)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec, prec = ctp / max(n_gt, 1), ctp / np.maximum(ctp + cfp, 1)
    # AP50: area under the monotone precision envelope (COCO-style 101-point interpolation)
    env = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    ap = np.mean([env[rec >= r].max() if np.any(rec >= r) else 0 for r in np.linspace(0, 1, 101)])
    k = s >= conf
    P = tp[k].sum() / max(k.sum(), 1)
    R = tp[k].sum() / max(n_gt, 1)
    return {"AP50": round(ap, 4), f"P@{conf}": round(P, 4), f"R@{conf}": round(R, 4),
            f"F1@{conf}": round(2 * P * R / max(P + R, 1e-9), 4), "boxes": int(k.sum()), "gt": n_gt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef-preds", type=Path, default=ROOT / "runs/hef_eval/preds_hef.json")
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/apple_yolov8s_r2/weights/best.pt")
    ap.add_argument("--images", type=Path, default=ROOT / "data/yolo/images/val")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/yolo/labels/val")
    ap.add_argument("--conf", type=float, default=0.35, help="operating point used on the car")
    args = ap.parse_args()

    gt = load_gt(args.labels)
    hef = {k: v for k, v in json.loads(args.hef_preds.read_text()).items() if k in gt}

    # PyTorch reference: same 0.2 floor and IoU 0.7 NMS as the HEF's on-chip NMS
    model = YOLO(args.weights)
    pt = {}
    for r in model.predict(str(args.images), imgsz=640, conf=0.2, iou=0.7, stream=True, verbose=False):
        pt[Path(r.path).stem] = [[*b, s] for b, s in zip(r.boxes.xyxyn.cpu().tolist(), r.boxes.conf.cpu().tolist())]

    res = {"pytorch_fp32": score(pt, gt, args.conf), "hef_int8_on_hailo": score(hef, gt, args.conf)}
    for k, v in res.items():
        print(f"{k:18s} " + "  ".join(f"{m}={x}" for m, x in v.items()))
    d = res["hef_int8_on_hailo"]["AP50"] - res["pytorch_fp32"]["AP50"]
    print(f"AP50 change from quantisation: {d:+.4f}")


if __name__ == "__main__":
    main()
