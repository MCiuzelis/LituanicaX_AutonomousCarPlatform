"""Run a HEF over a folder of 1280x720 JPEGs on the Pi's Hailo and dump predictions as JSON.

Frames are downscaled to 640x360 with PIL bilinear (the car gets this from the ISP) and go
through the exact AppleDetector path used on the car.

    python3 eval_hef.py --hef ~/rc_car/models/apple_yolov8s.hef --images val_images --out preds_hef.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from apple_detector import AppleDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default=os.path.expanduser("~/rc_car/models/apple_yolov8s.hef"))
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.2, help="HEF NMS already cuts at 0.2")
    args = ap.parse_args()

    det = AppleDetector(args.hef, score_thr=args.conf, lores_size=(640, 360))
    preds = {}
    try:
        for p in sorted(args.images.glob("*.jpg")):
            im = np.asarray(Image.open(p).convert("RGB").resize((640, 360), Image.BILINEAR))
            preds[p.stem] = [[round(v, 5) for v in b[:5]] for b in det.detect(im)]
    finally:
        det.close()
    args.out.write_text(json.dumps(preds))
    print(f"{len(preds)} images, {sum(map(len, preds.values()))} boxes -> {args.out}")


if __name__ == "__main__":
    main()
