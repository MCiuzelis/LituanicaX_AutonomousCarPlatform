"""Run the trained detector over a video and write an annotated mp4 (visual check).

    uv run scripts/06_predict_video.py data/videos/20261010_131840_cam0_1536x864_60fps.mp4
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/apple_yolov8s_r2/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "runs" / f"pred_{args.video.stem}.mp4"

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for r in model.predict(str(args.video), conf=args.conf, imgsz=960, stream=True, verbose=False):
        vw.write(r.plot(line_width=2, font_size=10))
    vw.release()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
