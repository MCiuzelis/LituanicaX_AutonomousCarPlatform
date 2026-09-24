"""Train the single-class apple detector.

YOLOv8s @ 640 is the target: it is the architecture Hailo's Model Zoo ships for Hailo-8
(stock yolov8s_h8.hef measured 309 FPS on the car's Hailo-8), so compiling it is the
well-trodden path and leaves 3x headroom over the 100 FPS goal. The 1280x720 camera frame
is letterboxed to 640 (apples at the far end of the track are ~10-15 px after that).

    uv run scripts/04_train.py                      # yolov8s, 150 epochs
    uv run scripts/04_train.py --model yolov8n.pt   # faster/smaller fallback
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/yolo/data.yaml")
    ap.add_argument("--model", default=str(ROOT / "weights/yolov8s.pt"), help="COCO-pretrained start point")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="16 fits yolov8s@640 in 8 GB")
    ap.add_argument("--name", default="apple_yolov8s")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=40,
        cos_lr=True,
        close_mosaic=15,
        # Augmentation tuned for a low, fixed camera on a moving car:
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.45,  # lighting / apple colour variety (red, yellow, mixed)
        degrees=3.0,                        # slight chassis roll
        translate=0.1,
        scale=0.5,                          # apples near and far
        shear=0.0,
        perspective=0.0,
        flipud=0.0,                         # camera never upside-down
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        erasing=0.2,
        plots=True,
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    metrics = YOLO(best).val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, plots=False)
    print(f"best: {best}\nval mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f} "
          f"P={metrics.box.mp:.3f} R={metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
