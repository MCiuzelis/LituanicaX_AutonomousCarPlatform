"""Export the trained model to ONNX for the Hailo Dataflow Compiler + collect calibration images.

Output in export/:
    apple_yolov8s.onnx   static 1x3x640x640, opset 11, no NMS (Hailo adds its own NMS on-chip)
    calib/               training frames for INT8 quantisation calibration
Then compile on Linux (Hailo DFC does not run on Windows) with export/compile_hailo.sh.

    uv run scripts/05_export_hailo.py
"""
import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "runs/apple_yolov8s_r2/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=ROOT / "export")
    ap.add_argument("--n-calib", type=int, default=1024)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    onnx = YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, opset=11, simplify=True,
                                     dynamic=False, nms=False)
    dst = args.out / "apple_yolov8s.onnx"
    shutil.copy2(onnx, dst)
    print(f"onnx -> {dst}")

    calib = args.out / "calib"
    if calib.exists():
        shutil.rmtree(calib)
    calib.mkdir()
    frames = sorted((ROOT / "data/yolo/images/train").glob("*.jpg"))
    random.Random(0).shuffle(frames)
    for f in frames[: args.n_calib]:
        shutil.copy2(f, calib / f.name)
    print(f"calib: {min(len(frames), args.n_calib)} images -> {calib}")


if __name__ == "__main__":
    main()
