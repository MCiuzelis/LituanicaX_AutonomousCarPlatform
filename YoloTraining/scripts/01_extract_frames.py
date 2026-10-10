"""Extract frames from the driving videos, dropping blurry and redundant ones.

Keeps a frame only if the car is driving (motor_rps from the recording's <stem>_frames.csv,
when present), it differs enough from the last kept frame of the same video (mean abs diff
on a small grayscale thumbnail) and is not much blurrier than its neighbours. The motion
gate drops the static start/end of each clip and the tail after hitting apples, where the
car has stopped but the apples are still rolling. Output: data/frames/<video_stem>_<frame_idx>.jpg

    uv run scripts/01_extract_frames.py
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def thumb(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)


def sharpness(frame):
    g = cv2.cvtColor(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def driving_mask(vid: Path, n: int, min_rps: float, pad: int):
    """Per-frame bool: car driving, dilated by `pad` frames. None if the clip has no frames.csv."""
    csv_path = vid.with_name(f"{vid.stem}_frames.csv")
    if not csv_path.exists():
        return None
    moving = np.zeros(n, bool)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["frame"])
            if i < n and row["motor_rps"] and abs(float(row["motor_rps"])) >= min_rps:
                moving[i] = True
    if pad:
        moving = np.convolve(moving, np.ones(2 * pad + 1), "same") > 0
    return moving


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=ROOT / "data/videos")
    ap.add_argument("--out", type=Path, default=ROOT / "data/frames")
    ap.add_argument("--stride", type=int, default=3, help="look at every Nth frame (60fps -> 20fps)")
    ap.add_argument("--min-diff", type=float, default=16.0, help="min mean abs pixel diff vs last kept frame")
    ap.add_argument("--max-gap", type=int, default=60, help="force-keep after this many source frames")
    ap.add_argument("--min-rps", type=float, default=5.0, help="motor rev/s that counts as driving")
    ap.add_argument("--pad", type=int, default=0, help="also keep this many frames around driving ones")
    ap.add_argument("--blur-ratio", type=float, default=0.45,
                    help="drop frame if sharpness < ratio * running median sharpness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    total_kept = total_seen = 0
    for vid in sorted(args.videos.glob("*.mp4")):
        cap = cv2.VideoCapture(str(vid))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        moving = driving_mask(vid, n, args.min_rps, args.pad)
        last_thumb, last_idx = None, -10**9
        sharp_hist = []
        kept = seen = 0
        for idx in tqdm(range(n), desc=vid.stem, leave=False):
            if not cap.grab():
                break
            if idx % args.stride or (moving is not None and not moving[idx]):
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            seen += 1
            s = sharpness(frame)
            sharp_hist.append(s)
            med = float(np.median(sharp_hist[-100:]))
            if s < args.blur_ratio * med:
                continue
            t = thumb(frame)
            if last_thumb is not None and idx - last_idx < args.max_gap:
                if float(np.mean(np.abs(t - last_thumb))) < args.min_diff:
                    continue
            cv2.imwrite(str(args.out / f"{vid.stem}_{idx:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            last_thumb, last_idx = t, idx
            kept += 1
        cap.release()
        print(f"{vid.name}: kept {kept}/{seen} sampled ({n} total)")
        total_kept += kept
        total_seen += seen
    print(f"TOTAL kept {total_kept}/{total_seen} -> {args.out}")


if __name__ == "__main__":
    main()
