"""Extract frames from the driving videos, dropping blurry and redundant ones.

Keeps a frame only if it differs enough from the last kept frame of the same video
(mean abs diff on a small grayscale thumbnail) and is not much blurrier than its
neighbours. Output: data/frames/<video_stem>_<frame_idx>.jpg

    uv run scripts/01_extract_frames.py
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=ROOT / "data/videos")
    ap.add_argument("--out", type=Path, default=ROOT / "data/frames")
    ap.add_argument("--stride", type=int, default=3, help="look at every Nth frame (60fps -> 20fps)")
    ap.add_argument("--min-diff", type=float, default=16.0, help="min mean abs pixel diff vs last kept frame")
    ap.add_argument("--max-gap", type=int, default=60, help="force-keep after this many source frames")
    ap.add_argument("--blur-ratio", type=float, default=0.45,
                    help="drop frame if sharpness < ratio * running median sharpness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    total_kept = total_seen = 0
    for vid in sorted(args.videos.glob("*.mp4")):
        cap = cv2.VideoCapture(str(vid))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_thumb, last_idx = None, -10**9
        sharp_hist = []
        kept = seen = 0
        for idx in tqdm(range(n), desc=vid.stem, leave=False):
            if not cap.grab():
                break
            if idx % args.stride:
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
