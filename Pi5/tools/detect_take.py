#!/usr/bin/env python3
"""Run the apple detector over a recorded take (offline) -> <id>_det.csv in the track-map format.

Takes recorded before the track map existed have video + IMU + odometry but no detections. This
decodes the take's mp4 (OpenCV), scales each frame to the detector's lores size (INTER_AREA; live,
the ISP does this) and runs the HEF on the Hailo. The output has the raw columns of the live
`<id>_det.csv` (see vision/track_map.py, DET_COLUMNS); tools/replay_map.py then replays the map.

    pkill -f "[p]ython3 dashboard.py"          # only one process can own the Hailo
    python3 tools/detect_take.py recordings/<id> [recordings/<id2> ...] [--conf 0.35]

Writes results/trackmap/<id>/<id>_det.csv (the take folder itself is left untouched). Rows join
the take's _frames.csv on sensor_ts_ns. Needs python3-opencv + python3-hailort.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision"))
from vision.track_map import DET_RAW_COLUMNS, OUT_DIR, take_files   # noqa: E402


def detect_take(take, hef, conf, out_dir=None):
    import cv2
    from apple_detector import AppleDetector, hef_input_size, lores_fit
    f = take_files(take)
    meta = json.load(open(f["meta"]))
    st = meta["stream"]
    w, h = st["width"], st["height"]
    crop = st.get("scaler_crop") or [768, 432, 3072, 1728]
    rows = list(csv.DictReader(open(f["frames"])))
    lo = lores_fit(w, h, *hef_input_size(hef))
    out_dir = out_dir or os.path.join(OUT_DIR, meta["id"])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, meta["id"] + "_det.csv")
    det = AppleDetector(hef, score_thr=conf, lores_size=lo, depth=4)
    cap = cv2.VideoCapture(f["video"])
    n_frames = n_boxes = 0
    t0 = time.monotonic()
    with open(out + ".tmp", "w", newline="") as fo:
        wr = csv.writer(fo)
        wr.writerow(DET_RAW_COLUMNS)

        def finish():
            nonlocal n_boxes
            boxes, r = det.collect()
            ts, tm = int(r["sensor_ts_ns"]), int(r["t_mid_ns"])
            base = [r["frame"], ts, tm, len(boxes), *crop]
            if not boxes:
                wr.writerow(base + [-1] + [""] * 6)
            for i, (x1, y1, x2, y2, s, c) in enumerate(boxes):
                wr.writerow(base + [i, "%.5f" % x1, "%.5f" % y1, "%.5f" % x2, "%.5f" % y2, "%.4f" % s, c])
            n_boxes += len(boxes)

        for r in rows:
            ok, img = cap.read()
            if not ok:
                break
            small = cv2.resize(img, lo, interpolation=cv2.INTER_AREA)
            det.submit(np.ascontiguousarray(small[:, :, ::-1]), r)     # BGR -> RGB like picamera2 "BGR888"
            n_frames += 1
            while det.in_flight > 2:
                finish()
        while det.in_flight:
            finish()
    det.close()
    os.replace(out + ".tmp", out)
    dt = time.monotonic() - t0
    print("%s: %d frames, %.2f boxes/frame, %.0f fps -> %s" % (meta["id"], n_frames, n_boxes / max(1, n_frames),
                                                              n_frames / dt, out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--hef", default=os.path.join(ROOT, "models", "apple_yolov8s.hef"))
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    for t in a.takes:
        detect_take(t, a.hef, a.conf)


if __name__ == "__main__":
    main()
