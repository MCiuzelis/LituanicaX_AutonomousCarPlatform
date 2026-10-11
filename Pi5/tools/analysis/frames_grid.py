"""Grid of video frames (2 per row) with the live detections from _det.csv (joined on sensor_ts) drawn in:
green = used by the map, orange = beyond range, magenta = cut by the frame edge.

    python tools/analysis/frames_grid.py <dir with flat take files> <take id> out.jpg 2.0,8.5,14
(times = seconds from the take start)
"""
import csv, sys, os
import cv2
import numpy as np

take_dir, rid, out = sys.argv[1], sys.argv[2], sys.argv[3]
times = [float(x) for x in sys.argv[4].split(",")]
p = lambda s: os.path.join(take_dir, rid + s)
fr = list(csv.DictReader(open(p("_frames.csv"))))
ts = np.array([int(r["sensor_ts_ns"]) for r in fr])
dets = {}
for r in csv.DictReader(open(p("_det.csv"))):
    if int(r["box"]) >= 0:
        dets.setdefault(int(r["sensor_ts_ns"]), []).append((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["score"]), r["status"]))
det_ts = np.array(sorted(dets)) if dets else np.zeros(0)
cap = cv2.VideoCapture(p(".mp4"))
tiles = []
for tt in times:
    i = int(np.searchsorted(ts, ts[0] + tt * 1e9))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    if not ok:
        continue
    H, W = img.shape[:2]
    j = int(np.argmin(np.abs(det_ts - ts[i]))) if len(det_ts) else None
    n = 0
    if j is not None and abs(det_ts[j] - ts[i]) < 20e6:
        for x1, y1, x2, y2, s, st in dets[det_ts[j]]:
            col = (0, 255, 0) if st == "used" else (0, 200, 255) if st == "range" else (255, 0, 255)
            cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), col, 2)
            n += 1
    cv2.putText(img, "%s t=%.1fs boxes=%d" % (rid[9:15], tt, n), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    tiles.append(cv2.resize(img, (W // 2, H // 2)))
while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
