"""Is the lens model right at the edges of the image? For every apple the map tracked (_det.csv rows with a
landmark), its position from observations near the image centre (|x - 0.5| < 0.15) is the reference; every
observation is compared with it as a bearing seen from the camera at that frame (pose from the same row).
A bearing error that grows towards the edges = distortion not modelled there (the 2304x1296 / 56 fps mode sees
the full sensor, 96 deg; the calibration covered radius 1650 sensor px).

    python tools/analysis/edge_bearing.py recordings/<id> [...]
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np


def main():
    for take in sys.argv[1:]:
        rows = []
        for r in csv.DictReader(open(glob.glob(os.path.join(take, "*_det.csv"))[0])):
            if r["status"] not in ("used", "edge_fit") or not r["landmark"]:
                continue
            xc = (float(r["x1"]) + float(r["x2"])) / 2
            rows.append((r["landmark"], xc, float(r["world_x_mm"]), float(r["world_y_mm"]), float(r["pose_x_mm"]),
                         float(r["pose_y_mm"]), math.radians(float(r["pose_yaw_deg"])), float(r["obs_x_mm"]),
                         float(r["obs_y_mm"]), r["status"]))
        ref = defaultdict(list)
        for lid, xc, wx, wy, *_ in rows:
            if abs(xc - 0.5) < 0.15:
                ref[lid].append((wx, wy))
        ref = {k: np.median(np.array(v), 0) for k, v in ref.items() if len(v) >= 5}
        bins = defaultdict(list)
        for lid, xc, wx, wy, px, py, yaw, ox, oy, st in rows:
            if lid not in ref:
                continue
            rx, ry = ref[lid]
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = rx - px, ry - py
            fx, fy = c * dx + s * dy, -s * dx + c * dy                  # reference in the camera floor frame
            if fx < 300:
                continue
            err = math.degrees(math.atan2(oy, ox) - math.atan2(fy, fx))
            rng = math.hypot(ox, oy) - math.hypot(fx, fy)
            b = min(9, int(abs(xc - 0.5) / 0.05))                         # 0.05-wide bins from the centre
            bins[(b, 1 if xc > 0.5 else -1)].append((err, rng, st == "edge_fit"))
        print(os.path.basename(take.rstrip("/"))[:28])
        for (b, sd) in sorted(bins, key=lambda k: (k[1], k[0])):
            v = np.array([(e, r) for e, r, _ in bins[(b, sd)]])
            if len(v) < 20:
                continue
            print("   image x %s %.2f-%.2f: %5d obs, bearing error median %+5.2f deg (p25 %+5.2f p75 %+5.2f), range error median %+5.0f mm" % (
                "right" if sd > 0 else "left ", b * 0.05, b * 0.05 + 0.05, len(v), np.median(v[:, 0]),
                np.percentile(v[:, 0], 25), np.percentile(v[:, 0], 75), np.median(v[:, 1])))


if __name__ == "__main__":
    main()
