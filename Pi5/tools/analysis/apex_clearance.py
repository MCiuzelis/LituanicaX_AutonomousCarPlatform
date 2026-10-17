"""How close did the car come to the apples it passed? For every apple the map saw the car pass (Landmark.side,
from _landmarks.csv) the smallest distance from the camera floor point's path (_pose.csv, map frame), minus the
car's half width (110 mm) and the apple's radius (37 mm) = body clearance; split by whether the apple was on the
inside of a turn (the car yawing towards that side, |yaw rate| > 40 deg/s) and by autonomous / manual driving.

    python tools/analysis/apex_clearance.py recordings/<id> [...]
"""
import csv
import glob
import os
import sys

import numpy as np

HALF_W, APPLE_R = 110.0, 37.0


def main():
    for take in sys.argv[1:]:
        rid = os.path.basename(take.rstrip("/"))
        po = np.array([[float(r["t_ns"]) * 1e-9, float(r["x_mm"]), float(r["y_mm"]), float(r["yaw_deg"])]
                       for r in csv.DictReader(open(glob.glob(os.path.join(take, "*_pose.csv"))[0]))])
        dr = list(csv.DictReader(open(glob.glob(os.path.join(take, "*_drive.csv"))[0])))
        dt_ = np.array([int(r["t_ns"]) for r in dr]) * 1e-9
        auto = np.array([int(float(r["auto_state"] or 0)) for r in dr])
        yaw = np.unwrap(np.radians(po[:, 3]))
        rate = np.degrees(np.gradient(yaw, po[:, 0]))
        last = {}
        for r in csv.DictReader(open(glob.glob(os.path.join(take, "*_landmarks.csv"))[0])):
            if r["side"] in ("L", "R") and r["state"] == "confirmed":
                last[r["id"]] = (float(r["x_mm"]), float(r["y_mm"]), 1 if r["side"] == "L" else -1)
        res = []
        for lid, (x, y, sd) in last.items():
            d = np.hypot(po[:, 1] - x, po[:, 2] - y)
            i = int(np.argmin(d))
            clear = d[i] - HALF_W - APPLE_R
            j = min(len(dt_) - 1, int(np.searchsorted(dt_, po[i, 0])))
            # the turn around the passing moment (+-0.3 s)
            k = (po[:, 0] > po[i, 0] - 0.3) & (po[:, 0] < po[i, 0] + 0.3)
            wr = float(np.median(rate[k]))
            inner = abs(wr) > 40 and np.sign(wr) == sd
            res.append((clear, inner, abs(wr) > 40, auto[j] == 1, po[i, 0] - po[0, 0]))
        if not res:
            continue
        R = np.array(res, dtype=object)
        for name, sel in (("auto, inside of a turn", lambda r: r[3] and r[1]),
                          ("auto, outside of a turn", lambda r: r[3] and r[2] and not r[1]),
                          ("auto, straight", lambda r: r[3] and not r[2]),
                          ("manual, inside of a turn", lambda r: not r[3] and r[1])):
            c = np.array([r[0] for r in res if sel(r)], float)
            if len(c):
                print("%s %-26s %3d apples: clearance p10 %4.0f p50 %4.0f mm, touched (< 0) %d %s" % (
                    rid[:15], name, len(c), np.percentile(c, 10), np.median(c), (c < 0).sum(),
                    " ".join("%.1fs" % r[4] for r in res if sel(r) and r[0] < 0)[:120]))


if __name__ == "__main__":
    main()
