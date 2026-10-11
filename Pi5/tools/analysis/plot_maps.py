"""Top-down plot of the maps recorded live (<id>_map.json + _pose.csv): apples by side, fused path,
odometry-only path (grey).

    python tools/analysis/plot_maps.py <dir with flat take files> out.png 083517,083556
"""
import json, sys, os, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, out = sys.argv[1], sys.argv[2]
ids = sys.argv[3].split(",")
fig, axs = plt.subplots(1, len(ids), figsize=(7 * len(ids), 7))
for ax, rid in zip(np.atleast_1d(axs), ids):
    p = glob.glob(os.path.join(D, "*%s*_map.json" % rid))[0]
    m = json.load(open(p))
    col = {"L": "tab:blue", "R": "tab:orange", "": "0.6"}
    for l in m["landmarks"]:
        ax.plot(l["x"] / 1000, l["y"] / 1000, "o", ms=4, color=col[l["side"]], alpha=0.35 if l["state"] == "archived" else 1)
    tr = np.array(m["trail"]) / 1000
    ax.plot(tr[:, 0], tr[:, 1], "k-", lw=1)
    pose = list(csv.DictReader(open(p.replace("_map.json", "_pose.csv"))))
    o = np.array([[float(r["odo_x_mm"]), float(r["odo_y_mm"])] for r in pose]) / 1000
    ax.plot(o[:, 0], o[:, 1], "-", color="0.7", lw=1)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title("%s: %d apples (%d archived), resid %s mm" % (rid, len(m["landmarks"]),
                 sum(l["state"] == "archived" for l in m["landmarks"]), m["stats"]["resid_mm"]))
plt.tight_layout(); plt.savefig(out, dpi=80)
