#!/usr/bin/env python3
"""Could the car have driven this take by itself? Replays recorded takes through the track map
(rolling window, vision/track_map.py) and the planner (vision/planner.py) and compares, every 0.1 s:

  boundaries   the path the driver actually took over the next ~1.5 m (odometry, car frame) must lie
               between the left and the right boundary the map had at that moment: a boundary on the
               wrong side of the driven path is a map error the car would have followed
  planner      the curvature the planner commands vs the curvature the car really drove a moment later
               (yaw rate / speed after the actuation lag); the driver's line is the reference
  coverage     how far ahead a path was known and from what (centre line / one side / lost)

    python3 tools/autonomy_eval.py recordings/<id> [recordings/<id2> ...] [--csv out.csv] [--plot out.png]
                                   [--window-s 4] [--behind-mm 1000]

Needs the take's _det.csv (live detections, or tools/detect_take.py) like tools/replay_map.py.
numpy only (+ matplotlib for --plot).
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from tools.replay_map import load_take                               # noqa: E402
from vision.planner import Planner                                   # noqa: E402
from vision.track_map import TrackMapper, load_params                # noqa: E402

FUTURE_MM = 1500.0      # driven path checked against the boundaries
MOVING_MM_S = 300.0


def signed_lateral(F, Q):
    """Signed distance (left +) of points Q (M, 2) from the polyline F (N, 2) at their nearest
    segment; NaN where the nearest point is the polyline's start / end (beside the checked stretch)."""
    a, b = F[:-1], F[1:]
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-9)
    out = np.full(len(Q), np.nan)
    for i, q in enumerate(Q):
        tt = ((q - a) * ab).sum(1) / L2
        tc = np.clip(tt, 0, 1)
        c = a + tc[:, None] * ab
        d = np.hypot(*(q - c).T)
        j = int(np.argmin(d))
        if (j == 0 and tt[j] < 0) or (j == len(ab) - 1 and tt[j] > 1):
            continue
        cr = ab[j, 0] * (q - a[j])[1] - ab[j, 1] * (q - a[j])[0]
        out[i] = math.copysign(d[j], cr)
    return out


def evaluate(take, args):
    f, meta, events = load_take(take)
    p = load_params()
    p.window_s, p.window_behind_mm = args.window_s, args.behind_mm
    m = TrackMapper(params=p)
    planner = Planner()
    planner.p.gates = not args.no_gates
    ticks = []

    next_tick = None
    for t, _, kind, x in events:
        if kind == "imu":
            m.add_imu(t, *x)
        elif kind == "turns":
            m.add_turns(t, x)
        else:
            m.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
        m.step()
        if next_tick is None:
            next_tick = t
        while t >= next_tick:
            b = m.tick(next_tick)
            o = m.odo_pose(next_tick)
            prev = m.odo_pose(next_tick - 0.1)
            v = math.hypot(o[0] - prev[0], o[1] - prev[1]) / 0.1 if o and prev else 0.0
            plan = planner.update(b, v, next_tick, apples=m.local_apples) if b is not None else None
            ticks.append({"t": next_tick, "odo": o, "v": v, "bounds": b, "plan": plan,
                          "n_lms": sum(1 for lm in m.lms if lm.state == 1)})
            next_tick += 0.1

    # the car's real curvature and the driver's steering, by time
    fr = list(csv.DictReader(open(f["frames"])))
    ft = np.array([int(r["t_mid_ns"]) for r in fr]) * 1e-9
    v_fr = np.abs(np.array([float(r["motor_rps"]) for r in fr])) * m.mm_per_turn
    k_fr = np.radians([float(r["gyro_z_dps"]) for r in fr]) / np.maximum(v_fr, 1.0) * 1000.0
    st_fr = np.array([float(r["steer"]) for r in fr])
    odo_xy = np.array([k["odo"][:2] if k["odo"] else (np.nan, np.nan) for k in ticks])
    odo_th = np.array([k["odo"][2] if k["odo"] else np.nan for k in ticks])

    rows = []
    lag = planner.p.actuation_lag_s
    for i, k in enumerate(ticks):
        if k["odo"] is None or k["v"] < MOVING_MM_S or k["plan"] is None:
            continue
        # driven path over the next FUTURE_MM, in the car frame of this tick
        c, s = math.cos(odo_th[i]), math.sin(odo_th[i])
        rel = (odo_xy[i:] - odo_xy[i]) @ np.array([[c, -s], [s, c]])
        dist = np.r_[0, np.cumsum(np.hypot(*np.diff(rel, axis=0).T))]
        F = rel[dist <= FUTURE_MM]
        b = k["bounds"] or {}
        pl = k["plan"]
        sides = {"left": pl.get("left"), "right": pl.get("right")}     # what the planner followed
        left_err = right_err = np.nan
        clear_l = clear_r = np.nan
        if len(F) >= 4 and dist[-1] >= FUTURE_MM * 0.8:
            for key in ("left", "right"):
                Q = sides[key] if sides[key] is not None else np.zeros((0, 2))
                Q = np.asarray(Q, float).reshape(-1, 2)
                Q = Q[(Q[:, 0] > 0) & (Q[:, 0] < FUTURE_MM)] if len(Q) else Q
                if not len(Q):
                    continue
                sd = signed_lateral(F, Q)
                sd = sd[~np.isnan(sd)]
                if not len(sd):
                    continue
                if key == "left":
                    left_err, clear_l = float((sd < 0).any()), float(sd.min())
                else:
                    right_err, clear_r = float((sd > 0).any()), float(-sd.max())
        j = int(np.searchsorted(ft, k["t"] + lag))
        jj = slice(max(0, j - 3), j + 4)                       # +-50 ms around it
        real_k = float(np.median(k_fr[jj])) if j < len(ft) else np.nan
        human = float(np.mean(st_fr[max(0, int(np.searchsorted(ft, k["t"])) - 3):int(np.searchsorted(ft, k["t"])) + 4])) \
            if j < len(ft) else np.nan
        rows.append({"t": k["t"] - ticks[0]["t"], "v": k["v"], "mode": pl["mode"], "known_mm": pl["known_mm"],
                     "plan_k": pl.get("curvature"), "plan_steer": pl.get("steer"), "plan_v": pl.get("speed_mm_s"),
                     "real_k": real_k, "human_steer": human, "n_lms": k["n_lms"],
                     "width": (b or {}).get("width"), "left_err": left_err, "right_err": right_err,
                     "clear_l": clear_l, "clear_r": clear_r})
    return f["id"], rows, m


def summarize(rid, rows):
    n = len(rows)
    if not n:
        print("%s: no moving ticks" % rid)
        return {}
    a = lambda k: np.array([np.nan if r[k] is None else r[k] for r in rows], float)   # noqa: E731
    modes = [r["mode"] for r in rows]
    le, re_ = a("left_err"), a("right_err")
    pk, rk = a("plan_k"), a("real_k")
    ok = ~np.isnan(pk) & ~np.isnan(rk)
    corr = float(np.corrcoef(pk[ok], rk[ok])[0, 1]) if ok.sum() > 10 else np.nan
    turn = ok & (np.abs(rk) > 0.7)
    sign = float(np.mean(np.sign(pk[turn]) == np.sign(rk[turn]))) if turn.any() else np.nan
    known = a("known_mm")
    s = {"ticks": n, "gates": modes.count("gates") / n, "center": modes.count("center") / n, "one_side": (modes.count("left") + modes.count("right")) / n,
         "lost": modes.count("lost") / n, "known_p10": float(np.nanpercentile(known, 10)),
         "known_p50": float(np.nanpercentile(known, 50)),
         "left_checked": int((~np.isnan(le)).sum()), "left_wrong": float(np.nanmean(le)) if (~np.isnan(le)).any() else np.nan,
         "right_checked": int((~np.isnan(re_)).sum()), "right_wrong": float(np.nanmean(re_)) if (~np.isnan(re_)).any() else np.nan,
         "k_corr": corr, "k_mae": float(np.nanmedian(np.abs(pk[ok] - rk[ok]))) if ok.any() else np.nan,
         "turn_sign_agree": sign, "lms_p50": float(np.median(a("n_lms"))), "lms_max": float(np.max(a("n_lms"))),
         "width_p50": float(np.nanmedian(a("width")))}
    print("%s: %d moving ticks | path: gates %.0f%% centre %.0f%% one side %.0f%% lost %.0f%% | known ahead p10 %.2f p50 %.2f m | "
          "boundary on the wrong side of the driven path: left %.1f%% of %d, right %.1f%% of %d | planner vs driven "
          "curvature: r %.2f, median |err| %.2f 1/m, turn direction agrees %.0f%% | apples in map p50 %.0f max %.0f | "
          "width %.2f m" % (rid, n, 100 * s["gates"], 100 * s["center"], 100 * s["one_side"], 100 * s["lost"], s["known_p10"] / 1000,
                            s["known_p50"] / 1000, 100 * s["left_wrong"], s["left_checked"], 100 * s["right_wrong"],
                            s["right_checked"], s["k_corr"], s["k_mae"], 100 * s["turn_sign_agree"], s["lms_p50"],
                            s["lms_max"], s["width_p50"] / 1000))
    return s


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(results), 1, figsize=(14, 3.2 * len(results)), squeeze=False)
    for ax, (rid, rows) in zip(axs[:, 0], results):
        t = np.array([r["t"] for r in rows])
        g = lambda k: np.array([np.nan if r[k] is None else r[k] for r in rows], float)   # noqa: E731
        ax.plot(t, g("real_k"), color="k", lw=1.2, label="driven curvature (after lag)")
        ax.plot(t, g("plan_k"), color="tab:green", lw=1.2, label="planner curvature")
        ax.plot(t, -1.45 * g("human_steer"), color="0.6", lw=0.8, label="driver steer x full lock")
        bad = (g("left_err") > 0) | (g("right_err") > 0)
        ax.plot(t[bad], np.zeros(bad.sum()), "rx", ms=5, label="boundary on wrong side")
        lost = np.array([r["mode"] == "lost" for r in rows])
        ax.plot(t[lost], np.full(lost.sum(), -2.4), "m|", ms=8, label="no path")
        ax.set_ylim(-2.6, 2.6)
        ax.set_ylabel("1/m (left +)")
        ax.set_title(rid)
        ax.grid(alpha=.3)
    axs[0, 0].legend(loc="upper right", fontsize=8, ncol=5)
    axs[-1, 0].set_xlabel("s")
    plt.tight_layout()
    plt.savefig(path, dpi=80)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--window-s", type=float, default=load_params().window_s)
    ap.add_argument("--behind-mm", type=float, default=load_params().window_behind_mm)
    ap.add_argument("--csv", help="per-tick rows of all takes")
    ap.add_argument("--plot", help="curvature plot (png)")
    ap.add_argument("--no-gates", action="store_true", help="planner on the map's chained boundaries only")
    args = ap.parse_args()
    results = []
    for take in args.takes:
        rid, rows, _ = evaluate(take, args)
        summarize(rid, rows)
        results.append((rid, rows))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = None
            for rid, rows in results:
                for r in rows:
                    if w is None:
                        w = csv.DictWriter(fh, ["take"] + list(r))
                        w.writeheader()
                    w.writerow(dict(r, take=rid))
    if args.plot:
        plot(results, args.plot)


if __name__ == "__main__":
    main()
