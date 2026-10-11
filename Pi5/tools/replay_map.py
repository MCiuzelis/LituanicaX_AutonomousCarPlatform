#!/usr/bin/env python3
"""Replay the track map over a recorded take: same code as the live dashboard, fed from the files.

    python3 tools/replay_map.py run recordings/<id> [--mm-per-turn 23.3] [--no-correct] [--pitch]
            [--attitude highpass|absolute|static] [--png] [--video [--every 2]]
    python3 tools/replay_map.py fit-scale recordings/<id> [recordings/<id2> ...]

Inputs: the take's _imu.csv (game_rv -> car attitude through the take's IMU mount), _drive.csv
(encoder turns) and detections: the take's own <id>_det.csv (recorded live with the track map on)
or results/trackmap/<id>/<id>_det.csv (tools/detect_take.py, for older takes). Only the raw box
columns are read, so a live take can be re-mapped with other parameters.

run       writes results/trackmap/<id>/ : the same _det / _pose / _landmarks / _track / _map files a
          live recording gets, map.png (top-down: apples by side, path, boundaries; odometry-only path
          in grey), and with --video an annotated mp4 (detections, mapped apples projected back into
          the image, boundaries, inset map) for review.
fit-scale estimates odometry mm_per_turn: apples do not move, so every apple tracked over several
          frames must stay put; linear least squares in the scale and a static camera pitch offset
          (the only other thing that makes still apples drift). --write saves it to config/odometry.json.
Runs on the Pi or a PC (numpy; the video / png need OpenCV).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from hardware.imu_frame import CarFrame                                # noqa: E402
from vision.camera_model import CameraModel                            # noqa: E402
from vision.track_map import (DEFAULT_ODO, ODO_CONFIG, OUT_DIR, MapLog, TrackMapper,  # noqa: E402
                              load_json, load_params, take_files, write_json)


def load_take(take):
    f = take_files(take)
    meta = json.load(open(f["meta"]))
    det_path = f["det"] if os.path.exists(f["det"]) else os.path.join(OUT_DIR, f["id"], f["id"] + "_det.csv")
    if not os.path.exists(det_path):
        sys.exit("no detections for %s: run tools/detect_take.py %s first" % (f["id"], take))
    mnt = meta["imu"]["mount"]
    frame = CarFrame(mnt["forward"], mnt["up"], mnt.get("level_gravity"), mnt.get("level_time"))
    events = []                                                       # (t, order, kind, payload)
    for r in csv.DictReader(open(f["imu"])):
        if r["sensor"] == "game_rv":
            q = [float(r["v%d" % k]) for k in range(4)]
            events.append((int(r["t_ns"]) * 1e-9, 0, "imu", frame.attitude(q)))
    for r in csv.DictReader(open(f["drive"])):
        events.append((int(r["t_ns"]) * 1e-9, 1, "turns", float(r["odo_turns"])))
    dets = {}
    for r in csv.DictReader(open(det_path)):
        k = int(r["sensor_ts_ns"])
        d = dets.setdefault(k, {"t": int(r["t_mid_ns"]) * 1e-9, "seq": r["seq"], "sensor_ts": k, "boxes": [],
                                "crop": tuple(int(r[c]) for c in ("crop_x", "crop_y", "crop_w", "crop_h"))})
        if int(r["box"]) >= 0:
            d["boxes"].append(tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2", "score")) + (int(r["cls"] or 0),))
    for d in dets.values():
        # boxes become available ~25 ms after the exposure (measured live: sensor -> boxes p50)
        events.append((d["t"] + 0.025, 2, "det", d))
    events.sort(key=lambda e: (e[0], e[1]))
    return f, meta, events


def replay(events, mapper, video_frames=None, on_tick=None):
    next_tick = None
    for t, _, kind, x in events:
        if kind == "imu":
            mapper.add_imu(t, *x)
        elif kind == "turns":
            mapper.add_turns(t, x)
        else:
            mapper.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
        mapper.step()
        if next_tick is None:
            next_tick = t
        while t >= next_tick:
            mapper.tick()
            if on_tick:
                on_tick(next_tick, mapper)
            next_tick += 0.1
    mapper.step()
    mapper.tick()
    return mapper


def make_mapper(args, cam=None):
    p = load_params()
    p.correct = not args.no_correct
    if args.pitch:
        p.estimate_pitch = True
    if args.attitude:
        p.attitude = args.attitude
    odo = load_json(ODO_CONFIG, DEFAULT_ODO)
    if args.mm_per_turn:
        odo["mm_per_turn"] = args.mm_per_turn
    return TrackMapper(cam or CameraModel.load(), p, odo)


# ---------------------------------------------------------------- drawing
SIDE_BGR = {1: (255, 160, 60), -1: (40, 140, 255), 0: (190, 190, 190)}


def draw_map(m, size=900, odo_trail=None, view=None, scale=None, car=True, img=None, bounds_world=None, path=None):
    """Top-down map (cv2 image). view = (cx, cy) world mm centre (default: fit everything).
    path: fused path to draw (default: the mapper's rolling trail)."""
    import cv2
    lms = [lm for lm in m.lms if lm.state > 0]
    path = path if path is not None else [(x, y) for _, x, y in m.trail]
    pts = [lm.x for lm in lms] + [np.array(p) for p in path]
    if odo_trail:
        pts += [np.array(p) for p in odo_trail]
    if view is None:
        P = np.array(pts) if pts else np.zeros((1, 2))
        lo, hi = P.min(0) - 500, P.max(0) + 500
        view = (lo + hi) / 2
        scale = scale or (size - 20) / max(hi - lo)
    if img is None:
        img = np.full((size, size, 3), 24, np.uint8)
    H = img.shape[0]

    def px(p):   # world X up the image, Y to the left
        return (int(H / 2 - (p[1] - view[1]) * scale), int(H / 2 - (p[0] - view[0]) * scale))
    grid = 1000
    g0 = (np.array(view) - H / 2 / scale) // grid * grid
    for k in range(int(H / scale / grid) + 2):
        for axis in (0, 1):
            v = g0[axis] + k * grid
            a, b = (np.array([v, -1e7]), np.array([v, 1e7])) if axis == 0 else (np.array([-1e7, v]), np.array([1e7, v]))
            cv2.line(img, px(a), px(b), (40, 40, 40), 1)
    if odo_trail and len(odo_trail) > 1:
        cv2.polylines(img, [np.array([px(p) for p in odo_trail], np.int32)], False, (90, 90, 90), 1)
    if len(path) > 1:
        cv2.polylines(img, [np.array([px(p) for p in path], np.int32)], False, (220, 220, 220), 1)
    b = bounds_world if bounds_world is not None else (m.bounds or {})
    for key, col in (("left_w", SIDE_BGR[1]), ("right_w", SIDE_BGR[-1]), ("center_w", (120, 220, 120))):
        pl = b.get(key, [])
        if len(pl) > 1:
            cv2.polylines(img, [np.array([px(p) for p in pl], np.int32)], False, col, 2)
    for lm in lms:
        col = SIDE_BGR[lm.side] if lm.state == 1 else tuple(c // 2 for c in SIDE_BGR[lm.side])
        r = max(2, int(40 * scale))
        cv2.circle(img, px(lm.x), r, col, -1 if lm.state == 1 else 1)
    if car:
        pose = m.pose()
        if pose is not None:
            c, s = math.cos(pose[2]), math.sin(pose[2])
            tri = [(pose[0] + 120 * c, pose[1] + 120 * s), (pose[0] - 200 * c - 70 * s, pose[1] - 200 * s + 70 * c),
                   (pose[0] - 200 * c + 70 * s, pose[1] - 200 * s - 70 * c)]
            cv2.fillPoly(img, [np.array([px(p) for p in tri], np.int32)], (60, 220, 255))
    return img


def write_video(take_f, meta, events, mapper, out_path, every=2, width=960):
    """Annotated review video: detections (green used / grey not), mapped apples projected back
    (filled = confirmed, colour = side), boundaries, inset map."""
    import cv2
    rows = list(csv.DictReader(open(take_f["frames"])))
    frame_t = [int(r["t_mid_ns"]) * 1e-9 for r in rows]
    cap = cv2.VideoCapture(take_f["video"])
    W = width
    Hh = int(round(W * meta["stream"]["height"] / meta["stream"]["width"]))
    fps = meta["stream"]["fps"] / every
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W + Hh, Hh))
    fi = 0
    img = None
    dets_by_ts = {}

    def grab(t_until):
        nonlocal fi, img
        while fi < len(rows) and frame_t[fi] <= t_until:
            ok, im = cap.read()
            if not ok:
                return
            if fi % every == 0:
                emit(fi, cv2.resize(im, (W, Hh), interpolation=cv2.INTER_AREA))
            fi += 1

    def emit(i, im):
        m = mapper
        f = m.last_frame
        t = frame_t[i]
        crop = tuple(meta["stream"]["scaler_crop"] or (768, 432, 3072, 1728))
        if f is not None and abs(f["t"] - t) < 0.05:
            for k, b in enumerate(f["boxes"]):
                used = k in f["idx"]
                cv2.rectangle(im, (int(b[0] * W), int(b[1] * Hh)), (int(b[2] * W), int(b[3] * Hh)),
                              (80, 220, 80) if used else (140, 140, 140), 1)
        ov = m.image_overlay(t, crop) if m.odo.last_t else {}
        for key, col in (("left", SIDE_BGR[1]), ("right", SIDE_BGR[-1])):
            pl = ov.get(key, [])
            if len(pl) > 1:
                cv2.polylines(im, [np.array([(int(x * W), int(y * Hh)) for x, y in pl], np.int32)], False, col, 2)
        for x, y, lid, side in ov.get("lms", []):
            cv2.circle(im, (int(x * W), int(y * Hh)), 5, SIDE_BGR[side], -1)
        s = m.stats
        txt = "t %.1fs  apples %d  corr %d  resid %smm  pitch_bias %.2f  w %.0fmm" % (
            t - frame_t[0], sum(1 for lm in m.lms if lm.state == 1), s["corrected"], m.resid_mm(), m.pitch_bias, m.width)
        cv2.putText(im, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        pose = m.pose()
        inset = draw_map(m, Hh, view=(pose[0] + 1200 * math.cos(pose[2]), pose[1] + 1200 * math.sin(pose[2]))
                         if pose else (0, 0), scale=Hh / 6000.0) if pose else np.zeros((Hh, Hh, 3), np.uint8)
        vw.write(np.hstack([im, inset]))

    for ev in events:
        t = ev[0]
        grab(t - 0.03)   # frames are drawn once the map has seen everything up to them
        kind, x = ev[2], ev[3]
        if kind == "imu":
            mapper.add_imu(t, *x)
        elif kind == "turns":
            mapper.add_turns(t, x)
        else:
            mapper.add_detections(x["t"], x["boxes"], x["crop"], x["seq"], x["sensor_ts"])
        mapper.step()
        if not hasattr(mapper, "_vt") or t >= mapper._vt:
            mapper.tick()
            mapper._vt = t + 0.1
    grab(1e18)
    vw.release()
    cap.release()


# ---------------------------------------------------------------- commands
def cmd_run(args):
    for take in args.takes:
        f, meta, events = load_take(take)
        out = os.path.join(OUT_DIR, f["id"])
        os.makedirs(out, exist_ok=True)
        m = make_mapper(args)
        if args.video:
            t0 = time.monotonic()
            write_video(f, meta, events, m, os.path.join(out, f["id"] + "_review.mp4"), every=args.every)
            print("video %.0f s" % (time.monotonic() - t0))
        else:
            m.log = MapLog(out, f["id"])
            odo_trail, path = [], []

            def on_tick(t, mm):
                o, q = mm.odo_pose(), mm.pose()
                if o and (not odo_trail or math.hypot(o[0] - odo_trail[-1][0], o[1] - odo_trail[-1][1]) > 20):
                    odo_trail.append(o[:2])
                if q and (not path or math.hypot(q[0] - path[-1][0], q[1] - path[-1][1]) > 20):
                    path.append(q[:2])
            t0 = time.monotonic()
            replay(events, m, on_tick=on_tick)
            dt = time.monotonic() - t0
            m.log.close(m.final_map())
            m.log = None
            if args.png:
                import cv2
                cv2.imwrite(os.path.join(out, "map.png"), draw_map(m, 1000, odo_trail=odo_trail, path=path))
            print_summary(f["id"], m, dt, events, path)


def print_summary(rid, m, dt, events, path=()):
    s = m.stats
    conf = [lm for lm in m.lms if lm.state > 0]
    side = {k: sum(1 for lm in conf if lm.side == k) for k in (1, -1, 0)}
    dur = events[-1][0] - events[0][0]
    o = m.odo_pose()
    print("%s: %.0f s take replayed in %.1f s (%.0f frames/s) | frames %d boxes %d used %d matched %d | corrections %d "
          "| apples created %d confirmed %d dropped %d forgotten %d kept %d (L %d R %d ?) %d | resid %s mm | pitch_bias %.2f | "
          "width %.0f | correction %s | path %.1f m | enc glitches %d" % (
              rid, dur, dt, s["frames"] / max(dt, 1e-6), s["frames"], s["boxes"], s["used"], s["matched"], s["corrected"],
              s["created"], s["confirmed"], s["dropped"], s["forgotten"], len(conf), side[1], side[-1], side[0], m.resid_mm(),
              m.pitch_bias, m.width, np.round(m.C, 1).tolist(), path_len(path) / 1000, s["enc_glitches"]))


def path_len(tr):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(tr[:-1], tr[1:]))


def cmd_fit_scale(args):
    """Apples do not move: for every apple tracked over several frames (correction off),
    k * unit-scale path + lever arm + R(heading) * (observation + delta * d obs / d pitch) is constant.
    Centred per apple, that is linear least squares in k (mm per turn) and delta (static camera
    pitch error, deg; the only other thing that makes a still apple seem to move as the car
    approaches it). Iterated: the associations are redone with the new k."""
    loaded = [load_take(t) for t in args.takes]
    k = args.mm_per_turn or load_json(ODO_CONFIG, DEFAULT_ODO)["mm_per_turn"]
    for it in range(args.iterations):
        rows_all, per_take = [], []
        for f, meta, events in loaded:
            args.mm_per_turn = k
            m = make_mapper(args)
            m.p.correct = False                # raw odometry: corrections would hide a bad scale
            m.p.estimate_pitch = False
            obs = _collect_tracks(m)
            replay(events, m)
            rows = _track_rows(obs, m, args.max_range)
            rows_all += rows
            per_take.append((f["id"], _solve_scale(rows, k)))
        sol = _solve_scale(rows_all, k)
        print("iteration %d (associated at %.2f mm/turn):" % (it + 1, k))
        for rid, sl in per_take:
            print("   %s  %.2f mm/turn  pitch offset %+.2f deg  (%d apples, %d obs, rms %.0f mm)"
                  % (rid, sl["k"], sl["delta"], sl["tracks"], sl["obs"], sl["rms"]))
        print("   all takes                            %.2f mm/turn  pitch offset %+.2f deg  (%d apples, %d obs, rms %.0f mm)"
              % (sol["k"], sol["delta"], sol["tracks"], sol["obs"], sol["rms"]), flush=True)
        k = sol["k"]
    if args.write:
        odo = load_json(ODO_CONFIG, DEFAULT_ODO)
        odo.update(mm_per_turn=round(float(k), 3), source="tools/replay_map.py fit-scale %s (%s); static pitch "
                   "offset seen %+.2f deg" % (" ".join(f[0]["id"] for f in loaded), time.strftime("%Y-%m-%dT%H:%M:%S"),
                                              sol["delta"]))
        write_json(ODO_CONFIG, odo)
        print("written", ODO_CONFIG)


def _collect_tracks(m):
    """mapper.on_frame hook: per observation (apple id, t, camera floor xy, pose, h)."""
    obs = []

    def on_frame(res):
        pose, h = res["pose"], res["cam"]["h"]
        for k, lm in res["lm"].items():
            obs.append((lm.id, res["t"], res["xy"][k][0], res["xy"][k][1], pose[0], pose[1], pose[2], h))
    m.on_frame = on_frame
    return obs


def _track_rows(obs, m, max_range):
    """Per apple: centred design rows for (k, delta) and the right-hand side, whitened."""
    by = {}
    for o in obs:
        by.setdefault(o[0], []).append(o)
    out = []
    lc, k_used = m.cam_ahead, m.mm_per_turn
    for lid, tr in by.items():
        tr = [o for o in tr if math.hypot(o[2], o[3]) < max_range]
        if len(tr) < 10:
            continue
        a = np.array(tr)
        th = a[:, 6]
        dirs = np.stack([np.cos(th), np.sin(th)], 1)
        pu = (a[:, 4:6] - lc * dirs) / k_used                      # rear axle path at unit scale
        if np.ptp(pu @ dirs.mean(0)) * k_used < 300:              # needs >= 0.3 m of travel while tracked
            continue
        xy, h = a[:, 2:4], a[:, 7]
        d = np.hypot(xy[:, 0], xy[:, 1])
        c, s = np.cos(th), np.sin(th)
        rot = lambda v: np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)   # noqa: E731
        ro = rot(xy)
        g = rot(xy / d[:, None]) * (-(d * d + h * h) / h * math.pi / 180.0)[:, None]
        rhs = -(lc * dirs + ro)
        sig = np.hypot(m.p.sigma_base_mm, (d * d + h * h) / h * math.hypot(m.p.sigma_px_rad, m.p.sigma_pitch_rad))
        for comp in (0, 1):
            A = np.stack([pu[:, comp] - pu[:, comp].mean(), g[:, comp] - g[:, comp].mean()], 1)
            b = rhs[:, comp] - rhs[:, comp].mean()
            out.append((lid, A / sig[:, None], b / sig, sig))
    return out


def _solve_scale(rows, k0):
    if not rows:
        return {"k": k0, "delta": 0.0, "tracks": 0, "obs": 0, "rms": float("nan")}
    A = np.concatenate([r[1] for r in rows])
    b = np.concatenate([r[2] for r in rows])
    sig = np.concatenate([r[3] for r in rows])
    keep = np.ones(len(b), bool)
    for _ in range(3):                                     # trim outliers (mis-associations)
        x, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        r = A @ x - b
        keep = np.abs(r) < 3.0 * max(1.0, 1.4826 * np.median(np.abs(r[keep])))
    res_mm = (A[keep] @ x - b[keep]) * sig[keep]
    return {"k": float(x[0]), "delta": float(x[1]), "tracks": len({r[0] for r in rows}), "obs": int(keep.sum() // 2),
            "rms": float(np.sqrt((res_mm ** 2).mean()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "fit-scale"):
        s = sub.add_parser(name)
        s.add_argument("takes", nargs="+")
        s.add_argument("--mm-per-turn", type=float)
        s.add_argument("--no-correct", action="store_true", help="odometry only (no camera pose correction)")
        s.add_argument("--pitch", action="store_true", help="also estimate the camera pitch bias (experimental)")
        s.add_argument("--attitude", choices=("highpass", "absolute", "static"),
                       help="default: config/track_map.json or highpass")
        if name == "run":
            s.add_argument("--png", action="store_true")
            s.add_argument("--video", action="store_true")
            s.add_argument("--every", type=int, default=2, help="video: every Nth frame")
        else:
            s.add_argument("--iterations", type=int, default=2)
            s.add_argument("--max-range", type=float, default=2000.0, help="observations closer than this (mm)")
            s.add_argument("--write", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run, "fit-scale": cmd_fit_scale}[a.cmd](a)


if __name__ == "__main__":
    main()
