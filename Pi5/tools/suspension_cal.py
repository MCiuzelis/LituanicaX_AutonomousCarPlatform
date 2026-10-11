#!/usr/bin/env python3
"""Suspension calibration: camera height / attitude above the floor vs the IMU on the camera bracket.

The IMU shares the camera's bracket, so it sees every camera ROTATION directly; what it cannot see
is how the camera's HEIGHT changes when the suspension moves (a few mm of height is a few % of
every floor distance). This measures both against a checkerboard on the floor:

  * car on level floor, ONE board (8x6 inner corners, 30 mm squares, like tools/camera_cal.py)
    flat on the floor ~0.4-0.7 m in front of the camera, fully in view
  * `capture`: press the car down and let go - front, rear, left, right, diagonals, all of it -
    slowly, for as long as you like (Ctrl-C ends the run and saves it); every frame gives the camera's exact pose above the floor
    (solvePnP on the board) and the IMU attitude at that frame's exposure time
  * `fit`: camera pitch above floor = p0 - pitch_gain * d_pitch - pitch_per_roll * d_roll  (rigid: gain ~1)
           camera roll above floor  = r0 - roll_gain * d_roll
           camera height            = h0 + h_per_pitch * d_pitch + h_per_abs_roll * |d_roll| + h_per_roll * d_roll
           (d_* = IMU pitch / roll change; a car pressed down on either side sits lower, hence |d_roll|)
    -> config/suspension.json, read by vision/track_map.py. Also compares h0 / p0 / r0 with
    config/camera.json (the static mount the track map starts from); --update-mount writes them there.

    pkill -f "[p]ython3 dashboard.py"                         # camera + IMU are the dashboard's
    python3 tools/suspension_cal.py capture                  # live readout while you push the car; Ctrl-C ends it
    python3 tools/suspension_cal.py fit results/suspension_cal/<ts> [--write] [--update-mount]
    python3 tools/suspension_cal.py check <image.png>        # board pose(s) in a 1280x720 image (test)
    scripts/dashboard_restart.sh

IMU angles are car frame (CarFrame: roll right side down +, pitch nose up +) relative to the level
reference in config/imu.json; only their changes matter. Needs python3-opencv.
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from vision.camera_model import CONFIG as CAM_CONFIG, CROP_1536, CameraModel   # noqa: E402
from vision.track_map import DEFAULT_SUSP, SUSP_CONFIG, camera_angles, write_json   # noqa: E402

RUNS = os.path.join(ROOT, "results", "suspension_cal")
SIZE = (1280, 720)                    # main stream from the 1536x864 raw mode = CROP_1536


def find_board(gray, pattern, last=None):
    """Board corners (image px) or None. With the last corners known, searches a window around
    them (~0.2 s on the Pi); otherwise, or if that fails, the whole frame with every stretch
    camera_cal uses (~1-3 s, only until the board is found again)."""
    import cv2
    from camera_cal import _find, _orient
    H, W = gray.shape
    if last is not None:
        lo, hi = last.min(0), last.max(0)
        pad = (hi - lo) * 0.4 + 20
        x0, y0 = np.maximum(lo - pad, 0).astype(int)
        x1, y1 = np.minimum(hi + pad, [W, H]).astype(int)
        c = _find(cv2, gray[y0:y1, x0:x1], pattern)
        if c is not None:
            return _orient(c + [x0, y0], pattern)
    c = _find(cv2, gray, pattern)
    return None if c is None else _orient(c, pattern)


def board_pose(gray, cam, pattern, square, crop=CROP_1536, size=SIZE, last=None):
    """Camera pose above the board plane -> dict(h, pitch, roll, heading, rms_px, corners) or None."""
    c = find_board(gray, pattern, last)
    if c is None:
        return None
    return pose_from_corners(c, cam, pattern, square, crop, size)


def pose_from_corners(corners, cam, pattern, square, crop=CROP_1536, size=SIZE):
    import cv2
    r, cc = np.mgrid[0:pattern[1], 0:pattern[0]]
    obj = np.stack([cc.ravel() * square, r.ravel() * square, np.zeros(r.size)], 1).astype(np.float64)
    K = cam.K(crop, size)
    ok, rv, tv = cv2.solvePnP(obj, corners.astype(np.float64), K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    ok, rv, tv = cv2.solvePnP(obj, corners.astype(np.float64), K, cam.dist, rv, tv, True, cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rv, tv, K, cam.dist)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - corners) ** 2).sum(1).mean()))
    R = cv2.Rodrigues(rv)[0]                       # board -> camera
    C = -R.T @ tv.ravel()                          # camera centre in board coords
    c2b = R.T                                      # camera -> board
    if C[2] < 0:                                   # board z must point up (towards the camera)
        F = np.diag([1.0, -1.0, -1.0])
        C, c2b = F @ C, F @ c2b
    pitch, roll, heading = camera_angles(c2b)
    return {"h": float(C[2]), "pitch": pitch, "roll": roll, "heading": heading, "rms_px": rms,
            "dist": float(np.hypot(C[0] - obj[:, 0].mean(), C[1] - obj[:, 1].mean())), "corners": corners}


# ---------------------------------------------------------------------------------- capture
def capture(args):
    import cv2
    from picamera2 import Picamera2
    from hardware import BNO085, CarFrame
    cam = CameraModel.load()
    frame = CarFrame.load()
    out = os.path.join(RUNS, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    imu = BNO085()
    imu.enable("game_rv", "gravity", "gyro", rate_hz=200)
    att, lock = [], threading.Lock()

    def on_rv(r):
        with lock:
            att.append((r.timestamp, frame.attitude(r.values[:4])))
            if len(att) > 4000:
                del att[:2000]
    imu.on("game_rv", on_rv)

    def att_at(t):
        with lock:
            a = list(att)
        if len(a) < 2 or not a[0][0] <= t <= a[-1][0] + 0.02:
            return None
        ts = [x[0] for x in a]
        i = max(1, min(len(a) - 1, int(np.searchsorted(ts, t))))
        (t0, a0), (t1, a1) = a[i - 1], a[i]
        f = 0.0 if t1 <= t0 else max(0.0, min(1.0, (t - t0) / (t1 - t0)))
        return [x + f * (y - x) for x, y in zip(a0, a1)]

    pc = Picamera2()
    pc.configure(pc.create_video_configuration(main={"size": SIZE, "format": "RGB888"}, raw={"size": (1536, 864)},
                                               buffer_count=4))
    pc.set_controls({"FrameDurationLimits": (33333, 33333)})
    pc.start()
    time.sleep(1.5)
    readout_ns = 1e9 / 120.13
    rows, saved, last, misses = [], 0, None, 0
    t_start = time.monotonic()
    t_end = t_start + args.seconds if args.seconds > 0 else float("inf")
    print("hold the car still for the first 3 s (rest pose), then press it down / let go in every direction; "
          "Ctrl-C when done (samples are saved)", flush=True)
    try:
        while time.monotonic() < t_end:
            req = pc.capture_request()
            try:
                md, img = req.get_metadata(), req.make_array("main")
            finally:
                req.release()
            t_mid = (md["SensorTimestamp"] - md.get("ExposureTime", 0) * 500 + readout_ns / 2) * 1e-9
            if tuple(md.get("ScalerCrop", CROP_1536)) != CROP_1536:
                sys.exit("unexpected ScalerCrop %s" % (md.get("ScalerCrop"),))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            p = board_pose(gray, cam, (args.cols, args.rows), args.square, last=last)
            a = att_at(t_mid)
            if p is None or a is None:
                misses += 1
                last = None if p is None else last
                print("  no board (hands / shadow over it? keep it fully in view)" if p is None else "  no IMU", flush=True)
                continue
            last = p["corners"]
            rows.append([int(t_mid * 1e9), *("%.3f" % x for x in (p["h"], p["pitch"], p["roll"], p["heading"],
                                                                    p["rms_px"], p["dist"], a[0], a[1]))])
            if saved < 3 and len(rows) in (1, 40, 120):
                im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.drawChessboardCorners(im, (args.cols, args.rows), p["corners"].reshape(-1, 1, 2).astype(np.float32), True)
                cv2.imwrite(os.path.join(out, "frame_%03d.jpg" % len(rows)), im)
                saved += 1
            print("  n %3d  h %6.1f mm  cam pitch %6.2f  roll %6.2f  | imu roll %6.2f  pitch %6.2f  | rms %.2f px"
                  % (len(rows), p["h"], p["pitch"], p["roll"], a[0], a[1], p["rms_px"]), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pc.stop()
        pc.close()
        imu.close()
    with open(os.path.join(out, "samples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "h_mm", "cam_pitch_deg", "cam_roll_deg", "cam_heading_deg", "rms_px", "board_dist_mm",
                    "imu_roll_deg", "imu_pitch_deg"])
        w.writerows(rows)
    with open(os.path.join(out, "capture.json"), "w") as f:
        json.dump({"board": {"pattern": [args.cols, args.rows], "square_mm": args.square}, "size": SIZE,
                   "crop": CROP_1536, "camera_config": json.load(open(CAM_CONFIG)),
                   "imu_config": frame.status()}, f, indent=1)
    dt = time.monotonic() - t_start
    print("%d samples in %.0f s (%.1f /s), %d frames without board / IMU" % (len(rows), dt, len(rows) / max(dt, 1e-6), misses))
    print("%d samples -> %s\nnext: python3 tools/suspension_cal.py fit %s --write" % (len(rows), out, out))


# -------------------------------------------------------------------------------------- fit
def fit(args):
    run = args.run.rstrip("/")
    S = np.array([[float(x) for x in r] for r in list(csv.reader(open(os.path.join(run, "samples.csv"))))[1:]])
    if len(S) < 20:
        sys.exit("only %d samples: capture again (board in view the whole time)" % len(S))
    t, h, cp, cr, rms, iroll, ipitch = S[:, 0] * 1e-9, S[:, 1], S[:, 2], S[:, 3], S[:, 5], S[:, 7], S[:, 8]
    good = rms < max(1.5, 3 * np.median(rms))
    # rest pose: the leading samples before anything moved (IMU within 0.3 deg, height within 1.5 mm of the first)
    moved = (np.abs(ipitch - ipitch[0]) > 0.3) | (np.abs(iroll - iroll[0]) > 0.3) | (np.abs(h - h[0]) > 1.5)
    n_rest = int(np.argmax(moved)) if moved.any() else len(t)
    rest = good & (np.arange(len(t)) < max(1, n_rest)) & (t < t[0] + args.rest_s)
    if rest.sum() < 1:
        sys.exit("no rest pose: keep the car still for the first seconds of the capture")
    r0, p0 = np.median(iroll[rest]), np.median(ipitch[rest])
    dr, dp = iroll - r0, ipitch - p0

    def lin(y, X):
        A = np.c_[np.ones(len(y)), X]
        c, *_ = np.linalg.lstsq(A[good], y[good], rcond=None)
        res = y[good] - A[good] @ c
        return c, float(np.sqrt((res ** 2).mean())), float(np.abs(res).max())
    (pc0, pg, ppr), p_rms, p_max = lin(cp, np.c_[dp, dr])
    (rc0, rg), r_rms, r_max = lin(cr, dr[:, None])
    # a car pressed down on either side sits lower: height goes with |roll| (+ a small one-sided part)
    (h0, kp, kra, kr), h_rms, h_max = lin(h, np.c_[dp, np.abs(dr), dr])
    span_p, span_r = np.ptp(dp[good]), np.ptp(dr[good])
    print("samples %d (%d good), rest pose from %d" % (len(S), good.sum(), rest.sum()))
    print("IMU motion covered: pitch %.2f deg (%.2f .. %.2f), roll %.2f deg%s" % (
        span_p, dp[good].min(), dp[good].max(), span_r,
        "   <- too little: press the car harder / in every direction" if min(span_p, span_r) < 1.5 else ""))
    if dp[good].max() < 1.0:
        print("note: no nose-up samples (rear not pressed): the model is extrapolated for acceleration squat")
    print("camera pitch = %.3f %+.3f * d_pitch %+.3f * d_roll  (rms %.3f, max %.2f deg)" % (pc0, pg, ppr, p_rms, p_max))
    print("camera roll  = %.3f %+.3f * d_roll  (rms %.3f, max %.2f deg)" % (rc0, rg, r_rms, r_max))
    print("camera h     = %.1f %+.2f * d_pitch %+.2f * |d_roll| %+.2f * d_roll mm  (rms %.2f, max %.1f mm)"
          % (h0, kp, kra, kr, h_rms, h_max))
    for name, g in (("pitch", -pg), ("roll", -rg)):
        if not 0.85 < g < 1.15:
            print("WARNING: %s gain %.2f is not ~1: the IMU does not turn with the camera (not on the same "
                  "bracket / loose mount?) or the motion range is too small" % (name, g))
    cm = CameraModel.load()
    hr, pr, rr = (float(np.median(x[rest])) for x in (h, cp, cr))     # measured at rest (not the fit's intercepts)
    print("rest pose (board) vs config/camera.json: h %.1f vs %.1f mm, pitch %.2f vs %.2f deg, roll %.2f vs %.2f deg"
          % (hr, cm.height_mm, pr, cm.pitch_deg, rr, cm.roll_deg))
    out = dict(DEFAULT_SUSP)
    out.update({
        "h_per_pitch_mm_deg": round(float(kp), 3), "h_per_roll_mm_deg": round(float(kr), 3),
        "h_per_abs_roll_mm_deg": round(float(kra), 3),
        "pitch_gain": round(float(-pg), 4), "pitch_per_roll": round(float(-ppr), 4), "roll_gain": round(float(-rg), 4),
        "rest": {"h_mm": round(hr, 2), "cam_pitch_deg": round(pr, 3), "cam_roll_deg": round(rr, 3),
                 "imu_roll_deg": round(float(r0), 3), "imu_pitch_deg": round(float(p0), 3)},
        "quality": {"samples": int(good.sum()), "pitch_span_deg": round(float(span_p), 2),
                    "roll_span_deg": round(float(span_r), 2), "pitch_rms_deg": round(p_rms, 3),
                    "roll_rms_deg": round(r_rms, 3), "h_rms_mm": round(h_rms, 2), "h_max_mm": round(h_max, 1),
                    "pitch_max_deg": round(p_max, 2), "nose_up_max_deg": round(float(dp[good].max()), 2),
                    "board_rms_px": round(float(np.median(rms)), 2)},
        "source": "tools/suspension_cal.py fit %s (%s)" % (run, time.strftime("%Y-%m-%dT%H:%M:%S")),
        "model": "camera pitch = rest - pitch_gain * d_pitch - pitch_per_roll * d_roll; camera roll = rest - roll_gain * "
                 "d_roll; height = rest + h_per_pitch * d_pitch + h_per_abs_roll * |d_roll| + h_per_roll * d_roll; "
                 "d_* = car pitch (nose up +) / roll (right side down +) change from the rest pose, deg",
    })
    with open(os.path.join(run, "fit.json"), "w") as f:
        json.dump(out, f, indent=1)
    if args.write:
        if os.path.exists(SUSP_CONFIG):
            shutil.copy(SUSP_CONFIG, SUSP_CONFIG + ".bak")
        write_json(SUSP_CONFIG, out)
        print("written", SUSP_CONFIG)
    if args.update_mount:
        c = json.load(open(CAM_CONFIG))
        shutil.copy(CAM_CONFIG, CAM_CONFIG + ".bak_" + time.strftime("%Y%m%d"))
        c["extrinsics"].update(height_mm=round(hr, 1), pitch_deg=round(pr, 3), roll_deg=round(rr, 3))
        c["extrinsics"]["mount_source"] = out["source"]
        write_json(CAM_CONFIG, c)
        print("written mount to", CAM_CONFIG)


def check(args):
    """Board pose(s) in a saved 1280x720 image of the 1536x864 crop (tests the pose math)."""
    import cv2
    from camera_cal import detect_boards
    img = cv2.imread(args.image)
    cam = CameraModel.load()
    size = (img.shape[1], img.shape[0])
    for c in detect_boards(img, (args.cols, args.rows)):
        p = pose_from_corners(c, cam, (args.cols, args.rows), args.square, CROP_1536, size)
        print("board at %.0f mm: h %.1f mm, pitch %.2f, roll %.2f, heading %.1f deg, rms %.2f px"
              % (p["dist"], p["h"], p["pitch"], p["roll"], p["heading"], p["rms_px"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "check"):
        s = sub.add_parser(name)
        s.add_argument("--cols", type=int, default=8)
        s.add_argument("--rows", type=int, default=6)
        s.add_argument("--square", type=float, default=30.0)
        if name == "capture":
            s.add_argument("--seconds", type=float, default=0.0, help="stop after this long (default: run until Ctrl-C)")
        else:
            s.add_argument("image")
    s = sub.add_parser("fit")
    s.add_argument("run")
    s.add_argument("--rest-s", type=float, default=5.0, help="rest pose: at most the first seconds, until the car moves")
    s.add_argument("--write", action="store_true", help="write config/suspension.json")
    s.add_argument("--update-mount", action="store_true", help="also write the rest pose into config/camera.json")
    a = ap.parse_args()
    {"capture": capture, "fit": fit, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
