"""Shareable videos of a take: the recorded video as is, and the same with the live detections drawn in.

    cd Pi5 && uv run --project ../YoloTraining python tools/analysis/overlay_video.py <take dir or mp4> [...]
        [--out DIR] [--hud] [--crf 20]

<take dir> holds <id>.mp4 + <id>_frames.csv + <id>_det.csv (e.g. YoloTraining/data/videos_20261016/<id>).
Writes <out>/<id>_clean.mp4 (the recording, remuxed with faststart so it streams / previews everywhere) and
<out>/<id>_boxes.mp4 (H.264, same resolution and frame rate). --plan instead writes <id>_plan.mp4: boxes with
their scores, the path the planner proposed (violet, from _planpath.jsonl: the live plans, 10 Hz, moved to each
frame's pose and projected with the frame's camera attitude), the apples it used as the left / right boundary (blue
/ orange dots) and a HUD. Boxes come from _det.csv (what the Hailo
saw live, every camera frame), joined on the video frame's sensor timestamp (nearest detection within
DET_TOL_CAM_FRAMES: the detector skips the odd camera frame). --hud adds speed / steering / autonomy state from
_frames.csv + _drive.csv in a corner. Needs opencv and ffmpeg on PATH.
"""
import argparse
import bisect
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np

DET_TOL_CAM_FRAMES = 2.5            # a detection this many camera frames from the video frame still counts (2304x1296
                                    # @ 56 fps: the detector ran at only 37-41 fps)
BOX_BGR = (60, 220, 60)
LABEL_MIN_H = 0.06                  # boxes at least this tall (fraction of the frame) get a score label
PATH_BGR, LEFT_BGR, RIGHT_BGR = (255, 80, 200), (255, 160, 40), (40, 150, 255)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Pi5/
AUTO_NAMES = {0: "manual", 1: "AUTO", 2: "braking"}


def take_paths(arg):
    if arg.endswith(".mp4"):
        d, tid = os.path.dirname(arg), os.path.basename(arg)[:-4]
    else:
        d = arg.rstrip("/\\")
        tid = os.path.basename(d)
    return d, tid, lambda suf: os.path.join(d, tid + suf)


def load_dets(path):
    """sensor_ts_ns -> [(x1, y1, x2, y2, score)] (normalised to the frame)."""
    dets = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ts = int(r["sensor_ts_ns"])
            lst = dets.setdefault(ts, [])
            if r["x1"]:
                lst.append((float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["score"])))
    return dets


def load_drive(path):
    """-> (t_ns array, rows) of _drive.csv, or None."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["t_ns"]) for r in rows]), rows


def draw_boxes(img, boxes, all_scores=False):
    H, W = img.shape[:2]
    import cv2
    th = max(2, W // 640)
    for x1, y1, x2, y2, s in boxes:
        big = y2 - y1 >= LABEL_MIN_H
        p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
        cv2.rectangle(img, p1, p2, BOX_BGR, th if big else 1)
        if not big and not all_scores:            # far apples: a box only (labels would cover the row)
            continue
        label = "apple %.2f" % s if big else "%.2f" % s
        fs = 0.4 * th if big else 0.32
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y = p1[1] - 3 if p1[1] - tht - 4 > 0 else p2[1] + tht + 3
        cv2.rectangle(img, (p1[0], y - tht - 3), (p1[0] + tw + 4, y + 3), BOX_BGR, -1)
        cv2.putText(img, label, (p1[0] + 2, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(img, fr, drv):
    import cv2
    H, W = img.shape[:2]
    lines = []
    rps = float(fr.get("motor_rps") or 0.0)
    if drv is not None:
        ts, rows = drv
        i = max(0, bisect.bisect_right(ts, int(fr["t_mid_ns"])) - 1)
        r = rows[i]
        st = int(float(r.get("auto_state") or 0))
        v = float(r.get("auto_v_meas") or 0.0) if st else abs(rps) * 55.0
        lines.append("%s  %.1f m/s" % (AUTO_NAMES.get(st, "?"), v / 1000.0))
        lines.append("throttle %+.2f  steer %+.2f" % (float(r["throttle_cmd"] or 0), float(r["steer"] or 0)))
        if st and r.get("plan_mode"):
            lines.append("plan %s  target %.1f m/s" % (r["plan_mode"], float(r.get("auto_v_target") or 0) / 1000.0))
    fs = W / 1536 * 0.8
    y = int(30 * fs / 0.8)
    for ln in lines:
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(32 * fs / 0.8)


class PlanOverlay:
    """The live plans of a take drawn into its frames: _planpath.jsonl (car frame at the plan's odometry time) ->
    map frame -> car frame of the video frame -> image (camera pitch / roll / height at the frame, the take's
    lens). Poses and attitude: _pose.csv, interpolated."""

    def __init__(self, p, crop):
        sys.path.insert(0, ROOT)
        from vision.camera_model import CameraModel
        from vision.track_map import cam_to_floor
        self.cam, self.cam_to_floor, self.crop = CameraModel.load(), cam_to_floor, crop
        self.plans = []
        with open(p("_planpath.jsonl")) as f:
            for ln in f:
                try:
                    self.plans.append(json.loads(ln))
                except ValueError:
                    pass
        self.pt = [q["t_ns"] for q in self.plans]
        with open(p("_pose.csv")) as f:
            rows = list(csv.DictReader(f))
        self.ps_t = np.array([int(r["t_ns"]) for r in rows], float)
        self.ps = np.array([[float(r["x_mm"]), float(r["y_mm"]), math.radians(float(r["yaw_deg"])),
                             float(r["cam_pitch_deg"]), float(r["cam_roll_deg"]), float(r["cam_h_mm"])] for r in rows])
        if len(self.ps):
            self.ps[:, 2] = np.unwrap(self.ps[:, 2])

    def pose_at(self, t_ns):
        """-> (x, y, yaw rad, cam pitch, cam roll, cam height) at t_ns, None outside the log."""
        if not len(self.ps_t) or t_ns < self.ps_t[0] - 1e8 or t_ns > self.ps_t[-1] + 1e8:
            return None
        return tuple(float(np.interp(t_ns, self.ps_t, self.ps[:, k])) for k in range(6))

    def project(self, P, att):
        """Car-frame floor points (N, 2) -> normalised frame coords (N, 2), NaN behind the camera."""
        R = self.cam_to_floor(att[3], att[4])
        c = np.c_[P, np.full(len(P), -att[5])] @ R
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(c[:, 2] > 50.0, c[:, 0] / c[:, 2], np.nan)
            y = np.where(c[:, 2] > 50.0, c[:, 1] / c[:, 2], np.nan)
        u, v = self.cam.distort(x, y)
        nx, ny = self.cam.from_sensor(u, v, self.crop)
        return np.stack([nx, ny], -1)

    def draw(self, img, t_mid_ns):
        """-> the plan drawn (dict) or None."""
        import cv2
        i = bisect.bisect_right(self.pt, t_mid_ns) - 1
        if i < 0 or t_mid_ns - self.pt[i] > 3e8:
            return None
        q = self.plans[i]
        pp, att = self.pose_at(q["t_ns"]), self.pose_at(t_mid_ns)
        if pp is None or att is None:
            return q
        cp, sp = math.cos(pp[2]), math.sin(pp[2])
        cf, sf = math.cos(att[2]), math.sin(att[2])

        def to_frame(pts):
            P = np.asarray(pts, float).reshape(-1, 2)
            if not len(P):
                return P
            Wd = P @ np.array([[cp, sp], [-sp, cp]]) + [pp[0], pp[1]]             # plan's car frame -> map
            return (Wd - [att[0], att[1]]) @ np.array([[cf, -sf], [sf, cf]])     # map -> this frame's car frame
        H, W = img.shape[:2]
        th = max(2, W // 400)
        path = to_frame(q.get("path") or [])
        if len(path) >= 2:
            # densify so the projected curve bends with the lens; drop what is behind / beside the camera
            seg = np.hypot(*np.diff(path, axis=0).T)
            cum = np.r_[0.0, np.cumsum(seg)]
            u = np.arange(0.0, cum[-1], 30.0)
            path = np.stack([np.interp(u, cum, path[:, 0]), np.interp(u, cum, path[:, 1])], 1)
            path = path[path[:, 0] > 120.0]
            if len(path) >= 2:
                uv = self.project(path, att)
                ok = ~np.isnan(uv).any(1) & (np.abs(uv - 0.5) < 1.5).all(1)
                pts = (uv[ok] * [W, H]).astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(img, [pts], False, (0, 0, 0), th + 4, cv2.LINE_AA)
                    cv2.polylines(img, [pts], False, PATH_BGR, th, cv2.LINE_AA)
        for key, col in (("left", LEFT_BGR), ("right", RIGHT_BGR)):
            A = to_frame(q.get(key) or [])
            if len(A):
                A = A[A[:, 0] > 120.0]
            if len(A):
                for x, y in self.project(A, att):
                    if np.isfinite(x) and np.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1:
                        cv2.circle(img, (int(x * W), int(y * H)), th + 3, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(img, (int(x * W), int(y * H)), th + 1, col, -1, cv2.LINE_AA)
        return q


def draw_plan_hud(img, q, drv, t_mid_ns):
    """Mode / speeds / steering bar top left, legend bottom left."""
    import cv2
    H, W = img.shape[:2]
    st, v, vt, steer = 0, 0.0, 0.0, 0.0
    if drv is not None:
        ts, rows = drv
        r = rows[max(0, bisect.bisect_right(ts, t_mid_ns) - 1)]
        st = int(float(r.get("auto_state") or 0))
        v = float(r.get("auto_v_meas") or 0.0)
        vt = float(r.get("auto_v_target") or 0.0)
        steer = float(r.get("steer") or 0.0)
    lines = ["%s   %.2f m/s" % ({1: "AUTONOMOUS", 2: "braking"}.get(st, "stopped / manual"), v / 1000.0)]
    if st == 1:
        lines[0] += "  (target %.2f)" % (vt / 1000.0)
    if q is not None:
        lines.append("plan: %s   steer %+.2f" % (q.get("mode", "?"), steer))
    fs = W / 1536 * 0.75
    lh = int(30 * fs / 0.75)
    bw = int(W * 0.36)
    cv2.rectangle(img, (0, 0), (bw, lh * (len(lines) + 1) + 6), (0, 0, 0), -1)
    y = lh
    for ln in lines:
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        y += lh
    x0, x1, yb = 12, bw - 12, y - lh // 3                     # steering bar: left of centre = steering left
    xm = (x0 + x1) // 2
    cv2.line(img, (x0, yb), (x1, yb), (90, 90, 90), 4)
    cv2.line(img, (xm, yb), (int(xm + steer * (x1 - x0) / 2), yb), PATH_BGR, 6)
    cv2.line(img, (xm, yb - 7), (xm, yb + 7), (255, 255, 255), 1)
    items = (("planned path", PATH_BGR), ("left boundary", LEFT_BGR), ("right boundary", RIGHT_BGR),
             ("detection + score", BOX_BGR))
    lg = int(24 * fs / 0.75)
    cv2.rectangle(img, (0, H - lg * len(items) - 10), (int(230 * fs / 0.75), H), (0, 0, 0), -1)
    for k, (txt, col) in enumerate(items):
        yy = H - 10 - (len(items) - 1 - k) * lg
        cv2.circle(img, (18, yy - 5), 6, col, -1)
        cv2.putText(img, txt, (32, yy), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 255, 255), 1, cv2.LINE_AA)


def render(arg, out_dir, hud, crf, plan=False):
    import cv2
    d, tid, p = take_paths(arg)
    os.makedirs(out_dir, exist_ok=True)
    clean = os.path.join(out_dir, tid + "_clean.mp4")
    boxed = os.path.join(out_dir, tid + ("_plan.mp4" if plan else "_boxes.mp4"))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", p(".mp4"), "-c", "copy", "-movflags", "+faststart", clean],
                   check=True)
    with open(p("_frames.csv")) as f:
        frames = list(csv.DictReader(f))
    with open(p("_meta.json")) as f:
        stream = json.load(f)["stream"]
    cam_fps = float(stream.get("camera_fps") or 120.0)
    po = PlanOverlay(p, tuple(stream.get("scaler_crop") or (768, 432, 3072, 1728))) if plan else None
    tol_ns = DET_TOL_CAM_FRAMES * 1e9 / cam_fps
    dets = load_dets(p("_det.csv"))
    dts = sorted(dets)
    drv = load_drive(p("_drive.csv")) if hud or plan else None
    cap = cv2.VideoCapture(p(".mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (W, H),
                            "-r", "%g" % fps, "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", boxed], stdin=subprocess.PIPE)
    n = hit = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n < len(frames):
            fr = frames[n]
            ts = int(fr["sensor_ts_ns"])
            j = bisect.bisect_left(dts, ts)
            best = min((k for k in (j - 1, j) if 0 <= k < len(dts)), key=lambda k: abs(dts[k] - ts), default=None)
            near = best is not None and abs(dts[best] - ts) <= tol_ns
            q = None
            if po is not None:
                q = po.draw(img, int(fr["t_mid_ns"]))
            if near:
                draw_boxes(img, dets[dts[best]], all_scores=plan)
                hit += 1
            if po is not None:
                draw_plan_hud(img, q, drv, int(fr["t_mid_ns"]))
            elif hud:
                draw_hud(img, fr, drv)
        enc.stdin.write(img.tobytes())
        n += 1
    cap.release()
    enc.stdin.close()
    enc.wait()
    print("%s: %d frames, %d with detections -> %s, %s" % (tid, n, hit, clean, boxed))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--out", default=None, help="output folder (default: <take dir>/../share)")
    ap.add_argument("--hud", action="store_true", help="speed / steering / autonomy text in the corner")
    ap.add_argument("--plan", action="store_true", help="<id>_plan.mp4: boxes + scores + the planned path + HUD")
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better, bigger)")
    a = ap.parse_args()
    for t in a.takes:
        d = take_paths(t)[0]
        render(t, a.out or os.path.join(os.path.dirname(os.path.abspath(d)), "share"), a.hud, a.crf, a.plan)


if __name__ == "__main__":
    sys.exit(main())
