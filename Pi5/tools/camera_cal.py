#!/usr/bin/env python3
"""Camera calibration from ONE view of several checkerboards lying flat on the floor.

Fits, in a single least-squares problem:
  * intrinsics in sensor pixels (one focal length, square pixels, principal point at the sensor
    centre unless --free-pp) + radial distortion k1, k2 (--k3, --tangential for more terms),
  * the camera mount: height above the floor, pitch, roll (camera at X = Y = 0, yaw 0 = forward),
  * each board's free position + rotation on the floor (boards may be placed anywhere),
  * optionally plumb lines: rough endpoints of edges that are straight in the room (door frames,
    furniture); their traced points must be straight after undistortion. They constrain the
    distortion where the boards do not reach (the upper half of the frame).
Writes config/camera.json, read by vision/camera_model.py.

Boards: 8x6 inner corners (the perimeter squares are cut by the printer margin and not used),
30 mm squares; set with --pattern / --square. Detection works on the 2x..4x vertically stretched
image (boards seen at a grazing angle are too flat for OpenCV's detector otherwise).

    # dashboard holds the camera: stop it, capture, restart
    pkill -f "[p]ython3 dashboard.py"; python3 tools/camera_cal.py capture; scripts/dashboard_restart.sh
    python3 tools/camera_cal.py fit results/camera_cal/<run> [--lines lines.json] [--write]

capture saves full.png (4608x2592, mean of 4 frames, whole sensor) plus one frame of each other raw
mode (bin2304 = 2304x1296 whole sensor, def1280 = the dashboard's 1280x720 from the 1536x864 crop
mode); fit calibrates on full.png and checks the result against the other frames (independent
detections, through each mode's ScalerCrop). lines.json: {"name": [[u0, v0], [u1, v1]], ...} in
full.png pixels, a few px accuracy is enough (the edge is searched +-24 px around the segment).
Needs python3-opencv and python3-scipy (apt).
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from vision.camera_model import CONFIG, SENSOR, CROP_FULL, CROP_1536, CameraModel  # noqa: E402

RUNS = os.path.join(ROOT, "results", "camera_cal")
FRAMES = {"full": ((4608, 2592), (4608, 2592), CROP_FULL, 4, True),     # main, raw, crop, n avg, still
          "bin2304": ((2304, 1296), (2304, 1296), CROP_FULL, 4, False),
          "def1280": ((1280, 720), (1536, 864), CROP_1536, 1, False)}


# ------------------------------------------------------------------------------------- capture
def capture(out_dir):
    from picamera2 import Picamera2
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    cam = Picamera2()
    info = {"camera": {k: str(v) for k, v in cam.camera_properties.items()}, "frames": {}}
    try:
        for name, (main, raw, crop, n, still) in FRAMES.items():
            mk = cam.create_still_configuration if still else cam.create_video_configuration
            cam.configure(mk(main={"size": main, "format": "RGB888"}, raw={"size": raw}))
            cam.start()
            time.sleep(2.0)                                  # AE / AWB settle
            acc, md = None, {}
            for _ in range(n):
                req = cam.capture_request()
                a, md = req.make_array("main").astype(np.float32), req.get_metadata()
                req.release()
                acc = a if acc is None else acc + a
            cam.stop()
            Image.fromarray((acc / n).round().astype(np.uint8)[:, :, ::-1]).save(os.path.join(out_dir, name + ".png"))
            info["frames"][name] = {"size": main, "raw": raw, "ScalerCrop": md.get("ScalerCrop"),
                                    "LensPosition": md.get("LensPosition"), "ExposureTime": md.get("ExposureTime")}
            print(name, info["frames"][name], flush=True)
    finally:
        cam.close()
    with open(os.path.join(out_dir, "capture.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("saved ->", out_dir)


# ----------------------------------------------------------------------------------- detection
def _find(cv2, gray, pattern):
    """One board in `gray` via the stretched image (vertical x2..x4, optionally x2 overall)."""
    for up in (2, 3, 4):
        for fx in (1, 2):
            if gray.size * up * fx * fx > 12e6:     # the detector's memory grows fast (Pi OOM)
                continue
            st = cv2.resize(gray, None, fx=fx, fy=up * fx, interpolation=cv2.INTER_CUBIC)
            ok, c = cv2.findChessboardCornersSB(st, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ok:
                return c.reshape(-1, 2) / [fx, up * fx]
    return None


def _orient(c, pattern):
    """Rows along the long side (pattern[0] per row), handedness fixed so that in the image the
    column direction x row direction is positive (col -> right, row -> nearer for an unrotated board)."""
    g = c.reshape(pattern[1], pattern[0], 2)
    a, b = g[0, -1] - g[0, 0], g[-1, 0] - g[0, 0]
    if a[0] * b[1] - a[1] * b[0] < 0:
        g = g[::-1]
    return g.reshape(-1, 2)


def detect_boards(img, pattern, max_boards=12, rois=None):
    """All boards in a BGR image -> list of (N, 2) corner arrays (image pixels)."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    boards = []
    if rois:
        cands = [(int(x0), int(y0), int(x1), int(y1)) for x0, y0, x1, y1 in rois]
    else:
        # coarse search on tiles of the half-size image; each hit is refined at full size below
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        sh, sw = small.shape
        work = small.copy()
        cands = []
        tw, th = sw // 2, sh // 3
        tiles = [(0, 0, sw, sh)] + [(x, y, x + tw, y + th) for y in range(0, sh - th + 1, th // 2)
                                    for x in range(0, sw - tw + 1, tw // 2)]
        for x0, y0, x1, y1 in tiles:
            while len(cands) < max_boards:
                c = _find(cv2, work[y0:y1, x0:x1], pattern)
                if c is None:
                    break
                c = c + [x0, y0]
                lo, hi = c.min(0), c.max(0)
                pad = (hi - lo) * 0.35 + 8
                cands.append(tuple(int(v) for v in np.r_[(lo - pad) * 2, (hi + pad) * 2]))
                hull = cv2.convexHull(c.astype(np.float32)).astype(np.int32)
                m = np.zeros_like(work)
                cv2.fillConvexPoly(m, hull, 255)
                work[cv2.dilate(m, np.ones((31, 31), np.uint8)) > 0] = 128
    for x0, y0, x1, y1 in cands:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        c = _find(cv2, gray[y0:y1, x0:x1], pattern)
        if c is not None:
            boards.append(_orient(c + [x0, y0], pattern))
    return boards


def trace_lines(img, lines, search=24, step=12):
    """Rough segments {name: [[u0, v0], [u1, v1]]} -> {name: (N, 2) sub-pixel edge points}."""
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    out = {}
    for name, (p0, p1) in lines.items():
        p0, p1 = np.array(p0, float), np.array(p1, float)
        t = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-t[1], t[0]])
        offs = np.arange(-search, search + 1)
        pts = []
        for s in np.arange(10, np.linalg.norm(p1 - p0) - 10, step):
            xy = (p0 + t * s + offs[:, None] * n).astype(np.float32)
            m = np.abs(cv2.remap(gx, xy[None, :, 0], xy[None, :, 1], cv2.INTER_LINEAR)[0] * n[0] +
                       cv2.remap(gy, xy[None, :, 0], xy[None, :, 1], cv2.INTER_LINEAR)[0] * n[1])
            i = int(np.argmax(m))
            if 0 < i < len(m) - 1 and m[i] > 15:
                den = m[i - 1] - 2 * m[i] + m[i + 1]
                pts.append(p0 + t * s + (offs[i] + (0.5 * (m[i - 1] - m[i + 1]) / den if den else 0)) * n)
        pts = np.array(pts)
        for _ in range(3):        # drop points that jumped to a neighbouring edge
            s, e = (pts - p0) @ t, (pts - p0) @ n
            r = e - np.polyval(np.polyfit(s, e, 2), s)
            pts = pts[np.abs(r) < max(4.5 * np.median(np.abs(r)), 0.8)]
        out[name] = pts
    return out


# ----------------------------------------------------------------------------------------- fit
class Problem:
    def __init__(self, boards, lines, pattern, square, didx, free_pp):
        self.boards, self.lines = [np.asarray(b, float) for b in boards], [np.asarray(l, float) for l in lines]
        r, c = np.mgrid[0:pattern[1], 0:pattern[0]]
        self.local = np.stack([-r.ravel() * square, -c.ravel() * square], 1)   # row -> -X, col -> -Y
        self.didx, self.free_pp = didx, free_pp

    def unpack(self, p):
        i = 1
        f = p[0]
        cx, cy = (p[1], p[2]) if self.free_pp else (SENSOR[0] / 2, SENSOR[1] / 2)
        i += 2 if self.free_pp else 0
        d = np.zeros(5)
        d[self.didx] = p[i:i + len(self.didx)]
        i += len(self.didx)
        h, pitch, roll = p[i:i + 3]
        return f, cx, cy, d, h, pitch, roll, p[i + 3:].reshape(-1, 3)

    def model(self, p):
        f, cx, cy, d, h, pitch, roll, _ = self.unpack(p)
        return CameraModel(f, f, cx, cy, d, h, np.degrees(pitch), np.degrees(roll))

    def board_world(self, b):
        bx, by, th = b
        ct, st = np.cos(th), np.sin(th)
        L = self.local
        return np.stack([bx + ct * L[:, 0] - st * L[:, 1], by + st * L[:, 0] + ct * L[:, 1], np.zeros(len(L))], 1)

    def residuals(self, p, wl=1.0):
        cam = self.model(p)
        bp = self.unpack(p)[-1]
        r = [(cam.project(self.board_world(b)) - c).ravel() for b, c in zip(bp, self.boards)]
        for L in self.lines:
            x, y = cam.undistort(L[:, 0], L[:, 1])
            q = np.stack([x, y], 1)
            q -= q.mean(0)
            r.append(wl * (q @ np.linalg.svd(q)[2][1]) * cam.fx)
        return np.concatenate(r)

    def init(self, f0, pitch0, h0):
        cam = CameraModel(f0, f0, SENSOR[0] / 2, SENSOR[1] / 2, [], h0, np.degrees(pitch0), 0)
        bp = []
        for c in self.boards:          # rigid 2D fit of the board to its corners back-projected on the floor
            g = cam.ground(c[:, 0], c[:, 1])
            if np.isnan(g).any():
                return None
            A, B = self.local - self.local.mean(0), g - g.mean(0)
            U, _, Vt = np.linalg.svd(A.T @ B)
            Rm = (U @ Vt).T
            if np.linalg.det(Rm) < 0:
                return None
            th = np.arctan2(Rm[1, 0], Rm[0, 0])
            bp += list(g.mean(0) - Rm @ self.local.mean(0)) + [th]
        return np.r_[f0, [SENSOR[0] / 2, SENSOR[1] / 2] if self.free_pp else [], np.zeros(len(self.didx)),
                     h0, pitch0, 0.0, bp]


def fit(boards, lines, pattern, square, didx, free_pp):
    from scipy.optimize import least_squares
    pb = Problem(boards, [], pattern, square, didx, free_pp)
    best = None
    for f0 in np.linspace(1200, 4000, 8):          # boards only: global start over f / pitch / height
        for pitch0 in np.radians([0, 10, 25]):
            for h0 in (80, 200, 500):
                p0 = pb.init(f0, pitch0, h0)
                if p0 is None:
                    continue
                res = least_squares(pb.residuals, p0, method="lm", max_nfev=3000)
                if best is None or res.cost < best.cost:
                    best = res
    pl = Problem(boards, lines, pattern, square, didx, free_pp)
    res = least_squares(pl.residuals, best.x, method="lm", max_nfev=20000) if lines else best
    nb = sum(b.size for b in pl.boards)
    s2 = (res.fun ** 2).sum() / max(len(res.fun) - len(res.x), 1)
    try:
        sd = np.sqrt(np.diag(np.linalg.inv(res.jac.T @ res.jac) * s2))
    except np.linalg.LinAlgError:
        sd = np.full(len(res.x), np.nan)
    return pl, res, res.fun[:nb].reshape(-1, 2), res.fun[nb:], sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", default=os.path.join(RUNS, time.strftime("%Y%m%d_%H%M%S")))
    f = sub.add_parser("fit")
    f.add_argument("run")
    f.add_argument("--lines", help="plumb lines json (default: <run>/lines.json if present)")
    f.add_argument("--pattern", default="8x6", help="inner corners per row x rows")
    f.add_argument("--square", type=float, default=30.0, help="square size, mm")
    f.add_argument("--k3", action="store_true")
    f.add_argument("--tangential", action="store_true")
    f.add_argument("--free-pp", action="store_true", help="fit the principal point (weak from one view)")
    f.add_argument("--rois", help="json list of [x0, y0, x1, y1] board regions in full.png, skips the search")
    f.add_argument("--write", action="store_true", help="write config/camera.json")
    a = ap.parse_args()
    if a.cmd == "capture":
        return capture(a.out)

    import cv2
    pattern = tuple(int(v) for v in a.pattern.lower().split("x"))
    img = cv2.imread(os.path.join(a.run, "full.png"))
    if img is None or img.shape[1] != SENSOR[0]:
        sys.exit("need %s/full.png at %dx%d" % (a.run, *SENSOR))
    rois = json.load(open(a.rois)) if a.rois else None
    boards = detect_boards(img, pattern, rois=rois)
    print("boards found: %d (%s)" % (len(boards), ", ".join("centre %.0f,%.0f" % tuple(b.mean(0)) for b in boards)))
    if len(boards) < 2:
        sys.exit("need at least 2 boards")
    lp = a.lines or os.path.join(a.run, "lines.json")
    seg = json.load(open(lp)) if os.path.exists(lp) else {}
    traced = trace_lines(img, seg) if seg else {}
    for k, v in traced.items():
        print("line %-18s %3d points" % (k, len(v)))
    didx = [0, 1] + ([4] if a.k3 else []) + ([2, 3] if a.tangential else [])
    pl, res, eb, el, sd = fit(boards, list(traced.values()), pattern, a.square, didx, a.free_pp)
    f_, cx, cy, d, h, pitch, roll, bp = pl.unpack(res.x)
    cam = pl.model(res.x)
    i0 = 1 + (2 if a.free_pp else 0) + len(didx)
    board_rms = float(np.sqrt((eb ** 2).sum(1).mean()))
    line_rms = float(np.sqrt((el ** 2).mean())) if len(el) else None
    print("fit: board rms %.2f px%s (full-res pixels)" % (board_rms, ", line rms %.2f px" % line_rms if line_rms else ""))
    print("  f %.1f +- %.1f px  pp (%.1f, %.1f)  dist %s" % (f_, sd[0], cx, cy, np.round(d, 4).tolist()))
    print("  camera height %.1f +- %.1f mm, pitch %.2f +- %.2f deg (down +), roll %.2f +- %.2f deg" % (
        h, sd[i0], np.degrees(pitch), np.degrees(sd[i0 + 1]), np.degrees(roll), np.degrees(sd[i0 + 2])))
    for i, b in enumerate(bp):
        print("  board %d at X %.0f Y %.0f mm, rotated %.1f deg, rms %.2f px" % (
            i, b[0], b[1], np.degrees(b[2]), np.sqrt((eb[i * 48:(i + 1) * 48] ** 2).sum(1).mean())))

    # independent check on the other captures (own detections, mapped through their ScalerCrop)
    capinfo = json.load(open(os.path.join(a.run, "capture.json"))) if os.path.exists(os.path.join(a.run, "capture.json")) else {}
    lens = (capinfo.get("frames", {}).get("full") or {}).get("LensPosition")
    checks = {}
    pred = [cam.project(pl.board_world(b)) for b in bp]
    for name, (size, _, crop, _, _) in FRAMES.items():
        p = os.path.join(a.run, name + ".png")
        if name == "full" or not os.path.exists(p):
            continue
        crop = tuple((capinfo.get("frames", {}).get(name) or {}).get("ScalerCrop") or crop)
        im = cv2.imread(p)
        sx = crop[2] / im.shape[1]
        errs = []
        for q in pred:
            roi = [(q[:, 0].min() - crop[0]) / sx - 60, (q[:, 1].min() - crop[1]) / sx - 40,
                   (q[:, 0].max() - crop[0]) / sx + 60, (q[:, 1].max() - crop[1]) / sx + 40]
            det = detect_boards(im, pattern, rois=[roi])
            if not det:
                continue
            det = det[0]
            u, v = CameraModel.to_sensor(det[:, 0], det[:, 1], crop, (im.shape[1], im.shape[0]))
            s = np.stack([u, v], 1)
            e = min(s - q, s[::-1] - q, key=lambda x: np.abs(x).max())   # start corner may differ by 180 deg
            if np.abs(e).max() < 40:
                errs.append(e / sx)                # in that frame's pixels
        if errs:
            e = np.concatenate(errs)
            checks[name] = {"boards": len(errs), "rms_px": round(float(np.sqrt((e ** 2).sum(1).mean())), 3),
                            "crop": list(crop), "size": [im.shape[1], im.shape[0]]}
            print("check %-8s %d boards, rms %.2f px of that frame" % (name, len(errs), checks[name]["rms_px"]))

    # how far out the data reach: beyond this radius the distortion is extrapolated
    pts = np.concatenate([np.concatenate(pl.boards)] + list(traced.values()))
    r_cov = float(np.max(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)))
    out = {
        "calibrated": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": os.path.relpath(a.run, ROOT),
        "camera": "IMX708 (wide lens), sensor %dx%d; LensPosition %s dioptre (manual focus, libcamera default)" % (*SENSOR, lens),
        "intrinsics": {"fx": round(f_, 3), "fy": round(f_, 3), "cx": round(cx, 3), "cy": round(cy, 3),
                       "dist": [round(float(x), 6) for x in d],
                       "units": "full-sensor pixels; per stream: CameraModel.K(ScalerCrop, size)",
                       "model": "OpenCV pinhole, dist = k1, k2, p1, p2, k3"},
        "extrinsics": {"height_mm": round(h, 1), "pitch_deg": round(float(np.degrees(pitch)), 3),
                       "roll_deg": round(float(np.degrees(roll)), 3),
                       "frame": "origin on the floor under the camera, X forward, Y left, Z up; pitch + = down, roll + = left side down"},
        "quality": {"board_rms_px": round(board_rms, 3), "line_rms_px": line_rms and round(line_rms, 3),
                    "boards": len(boards), "corners": int(sum(len(b) for b in boards)),
                    "lines": {k: len(v) for k, v in traced.items()},
                    "sd": {"f_px": round(float(sd[0]), 2), "height_mm": round(float(sd[i0]), 2),
                           "pitch_deg": round(float(np.degrees(sd[i0 + 1])), 3)},
                    "covered_radius_px": round(r_cov, 0),
                    "covered_note": "distortion is measured out to this radius from the centre; the 1536x864 "
                                    "crop modes (1280x720, 640x360) lie inside it, full-FOV corners are extrapolated",
                    "checks": checks},
        "target": {"pattern": list(pattern), "square_mm": a.square,
                   "boards": [[round(float(b[0]), 1), round(float(b[1]), 1), round(float(np.degrees(b[2])), 2)] for b in bp]},
    }
    with open(os.path.join(a.run, "result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    # visual check: the default-mode frame undistorted (straight edges must come out straight)
    p = os.path.join(a.run, "def1280.png")
    if os.path.exists(p):
        mx, my = cam.undistort_maps(CROP_1536, (1280, 720), f_scale=0.8)
        cv2.imwrite(os.path.join(a.run, "def1280_undistorted.jpg"), cv2.remap(cv2.imread(p), mx, my, cv2.INTER_LINEAR))
    if a.write:
        with open(CONFIG, "w") as fh:
            json.dump(out, fh, indent=1)
        print("written", CONFIG)
    else:
        print("result in %s/result.json (add --write to install it as %s)" % (a.run, CONFIG))


if __name__ == "__main__":
    main()
