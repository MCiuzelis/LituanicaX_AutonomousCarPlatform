"""Track map + planner in their own process (the dashboard starts it: dashboard.MapProcess).

The mapper costs ~2.5-5 ms of Python per detection frame on the Pi 5 (120 fps -> 30-60 % of a core);
inside the dashboard process that time held the GIL the camera / Hailo / recorder / web threads
need (measured 2026-10-11: detection fell from 120 to 20-40 fps as the map grew). Here it has its
own interpreter and core.

    python3 -m vision.map_worker <fd>      (fd: one end of a socketpair; multiprocessing Connection)

parent -> worker (pickled tuples)
  ("data", imu, turns, dets)   imu [(t, qi, qj, qk, qr)] game_rv in the board frame, turns [(t, turns)],
                               dets [(t_mid, boxes, crop, seq, sensor_ts)]
  ("frame",)                   config/imu.json changed ("set level"): reload the car frame
  ("reset",) ("start_log", dir, id) ("stop_log",) -> ("stopped", {"files", "meta"} | None) ("quit",)
worker -> parent
  ("snap", json_text, status)  at SNAP_HZ: TrackMapper.snapshot() + the planner's output ("plan") and the
                               worker's own status
worker -> drive server (unix datagram, drive_server.SOCK): {"plan": {"s", "v", "m"}} at PLAN_HZ. The drive
server only acts on it while autonomy is armed (SPACE held on the page in autonomous mode).
  ("stopped", result)
"""
import json
import math
import os
import socket
import sys
import time
from collections import deque

import numpy as np
from multiprocessing.connection import Connection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)

from hardware.imu_frame import CarFrame                               # noqa: E402
from vision.planner import Planner, plan_with_map                     # noqa: E402
from vision.track_map import MapLog, TrackMapper                      # noqa: E402

PLAN_HZ = 50                 # map tick + planner; each plan goes straight to the drive server (autonomy). 50 (was 20):
                             # ~15 ms less delay on average; map + plans cost ~0.5 core at 2.4 GHz (tools/analysis/map_bench.py)
SNAP_HZ = 10                 # snapshot for the page
STALE_INPUT_S = 0.15         # odometry older than this: no plan is sent
MAX_QUEUE_S = 0.08           # detection frames this far behind the newest are skipped (load shedding)
DRIVE_SOCK = os.environ.get("RC_DRIVE_SOCK", "/tmp/rc_drive.sock")   # drive_server.SOCK (tools/worker_rt_replay.py: a dummy)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(len(s) * p))], 2)


class Worker:
    def __init__(self, conn):
        self.conn = conn
        self.frame = CarFrame.load()
        self.mapper = TrackMapper()
        self.planner = Planner()
        self.error = None
        self.step_ms, self.tick_ms = deque(maxlen=300), deque(maxlen=50)
        self.backlog = 0
        self.n_ticks = self.plans_sent = 0
        self.last_plan = None
        self._last_planpath = -1.0
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def handle(self, msg):
        kind = msg[0]
        m = self.mapper
        if kind == "data":
            _, imu, turns, dets = msg
            for t, *q in imu:
                m.add_imu(t, *self.frame.attitude(q))
            for t, tr in turns:
                m.add_turns(t, tr)
            for d in dets:
                m.add_detections(*d)
            self.backlog = len(m._dets)
        elif kind == "frame":
            self.frame = CarFrame.load()
        elif kind == "planner":              # dashboard: live PlannerParams changes (steer gain)
            for k, v in msg[1].items():
                setattr(self.planner.p, k, v)
        elif kind == "reset":
            m.reset()
        elif kind == "start_log":
            if m.log is None:
                m.reset()
                m.log = MapLog(msg[1], msg[2])
        elif kind == "stop_log":
            self.conn.send(("stopped", self.stop_log()))
        elif kind == "quit":
            return False
        return True

    def stop_log(self):
        m = self.mapper
        lg, m.log = m.log, None
        if lg is None:
            return None
        final = m.final_map()
        files = lg.close(final)
        return {"files": files, "meta": {
            "what": "track map (vision/track_map.py) run live during the take (rolling window: window_s / "
                    "window_behind_mm); reset at the take's start, world frame = camera floor point at reset, "
                    "X = car heading, Y left, mm",
            "params": final["params"], "odometry": final["odometry"], "suspension": final["suspension"],
            "stats": final["stats"], "log_rows": lg.rows, "log_dropped": lg.dropped,
            "replay": "tools/replay_map.py run recordings/<id> re-maps the take from its raw box columns; "
                      "tools/autonomy_eval.py replays it through the planner",
            "det_join": "_det.csv rows join _frames.csv on sensor_ts_ns (detection runs on every camera "
                        "frame, the video keeps every frame_skip-th)",
            "mm_per_turn": m.mm_per_turn,
            # the autonomy settings in effect at the end of the take (dashboard tuning sliders)
            "planner_params": {k: v for k, v in vars(self.planner.p).items()},
            "drive_tuning": _read_json(os.path.join(ROOT, "config", "drive.json"))}}

    def tick(self, snapshot):
        """PLAN_HZ: map tick + planner, the plan straight to the drive server; every PLAN_HZ / SNAP_HZ-th time also the
        snapshot for the page."""
        m = self.mapper
        b = m.tick()
        t = m.odo.last_t
        p, v = None, 0.0
        if b is not None and t is not None:
            o, prev = m.odo_pose(t), m.odo_pose(t - 0.2)
            v = math.hypot(o[0] - prev[0], o[1] - prev[1]) / 0.2 if o and prev else 0.0
            age = time.monotonic() - t
            p = plan_with_map(self.planner, m, v, t, age_s=age)
            # what the page draws as the boundaries = what the planner follows (car frame)
            m.display_bounds = {"left": p["left"].tolist(), "right": p["right"].tolist()}
            # only a plan made on current data goes to the car (IMU / encoder feed stalled -> none: the
            # drive server brakes to a stop after its AUTO_STALE_S)
            sent = age < STALE_INPUT_S
            if sent:
                self._send_plan(p, age)
            if m.log is not None:
                st = p.get("steer")
                tr = p.get("tracking") or {}
                m.log.put("plan", [[int(t * 1e9), p["mode"], p.get("why", ""), "" if st is None else "%.3f" % st,
                                    "%.0f" % (p.get("speed_mm_s") or 0), "%.0f" % p["known_mm"],
                                    p.get("confidence", ""), len(p["left_idx"]), len(p["right_idx"]),
                                    len(m.local_ids), sum(1 for lm in m.lms if lm.state == 0), "%.0f" % v,
                                    "%.0f" % (age * 1000), int(sent), "%.1f" % math.degrees(p.get("yaw_rate", 0.0)),
                                    "" if p.get("curvature") is None else "%.3f" % p["curvature"],
                                    tr.get("e_mm", ""), tr.get("psi_deg", ""), tr.get("k_ff", ""),
                                    m.stats.get("skipped", 0),
                                    (p.get("limits") or {}).get("curve", ""), (p.get("limits") or {}).get("known", ""),
                                    (p.get("limits") or {}).get("conf", ""), p.get("n_virtual", 0),
                                    p.get("virtual_used", 0),
                                    "" if p.get("target") is None else "%.0f" % p["target"][0],
                                    "" if p.get("target") is None else "%.0f" % p["target"][1],
                                    "%.0f" % (p.get("lookahead_mm") or 0)]])
                if t - self._last_planpath >= 0.1:
                    self._last_planpath = t
                    rd = lambda a: [[round(float(x)), round(float(y))] for x, y in np.asarray(a).reshape(-1, 2)]  # noqa: E731
                    m.log.put("planpath", {
                        "t_ns": int(t * 1e9), "mode": p["mode"], "steer": None if st is None else round(st, 3),
                        "v": round(p.get("speed_mm_s") or 0), "why": p.get("why", ""),
                        "path": rd(p["path"][::3]) if p.get("path") is not None else [],
                        "left": rd(p["left"]), "right": rd(p["right"]),
                        "target": None if p.get("target") is None else [round(p["target"][0]), round(p["target"][1])],
                        "tracking": p.get("tracking"), "limits": p.get("limits")})
        self.last_plan = p
        if not snapshot:
            return
        snap = m.snapshot()
        if snap.get("up") and p is not None:
            pose = m.pose(t)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            w = lambda pl: [[round(pose[0] + c * x - s * y), round(pose[1] + s * x + c * y)] for x, y in pl]  # noqa: E731
            snap["plan"] = {"mode": p["mode"], "steer": None if p.get("steer") is None else round(p["steer"], 3),
                            "speed_mm_s": round(p.get("speed_mm_s") or 0), "known_mm": round(p["known_mm"]),
                            "confidence": p.get("confidence"), "limits": p.get("limits"), "why": p.get("why", ""),
                            "path": w(p["path"]) if p.get("path") is not None else [],
                            "left": w(p["left"]), "right": w(p["right"]),
                            "speed_mm_s_now": round(v)}
            # the boundaries drawn on the map are the planner's sides (ordered along its path), not the
            # map's own straight-on chaining (which joins the inner row to the outer one in corners)
            snap["left"], snap["right"], snap["center"] = snap["plan"]["left"], snap["plan"]["right"], []
        st, sm = self.step_ms, self.tick_ms
        status = {"up": bool(snap.get("up")), "error": self.error, "stats": snap.get("stats"),
                  "apples": sum(1 for x in snap.get("lms", []) if x[3] == 1), "width_mm": snap.get("width"),
                  "pitch_bias": snap.get("pitch_bias"), "sigma_xy": snap.get("sigma_xy"), "cam": snap.get("cam"),
                  "mm_per_turn": m.mm_per_turn, "odo_source": m.odo_cfg.get("source"),
                  "susp_source": m.susp.get("source"), "attitude": m.p.attitude, "logging": m.log is not None,
                  "window_s": m.p.window_s, "plan": {k: v for k, v in (snap.get("plan") or {}).items()
                                                     if k in ("mode", "steer", "speed_mm_s", "known_mm", "confidence")},
                  "loop_ms": {"p50": _pct(list(st), 0.5), "max": _pct(list(st), 1.0)},
                  "tick_ms": _pct(list(sm), 0.5), "backlog": self.backlog, "pid": os.getpid(),
                  "plans_sent": self.plans_sent}
        self.conn.send(("snap", json.dumps(snap, separators=(",", ":")), status))

    def _send_plan(self, p, age):
        steer = p.get("steer")
        msg = {"plan": {"s": round(steer or 0.0, 4), "v": round(p.get("speed_mm_s") or 0.0),
                        "m": p["mode"] if steer is not None else "lost", "a": round(age * 1000)}}
        try:
            self.sock.sendto(json.dumps(msg).encode(), DRIVE_SOCK)
            self.plans_sent += 1
        except OSError:
            pass                                # drive server restarting: it brakes on its own

    def run(self):
        next_tick = time.monotonic()
        while True:
            try:
                wait = 0.0 if self.mapper._dets else 0.004
                while self.conn.poll(wait):
                    if not self.handle(self.conn.recv()):
                        return
                    wait = 0.0
                # map frames only until the next plan is due (the plan goes first), skipping frames that
                # fell more than MAX_QUEUE_S behind: a late plan is worth less than a few skipped frames
                t0 = time.monotonic()
                n = self.mapper.step(budget_s=max(0.002, next_tick - t0), max_queue_s=MAX_QUEUE_S)
                if n:
                    self.step_ms.append((time.monotonic() - t0) * 1000 / n)
                t0 = time.monotonic()
                if t0 >= next_tick:
                    next_tick = max(next_tick + 1.0 / PLAN_HZ, t0)
                    self.n_ticks += 1
                    a = time.monotonic()
                    self.tick(snapshot=self.n_ticks % (PLAN_HZ // SNAP_HZ) == 0)
                    self.tick_ms.append((time.monotonic() - a) * 1000)
                self.error = None
            except (EOFError, OSError):
                return                          # parent gone
            except Exception as e:  # noqa: BLE001 - keep mapping, report it
                if self.error is None:
                    import traceback
                    traceback.print_exc()
                self.error = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "")
                time.sleep(0.01)


def main():
    conn = Connection(int(sys.argv[1]))
    try:
        os.nice(2)                               # below the dashboard (camera / Hailo), above nothing
    except OSError:
        pass
    w = Worker(conn)
    try:
        w.run()
    finally:
        if w.mapper.log is not None:
            w.stop_log()


if __name__ == "__main__":
    main()
