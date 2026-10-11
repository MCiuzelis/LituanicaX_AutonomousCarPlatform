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
  ("snap", json_text, status)  at TICK_HZ: TrackMapper.snapshot() + the planner's suggestion ("plan",
                               shadow mode: shown, not driven) and the worker's own status
  ("stopped", result)
"""
import json
import math
import os
import sys
import time
from collections import deque
from multiprocessing.connection import Connection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)

from hardware.imu_frame import CarFrame                               # noqa: E402
from vision.planner import Planner                                    # noqa: E402
from vision.track_map import MapLog, TrackMapper                      # noqa: E402

TICK_HZ = 10


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
            "mm_per_turn": m.mm_per_turn}}

    def tick(self):
        m = self.mapper
        b = m.tick()
        snap = m.snapshot()
        if snap.get("up") and b is not None:
            t = m.odo.last_t
            o, prev = m.odo_pose(t), m.odo_pose(t - 0.2)
            v = math.hypot(o[0] - prev[0], o[1] - prev[1]) / 0.2 if o and prev else 0.0
            p = self.planner.update(b, v, t, apples=m.local_apples)
            pose = m.pose(t)
            c, s = math.cos(pose[2]), math.sin(pose[2])
            w = lambda pl: [[round(pose[0] + c * x - s * y), round(pose[1] + s * x + c * y)] for x, y in pl]  # noqa: E731
            snap["plan"] = {"mode": p["mode"], "steer": None if p.get("steer") is None else round(p["steer"], 3),
                            "speed_mm_s": round(p.get("speed_mm_s") or 0), "known_mm": round(p["known_mm"]),
                            "path": w(p["path"]) if p.get("path") is not None else [],
                            "left": w(p["left"]) if p.get("left") is not None and p["mode"] == "gates" else [],
                            "right": w(p["right"]) if p.get("right") is not None and p["mode"] == "gates" else [],
                            "speed_mm_s_now": round(v)}
        st, sm = self.step_ms, self.tick_ms
        status = {"up": bool(snap.get("up")), "error": self.error, "stats": snap.get("stats"),
                  "apples": sum(1 for x in snap.get("lms", []) if x[3] == 1), "width_mm": snap.get("width"),
                  "pitch_bias": snap.get("pitch_bias"), "sigma_xy": snap.get("sigma_xy"), "cam": snap.get("cam"),
                  "mm_per_turn": m.mm_per_turn, "odo_source": m.odo_cfg.get("source"),
                  "susp_source": m.susp.get("source"), "attitude": m.p.attitude, "logging": m.log is not None,
                  "window_s": m.p.window_s, "plan": {k: v for k, v in (snap.get("plan") or {}).items()
                                                     if k in ("mode", "steer", "speed_mm_s", "known_mm")},
                  "loop_ms": {"p50": _pct(list(st), 0.5), "max": _pct(list(st), 1.0)},
                  "tick_ms": _pct(list(sm), 0.5), "backlog": self.backlog, "pid": os.getpid()}
        self.conn.send(("snap", json.dumps(snap, separators=(",", ":")), status))

    def run(self):
        next_tick = time.monotonic()
        while True:
            try:
                while self.conn.poll(0.004):
                    if not self.handle(self.conn.recv()):
                        return
                t0 = time.monotonic()
                n = self.mapper.step()
                if n:
                    self.step_ms.append((time.monotonic() - t0) * 1000 / n)
                if t0 >= next_tick:
                    next_tick = max(next_tick + 1.0 / TICK_HZ, t0)
                    a = time.monotonic()
                    self.tick()
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
