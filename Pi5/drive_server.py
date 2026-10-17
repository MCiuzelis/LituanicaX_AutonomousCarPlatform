#!/usr/bin/env python3
"""Drive server: throttle, steering and the motor encoder in their own process, nothing else.

Started (and stopped) by dashboard.py; it can also run alone: `python3 drive_server.py`.
Keyboard commands must never wait for the camera, the Hailo, the recorder or the track map, and
in one Python process they all share one GIL (measured 2026-10-11: with the map in the dashboard
process the detector fell to 20-40 fps and commands stalled). Here the only Python running is the
command path, the dead-man and the encoder sampler; the network thread runs SCHED_FIFO.

Links (one JSON object per message / line; the freshest command wins, see Control):
  ws://<pi>:8082/ws/ctl   the dashboard page (from a Web Worker, 20 Hz + on every key change):
                          {"t": -1..1, "s": -1..1, "n": seq} manual, or {"arm": 1|0, "max": level, "n": seq}
                          (autonomous mode: SPACE held / released; max = the page's manual throttle level)
                          ->  ack {"a": seq, "st": stopped} at once (the page measures the round trip),
                          health {"h": 1, ...} at HEALTH_HZ
  tcp <pi>:8081           car_client.py on a PC: {"t","s"} lines in, health lines out (as before)
  unix datagram SOCK      the map worker: {"plan": {"s": steer, "v": mm/s, "m": mode}} at 20 Hz (autonomy);
                          the dashboard: {"health": {"v", "temp"}} (for the health lines), {"stop": true};
                          {"link": "auto", "t", "s"} raw commands (a human command in the last
                          HUMAN_PRIORITY_S wins over them)
State for the dashboard / recorder: shared memory SHM (float64 array, FIELDS; seqlock), read with
DriveState below.

Safety: throttle capped at MAX_THROTTLE (autonomy: at the page's manual level); no message on any
link for DEADMAN_S -> neutral + centre; autonomy brakes to a stop the moment it is not armed.
The process exits (neutral) when its parent (the dashboard) dies.
"""
import asyncio
import json
import logging
import math
import os
import signal
import socket
import sys
import threading
import time
from multiprocessing import shared_memory

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MAX_THROTTLE = 0.40          # fraction of full forward the server will ever apply
DEADMAN_S = 0.4              # stop if no command arrives on any link for this long
HUMAN_PRIORITY_S = 0.5       # an "auto" command is ignored while a human link sent one this recently
WS_PORT, TCP_PORT = 8082, 8081
HEALTH_HZ = 5
PUBLISH_HZ = 200             # encoder values -> shared memory
ENCODER_HZ = 500             # motor encoder sampling (25 rev/s max = 18 deg per sample; free-running was ~1.5 kHz
                             # and a quarter of a core)
RT_PRIORITY = 20             # SCHED_FIFO for the network + watchdog threads (BNO085 worker: higher)
SHM = "rc_drive_state"
SOCK = "/tmp/rc_drive.sock"
LINKS = ("ws", "tcp", "auto")
HUMAN = ("ws", "tcp")
FIELDS = ("seq", "t", "turns", "rps", "steer_pos", "esc_us", "throttle", "steer", "stopped", "active",
          "last_ws", "last_tcp", "last_auto", "deadman", "commands", "pid", "max_throttle",
          "gap_ws_ms", "gap_tcp_ms", "late_ws", "late_tcp", "rt", "ws_clients", "started",
          "armed", "arm_max", "arms", "auto_state", "auto_v", "plan_t", "plan_steer", "plan_v", "plan_mode",
          "auto_u", "auto_vmeas")
IDX = {k: i for i, k in enumerate(FIELDS)}
LATE_MS = 150                # a gap between commands of one link above this counts as late
# autonomy (Control.auto_loop)
AUTO_HZ = 50
ARM_S = 0.25                 # one arm message (page: 20 Hz while SPACE is held) keeps autonomy armed this long
AUTO_STALE_S = 0.3           # planner silent this long -> brake to a stop
AUTO_OFF, AUTO_DRIVING, AUTO_BRAKING = 0, 1, 2
# speed control (2026-10-11 evening; before: bang-bang ceiling / coast / brake, 10-24 switches a second):
# PI on the throttle as a fraction u of the page's level, target rate-limited, measured speed filtered
KP_PER_MM = 0.0006           # u per mm/s of error (500 mm/s short -> +0.3)
KI_PER_MM = 0.0012           # u per mm/s of error per second (500 mm/s short -> u +0.6 per s)
U_START = 0.6                # u when starting from standstill / after a brake
SPEED_FILTER = 0.25          # low-pass of the motor speed per AUTO_HZ step (~70 ms)
TARGET_UP_MM_S2 = 1500.0     # the target may rise this fast ...
TARGET_DOWN_MM_S2 = 3000.0   # ... and fall this fast (a 0 = "stop" from the planner still brakes at once)
BRAKE_ABOVE_MM_S = 250       # this far above target: brake (strength grows with the excess)
BRAKE_GAIN_MM_S = 800        # brake fraction = excess / this, clipped to AUTO_BRAKE_MIN..MAX
AUTO_BRAKE_MIN = 0.25        # (the strongest: DRIVE_TUNABLES brake_max)
AUTO_STOP_BRAKE = 0.6        # brake while stopping (released / lost / planner says stop) ...
STOP_MM_S = 120              # ... until the motor says the car is slower than this, then neutral
# live-tunable speed control (dashboard "autonomy tuning", saved in config/drive.json): key -> (default, min,
# max, step, label, help). 2026-10-16 defaults are more aggressive than the constants above (the car crawled
# through corners: the target rose at 1.5 m/s^2 and the PI barely pushed): accelerate faster, brake earlier
DRIVE_CONFIG = os.path.join(ROOT, "config", "drive.json")
DRIVE_TUNABLES = {
    "accel_mm_s2": (3000.0, 500.0, 8000.0, 100.0, "accel mm/s\u00b2",
                    "how fast the speed target may rise (was 1500): higher = gets going / out of corners faster"),
    "decel_mm_s2": (4000.0, 1000.0, 10000.0, 100.0, "target drop mm/s\u00b2", "how fast the speed target may fall"),
    "kp": (0.0010, 0.0, 0.004, 0.0001, "throttle P", "throttle (fraction of the ceiling) per mm/s too slow (was 0.0006)"),
    "ki": (0.0020, 0.0, 0.008, 0.0001, "throttle I", "throttle build-up per mm/s too slow per second (was 0.0012)"),
    "u_start": (0.8, 0.2, 1.0, 0.05, "start throttle", "fraction of the ceiling when starting / after braking (was 0.6)"),
    "brake_above_mm_s": (150.0, 50.0, 600.0, 10.0, "brake above mm/s",
                         "this much faster than the target: brake instead of coasting (was 250)"),
    "brake_gain_mm_s": (500.0, 150.0, 2000.0, 50.0, "brake softness mm/s",
                        "brake = excess speed / this (smaller = harder, was 800)"),
    "brake_max": (1.0, 0.3, 1.0, 0.05, "max brake",
                  "strongest brake the speed control uses (was 0.8; 2026-10-16: brake 0.4-0.8 slowed only ~1.2 m/s^2)"),
}


def load_drive_tuning(path=DRIVE_CONFIG):
    """DRIVE_TUNABLES defaults overridden by config/drive.json (clipped to the ranges)."""
    t = {k: v[0] for k, v in DRIVE_TUNABLES.items()}
    try:
        with open(path) as f:
            over = json.load(f)
    except (OSError, ValueError):
        over = {}
    for k, v in over.items():
        if k in DRIVE_TUNABLES:
            try:
                t[k] = max(DRIVE_TUNABLES[k][1], min(DRIVE_TUNABLES[k][2], float(v)))
            except (TypeError, ValueError):
                pass
    return t


PLAN_MODES = ("", "gates", "center", "left", "right", "lost", "none", "row", "memory", "crawl")   # append only: the index is shared

log = logging.getLogger("drive")


# ---------------------------------------------------------------- shared state
class DriveState:
    """Reader side (dashboard): consistent snapshot of the drive server's state."""

    def __init__(self, create=False):
        n = len(FIELDS) * 8
        if create:
            try:
                old = shared_memory.SharedMemory(SHM)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(SHM, create=True, size=n)
        else:
            self.shm = shared_memory.SharedMemory(SHM)
            try:                       # Python < 3.13 registers attached segments too: do not unlink ours
                from multiprocessing import resource_tracker
                resource_tracker.unregister(self.shm._name, "shared_memory")
            except Exception:  # noqa: BLE001
                pass
        self.a = np.ndarray((len(FIELDS),), np.float64, self.shm.buf)
        if create:
            self.a[:] = 0.0

    def read(self):
        """dict of FIELDS (retries while the writer is mid-update)."""
        for _ in range(100):
            s0 = self.a[0]
            v = self.a.copy()
            if s0 == self.a[0] and int(s0) % 2 == 0:
                return dict(zip(FIELDS, v.tolist()))
            time.sleep(0.0001)
        return dict(zip(FIELDS, v.tolist()))

    def close(self):
        self.a = None
        self.shm.close()


# ---------------------------------------------------------------- control arbiter
class Control:
    """Applies the freshest manual command; runs autonomy while armed; a watchdog stops the car when
    every link goes quiet.

    Autonomy: the page arms it while SPACE is held (ARM messages at 20 Hz, each valid for ARM_S, with
    the page's manual throttle level as the throttle ceiling). The planner (vision/map_worker.py)
    sends steering + a target speed (PLAN messages) at 20 Hz; auto_loop turns the target speed into
    throttle / brake against the motor encoder at AUTO_HZ. Not armed any more (space released, link
    lost), planner quiet for AUTO_STALE_S, or the planner asking for 0 -> brake to a stop, then neutral.
    Any manual command (W/A/S/D, car_client) takes over at once and disarms."""

    def __init__(self, drive, state):
        self.drive, self.st = drive, state
        self.lock = threading.Lock()
        self.last = {k: 0.0 for k in LINKS}        # any message that keeps a link alive (dead-man)
        self.last_manual = {k: 0.0 for k in HUMAN}
        self.gap_max = {k: 0.0 for k in HUMAN}     # longest gap between messages, last window
        self.gap_win = {k: [] for k in HUMAN}
        self.late = {k: 0 for k in HUMAN}
        self.throttle = self.steer = 0.0
        self.active = None
        self.stopped = True
        self.deadman = self.commands = 0
        self.health = {}
        self.mm_per_turn = _mm_per_turn()
        self.arm_by = {}                           # (link, page) -> (valid until, throttle ceiling)
        self.arm_max, self.arms = 0.0, 0
        self.u = 0.0                               # speed controller: throttle as a fraction of the ceiling
        self.vt_f = 0.0                            # rate-limited target speed
        self.v_f = 0.0                             # filtered measured speed
        self.plan = None                           # (time, steer, target speed mm/s, mode)
        self.auto_state, self.auto_v = AUTO_OFF, 0.0
        self.tune = load_drive_tuning()
        self._run = True

    # ---- inputs
    def _touch(self, link, now):
        """Caller holds the lock: link alive + gap statistics."""
        if link in HUMAN and self.last[link]:
            gap = (now - self.last[link]) * 1000
            if gap < 2000:                          # not a reconnect after a pause
                w = self.gap_win[link]
                w.append((now, gap))
                while w and now - w[0][0] > 10.0:
                    w.pop(0)
                self.gap_max[link] = max(g for _, g in w)
                if gap > LATE_MS:
                    self.late[link] += 1
        self.last[link] = now

    def command(self, link, throttle, steer):
        """Manual command (or a raw "auto" t / s from the local socket)."""
        now = time.monotonic()
        t = max(-1.0, min(MAX_THROTTLE, float(throttle)))
        s = max(-1.0, min(1.0, float(steer)))
        if not (math.isfinite(t) and math.isfinite(s)):
            return False
        with self.lock:
            if link == "auto" and any(now - self.last_manual[h] < HUMAN_PRIORITY_S for h in HUMAN):
                return False
            self._touch(link, now)
            if link in HUMAN and (t != 0.0 or s != 0.0):
                # a key is pressed: the human takes over at once. A neutral command (t = s = 0: driving
                # enabled, no key, e.g. a second dashboard tab) only keeps the link alive.
                self.last_manual[link] = now
                if self.auto_state != AUTO_OFF or self.arm_by:
                    log.info("manual %s command: autonomy off", link)
                self.arm_by.clear()
                self.auto_state = AUTO_OFF
            elif link in HUMAN and self._auto_active(now):
                return True                       # neutral keep-alive while autonomy drives: nothing to apply
            self.commands += 1
            self._apply(link, t, s)
        return True

    def arm(self, link, on, max_level, client=None):
        """Page: SPACE held in autonomous mode (on) / released (off). Keeps the link alive either way.
        Kept per page (client): a second dashboard tab sending "not armed" cannot cancel the page whose
        SPACE is held (2026-10-11: two open tabs made autonomy drop 10-20 times a second)."""
        now = time.monotonic()
        key = (link, client)
        with self.lock:
            self._touch(link, now)
            if on:
                until, _ = self.arm_by.get(key, (0.0, 0.0))
                if now > until:
                    self.arms += 1
                    log.info("autonomy armed by %s%s (throttle ceiling %.0f %%)", link,
                             "" if client is None else " #%s" % client, 100 * float(max_level))
                self.arm_by[key] = (now + ARM_S, max(0.0, min(MAX_THROTTLE, float(max_level))))
                self.last_manual[link] = 0.0      # keys released + SPACE held: autonomy takes over now
            else:
                self.arm_by.pop(key, None)
            if not on and self.auto_state == AUTO_OFF and not self.stopped and self.active in HUMAN:
                # "not armed" = no key held in autonomous mode: it must stop a manual command like a
                # neutral one would (2026-10-11: W released in autonomous mode kept the W throttle on
                # until S, because this message kept the link alive but applied nothing)
                self._apply(link, 0.0, 0.0)

    def _armed(self, now):
        """Caller holds the lock -> (armed, throttle ceiling)."""
        fresh = [m for until, m in self.arm_by.values() if now < until]
        return (bool(fresh), max(fresh) if fresh else 0.0)

    def _auto_active(self, now):
        return self.auto_state != AUTO_OFF and self._armed(now)[0]

    def set_tuning(self, cfg):
        """Dashboard: {key: value} of DRIVE_TUNABLES (clipped), applied at the next auto_loop step."""
        for k, v in (cfg or {}).items():
            if k in DRIVE_TUNABLES:
                try:
                    self.tune[k] = max(DRIVE_TUNABLES[k][1], min(DRIVE_TUNABLES[k][2], float(v)))
                except (TypeError, ValueError):
                    pass
        log.info("speed control: %s", ", ".join("%s %g" % kv for kv in sorted(self.tune.items())))

    def set_plan(self, steer, v, mode):
        """Planner (map worker): steering -1..1 and target speed mm/s (0 = stop)."""
        s, v = max(-1.0, min(1.0, float(steer))), max(0.0, float(v))
        if math.isfinite(s) and math.isfinite(v):
            now = time.monotonic()
            self.plan = (now, s, v, str(mode)[:12])
            # while autonomy drives, the new steering goes out now, not at the next auto_loop step (<= 20 ms)
            with self.lock:
                if (self.auto_state == AUTO_DRIVING and v > 0 and self._armed(now)[0]
                        and not any(now - self.last_manual[h] < HUMAN_PRIORITY_S for h in HUMAN)):
                    self._apply("auto", self.throttle, s)

    # ---- outputs
    def _apply(self, link, t, s):
        """Caller holds the lock."""
        self.throttle, self.steer, self.active, self.stopped = t, s, link, False
        if t >= 0:
            self.drive.forward(t)
        else:
            self.drive.brake(-t)
        self.drive.steer(s)
        self._publish()

    def stop(self, why):
        with self.lock:
            self._neutral(why)

    def _neutral(self, why):
        """Caller holds the lock."""
        if not self.stopped:
            log.warning("%s -> stop", why)
        self.drive.stop()
        self.drive.steer(0)
        self.throttle = self.steer = 0.0
        self.active, self.stopped = None, True
        self._publish()

    def watchdog(self):
        _set_rt("watchdog")
        while self._run:
            time.sleep(0.02)
            now = time.monotonic()
            if (not self.stopped and self.auto_state == AUTO_OFF
                    and all(now - v >= DEADMAN_S for v in self.last.values())):
                with self.lock:
                    self.deadman += 1
                self.stop("dead-man: no command for %.0f ms on any link" % (DEADMAN_S * 1000))

    def auto_loop(self):
        """Autonomy at AUTO_HZ: target speed -> throttle (capped at the page's manual level) / brake."""
        _set_rt("auto")
        while self._run:
            time.sleep(1.0 / AUTO_HZ)
            with self.lock:
                now = time.monotonic()
                if any(now - self.last_manual[h] < HUMAN_PRIORITY_S for h in HUMAN):
                    continue                        # manual driving: autonomy stays out of it
                armed, self.arm_max = self._armed(now)
                p = self.plan
                tn = self.tune
                fresh = p is not None and now - p[0] < AUTO_STALE_S
                v = abs(self.drive.rps) * self.mm_per_turn
                self.v_f += (v - self.v_f) * SPEED_FILTER
                if armed and fresh and p[2] > 0:
                    s = p[1]
                    if self.auto_state != AUTO_DRIVING:
                        log.info("autonomy driving (%s)", p[3])
                        self.vt_f = max(self.v_f, 0.0)
                        self.u = tn["u_start"] if self.v_f < STOP_MM_S else self.u
                    # target speed changes smoothly (the planner's confidence makes it jump)
                    dv = p[2] - self.vt_f
                    self.vt_f += max(-tn["decel_mm_s2"] / AUTO_HZ, min(tn["accel_mm_s2"] / AUTO_HZ, dv))
                    vt = self.vt_f
                    err = vt - self.v_f
                    if err < -tn["brake_above_mm_s"]:      # much too fast: brake in proportion, then resume low
                        t = -min(tn["brake_max"], max(AUTO_BRAKE_MIN, -err / tn["brake_gain_mm_s"]))
                        self.u = min(self.u, tn["u_start"])
                    else:                                  # PI on the throttle, 0..1 of the page's level
                        # no integrating up while P alone already asks for the full level (anti-windup: the I
                        # term built up during a long acceleration overshot the target, then braked)
                        if err < 0 or self.u + tn["kp"] * err < 1.0:
                            self.u = max(0.0, min(1.0, self.u + tn["ki"] * err / AUTO_HZ))
                        t = self.arm_max * max(0.0, min(1.0, self.u + tn["kp"] * err))
                    self.auto_state, self.auto_v = AUTO_DRIVING, vt
                    self._apply("auto", t, s)
                elif self.auto_state != AUTO_OFF or (armed and not self.stopped):
                    # space released / arming lost / planner quiet or asking for 0: brake to a stop
                    s = p[1] if fresh else self.steer
                    if v > STOP_MM_S:
                        if self.auto_state != AUTO_BRAKING:
                            log.info("autonomy: %s -> braking", "planner stop" if armed and fresh else
                                     "planner quiet" if armed else "released")
                        self.auto_state, self.auto_v = AUTO_BRAKING, 0.0
                        self._apply("auto", -AUTO_STOP_BRAKE, s)
                    else:
                        self.auto_state, self.auto_v = AUTO_OFF, 0.0
                        self._neutral("autonomy stopped")

    def publisher(self):
        while self._run:
            with self.lock:
                self._publish()
            time.sleep(1.0 / PUBLISH_HZ)

    def _publish(self):
        """Caller holds self.lock."""
        a, d = self.st.a, self.drive
        a[0] += 1                                   # odd: writing
        now = time.monotonic()
        a[IDX["t"]] = now
        a[IDX["turns"]], a[IDX["rps"]] = d.turns, d.rps
        a[IDX["steer_pos"]] = d.steering.position
        a[IDX["esc_us"]] = d.esc.current_us or 0.0
        a[IDX["throttle"]], a[IDX["steer"]] = self.throttle, self.steer
        a[IDX["stopped"]] = 1.0 if self.stopped else 0.0
        a[IDX["active"]] = LINKS.index(self.active) + 1 if self.active else 0
        for k in LINKS:
            a[IDX["last_" + k]] = self.last[k]
        for k in HUMAN:
            a[IDX["gap_%s_ms" % k]] = self.gap_max[k]
            a[IDX["late_" + k]] = self.late[k]
        a[IDX["deadman"]], a[IDX["commands"]] = self.deadman, self.commands
        a[IDX["armed"]] = 1.0 if self._armed(now)[0] else 0.0
        a[IDX["arm_max"]], a[IDX["arms"]] = self.arm_max, self.arms
        a[IDX["auto_state"]], a[IDX["auto_v"]] = self.auto_state, self.auto_v
        a[IDX["auto_u"]], a[IDX["auto_vmeas"]] = self.u, self.v_f
        p = self.plan
        a[IDX["plan_t"]] = p[0] if p else 0.0
        a[IDX["plan_steer"]], a[IDX["plan_v"]] = (p[1], p[2]) if p else (0.0, 0.0)
        a[IDX["plan_mode"]] = PLAN_MODES.index(p[3]) if p and p[3] in PLAN_MODES else 0
        a[0] += 1

    def health_line(self):
        now = time.monotonic()
        h = {"h": 1, "v": self.health.get("v"), "temp": self.health.get("temp"), "rps": round(self.drive.rps, 2),
             "us": self.drive.esc.current_us, "st": self.drive.steering.position, "act": self.active,
             "stop": self.stopped, "lk": {k: int((now - v) * 1000) for k, v in self.last.items() if v},
             "auto": self.auto_state, "armed": self._armed(now)[0]}
        return json.dumps(h, separators=(",", ":"))


def _mm_per_turn():
    try:
        with open(os.path.join(ROOT, "config", "odometry.json")) as f:
            return float(json.load(f)["mm_per_turn"])
    except (OSError, ValueError, KeyError):
        return 55.0


def _set_rt(name):
    """SCHED_FIFO for the calling thread (the gpio group may use rtprio, /etc/security/limits.d)."""
    try:
        os.sched_setscheduler(threading.get_native_id(), os.SCHED_FIFO, os.sched_param(RT_PRIORITY))
        return True
    except (OSError, AttributeError) as e:
        log.warning("%s: no realtime priority (%s)", name, e)
        return False


# ---------------------------------------------------------------- links
def serve_tcp_conn(conn, ctl):
    conn.settimeout(0.2)
    buf, next_health = b"", time.monotonic()
    try:
        while True:
            try:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        try:
                            m = json.loads(line)
                            ctl.command("tcp", m.get("t", 0.0), m.get("s", 0.0))
                        except (ValueError, TypeError, AttributeError):
                            pass
            except socket.timeout:
                pass
            if time.monotonic() >= next_health:
                next_health += 1.0 / HEALTH_HZ
                conn.sendall((ctl.health_line() + "\n").encode())
    except OSError as e:
        log.info("tcp link closed: %s", e)
    finally:
        conn.close()


def tcp_server(ctl):
    _set_rt("tcp")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", TCP_PORT))
    srv.listen(2)
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log.info("tcp client %s", addr)
        threading.Thread(target=serve_tcp_conn, args=(conn, ctl), daemon=True).start()


def local_server(ctl):
    """Datagrams from the dashboard (same machine)."""
    _set_rt("local")
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.bind(SOCK)
    while True:
        data = s.recv(4096)
        try:
            m = json.loads(data)
            if "plan" in m:
                p = m["plan"]
                ctl.set_plan(p.get("s", 0.0), p.get("v", 0.0), p.get("m", ""))
            elif "health" in m:
                ctl.health = m["health"]
            elif "cfg" in m:
                ctl.set_tuning(m["cfg"])
            elif m.get("stop"):
                ctl.stop("dashboard asked")
            else:
                ctl.command(m.get("link", "auto"), m.get("t", 0.0), m.get("s", 0.0))
        except (ValueError, TypeError, AttributeError):
            pass


async def ws_ctl(request):
    from aiohttp import web
    ctl, state = request.app["ctl"], request.app["state"]
    ws = web.WebSocketResponse(heartbeat=2.0, max_msg_size=4096)
    await ws.prepare(request)                       # (aiohttp's server sets TCP_NODELAY itself)
    state.a[IDX["ws_clients"]] += 1
    cid = id(ws) % 10000                            # one arming slot per page
    log.info("ws client %s", request.remote)

    async def health():
        while not ws.closed:
            await asyncio.sleep(1.0 / HEALTH_HZ)
            await ws.send_str(ctl.health_line())
    task = asyncio.ensure_future(health())
    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                m = json.loads(msg.data)
                if "arm" in m:
                    ctl.arm("ws", bool(m["arm"]), m.get("max", 0.0), client=cid)
                else:
                    ctl.command("ws", m.get("t", 0.0), m.get("s", 0.0))
                if "n" in m:
                    await ws.send_str('{"a":%d,"st":%d}' % (int(m["n"]), ctl.stopped))
            except (ValueError, TypeError, AttributeError):
                pass
    finally:
        task.cancel()
        with ctl.lock:
            ctl.arm_by.pop(("ws", cid), None)      # page gone: its SPACE is not held any more
        state.a[IDX["ws_clients"]] -= 1
    return ws


async def serve_ws(ctl, state):
    from aiohttp import web
    app = web.Application()
    app.update(ctl=ctl, state=state)
    app.add_routes([web.get("/ws/ctl", ws_ctl)])
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, port=WS_PORT).start()
    log.info("ws://0.0.0.0:%d/ws/ctl, tcp :%d, %s", WS_PORT, TCP_PORT, SOCK)
    await asyncio.Event().wait()


# ---------------------------------------------------------------- main
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s drive: %(message)s")
    sys.setswitchinterval(0.0005)            # the network thread gets the GIL back fast from the encoder thread
    from hardware import Drive
    state = DriveState(create=True)
    drive = Drive(max_throttle=MAX_THROTTLE, sample_hz=ENCODER_HZ)
    ctl = Control(drive, state)
    a = state.a
    a[IDX["pid"]], a[IDX["max_throttle"]], a[IDX["started"]] = os.getpid(), MAX_THROTTLE, time.time()
    ppid = os.getppid()
    done = threading.Event()

    def shutdown(*_):
        done.set()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    for target in (ctl.watchdog, ctl.auto_loop, ctl.publisher, lambda: tcp_server(ctl), lambda: local_server(ctl)):
        threading.Thread(target=target, daemon=True).start()

    def net():
        a[IDX["rt"]] = 1.0 if _set_rt("ws") else 0.0
        asyncio.run(serve_ws(ctl, state))
    threading.Thread(target=net, daemon=True).start()
    standalone = "--standalone" in sys.argv or ppid == 1
    try:
        while not done.wait(0.25):
            if not standalone and os.getppid() != ppid:
                log.warning("dashboard gone -> stopping")
                break
    finally:
        ctl._run = False
        ctl.stop("drive server exiting")
        drive.close()
        try:
            os.unlink(SOCK)
        except OSError:
            pass
        state.close()
        state.shm.unlink()


if __name__ == "__main__":
    main()
