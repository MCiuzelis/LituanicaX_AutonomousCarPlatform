#!/usr/bin/env python3
"""Drive server: throttle, steering and the motor encoder in their own process, nothing else.

Started (and stopped) by dashboard.py; it can also run alone: `python3 drive_server.py`.
Keyboard commands must never wait for the camera, the Hailo, the recorder or the track map, and
in one Python process they all share one GIL (measured 2026-10-11: with the map in the dashboard
process the detector fell to 20-40 fps and commands stalled). Here the only Python running is the
command path, the dead-man and the encoder sampler; the network thread runs SCHED_FIFO.

Links (one JSON object per message / line; the freshest command wins, see Control):
  ws://<pi>:8082/ws/ctl   the dashboard page (from a Web Worker, 20 Hz + on every key change):
                          {"t": -1..1, "s": -1..1, "n": seq}  ->  ack {"a": seq, "st": stopped} at once
                          (the page measures the round trip), health {"h": 1, ...} at HEALTH_HZ
  tcp <pi>:8081           car_client.py on a PC: {"t","s"} lines in, health lines out (as before)
  unix datagram SOCK      the dashboard: {"link": "auto", "t", "s"} (autonomy; any human command in
                          the last HUMAN_PRIORITY_S wins over it), {"health": {"v", "temp"}} (for the
                          health lines), {"stop": true}
State for the dashboard / recorder: shared memory SHM (float64 array, FIELDS; seqlock), read with
DriveState below.

Safety: throttle capped at MAX_THROTTLE; no command on any link for DEADMAN_S -> neutral + centre.
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
          "gap_ws_ms", "gap_tcp_ms", "late_ws", "late_tcp", "rt", "ws_clients", "started")
IDX = {k: i for i, k in enumerate(FIELDS)}
LATE_MS = 150                # a gap between commands of one link above this counts as late

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
    """Applies the freshest command; human links beat "auto"; a watchdog stops the car when all go quiet."""

    def __init__(self, drive, state):
        self.drive, self.st = drive, state
        self.lock = threading.Lock()
        self.last = {k: 0.0 for k in LINKS}
        self.gap_max = {k: 0.0 for k in HUMAN}     # longest gap between commands, last window
        self.gap_win = {k: [] for k in HUMAN}
        self.late = {k: 0 for k in HUMAN}
        self.throttle = self.steer = 0.0
        self.active = None
        self.stopped = True
        self.deadman = self.commands = 0
        self.health = {}
        self._run = True

    def command(self, link, throttle, steer):
        now = time.monotonic()
        t = max(-1.0, min(MAX_THROTTLE, float(throttle)))
        s = max(-1.0, min(1.0, float(steer)))
        if not (math.isfinite(t) and math.isfinite(s)):
            return False
        with self.lock:
            if link == "auto" and any(now - self.last[h] < HUMAN_PRIORITY_S for h in HUMAN):
                return False
            if link in HUMAN and self.last[link]:
                gap = (now - self.last[link]) * 1000
                if gap < 2000:                      # not a reconnect after a pause
                    w = self.gap_win[link]
                    w.append((now, gap))
                    while w and now - w[0][0] > 10.0:
                        w.pop(0)
                    self.gap_max[link] = max(g for _, g in w)
                    if gap > LATE_MS:
                        self.late[link] += 1
            self.last[link] = now
            self.throttle, self.steer, self.active, self.stopped = t, s, link, False
            self.commands += 1
            if t >= 0:
                self.drive.forward(t)
            else:
                self.drive.brake(-t)
            self.drive.steer(s)
            self._publish()
        return True

    def stop(self, why):
        with self.lock:
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
            if not self.stopped and all(now - v >= DEADMAN_S for v in self.last.values()):
                with self.lock:
                    self.deadman += 1
                self.stop("dead-man: no command for %.0f ms on any link" % (DEADMAN_S * 1000))

    def publisher(self):
        while self._run:
            with self.lock:
                self._publish()
            time.sleep(1.0 / PUBLISH_HZ)

    def _publish(self):
        """Caller holds self.lock."""
        a, d = self.st.a, self.drive
        a[0] += 1                                   # odd: writing
        a[IDX["t"]] = time.monotonic()
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
        a[0] += 1

    def health_line(self):
        now = time.monotonic()
        h = {"h": 1, "v": self.health.get("v"), "temp": self.health.get("temp"), "rps": round(self.drive.rps, 2),
             "us": self.drive.esc.current_us, "st": self.drive.steering.position, "act": self.active,
             "stop": self.stopped, "lk": {k: int((now - v) * 1000) for k, v in self.last.items() if v}}
        return json.dumps(h, separators=(",", ":"))


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
            if "health" in m:
                ctl.health = m["health"]
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
                ctl.command("ws", m.get("t", 0.0), m.get("s", 0.0))
                if "n" in m:
                    await ws.send_str('{"a":%d,"st":%d}' % (int(m["n"]), ctl.stopped))
            except (ValueError, TypeError, AttributeError):
                pass
    finally:
        task.cancel()
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
    for target in (ctl.watchdog, ctl.publisher, lambda: tcp_server(ctl), lambda: local_server(ctl)):
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
