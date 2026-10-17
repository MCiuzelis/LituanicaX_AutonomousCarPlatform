#!/usr/bin/env python3
"""Real-time load test of the map worker (vision/map_worker.py) on the car: starts the real worker process
and feeds it a recorded take at its real pace, the way the dashboard's MapProcess does (10 ms batches of
game_rv quaternions, encoder turns and detection frames, each arriving when it did live), then reports
how the worker kept up: plan interval, input age (now - odometry time) and plans that would have been
dropped as stale (> map_worker.STALE_INPUT_S).

    python3 tools/worker_rt_replay.py recordings/<id> [--from 0] [--to 30] [--burn 0]

--burn N runs N busy-loop processes alongside (the camera / Hailo / recorder load of the dashboard).
Plans go to a dummy socket, never to the drive server. Run it with the dashboard stopped.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Connection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from vision.track_map import take_files                               # noqa: E402

DUMMY_SOCK = "/tmp/rc_drive_replay.sock"


def burn():
    while True:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=1e9)
    ap.add_argument("--burn", type=int, default=0)
    a = ap.parse_args()
    f = take_files(a.take)
    events = []                                                        # (t, kind, payload)
    for r in csv.DictReader(open(f["imu"])):
        if r["sensor"] == "game_rv":
            events.append((int(r["t_ns"]) * 1e-9, 0, (float(r["v0"]), float(r["v1"]), float(r["v2"]), float(r["v3"]))))
    for r in csv.DictReader(open(f["drive"])):
        events.append((int(r["t_ns"]) * 1e-9, 1, float(r["odo_turns"])))
    dets = {}
    for r in csv.DictReader(open(f["det"])):
        k = int(r["sensor_ts_ns"])
        d = dets.setdefault(k, [int(r["t_mid_ns"]) * 1e-9, [], tuple(int(r[c]) for c in ("crop_x", "crop_y", "crop_w", "crop_h")),
                                int(r["seq"]), k])
        if int(r["box"]) >= 0:
            d[1].append(tuple(float(r[c]) for c in ("x1", "y1", "x2", "y2", "score")) + (int(r["cls"] or 0),))
    for d in dets.values():
        events.append((d[0] + 0.025, 2, tuple(d)))                     # boxes arrive ~25 ms after the exposure
    events.sort(key=lambda e: (e[0], e[1]))
    t_rec0 = events[0][0]
    events = [e for e in events if a.t_from <= e[0] - t_rec0 <= a.t_to]
    t_rec0 = events[0][0]

    # the worker, as dashboard.MapProcess starts it, with plans going to a dummy socket
    a_sock, b_sock = socket.socketpair()
    conn = Connection(a_sock.detach())
    fd = b_sock.detach()
    env = dict(os.environ, RC_DRIVE_SOCK=DUMMY_SOCK)
    proc = subprocess.Popen([sys.executable, "-m", "vision.map_worker", str(fd)], cwd=ROOT, pass_fds=(fd,), env=env)
    os.close(fd)
    burners = [mp.Process(target=burn, daemon=True) for _ in range(a.burn)]
    for b in burners:
        b.start()
    try:
        os.unlink(DUMMY_SOCK)
    except OSError:
        pass
    rx = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    rx.bind(DUMMY_SOCK)
    rx.settimeout(0.2)
    plans = []

    def receiver():
        while True:
            try:
                data = rx.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            plans.append((time.monotonic(), json.loads(data)))
    threading.Thread(target=receiver, daemon=True).start()
    snaps = []

    def reader():
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return
            if msg[0] == "snap":
                snaps.append((time.monotonic(), msg[2]))
    threading.Thread(target=reader, daemon=True).start()

    time.sleep(1.0)                                                    # worker start-up
    shift = time.monotonic() + 0.05 - t_rec0                           # recording time -> now
    i = 0
    t_end = events[-1][0] + shift
    while time.monotonic() < t_end:
        now = time.monotonic()
        imu, turns, dd = [], [], []
        while i < len(events) and events[i][0] + shift <= now:
            t, kind, x = events[i]
            if kind == 0:
                imu.append((t + shift, *x))
            elif kind == 1:
                turns.append((t + shift, x))
            else:
                dd.append((x[0] + shift, x[1], x[2], x[3], x[4]))
            i += 1
        conn.send(("data", imu, turns[-1:], dd))
        time.sleep(max(0.0, 0.01 - (time.monotonic() - now)))
    time.sleep(0.3)
    conn.send(("quit",))
    proc.wait(5)
    rx.close()
    for b in burners:
        b.terminate()
    # report: plans that reached the (dummy) drive server and the worker's own status
    tt = [t for t, _ in plans]
    gaps = sorted((b - a_) * 1000 for a_, b in zip(tt[:-1], tt[1:]))
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")   # noqa: E731
    dur = events[-1][0] - events[0][0]
    print("%s %.0f-%.0f s (%.1f s), burn %d: %d plans sent (%.1f / s)" % (f["id"][:15], a.t_from, min(a.t_to, dur + a.t_from), dur,
                                                                         a.burn, len(plans), len(plans) / max(dur, 1e-9)))
    ages = sorted(m["plan"].get("a", 0) for _, m in plans)
    print("  input age of sent plans ms: p50 %.0f p90 %.0f p99 %.0f max %.0f" % (q(ages, .5), q(ages, .9), q(ages, .99),
                                                                              ages[-1] if ages else float("nan")))
    print("  interval between sent plans ms: p50 %.0f p90 %.0f p99 %.0f max %.0f; gaps > 300 ms (drive server brakes): %d" % (
        q(gaps, .5), q(gaps, .9), q(gaps, .99), gaps[-1] if gaps else float("nan"), sum(1 for g in gaps if g > 300)))
    if snaps:
        st = snaps[-1][1]
        print("  worker: loop ms/frame p50 %s max %s, tick ms %s, frames skipped %s, backlog %s" % (
            st["loop_ms"]["p50"], st["loop_ms"]["max"], st["tick_ms"], (st.get("stats") or {}).get("skipped"), st["backlog"]))


if __name__ == "__main__":
    main()
