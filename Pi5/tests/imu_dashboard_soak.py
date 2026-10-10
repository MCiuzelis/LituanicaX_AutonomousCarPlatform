#!/usr/bin/env python3
"""Soak test of the IMU as the dashboard serves it (dashboard must be running).

    python3 tests/imu_dashboard_soak.py [--minutes 15] [--vision] [--host localhost:8080]

Acts like a browser: /ws/imu, /ws/det, /ws and the MJPEG stream all connected, optionally with
Hailo detection on for load. Fails on any of:
  * a /ws/imu disconnect, or a gap > 100 ms between its messages (25 Hz expected)
  * an attitude sample older than 50 ms in a message (IMU data stale)
  * the dashboard's own 100 Hz game_rv dropout counter (gap > 50 ms) going up
  * IMU hub resets, worker restarts or lost (sequence-gap) samples going up
Writes results/imu_dashboard_soak_<ts>.json.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

import aiohttp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSG_GAP_S, MAX_AGE_MS = 0.1, 50.0


async def messages(ws, t0, dur):
    """Text frames until the socket closes or `dur` is over; a timeout (no message for 0.5 s)
    yields None so a silent socket can neither hang the test nor hide a stall."""
    while time.monotonic() - t0 < dur:
        try:
            m = await ws.receive(timeout=0.5)
        except asyncio.TimeoutError:
            yield None
            continue
        if m.type != aiohttp.WSMsgType.TEXT:
            return
        yield m


async def watch_imu(s, url, dur, r):
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        r["conns"] += 1
        async with s.ws_connect(url) as ws:
            last = time.monotonic()
            async for m in messages(ws, t0, dur):
                now = time.monotonic()
                if m is None:
                    continue                     # the gap is recorded when the next message comes
                if now - last > MSG_GAP_S:
                    r["msg_gaps"].append([round(now - t0, 1), round((now - last) * 1000)])
                last = now
                r["msgs"] += 1
                age = json.loads(m.data).get("age_ms", 0)
                r["max_age_ms"] = max(r["max_age_ms"], age)
                if age > MAX_AGE_MS:
                    r["stale"].append([round(now - t0, 1), age])
            if time.monotonic() - last > MSG_GAP_S and time.monotonic() - t0 < dur:
                r["msg_gaps"].append([round(time.monotonic() - t0, 1), round((time.monotonic() - last) * 1000)])
        if time.monotonic() - t0 < dur:
            r["closes"].append([round(time.monotonic() - t0, 1), ws.close_code])


async def watch_other(s, url, dur, r):
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        async with s.ws_connect(url) as ws:
            async for m in messages(ws, t0, dur):
                r["msgs"] += m is not None
        if time.monotonic() - t0 < dur:
            r["closes"].append([round(time.monotonic() - t0, 1), ws.close_code])


async def mjpeg(s, url, dur, r):
    """Camera load like a browser; a stall (> 5 s without bytes) is recorded and reconnected."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        try:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=None, sock_read=5)) as resp:
                async for chunk in resp.content.iter_chunked(65536):
                    r["bytes"] += len(chunk)
                    if time.monotonic() - t0 > dur:
                        break
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            r.setdefault("errors", []).append([round(time.monotonic() - t0, 1), repr(e)[:100]])
            await asyncio.sleep(0.5)


def imu_counters(t):
    i = t.get("imu") or {}
    return {"up": i.get("up"), "dropouts": (i.get("gaps") or {}).get("count"),
            "max_gap_ms": (i.get("gaps") or {}).get("max_ms"), "hub_resets": i.get("hub_resets"),
            "worker_restarts": i.get("worker_restarts"),
            "lost": sum(v["lost"] for v in (i.get("sensors") or {}).values()),
            "min_hz": min((v["hz"] for v in (i.get("sensors") or {}).values()), default=None)}


async def poll_server(s, base, dur, r):
    """Dashboard's own view of the 100 Hz stream, once a second; records every change."""
    t0, prev = time.monotonic(), None
    while time.monotonic() - t0 < dur:
        try:
            c = imu_counters(await (await s.get(base + "/api/telemetry")).json())
        except Exception as e:  # noqa: BLE001
            r["errors"].append([round(time.monotonic() - t0, 1), str(e)[:100]])
            c = None
        if c:
            r["min_hz"] = min(r["min_hz"] or 1e9, c["min_hz"] or 0)
            key = {k: c[k] for k in ("up", "dropouts", "hub_resets", "worker_restarts", "lost")}
            if prev is not None and key != prev:
                r["changes"].append([round(time.monotonic() - t0, 1), key])
            if r["start"] is None:
                r["start"] = c
            prev, r["end"] = key, c
        if int(time.monotonic() - t0) % 60 == 0 and c:
            print("  %4.0f s  %s" % (time.monotonic() - t0, c), flush=True)
        await asyncio.sleep(1.0)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--vision", action="store_true", help="turn Hailo detection on for load (off after)")
    ap.add_argument("--host", default="localhost:8080")
    a = ap.parse_args()
    base, dur = "http://" + a.host, a.minutes * 60
    res = {"started": datetime.now().isoformat(timespec="seconds"), "minutes": a.minutes, "vision": a.vision,
           "ws_imu": {"conns": 0, "closes": [], "msg_gaps": [], "stale": [], "msgs": 0, "max_age_ms": 0},
           "ws_det": {"closes": [], "msgs": 0}, "ws": {"closes": [], "msgs": 0}, "mjpeg": {"bytes": 0},
           "server": {"changes": [], "errors": [], "start": None, "end": None, "min_hz": None}}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=10)) as s:   # default total=300 s kills long streams
        if a.vision:
            await s.post(base + "/api/vision", json={"on": True})
            await asyncio.sleep(3)
        try:
            await asyncio.gather(watch_imu(s, base + "/ws/imu", dur, res["ws_imu"]),
                                 watch_other(s, base + "/ws/det", dur, res["ws_det"]),
                                 watch_other(s, base + "/ws", dur, res["ws"]),
                                 mjpeg(s, base + "/stream.mjpg", dur, res["mjpeg"]),
                                 poll_server(s, base, dur, res["server"]))
            t = await (await s.get(base + "/api/telemetry")).json()
            res["vision_fps"] = (t.get("vision") or {}).get("fps")
            res["cpu"] = t.get("cpu")
        finally:
            if a.vision:
                await s.post(base + "/api/vision", json={"on": False})
    w, sv = res["ws_imu"], res["server"]
    d = lambda k: (sv["end"] or {}).get(k, 0) - (sv["start"] or {}).get(k, 0)   # noqa: E731
    checks = {"ws_imu no disconnects": not w["closes"], "ws_imu no message gaps > 100 ms": not w["msg_gaps"],
              "imu data never older than 50 ms": not w["stale"], "no 100 Hz dropouts": d("dropouts") == 0,
              "no hub resets / worker restarts": d("hub_resets") == 0 and d("worker_restarts") == 0,
              "no lost samples": d("lost") == 0, "server telemetry never failed": not sv["errors"],
              "other sockets stayed up": not res["ws_det"]["closes"] and not res["ws"]["closes"],
              "camera stream never stalled": not res["mjpeg"].get("errors")}
    res["checks"] = checks
    res["ws_imu_rate"] = round(w["msgs"] / dur, 2)
    print("\n/ws/imu: %d msgs (%.2f/s), max sample age %.1f ms; server: %s -> %s, min rate %.1f Hz; mjpeg %.0f MB%s"
          % (w["msgs"], res["ws_imu_rate"], w["max_age_ms"], sv["start"], sv["end"], sv["min_hz"] or 0,
             res["mjpeg"]["bytes"] / 1e6, "; vision %s fps" % res.get("vision_fps") if a.vision else ""))
    for k, ok in checks.items():
        print("  [%s] %s" % ("PASS" if ok else "FAIL", k))
    for k in ("closes", "msg_gaps", "stale"):
        if w[k]:
            print("  ws_imu %s: %s" % (k, w[k][:20]))
    if sv["changes"]:
        print("  server counter changes:", sv["changes"][:20])
    out = os.path.join(ROOT, "results", "imu_dashboard_soak_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    print("results ->", out)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
