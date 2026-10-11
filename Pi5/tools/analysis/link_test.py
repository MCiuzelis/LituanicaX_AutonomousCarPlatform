"""PC-side link test: command round trips to the drive server (NEUTRAL commands only: t=0, s=0, the car
does not move) while pulling the preview video like the page does; --rec also records a take (delete it
afterwards). Needs aiohttp + numpy: uv run --project YoloTraining --with aiohttp python ...

    python tools/analysis/link_test.py 192.168.88.126 60 960 [--rec]
2026-10-11: 1128 commands, 0 lost, rtt p50 4 ms / max 21 ms with detection + recording + 960 preview.
"""
import asyncio, json, sys, time, struct
import aiohttp
import numpy as np

HOST = sys.argv[1]
SECS = float(sys.argv[2])
SIZE = sys.argv[3] if len(sys.argv) > 3 else "640"
REC = "--rec" in sys.argv


async def ctl(session, out):
    sent, rtt, gaps = {}, [], []
    async with session.ws_connect("ws://%s:8082/ws/ctl" % HOST) as ws:
        async def rx():
            async for m in ws:
                d = json.loads(m.data)
                if "a" in d and d["a"] in sent:
                    rtt.append((time.perf_counter() - sent.pop(d["a"])) * 1000)
        task = asyncio.ensure_future(rx())
        t_end, n, last = time.perf_counter() + SECS, 0, None
        while time.perf_counter() < t_end:
            n += 1
            now = time.perf_counter()
            if last:
                gaps.append((now - last) * 1000)
            last = now
            sent[n] = now
            await ws.send_str(json.dumps({"t": 0.0, "s": 0.0, "n": n}))
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)
        task.cancel()
    out["rtt"], out["lost"], out["sent"], out["send_gaps"] = rtt, len(sent), n, gaps


async def video(session, out):
    ages, kb, t = [], [], []
    async with session.ws_connect("ws://%s:8080/ws/video?w=%s" % (HOST, SIZE), max_msg_size=0) as ws:
        t_end = time.perf_counter() + SECS
        async for m in ws:
            b = m.data
            n = struct.unpack("<I", b[:4])[0]
            h = json.loads(b[4:4 + n])
            ages.append(h.get("age_ms") or 0)
            kb.append((len(b) - 4 - n) / 1024)
            t.append(time.perf_counter())
            await ws.send_str("a")
            if time.perf_counter() > t_end:
                break
    out["video_fps"] = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0
    out["video_age"], out["video_kb"] = ages, kb


async def main():
    out = {}
    async with aiohttp.ClientSession() as s:
        if REC:
            await (await s.post("http://%s:8080/api/record" % HOST, json={"on": True})).json()
        await asyncio.gather(ctl(s, out), video(s, out))
        tel = await (await s.get("http://%s:8080/api/telemetry" % HOST)).json()
        if REC:
            r = await (await s.post("http://%s:8080/api/record" % HOST, json={"on": False})).json()
            out["rec_id"] = r.get("file")
    r = np.array(out["rtt"])
    print("commands: sent %d, acked %d, lost %d | rtt ms p50 %.1f p90 %.1f p99 %.1f max %.1f | >100 ms: %d, >250 ms: %d" % (
        out["sent"], len(r), out["lost"], *np.percentile(r, [50, 90, 99]), r.max(), (r > 100).sum(), (r > 250).sum()))
    g = np.array(out["send_gaps"])
    print("client send gaps ms: p50 %.1f max %.1f" % (np.median(g), g.max()))
    a = np.array(out["video_age"])
    print("video %s: %.1f fps, %.0f kB/frame, frame age on car p50 %.1f max %.1f ms" % (SIZE, out["video_fps"], np.mean(out["video_kb"]), np.median(a), a.max()))
    v, c = tel["vision"], tel["control"]
    print("car: det %.1f fps dropped %.1f%% lat p50 %s | cpu %s | control: late %s gap_max %s deadman %s | rec %s" % (
        v["fps"], v["dropped_pct"], v["lat_ms"]["p50"], tel["cpu"]["util"], c["late"], c["gap_max_ms"], c["deadman"],
        out.get("rec_id")))


asyncio.run(main())
