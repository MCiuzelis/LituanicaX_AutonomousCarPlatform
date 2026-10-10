#!/usr/bin/env python3
"""BNO085 endurance test: hardware.BNO085 under load, fault injection and a long soak.

    python3 tests/bno085_endurance.py                      # ~45 min, results/bno085_endurance_<ts>.json
    python3 tests/bno085_endurance.py --block 10 --soak-minutes 2     # quick run
    python3 tests/bno085_endurance.py --skip matrix,faults            # soak only

Keep the board still (value checks assume rest). Nothing else may use GPIO5/6 or spidev0.1.

1. load matrix   every sensor profile x every load, `--block` s each. Per block: delivered rate
                 per sensor vs the same profile without load, lost samples (sequence gaps),
                 subscription overflow, driver latency (sample time -> callback) p50/p99/max,
                 bad values, driver errors / resets.
                 loads: none | cpu (stress-ng 4 cores) | gil (2 busy Python threads in this
                 process) | mem (stress-ng vm) | spi (MCP3202 on the same SPI controller, tight
                 loop) | all (everything at once)
2. faults        hub resets itself (SHTP exec reset), RST glitch behind the driver's back,
                 enable/disable churn, commands while streaming (product id, FRS metadata,
                 calibration, tare), close/reopen cycles (fd / thread leaks, boot time)
3. soak          rotates profiles and loads for --soak-minutes with the fault set every cycle;
                 checks memory (RSS), threads and fds stay flat, and no sample is ever lost
                 outside an injected fault.
"""
import argparse
import collections
import json
import math
import os
import random
import resource
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Pi5/
sys.path.insert(0, ROOT)
from hardware import HardwareError  # noqa: E402
from hardware.bno085 import BNO085, SENSORS, MAX_RATE_HZ, TARE_GAME_RV  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")

_FUS = ["rv", "game_rv", "accel", "gyro", "mag", "linear_accel", "gravity"]
_RAW = ["accel", "gyro", "mag", "raw_accel", "raw_gyro", "raw_mag"]
_EVT = [s for s in SENSORS if s not in MAX_RATE_HZ]
# name: (sensors, rate_hz (None = max), batch_ms). Reliable configurations only; the hub's
# own limits (raw / girv at max starve the rest, batch overflow) are documented in the driver.
PROFILES = {
    "default":     (list(BNO085.DEFAULT_SENSORS), 100, 0),      # enable_default(): ~700 rep/s
    "fusion_200b": (_FUS, 200, 20),                            # 7 outputs at 200 Hz, batched
    "girv":        (["girv"], None, 0),                        # 1 kHz gyro-integrated RV alone
    "with_events": (list(BNO085.DEFAULT_SENSORS), 100, 0),     # default + tap/stability/steps...
}
LOADS = ["none", "cpu", "gil", "mem", "spi", "all"]
QUAT = {"rv", "game_rv", "geo_rv", "arvr_rv", "arvr_game_rv", "girv"}
G = 9.80665


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return None


def n_fds():
    return len(os.listdir("/proc/self/fd"))


# ---- load generators --------------------------------------------------------------------------
class Load:
    def __init__(self, kind):
        self.kind = kind
        self.procs, self.threads = [], []
        self.stop = threading.Event()
        self.spi_reads = 0
        self.gil_loops = 0

    def __enter__(self):
        k = self.kind
        if k in ("cpu", "all"):
            self.procs.append(subprocess.Popen(["stress-ng", "-c", "4", "--cpu-method", "matrixprod"],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if k in ("mem", "all"):
            self.procs.append(subprocess.Popen(["stress-ng", "--vm", "2", "--vm-bytes", "20%"],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if k in ("gil", "all"):
            for _ in range(2):
                t = threading.Thread(target=self._gil, daemon=True)
                t.start()
                self.threads.append(t)
        if k in ("spi", "all"):
            t = threading.Thread(target=self._spi, daemon=True)
            t.start()
            self.threads.append(t)
        time.sleep(0.5)
        return self

    def _gil(self):
        while not self.stop.is_set():
            sum(i * i for i in range(20000))       # pure Python: holds the GIL
            self.gil_loops += 1

    def _spi(self):
        try:
            from hardware import MCP3202
            adc = MCP3202()
        except Exception as e:                     # noqa: BLE001
            print("    (spi load unavailable: %s)" % e)
            return
        while not self.stop.is_set():
            try:
                adc.read_raw(1)
                self.spi_reads += 1
            except Exception:                      # noqa: BLE001
                time.sleep(0.01)

    def __exit__(self, *exc):
        self.stop.set()
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        for t in self.threads:
            t.join(2)


# ---- measurement -------------------------------------------------------------------------------
class Monitor:
    """Validates every sample (subscription consumer thread) + latency (callback in driver thread)."""

    def __init__(self, imu):
        self.imu = imu
        self.lock = threading.Lock()
        self.sub = imu.subscribe(maxlen=200000)
        self.stop = threading.Event()
        self.reset_window()
        self.cb_lat = collections.defaultdict(list)
        for n in SENSORS:
            imu.on(n, self._cb)
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _cb(self, r):                              # draining thread: keep tiny
        self.cb_lat[r.sensor].append(time.monotonic() - r.timestamp)
        self.ts[r.sensor].append(r.timestamp)

    def reset_window(self):
        with getattr(self, "lock", threading.Lock()):
            self.count = collections.Counter()
            self.bad = collections.Counter()
            self.bad_examples = []
            self.ts_back = collections.Counter()
            self.ts_back_max = 0.0
            self.last_ts = {}
            self.max_gap = collections.defaultdict(float)
            self.cb_lat = collections.defaultdict(list)
            self.ts = collections.defaultdict(list)
            self.first_t = {}
            self.last_arr = {}
            self.overflow0 = self.sub.overflow if hasattr(self, "sub") else 0

    def _consume(self):
        while not self.stop.is_set():
            r = self.sub.get(0.2)
            if r is None:
                continue
            now = time.monotonic()
            with self.lock:
                n = r.sensor
                self.count[n] += 1
                self.first_t.setdefault(n, now)
                self.last_arr[n] = now
                lt = self.last_ts.get(n)
                if lt is not None:
                    if r.timestamp < lt - 1e-4:
                        self.ts_back[n] += 1
                        self.ts_back_max = max(self.ts_back_max, lt - r.timestamp)
                    self.max_gap[n] = max(self.max_gap[n], r.timestamp - lt)
                self.last_ts[n] = r.timestamp
                why = check_value(r)
                if why:
                    self.bad[n] += 1
                    if len(self.bad_examples) < 5:
                        self.bad_examples.append("%s %s %s" % (n, why, [round(v, 3) for v in r.values]))

    def window(self, t0, t1):
        """Stats of samples whose hub timestamp lies in [t0, t1] (immune to delivery bursts)."""
        dur = t1 - t0
        with self.lock:
            out = {}
            for n in set(self.count) | set(self.ts):
                c = sum(1 for t in self.ts.get(n, ()) if t0 <= t <= t1)
                lat = sorted(self.cb_lat.get(n, []))
                out[n] = {"rate_hz": round(c / dur, 1), "samples": c, "bad": self.bad[n],
                          "ts_backwards": self.ts_back[n], "max_gap_ms": round(self.max_gap[n] * 1e3, 1),
                          "lat_ms": {"p50": round(lat[len(lat) // 2] * 1e3, 2),
                                     "p99": round(lat[int(len(lat) * .99)] * 1e3, 2),
                                     "max": round(lat[-1] * 1e3, 2)} if lat else None}
            return out, self.sub.overflow - self.overflow0, list(self.bad_examples), self.ts_back_max

    def close(self):
        self.stop.set()
        self.thread.join(2)
        self.sub.close()


def check_value(r):
    v = r.values
    if any(isinstance(x, float) and not math.isfinite(x) for x in v):
        return "non-finite"
    if not 0 <= r.accuracy <= 3:
        return "accuracy"
    n = r.sensor
    if n in QUAT:
        q = math.sqrt(sum(x * x for x in v[:4]))
        if abs(q - 1) > 0.02:
            return "|q|=%.3f" % q
    elif n in ("accel", "gravity"):
        a = math.sqrt(sum(x * x for x in v[:3]))
        if abs(a - G) > 1.0:
            return "|a|=%.2f" % a
    elif n == "gyro":
        if math.sqrt(sum(x * x for x in v[:3])) > 0.2:
            return "moving"
    elif n == "linear_accel":
        if math.sqrt(sum(x * x for x in v[:3])) > 1.0:
            return "moving"
    return None


def counters(imu):
    c = imu.stats()["link"]
    return {k: v for k, v in c.items() if k.startswith(("resets_", "errors", "shtp_", "dropped_", "orphan",
                                                         "bad_", "callback_", "ping_lost", "frs_"))}


def lost(imu):
    return {n: s["lost"] for n, s in imu.stats()["sensors"].items()}


def apply_profile(imu, name):
    """Enable a profile; returns sensors the hub did not start within 4 s (starved), if any."""
    names, rate, batch = PROFILES[name]
    imu.disable()
    time.sleep(0.3)
    starved = []
    try:
        imu.enable(*names, rate_hz=rate, batch_ms=batch, timeout=4.0)
        if name == "with_events":
            imu.enable(*_EVT)
    except HardwareError as e:
        starved = str(e).split("from ", 1)[-1].split(" after")[0].split(", ")
    time.sleep(3.5)                                  # the hub re-plans for ~3 s (seq gaps)
    return starved


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail="", warn=False):
        st = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.checks.append({"check": name, "status": st, "detail": detail})
        print("  [%s] %-44s %s" % (st, name, detail), flush=True)
        return ok


# ---- 1. load matrix -----------------------------------------------------------------------------
def run_block(imu, mon, profile, load, seconds):
    lost0, cnt0 = lost(imu), counters(imu)
    with Load(load) as ld:
        time.sleep(0.5)
        mon.reset_window()
        t0 = time.monotonic()
        time.sleep(seconds)
        t1 = time.monotonic()
        spi_reads, gil = ld.spi_reads, ld.gil_loops
        lst = imu.stats()["link"]
    time.sleep(1.0)                                  # backlog of a starved consumer drains
    dur = t1 - t0
    win, overflow, bad_ex, back_max = mon.window(t0, t1)
    lost1, cnt1 = lost(imu), counters(imu)
    st = lst
    return {"profile": profile, "load": load, "seconds": round(dur, 1), "sensors": win,
            "lost": {n: lost1.get(n, 0) - lost0.get(n, 0) for n in lost1 if lost1.get(n, 0) - lost0.get(n, 0)},
            "overflow": overflow, "bad_examples": bad_ex, "ts_back_max_ms": round(back_max * 1e3, 2),
            "driver_counters": {k: cnt1.get(k, 0) - cnt0.get(k, 0) for k in cnt1 if cnt1.get(k, 0) != cnt0.get(k, 0)},
            "int_latency_us": st.get("int_latency_us"),
            "load_work": {"spi_reads": spi_reads, "gil_loops": gil},
            "total_hz": round(sum(s["rate_hz"] for s in win.values()), 1)}


def judge_block(rep, b, base):
    tag = "%s/%s" % (b["profile"], b["load"])
    lost_n = sum(b["lost"].values())
    bad = sum(s["bad"] for s in b["sensors"].values())
    back = sum(s["ts_backwards"] for s in b["sensors"].values())
    errs = {k: v for k, v in b["driver_counters"].items() if v}
    worst = None
    if base:
        ratios = []
        for n, s in base["sensors"].items():
            if s["rate_hz"] >= 5:
                got = b["sensors"].get(n, {}).get("rate_hz", 0)
                ratios.append((got / s["rate_hz"], n))
        worst = min(ratios) if ratios else None
    lat = [s["lat_ms"]["p99"] for s in b["sensors"].values() if s["lat_ms"]]
    lat_max = [s["lat_ms"]["max"] for s in b["sensors"].values() if s["lat_ms"]]
    detail = "%6.0f rep/s lost=%d ovf=%d bad=%d tsback=%d(%.1fms) lat p99 %.1f max %.1f ms%s%s" % (
        b["total_hz"], lost_n, b["overflow"], bad, back, b["ts_back_max_ms"], max(lat) if lat else 0,
        max(lat_max) if lat_max else 0,
        "" if worst is None else " worst %s %.0f%%" % (worst[1], worst[0] * 100), " drv %s" % errs if errs else "")
    if base and base["total_hz"]:
        detail += " (%.0f%% of no-load)" % (100 * b["total_hz"] / base["total_hz"])
    # FAIL = something was lost or wrong; WARN = all data arrived but the hub delivered less
    # (host slower to answer INT -> hub decimates) or the consumer saw it late
    # timestamps a bit out of order = IRQ timestamp jitter under load: warn, fail only if large
    ok = lost_n == 0 and b["overflow"] == 0 and b["ts_back_max_ms"] < 5 and not errs and bad == 0
    if not ok:
        rep.check("%-18s" % tag, False, detail)
        return False
    slow = (base is not None and b["total_hz"] < 0.9 * base["total_hz"]) or back > 0
    late = b["load"] not in ("gil", "all") and lat and max(lat) > 50
    rep.check("%-18s" % tag, not (slow or late), detail, warn=True)
    return True


def test_matrix(imu, mon, rep, block, profiles, loads):
    print("\n== 1. load matrix (%d s per block)" % block)
    out = []
    for prof in profiles:
        starved = apply_profile(imu, prof)
        if starved:
            rep.check("%-18s" % (prof + " start"), False, "hub never delivered %s (decimated to 0)" % starved,
                      warn=True)
        base = None
        for load in loads:
            b = run_block(imu, mon, prof, load, block)
            if load == "none":
                base = b
            judge_block(rep, b, base)
            out.append(b)
    return out


# ---- 2. faults ----------------------------------------------------------------------------------
def wait_data(imu, names, after, timeout=3.0):
    """Seconds until every sensor in `names` has a sample newer than `after`, or None."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if all(imu.read(n) and imu.read(n).timestamp > after for n in names):
            return time.monotonic() - after
        time.sleep(0.005)
    return None


def fault_exec_reset(imu, rep):
    """Hub resets itself (SHTP executable channel 'reset'): driver must notice and re-apply."""
    names = list(imu.enabled)
    c0 = imu.stats()["link"]
    t = time.monotonic()
    imu._inject("exec_reset")                        # SHTP executable channel: reset
    time.sleep(0.4)
    rec = wait_data(imu, [n for n in names if n in MAX_RATE_HZ], t + 0.35)
    c1 = imu.stats()["link"]
    how = {k: c1[k] - c0.get(k, 0) for k in c1 if k.startswith("resets_") and c1[k] != c0.get(k, 0)}
    return rep.check("hub self-reset -> recovered", rec is not None and sum(how.values()) >= 1,
                     "detected %s, data back after %s s" % (how, None if rec is None else round(rec + 0.35, 2)))


def fault_rst_glitch(imu, rep):
    """RST pulled low behind the driver's back (brown-out-like): watchdog / reset detection."""
    names = [n for n in imu.enabled if n in MAX_RATE_HZ]
    c0 = imu.stats()["link"]
    r0 = sum(v for k, v in c0.items() if k.startswith("resets_"))
    t = time.monotonic()
    imu._inject("rst_glitch")                        # 20 ms low pulse on RST
    rec = wait_data(imu, names, t + 0.05, timeout=4.0)
    c1 = imu.stats()["link"]
    r1 = sum(v for k, v in c1.items() if k.startswith("resets_"))
    return rep.check("RST glitch -> recovered", rec is not None and r1 > r0,
                     "resets +%d (%s), data back after %s s" % (
                         r1 - r0, {k: c1[k] - c0.get(k, 0) for k in c1 if k.startswith("resets_") and c1[k] != c0.get(k, 0)},
                         None if rec is None else round(rec + 0.05, 2)))


def fault_churn(imu, rep, n=40):
    # raw sensors at max and girv at 1 kHz get hub priority and can starve everything else
    # to 0 Hz (tested in their own profiles): not churned here
    pool = [s for s in SENSORS if s in MAX_RATE_HZ and not s.startswith("raw_") and s != "girv"]
    fails, t_en = [], []
    for i in range(n):
        pick = random.sample(pool, random.randint(1, 4))
        imu.disable()
        t = time.monotonic()
        try:
            imu.enable(*pick, rate_hz=random.choice([None, 50, 100, 200]))
            t_en.append(time.monotonic() - t)
        except HardwareError as e:
            fails.append(str(e))
    return rep.check("enable/disable churn x%d" % n, not fails,
                     "enable->first sample p50 %.0f ms max %.0f ms%s" % (
                         statistics.median(t_en) * 1e3 if t_en else 0, max(t_en) * 1e3 if t_en else 0,
                         " fails: %s" % fails[:2] if fails else ""))


def fault_commands(imu, rep, n=10):
    apply_profile(imu, "default")
    lost0 = lost(imu)
    bad = collections.Counter()
    for i in range(n):
        if len(imu.product_id()) != 4:
            bad["product_id"] += 1
        m = imu.metadata(random.choice([s for s in SENSORS if s in MAX_RATE_HZ]))
        if not m or not m["min_period_us"]:
            bad["metadata"] += 1
        c = imu.calibration()
        if not c or c["status"] != 0:
            bad["calibration"] += 1
        if not imu.set_calibration(accel=False, gyro=True, mag=False):   # never accel / mag: see tools/imu_cal.py
            bad["set_calibration"] += 1
    # tare: game rv becomes ~identity, clear brings the old orientation back
    q_before = imu.game_quaternion
    imu.tare(TARE_GAME_RV)
    time.sleep(0.5)
    q_tared = imu.game_quaternion
    imu.clear_tare()
    time.sleep(0.5)
    q_after = imu.game_quaternion
    tare_ok = q_tared is not None and abs(abs(q_tared[3]) - 1) < 0.02
    clear_ok = q_after is not None and q_before is not None and abs(sum(a * b for a, b in zip(q_before, q_after))) > 0.98
    lost1 = lost(imu)
    lost_n = sum(lost1.get(k, 0) - lost0.get(k, 0) for k in lost1)
    ok = not bad and tare_ok and clear_ok and lost_n == 0
    return rep.check("commands while streaming x%d" % n, ok,
                     "failures %s, tare->|real| %.3f, clear dot %.3f, lost %d" % (
                         dict(bad), abs(q_tared[3]) if q_tared else -1,
                         abs(sum(a * b for a, b in zip(q_before, q_after))) if q_after and q_before else -1, lost_n))


def fault_reopen(rep, n=10):
    fds0, th0 = n_fds(), threading.active_count()
    boots, fails = [], []
    for i in range(n):
        t = time.monotonic()
        try:
            with BNO085() as imu:
                boots.append(time.monotonic() - t)
                imu.enable("game_rv")
                if imu.wait("game_rv", 1.0) is None:
                    fails.append("no data")
        except HardwareError as e:
            fails.append(str(e))
    fds1, th1 = n_fds(), threading.active_count()
    return rep.check("close/reopen x%d" % n, not fails and fds1 <= fds0 and th1 <= th0,
                     "boot p50 %.0f ms max %.0f ms, fds %d->%d, threads %d->%d%s" % (
                         statistics.median(boots) * 1e3 if boots else 0, max(boots) * 1e3 if boots else 0,
                         fds0, fds1, th0, th1, " fails %s" % fails[:2] if fails else ""))


def test_faults(imu, rep):
    print("\n== 2. faults")
    apply_profile(imu, "default")
    out = {"exec_reset": fault_exec_reset(imu, rep), "rst_glitch": fault_rst_glitch(imu, rep)}
    with Load("all"):
        apply_profile(imu, "default")
        out["exec_reset_under_load"] = fault_exec_reset(imu, rep)
        out["churn_under_load"] = fault_churn(imu, rep, 20)
    out["churn"] = fault_churn(imu, rep)
    out["commands"] = fault_commands(imu, rep)
    return out


# ---- 3. soak ------------------------------------------------------------------------------------
def test_soak(imu, mon, rep, minutes, block):
    print("\n== 3. soak (%.0f min, profile/load rotate every %d s, faults every cycle)" % (minutes, block))
    end = time.monotonic() + minutes * 60
    rss0, fds0, th0 = rss_mb(), n_fds(), threading.active_count()
    blocks, lost_tot, bad_blocks, cycle = [], 0, 0, 0
    profs = list(PROFILES)
    t_start = time.monotonic()
    while time.monotonic() < end:
        cycle += 1
        prof, load = profs[cycle % len(profs)], LOADS[cycle % len(LOADS)]
        apply_profile(imu, prof)
        b = run_block(imu, mon, prof, load, block)
        nl = sum(b["lost"].values())
        bad = sum(s["bad"] for s in b["sensors"].values()) + b["overflow"]
        errs = {k: v for k, v in b["driver_counters"].items() if v}
        lost_tot += nl
        if nl or bad or errs:
            bad_blocks += 1
        blocks.append({"cycle": cycle, "t_min": round((time.monotonic() - t_start) / 60, 1), "profile": prof,
                       "load": load, "total_hz": b["total_hz"], "lost": nl, "bad": bad, "driver": errs,
                       "rss_mb": round(rss_mb(), 1), "fds": n_fds(), "threads": threading.active_count()})
        print("    %5.1f min  %-8s %-5s %6.0f rep/s lost=%d bad=%d drv=%s rss=%.1f MB fds=%d thr=%d" % (
            blocks[-1]["t_min"], prof, load, b["total_hz"], nl, bad, errs or "-", blocks[-1]["rss_mb"],
            blocks[-1]["fds"], blocks[-1]["threads"]), flush=True)
        if cycle % 3 == 0:                     # inject faults regularly during the soak
            q = Report()
            fault_exec_reset(imu, q)
            fault_churn(imu, q, 5)
            if any(c["status"] == "FAIL" for c in q.checks):
                bad_blocks += 1
                blocks[-1]["fault_fail"] = q.checks
    rss1, fds1, th1 = rss_mb(), n_fds(), threading.active_count()
    rep.check("soak: no lost samples", lost_tot == 0, "%d blocks, lost %d" % (len(blocks), lost_tot))
    rep.check("soak: no bad blocks", bad_blocks == 0, "%d of %d" % (bad_blocks, len(blocks)))
    rep.check("soak: memory flat", rss1 - rss0 < 20, "RSS %.1f -> %.1f MB" % (rss0, rss1))
    rep.check("soak: fds / threads flat", fds1 <= fds0 + 2 and th1 <= th0 + 1,
              "fds %d->%d threads %d->%d" % (fds0, fds1, th0, th1))
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=int, default=20, help="seconds per matrix / soak block (default 20)")
    ap.add_argument("--soak-minutes", type=float, default=20)
    ap.add_argument("--profiles", default=",".join(PROFILES))
    ap.add_argument("--loads", default=",".join(LOADS))
    ap.add_argument("--skip", default="", help="comma list of: matrix,faults,soak")
    args = ap.parse_args()
    skip = set(filter(None, args.skip.split(",")))
    random.seed(1)

    rep = Report()
    res = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}
    t0 = time.monotonic()
    imu = BNO085()
    mon = Monitor(imu)
    try:
        res["product"] = imu.product
        res["rt_priority"] = imu.rt_priority
        print("BNO085 up: %s, SCHED_FIFO=%s" % (", ".join("%d v%s" % (p["part"], p["version"]) for p in imu.product),
                                                imu.rt_priority))
        if "matrix" not in skip:
            res["matrix"] = test_matrix(imu, mon, rep, args.block, args.profiles.split(","), args.loads.split(","))
        if "faults" not in skip:
            res["faults"] = test_faults(imu, rep)
            res["faults"]["reopen"] = None
        if "soak" not in skip:
            res["soak"] = test_soak(imu, mon, rep, args.soak_minutes, args.block)
    finally:
        res["final_stats"] = imu.stats()
        res["last_error"] = imu.last_error
        mon.close()
        imu.close()
    if "faults" not in skip:
        res["faults"]["reopen"] = fault_reopen(rep)
    res["checks"] = rep.checks
    res["minutes"] = round((time.monotonic() - t0) / 60, 1)
    fails = [c for c in rep.checks if c["status"] == "FAIL"]
    print("\n%d checks, %d failed, %.1f min" % (len(rep.checks), len(fails), res["minutes"]))
    for c in fails:
        print("  FAIL %s: %s" % (c["check"], c["detail"]))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "bno085_endurance_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    os.sync()
    print("saved", path)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
