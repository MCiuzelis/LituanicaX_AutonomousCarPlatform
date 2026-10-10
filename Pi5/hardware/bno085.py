"""BNO085 9-DOF IMU (Adafruit 4754 breakout) over SPI: SHTP / SH-2 driver, no vendor library.

    from hardware import BNO085

    with BNO085() as imu:
        imu.enable_default()                       # orientation + motion at 100 Hz (recommended)
        # or: imu.enable("rv", "accel", "gyro", rate_hz=200)   (rate_hz=None = sensor max)
        print(imu.quaternion, imu.acceleration, imu.euler)
        q = imu.wait("rv")                         # block until the next sample
        with imu.subscribe("accel", "gyro") as sub:
            for r in sub:                          # every sample, nothing dropped
                print(r.sensor, r.values, r.timestamp)

How it works: a worker process (`python -m hardware.bno085 --worker`, started and supervised
by this class) owns the SPI device and the INT/RST lines and talks SHTP to the hub. It runs
with SCHED_FIFO priority when allowed (scripts/setup_spi0_mode3.sh grants that to the gpio
group) and is immune to whatever the calling program does with the GIL. Samples go into a
shared-memory ring (~30 s deep at full rate); this process drains it from a background thread
and on every API call, so values are fresh even if that thread is starved. If the hub resets
or goes quiet the worker resets it and re-applies the configuration; if the worker dies it is
restarted. Your script needs no `if __name__ == "__main__"` guard.

Wiring (P0 + P1 jumpered high = SPI): SCL->GPIO11 SCLK, SDA->GPIO9 MISO, DI->GPIO10 MOSI,
CS->GPIO7 (SPI0 CE1, /dev/spidev0.1), INT->GPIO6, RST->GPIO5. SPI mode 3, max 3 MHz. The
MCP3202 on CE0 must also run in mode 3 (scripts/setup_spi0_mode3.sh), otherwise every switch
between the two devices adds a clock edge and garbles the next transfer.

Sensors (name: Reading.values, max rate measured on this board):
    accel, linear_accel, gravity        m/s^2 (x, y, z)                         500 / 400 / 400 Hz
    gyro                                rad/s (x, y, z)                         400 Hz
    gyro_uncal                          rad/s (x, y, z, bias x, y, z)           400 Hz
    mag / mag_uncal                     uT (x, y, z[, bias x, y, z])            100 Hz
    rv, geo_rv, arvr_rv                 quaternion (i, j, k, real, accuracy rad) 400 / 100 / 400 Hz
    game_rv, arvr_game_rv               quaternion (i, j, k, real)              400 Hz
    girv  (gyro-integrated RV)          (i, j, k, real, wx, wy, wz rad/s)       1000 Hz
    raw_accel, raw_gyro, raw_mag        ADC counts (x, y, z); need their calibrated sensor on,
                                        which enable() does for you
    tap, shake, step_detector, step_counter, stability, significant_motion   events (1 value)

Hub limits worth knowing (measured, firmware 3.2.13):
  * The hub's CPU caps total output at roughly 800-1200 reports/s. Each sensor reaches its max
    alone; with several enabled the hub decimates (fusion outputs share one rate). Raw sensors
    at full rate and girv at 1 kHz have priority and can starve every other output to ~0 Hz.
    stats() shows delivered rates; `lost` (sequence gaps) is real loss, decimation does not
    create gaps. For ~3 s after a configuration change the hub re-plans and skips sequence
    numbers; those gaps are reported separately as `settle_gaps`.
  * gyro / gyro_uncal / arvr_* fall back to 100 Hz if asked for < 2500 us. enable() never asks
    for less than the measured max interval, so rate_hz above max just gives max.
  * The breakout ties WAKE high: the hub only accepts writes during a transfer it started by
    asserting INT, and a write sent otherwise is silently lost. The worker keeps a cheap Get
    Feature request ("ping") in flight; when data may stop it reads only the head of each
    packet so the remainder (a continuation) guarantees another INT, and that "anchor" is only
    read together with a write or ping. Idle cost: ~20 tiny transfers/s.
  * A new control request makes the hub drop a response it has not sent yet (FRS fragments,
    product id, command responses). So exactly one response-generating request is in flight
    at a time; the next one rides on the transfer that reads the last bytes of the answer.
    Commands without a response (tare) cannot keep INT coming themselves (a ping appended
    to the same packet is ignored), so they ride on the head read of a ping's answer, sized
    so the rest of that answer stays pending.
  * Batching (enable(batch_ms=20)) packs ~10 reports per packet: ~50 % more total output and
    almost immune to host load. Below 20 ms it has no effect on this firmware. The hub's batch
    buffer is small: above ~1100 reports/s total (raw sensors at full rate, or 15+ sensors)
    it overflows and drops samples (seq gaps), so batch only moderate loads.
  * Timestamps: hub delays + the kernel's INT edge timestamp; heavy CPU load adds IRQ jitter
    of up to ~1 ms (consecutive samples can then appear slightly out of order).
"""
import collections
import json
import math
import mmap
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from .base import HardwareError, Subsystem

GPIOCHIP, PIN_RST, PIN_INT = "/dev/gpiochip0", 5, 6
SPI_BUS, SPI_CS, SPI_HZ = 0, 1, 3_000_000

CH_COMMAND, CH_EXEC, CH_CONTROL, CH_REPORTS, CH_WAKE, CH_GIRV = range(6)

# name: (report id, int16 fields, Q point, max-rate interval us, base sensor for raw)
_PERIODIC = {
    "accel":        (0x01, 3, 8, 2000, None),
    "gyro":         (0x02, 3, 9, 2500, None),
    "mag":          (0x03, 3, 4, 10000, None),
    "linear_accel": (0x04, 3, 8, 2500, None),
    "rv":           (0x05, 5, 14, 2500, None),
    "gravity":      (0x06, 3, 8, 2500, None),
    "gyro_uncal":   (0x07, 6, 9, 2500, None),
    "game_rv":      (0x08, 4, 14, 2500, None),
    "geo_rv":       (0x09, 5, 14, 10000, None),
    "mag_uncal":    (0x0F, 6, 4, 10000, None),
    "raw_accel":    (0x14, 3, 0, 2000, "accel"),
    "raw_gyro":     (0x15, 3, 0, 1000, "gyro"),
    "raw_mag":      (0x16, 3, 0, 10000, "mag"),
    "arvr_rv":      (0x28, 5, 14, 2500, None),
    "arvr_game_rv": (0x29, 4, 14, 2500, None),
    "girv":         (0x2A, 7, 14, 1000, None),
}
# event-type reports: name: (report id, default interval us)
_EVENTS = {
    "tap": (0x10, 10000), "step_counter": (0x11, 100000), "significant_motion": (0x12, 100000),
    "stability": (0x13, 50000), "step_detector": (0x18, 10000), "shake": (0x19, 10000),
}
SENSORS = list(_PERIODIC) + list(_EVENTS)
_ID = {n: v[0] for n, v in list(_PERIODIC.items()) + list(_EVENTS.items())}
_NAME = {v: k for k, v in _ID.items()}
_NVALS = {n: (v[1] if n != "girv" else 7) for n, v in _PERIODIC.items()}
_NVALS.update({n: 1 for n in _EVENTS})
_INTS = {n for n, v in _PERIODIC.items() if v[2] == 0} | set(_EVENTS)
MAX_RATE_HZ = {n: 1e6 / v[3] for n, v in _PERIODIC.items()}
STABILITY = ("unknown", "on_table", "stationary", "stable", "motion", "reserved")

# report lengths (incl. 4-byte header); replaced by the hub's advertisement at boot
_DEFAULT_LEN = {0x01: 10, 0x02: 10, 0x03: 10, 0x04: 10, 0x05: 14, 0x06: 10, 0x07: 16, 0x08: 12,
                0x09: 14, 0x0A: 8, 0x0B: 8, 0x0C: 6, 0x0D: 6, 0x0E: 6, 0x0F: 16, 0x10: 5, 0x11: 12,
                0x12: 6, 0x13: 6, 0x14: 16, 0x15: 16, 0x16: 14, 0x18: 8, 0x19: 6, 0x1C: 6, 0x1E: 16,
                0x28: 14, 0x29: 12, 0x2A: 14}

# FRS metadata record per report id (SH-2 reference manual); word 4 = minimum period (us)
FRS_METADATA = {0x01: 0xE302, 0x02: 0xE306, 0x03: 0xE309, 0x04: 0xE303, 0x05: 0xE30B, 0x06: 0xE304,
                0x07: 0xE307, 0x08: 0xE30C, 0x09: 0xE30D, 0x0F: 0xE30A, 0x14: 0xE301, 0x15: 0xE305,
                0x16: 0xE308, 0x28: 0xE322, 0x29: 0xE323, 0x2A: 0xE324}

PING_ID = 0x13           # Get Feature of the stability classifier: cheap, always answered
PING_PERIOD = 0.05       # s between pings while data flows (keeps a write slot coming)
IDLE_HOLD = 0.1          # s an idle anchor (rest of a ping answer) is left unread
DATA_IDLE = 0.1          # s without data after which the link counts as idle
WATCHDOG = 1.0           # s without any data while periodic sensors are on -> reset hub
PEEK = 7                 # bytes read of a new packet while careful (header + 3 payload bytes)
CHAIN_HOLD = 0.5         # s after a write during which reads are careful
CTL_PERIOD = 0.002       # s between control-socket checks while streaming
NOTIFY_PERIOD = 0.002    # s between "new data" wake-ups of the parent
OUT_TIMEOUT = {"ping": 0.5, "feature": 0.5, "product": 2.0, "frs": 1.0, "cmd": 1.5}
SETTLE = 3.0             # s after a config change in which seq gaps are the hub re-planning
FREE_QUIET = 0.02        # s without control writes after a command nobody answers (tare)
STALL = 0.3              # s with writes queued and no INT -> reset hub

# shared-memory ring: header (u64 records written) + fixed records written by the worker
_REC = struct.Struct("<BBHxxxxd7d")      # sensor id, accuracy, seq, timestamp, 7 values
_STAMP = struct.Struct("<Q")             # record index + 1, written after the record body
_RSZ = 8 + _REC.size                     # 80 bytes
_RHDR = 64

# Tare basis
TARE_RV, TARE_GAME_RV, TARE_GEO_RV, TARE_GIRV, TARE_ARVR_RV, TARE_ARVR_GAME_RV = range(6)


@dataclass(frozen=True)
class Reading:
    sensor: str
    values: tuple          # see module docstring for units / layout
    accuracy: int          # 0 unreliable .. 3 high (status bits of the report)
    timestamp: float       # sample time on the time.monotonic() clock (s), hub-timestamped
    seq: int               # report sequence number 0..255 (girv: running count)


def quaternion_to_euler(q):
    """(i, j, k, real) -> (roll, pitch, yaw) in degrees (ZYX convention)."""
    x, y, z, w = q[:4]
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.degrees(math.asin(sp))
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


# =================================================================================================
# Caller side
# =================================================================================================
class Subscription:
    """Every sample of the given sensors, in arrival order. Iterate it or call get()/drain().

    Holds `maxlen` readings; if the consumer falls behind, the oldest are dropped and counted
    in `.overflow`.
    """

    def __init__(self, owner, names, maxlen):
        self._owner, self.names, self.overflow = owner, set(names), 0
        self._q = collections.deque(maxlen=maxlen)
        self._ev = threading.Event()

    def _push(self, r):
        if len(self._q) == self._q.maxlen:
            self.overflow += 1
        self._q.append(r)
        self._ev.set()

    def get(self, timeout=None):
        """Next reading, or None after `timeout` seconds."""
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                return self._q.popleft()
            except IndexError:
                pass
            self._ev.clear()
            self._owner._drain()
            if self._q:
                continue
            left = None if end is None else end - time.monotonic()
            if left is not None and left <= 0:
                return None
            self._ev.wait(0.05 if left is None else min(left, 0.05))

    def drain(self):
        """All buffered readings (non-blocking)."""
        self._owner._drain()
        out = []
        while True:
            try:
                out.append(self._q.popleft())
            except IndexError:
                return out

    def __iter__(self):
        while True:
            r = self.get(1.0)
            if r is not None:
                yield r
            elif self._owner._closed:
                return

    def close(self):
        self._owner._unsubscribe(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BNO085(Subsystem):
    name = "bno085"

    def __init__(self, spi_bus=SPI_BUS, spi_cs=SPI_CS, rst_pin=PIN_RST, int_pin=PIN_INT,
                 spi_hz=SPI_HZ, gpiochip=GPIOCHIP, realtime=True, ring_records=65536,
                 boot_timeout=3.0):
        self._wcfg = {"spi_bus": spi_bus, "spi_cs": spi_cs, "rst_pin": rst_pin, "int_pin": int_pin,
                      "spi_hz": spi_hz, "gpiochip": gpiochip, "realtime": realtime,
                      "ring_records": ring_records,
                      # dynamic calibration after every hub boot: the hub boots learning accel + mag and
                      # auto-saves to flash, which would overwrite the calibration from tools/imu_cal.py.
                      # Gyro only ([accel, gyro, mag, planar]); set_calibration() overrides it.
                      "me_cal": [0, 1, 0, 0]}
        self.spi_dev = "/dev/spidev%d.%d" % (spi_bus, spi_cs)
        self.rst_pin, self.int_pin = rst_pin, int_pin
        self._boot_timeout = boot_timeout
        self._lock = threading.RLock()           # parent-side state
        self._cond = threading.Condition(self._lock)
        self._send_lock = threading.Lock()
        self._config = {}                        # name -> [interval_us, batch_us, auto]
        self._latest = {}
        self._subs = []
        self._callbacks = collections.defaultdict(list)
        self._pstats = {}                        # name -> parent-side delivery stats
        self._pending = {}                       # req id -> [Event, result]
        self._req = 0
        self.counters = collections.Counter()    # parent side: ring_overrun, worker_restarts
        self.product = []
        self.rt_priority = False
        self.last_error = None
        self._closed = False
        self._proc = None
        self.started = time.monotonic()
        self._spawn()
        self._thread = threading.Thread(target=self._receiver, name="bno085-rx", daemon=True)
        self._thread.start()
        self.product = self.product_id() or []

    # ---- worker management -------------------------------------------------------------------
    def _spawn(self):
        cap = self._wcfg["ring_records"]
        size = _RHDR + cap * _RSZ
        memfd = os.memfd_create("bno085-ring")
        os.ftruncate(memfd, size)
        ring = mmap.mmap(memfd, size)
        nr, nw = os.pipe()
        os.set_blocking(nr, False)
        os.set_blocking(nw, False)
        ps, cs = socket.socketpair()
        cfg = dict(self._wcfg, ctl_fd=cs.fileno(), notify_fd=nw, ring_fd=memfd, ppid=os.getpid())
        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            proc = subprocess.Popen([sys.executable, "-m", __name__, "--worker", json.dumps(cfg)],
                                    pass_fds=(cs.fileno(), nw, memfd), env=env, stdin=subprocess.DEVNULL)
        finally:
            cs.close()
            os.close(nw)
            os.close(memfd)
        from multiprocessing.connection import Connection
        conn = Connection(ps.detach())
        ok = conn.poll(self._boot_timeout + 3.0)
        hello = conn.recv() if ok else None
        if not hello or not hello[1]:
            err = hello[2] if hello else "worker did not start"
            proc.kill()
            proc.wait()
            conn.close()
            os.close(nr)
            ring.close()
            raise HardwareError("BNO085: %s" % err)
        with self._lock:
            self._proc, self._conn, self._notify_r, self._ring = proc, conn, nr, ring
            self._cap, self._ridx = cap, 0
            self.rt_priority = hello[2].get("rt", False)
        if self._config:
            self._send_config()

    def _respawn(self, why):
        self.counters["worker_restarts"] += 1
        self.last_error = "%s: worker %s" % (time.strftime("%H:%M:%S"), why)
        with self._lock:
            old = (self._proc, self._conn, self._notify_r, self._ring)
            for p in self._pending.values():
                p[0].set()
            self._pending.clear()
        try:
            old[0].kill()
            old[0].wait(2)
        except Exception:                        # noqa: BLE001
            pass
        for c in (old[1].close, lambda: os.close(old[2]), old[3].close):
            try:
                c()
            except Exception:                    # noqa: BLE001
                pass
        while not self._closed:
            try:
                self._spawn()
                return
            except HardwareError as e:
                self.last_error = "%s: respawn failed: %s" % (time.strftime("%H:%M:%S"), e)
                time.sleep(1.0)

    def _receiver(self):
        while not self._closed:
            proc, conn, nr = self._proc, self._conn, self._notify_r
            try:
                r, _, _ = select.select([conn.fileno(), nr], [], [], 0.2)
            except (OSError, ValueError):
                r = []
                time.sleep(0.05)
            if self._closed:
                return
            try:
                if conn.fileno() in r:
                    while conn.poll(0):
                        self._on_reply(conn.recv())
                if nr in r:
                    while True:
                        try:
                            if not os.read(nr, 4096):
                                break
                        except BlockingIOError:
                            break
            except (EOFError, OSError):
                pass
            self._drain()
            if proc.poll() is not None and not self._closed:
                self._respawn("exited with %s" % proc.returncode)

    def _on_reply(self, msg):
        rid, ok, result = msg
        with self._lock:
            p = self._pending.pop(rid, None)
        if p:
            p[1] = result if ok else None
            if not ok:
                self.last_error = "%s: %s" % (time.strftime("%H:%M:%S"), result)
            p[0].set()

    def _request(self, op, args=None, timeout=2.0):
        """Send one request to the worker and wait for its reply (None on timeout/failure)."""
        with self._lock:
            self._req += 1
            rid = self._req
            slot = [threading.Event(), None]
            self._pending[rid] = slot
            conn = self._conn
        try:
            with self._send_lock:
                conn.send((rid, op, args))
        except (OSError, ValueError):
            with self._lock:
                self._pending.pop(rid, None)
            return None
        if not slot[0].wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            return None
        return slot[1]

    # ---- ring -> readings ----------------------------------------------------------------------
    def _drain(self):
        """Pull every new record out of the shared ring (any thread may call this)."""
        if self._closed:
            return
        with self._lock:
            ring, cap = self._ring, self._cap
            n = 0
            while True:
                off = _RHDR + (self._ridx % cap) * _RSZ
                stamp = _STAMP.unpack_from(ring, off)[0]
                want = self._ridx + 1
                if stamp == want:
                    sid, acc, seq, ts, *vals = _REC.unpack_from(ring, off + 8)
                    if _STAMP.unpack_from(ring, off)[0] != want:     # overwritten while reading
                        continue
                    self._ridx = want
                    name = _NAME.get(sid)
                    if name:
                        v = vals[:_NVALS[name]]
                        if name in _INTS:
                            v = [int(x) for x in v]
                        self._deliver(Reading(name, tuple(v), acc, ts, seq))
                        n += 1
                elif stamp > want:                    # the worker lapped us: skip ahead
                    written = _STAMP.unpack_from(ring, 0)[0]
                    new = max(want, written - cap // 2)
                    self.counters["ring_overrun"] += new - self._ridx
                    self._ridx = new
                else:
                    break
            if n:
                self._cond.notify_all()

    def _deliver(self, r):
        """Lock held."""
        s = self._pstats.get(r.sensor)
        if s is None:
            s = self._pstats[r.sensor] = {"count": 0, "win": collections.deque(maxlen=512), "last": None}
        s["count"] += 1
        s["win"].append(r.timestamp)
        s["last"] = time.monotonic()
        self._latest[r.sensor] = r
        for sub in self._subs:
            if r.sensor in sub.names:
                sub._push(r)
        for cb in self._callbacks.get(r.sensor, ()):
            try:
                cb(r)
            except Exception:                       # noqa: BLE001 - user code must not kill delivery
                self.counters["callback_errors"] += 1

    def _wait(self, pred, timeout):
        end = time.monotonic() + timeout
        while True:
            self._drain()
            with self._cond:
                if pred():
                    return True
                left = end - time.monotonic()
                if left <= 0 or self._closed:
                    return False
                self._cond.wait(min(left, 0.02))

    # ---- public API ------------------------------------------------------------------------------
    def available(self):
        return not self._closed and self._proc is not None and self._proc.poll() is None

    def enable(self, *names, rate_hz=None, batch_ms=0, timeout=3.0):
        """Enable sensors. rate_hz=None -> max rate; capped at the max. Returns {name: Hz asked}.

        Waits until each periodic sensor delivers its first sample (HardwareError if one does
        not). Raw sensors also enable their calibrated base sensor if it is off.
        """
        if not names:
            raise ValueError("enable() needs sensor names: %s" % ", ".join(SENSORS))
        out = {}
        with self._lock:
            for n in names:
                if n not in _ID:
                    raise ValueError("unknown sensor %r (have: %s)" % (n, ", ".join(SENSORS)))
                iv = self._interval_for(n, rate_hz)
                base = _PERIODIC.get(n, (0, 0, 0, 0, None))[4]
                if base and base not in self._config:   # base first: the hub stalls (no INT) if
                    # a raw sensors Set Feature arrives before its calibrated sensors
                    self._config[base] = [self._interval_for(base, rate_hz), int(batch_ms * 1000), True]
                self._config[n] = [iv, int(batch_ms * 1000), False]
                out[n] = round(1e6 / iv, 1)
        self._send_config()                  # returns once the hub has been sent the new config
        self._drain()
        with self._lock:
            for n in names:
                self._pstats.pop(n, None)    # count only samples from now on
        wait = [n for n in names if n in _PERIODIC]
        if wait and not self._wait(lambda: all(self._pstats.get(n, {}).get("count") for n in wait), timeout):
            missing = [n for n in wait if not self._pstats.get(n, {}).get("count")]
            st = (self._request("stats", None, 1.0) or {}).get("sensors", {})
            acked = [n for n in missing if st.get(n, {}).get("applied_us")]
            hogs = [n for n in self._config if n == "girv" or n.startswith("raw_")]
            raise HardwareError("BNO085: no data from %s after enable%s" % (", ".join(missing),
                                "; the hub accepted %s but delivers nothing%s" % (
                                    ", ".join(acked), " (starved by %s?)" % ", ".join(hogs) if hogs else "")
                                if acked else "; the hub did not acknowledge it"))
        return out

    DEFAULT_SENSORS = ("rv", "game_rv", "accel", "gyro", "mag", "linear_accel", "gravity")

    def enable_default(self, rate_hz=100):
        """The recommended setup: orientation + motion at 100 Hz each (~700 reports/s total),
        well inside what the hub sustains, so every sample arrives even with the Pi under load."""
        return self.enable(*self.DEFAULT_SENSORS, rate_hz=rate_hz)

    def enable_all(self, rate_hz=None, include_raw=False, include_events=True):
        names = [n for n in _PERIODIC if include_raw or not n.startswith("raw_")]
        if include_events:
            names += list(_EVENTS)
        return self.enable(*names, rate_hz=rate_hz)

    def disable(self, *names):
        """Disable sensors (all if none given)."""
        with self._lock:
            for n in names or list(self._config):
                self._config.pop(n, None)
                self._latest.pop(n, None)
            for b, c in list(self._config.items()):  # auto-enabled bases nobody needs any more
                if c[2] and not any(_PERIODIC.get(r, (0,) * 5)[4] == b for r in self._config):
                    self._config.pop(b)
                    self._latest.pop(b, None)
        self._send_config()

    def _send_config(self):
        with self._lock:
            cfg = {n: c[:2] for n, c in self._config.items()}
        self._request("config", cfg, timeout=4.0)

    @property
    def enabled(self):
        with self._lock:
            return {n: round(1e6 / c[0], 1) for n, c in self._config.items()}

    def read(self, name):
        """Latest Reading of a sensor, or None if none yet."""
        self._drain()
        return self._latest.get(name)

    def wait(self, name, timeout=1.0):
        """Block until a newer sample of `name` arrives; returns it (None on timeout)."""
        self._drain()
        old = self._latest.get(name)
        if self._wait(lambda: self._latest.get(name) is not old, timeout):
            return self._latest.get(name)
        return None

    def subscribe(self, *names, maxlen=100000):
        """Subscription receiving every sample of `names` (all sensors if none)."""
        sub = Subscription(self, names or SENSORS, maxlen)
        with self._lock:
            self._subs.append(sub)
        return sub

    def _unsubscribe(self, sub):
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)

    def on(self, name, callback):
        """callback(Reading) for every sample of `name`; runs in the draining thread (usually the
        driver's receiver thread). Keep it short."""
        with self._lock:
            self._callbacks[name].append(callback)

    def _vals(self, name, n=None):
        r = self.read(name)
        return None if r is None else r.values[:n] if n else r.values

    acceleration = property(lambda self: self._vals("accel"))
    gyro = property(lambda self: self._vals("gyro"))
    magnetic = property(lambda self: self._vals("mag"))
    linear_acceleration = property(lambda self: self._vals("linear_accel"))
    gravity = property(lambda self: self._vals("gravity"))
    quaternion = property(lambda self: self._vals("rv", 4))
    game_quaternion = property(lambda self: self._vals("game_rv", 4))
    geomagnetic_quaternion = property(lambda self: self._vals("geo_rv", 4))

    @property
    def euler(self):
        """(roll, pitch, yaw) in degrees from the rotation vector (game_rv if rv is off)."""
        q = self.quaternion or self.game_quaternion
        return None if q is None else quaternion_to_euler(q)

    @property
    def stability(self):
        r = self.read("stability")
        return None if r is None else STABILITY[min(r.values[0], 5)]

    @property
    def steps(self):
        r = self.read("step_counter")
        return None if r is None else r.values[0]

    # ---- hub commands --------------------------------------------------------------------------
    def product_id(self, timeout=3.0):
        """[{part, version, build, reset_cause}] for the 4 firmware components."""
        r = self._request("product_id", None, timeout)
        if r:
            self.product = r
        return r

    def read_frs(self, record, words=16, timeout=1.5):
        """First `words` 32-bit words of an FRS record, or None (empty / unknown record)."""
        return self._request("frs", (record, words), timeout)

    def metadata(self, name):
        """FRS metadata of a sensor: min_period_us / max_rate_hz / power_mA / raw words."""
        w = self.read_frs(FRS_METADATA[_ID[name]], words=16)
        if not w:
            return None
        return {"min_period_us": w[4], "max_rate_hz": round(1e6 / w[4], 1) if w[4] else None,
                "power_mA": round((w[3] & 0xFFFF) / 1024, 3) if w[3] is not None else None, "words": w}

    def command(self, cmd, params=(), response=True, timeout=2.5):
        """SH-2 command (0xF2). Returns the response's R0..R10 bytes (or None)."""
        r = self._request("command", (cmd, list(params), response), timeout)
        return None if r is None or not response else bytes(r)

    def tare(self, basis=TARE_GAME_RV, axes=0x07):
        """Tare now: current orientation becomes the reference (axes bit0 x, 1 y, 2 z)."""
        self.command(0x03, [0x00, axes, basis], response=False)

    def persist_tare(self):
        self.command(0x03, [0x01], response=False)

    def clear_tare(self):
        """Reorientation to identity (undoes tare until the next reset)."""
        self.command(0x03, [0x02, 0, 0, 0, 0, 0, 0, 0x00, 0x40], response=False)   # (0, 0, 0, 1) Q14

    def calibration(self):
        """Dynamic calibration enables {accel, gyro, mag, planar} as reported by the hub."""
        r = self.command(0x07, [0, 0, 0, 0x01])
        return None if r is None else {"status": r[0], "accel": bool(r[1]), "gyro": bool(r[2]),
                                       "mag": bool(r[3]), "planar": bool(r[4])}

    def set_calibration(self, accel=True, gyro=True, mag=True, planar=False):
        """Dynamic calibration enables. The hub keeps them in RAM only (boots with gyro off), so the
        driver re-applies them after every hub reset and worker restart. True on success."""
        flags = [int(accel), int(gyro), int(mag), int(planar)]
        self._wcfg["me_cal"] = flags         # a respawned worker gets them in its start config
        r = self._request("me_cal", flags, timeout=2.5)
        return r is not None and r[0] == 0

    def save_calibration(self):
        """Save the dynamic calibration data (DCD) to the hub's flash. True on success."""
        r = self.command(0x06)
        return r is not None and r[0] == 0

    def set_periodic_calibration_save(self, enabled):
        self.command(0x09, [0 if enabled else 1], response=False)

    def reset(self):
        """Hardware reset of the hub; the configuration is re-applied automatically."""
        return self._request("reset", None, timeout=5.0)

    def _inject(self, fault):
        """Test hook: 'exec_reset' (hub resets itself) or 'rst_glitch' (RST pulsed behind the
        driver's back)."""
        return self._request("inject", fault, timeout=2.0)

    # ---- monitoring -------------------------------------------------------------------------------
    def stats(self):
        """{'sensors': {name: asked/applied/delivered Hz, samples, lost, age}, 'link': counters}."""
        w = self._request("stats", None, timeout=1.0) or {"sensors": {}, "link": {}}
        self._drain()
        now = time.monotonic()
        out = {}
        with self._lock:
            for n, c in self._config.items():
                ws = w["sensors"].get(n, {})
                p = self._pstats.get(n, {})
                win = p.get("win") or ()
                rate = (len(win) - 1) / (win[-1] - win[0]) if len(win) > 1 and win[-1] > win[0] else 0.0
                out[n] = {"asked_hz": round(1e6 / c[0], 1),
                          "applied_hz": round(1e6 / ws["applied_us"], 1) if ws.get("applied_us") else None,
                          "rate_hz": round(rate, 1), "samples": p.get("count", 0), "lost": ws.get("lost", 0),
                          "settle_gaps": ws.get("settle_gaps", 0),
                          "age_s": round(now - p["last"], 3) if p.get("last") else None}
            link = dict(w["link"])
            link.update(self.counters)
            link["uptime_s"] = round(now - self.started, 1)
            link["rt_priority"] = self.rt_priority
        return {"sensors": out, "link": link}

    def status(self):
        st = self.stats()
        return {"available": self.available(), "spi": self.spi_dev, "int_pin": self.int_pin,
                "rst_pin": self.rst_pin, "product": self.product, "enabled": self.enabled,
                "link": st["link"], "last_error": self.last_error}

    def close(self):
        if self._closed:
            return
        try:
            with self._send_lock:
                self._conn.send((0, "close", None))
        except Exception:                            # noqa: BLE001
            pass
        self._closed = True
        try:
            self._proc.wait(2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(1.0)
        for c in (self._conn.close, lambda: os.close(self._notify_r), self._ring.close):
            try:
                c()
            except Exception:                        # noqa: BLE001
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:                            # noqa: BLE001
            pass

    def _interval_for(self, name, rate_hz):
        if name in _EVENTS:
            return _EVENTS[name][1] if rate_hz is None else max(1000, int(1e6 / rate_hz))
        mx = _PERIODIC[name][3]
        return mx if rate_hz is None else max(mx, int(round(1e6 / rate_hz)))


# =================================================================================================
# Worker process: the SHTP engine
# =================================================================================================
class _Hub:
    def __init__(self, cfg, conn, notify_fd, ring):
        import gpiod
        import spidev
        from gpiod.line import Bias, Direction, Edge, Value
        self._V = Value
        self.rst_pin, self.int_pin = cfg["rst_pin"], cfg["int_pin"]
        self._gpio = gpiod.request_lines(cfg["gpiochip"], consumer="bno085", config={
            self.rst_pin: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
            self.int_pin: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP,
                                             edge_detection=Edge.FALLING)})
        self._spi = spidev.SpiDev()
        self._spi.open(cfg["spi_bus"], cfg["spi_cs"])
        self._spi.mode = 3
        self._spi.max_speed_hz = cfg["spi_hz"]
        self.conn, self.notify_fd, self.ring = conn, notify_fd, ring
        self.cap = cfg["ring_records"]
        self.widx = 0
        self.config = {}                     # name -> (interval, batch)
        self.sstate = {}                     # name -> {"seq", "lost", "count", "applied_us"}
        self.counters = collections.Counter()
        self.lat = collections.deque(maxlen=4096)
        self.pending = []                    # [rid, deadline, check() -> (done, result)]
        self.report_len = dict(_DEFAULT_LEN)
        self.txq = collections.deque()
        self.tx_seq = [0] * 6
        self.cmd_seq = 0
        self.me_cal = cfg.get("me_cal")      # [accel, gyro, mag, planar] re-sent after every boot
        self.product = []
        self.frs = {}
        self.responses = collections.deque(maxlen=64)
        self.closing = False
        self.ctl_ready = False
        self.last_error = None
        tr = os.environ.get("BNO085_TRACE")       # debugging: log every transfer to this file
        self.trace = open(tr, "w", buffering=1) if tr else None
        self._ctl_poll = select.poll()
        self._ctl_poll.register(conn.fileno(), select.POLLIN)
        self._both_poll = select.poll()
        self._both_poll.register(conn.fileno(), select.POLLIN)
        self._both_poll.register(self._gpio.fd, select.POLLIN)
        self._reset_state()

    def _reset_state(self):
        self.ready = False
        self.booting = False
        self.advert = False
        self.init_seen = False
        self.exec_reset = False
        self.partial = {}
        self.rx_seq = {}
        self.read_size = 64
        self.recent_len = []
        self.out = None                  # the one response-generating request in flight
        self.quiet_until = 0.0
        self.settle_until = time.monotonic() + SETTLE
        self.last_ping = 0.0
        self.last_data = 0.0
        self.last_xfer = 0.0
        self.prev_xfer_start = 0.0
        self.last_edge = None
        self.request_reset = None
        self.cfg_time = time.monotonic()
        self.chain_until = 0.0
        self.notify_due = False

    # ---- main loop ---------------------------------------------------------------------------
    def run(self):
        # Housekeeping is rate-limited so a transfer costs as few syscalls as possible: under
        # memory-bandwidth pressure every syscall and Python op gets several times slower.
        next_ctl = next_ppid = last_notify = 0.0
        while not self.closing:
            try:
                now = time.monotonic()
                if now >= next_ctl or self.ctl_ready:
                    self.ctl_ready = False
                    next_ctl = now + CTL_PERIOD
                    self._handle_ctl()
                    if self.closing:
                        break
                self._step()
                if self.pending:
                    self._check_pending()
                if self.notify_due and now - last_notify >= NOTIFY_PERIOD:
                    self.notify_due = False
                    last_notify = now
                    try:
                        os.write(self.notify_fd, b"\0")
                    except BlockingIOError:
                        pass
                if now >= next_ppid:
                    next_ppid = now + 1.0
                    if os.getppid() != self.ppid:
                        return
            except (EOFError, BrokenPipeError, ConnectionResetError):
                return                           # parent gone
            except Exception as e:               # noqa: BLE001 - SPI/GPIO/boot trouble: start over
                self.counters["errors"] += 1
                self.last_error = "%s: %r" % (time.strftime("%H:%M:%S"), e)
                self.ready = False
                time.sleep(0.1)
                self.request_reset = "error"

    def _handle_ctl(self):
        while self.conn.poll(0):
            rid, op, args = self.conn.recv()
            self._op(rid, op, args)

    def _reply(self, rid, ok, result):
        self.conn.send((rid, ok, result))

    def _op(self, rid, op, args):
        now = time.monotonic()
        if op == "close":
            self.closing = True
        elif op == "config":
            new = {n: tuple(v) for n, v in args.items()}
            for n in list(self.config):
                if n not in new:
                    self.config.pop(n)
                    self.sstate.pop(n, None)
                    self._queue(CH_CONTROL, struct.pack("<BBBHIII", 0xFD, _ID[n], 0, 0, 0, 0, 0),
                                ("feature", _ID[n]))
            for n, v in new.items():
                if self.config.get(n) != v:
                    self.config[n] = v
                    self.sstate[n] = {"seq": None, "lost": 0, "settle_gaps": 0, "count": 0, "applied_us": None}
                    self._send_feature(n)
            self.settle_until = now + SETTLE
            self.pending.append([rid, now + 3.0, lambda: (not any(k[0] == "feature" for _, _, k in self.txq), True)])
        elif op == "stats":
            lat = sorted(self.lat)
            link = dict(self.counters)
            link["int_latency_us"] = ({"p50": round(lat[len(lat) // 2] * 1e6), "p99": round(lat[int(len(lat) * .99)] * 1e6),
                                       "max": round(lat[-1] * 1e6)} if lat else None)
            link["worker_last_error"] = self.last_error
            self._reply(rid, True, {"sensors": {n: dict(s) for n, s in self.sstate.items()}, "link": link})
        elif op == "product_id":
            self.product = []
            self._queue(CH_CONTROL, bytes([0xF9, 0]), ("product",))
            self.pending.append([rid, now + 2.5, lambda: (len(self.product) >= 4, list(self.product))])
        elif op == "frs":
            record, words = args
            self.frs.pop(record, None)
            self._queue(CH_CONTROL, struct.pack("<BBHHH", 0xF4, 0, 0, record, words), ("frs", record))

            def check(record=record, words=words):
                rec = self.frs.get(record)
                if not rec or rec["status"] is None:
                    return False, None
                ok = rec["status"] in (3, 6, 7)
                return True, [rec["words"].get(i) for i in range(words)] if ok else None
            self.pending.append([rid, now + 1.2, check])
        elif op == "command":
            cmd, params, response = args
            params = list(params) + [0] * (9 - len(params))
            seq = self.cmd_seq
            self.cmd_seq = (seq + 1) & 0xFF
            self._queue(CH_CONTROL, bytes([0xF2, seq, cmd] + params), ("cmd", cmd) if response else ("free", rid))
            if response:
                def check(cmd=cmd, seq=seq):
                    for r in self.responses:
                        if r[0] == cmd and r[1] == seq:
                            return True, list(r[2])
                    return False, None
                self.pending.append([rid, now + 2.0, check])
        elif op == "me_cal":
            self.me_cal = list(args)
            seq = self._queue_me_cal()

            def check(seq=seq):
                for r in self.responses:
                    if r[0] == 0x07 and r[1] == seq:
                        return True, list(r[2])
                return False, None
            self.pending.append([rid, now + 2.0, check])
        elif op == "reset":
            self.request_reset = "user"
            self.pending.append([rid, now + 4.0, lambda: (self.ready and not self.request_reset, True)])
        elif op == "inject":
            if args == "exec_reset":
                self._queue(CH_EXEC, bytes([0x01]), ("reset",))
            elif args == "rst_glitch":
                self._gpio.set_value(self.rst_pin, self._V.INACTIVE)
                time.sleep(0.02)
                self._gpio.set_value(self.rst_pin, self._V.ACTIVE)
            self._reply(rid, True, None)
        elif op == "raw":                    # debugging: (channel, hex payload, answer kind tuple)
            ch, payload, kind = args
            self._queue(ch, bytes.fromhex(payload), tuple(kind))
            self._reply(rid, True, None)
        else:
            self._reply(rid, False, "unknown op %r" % op)

    def _check_pending(self):
        if not self.pending:
            return
        now = time.monotonic()
        keep = []
        for p in self.pending:
            done, res = p[2]()
            if done:
                self._reply(p[0], True, res)
            elif now > p[1]:
                self._reply(p[0], True, None)
            else:
                keep.append(p)
        self.pending = keep

    # ---- boot / recovery -------------------------------------------------------------------
    def boot(self, why):
        """Hard reset, wait for the hub to initialise, re-apply the configuration."""
        self._reset_state()
        self.booting = True
        self.tx_seq = [0] * 6
        self.txq.clear()                     # stale writes; the config is re-sent below
        for s in self.sstate.values():
            s["seq"] = None
        if why != "start":
            self.counters["resets_" + why] += 1
        g = self._gpio
        g.set_value(self.rst_pin, self._V.INACTIVE)
        time.sleep(0.002)
        self._spi.xfer2([0])                 # SCK to its mode-3 idle level while the chip ignores it
        time.sleep(0.01)
        while g.wait_edge_events(0):
            g.read_edge_events()
        g.set_value(self.rst_pin, self._V.ACTIVE)
        end = time.monotonic() + 2.0
        while time.monotonic() < end and not (self.init_seen and self.advert and self.exec_reset):
            if self._wait_int(0.05):
                self._transfer()
        self.booting = False
        if not (self.init_seen and self.advert and self.exec_reset):
            raise HardwareError("no SHTP traffic after reset (INT %s)" % ("low" if self._int_low() else "high"))
        for n in self.config:
            self._send_feature(n)
        if self.me_cal:
            self._queue_me_cal()
        self.ready = True

    def _queue_me_cal(self):
        """Configure ME Calibration (command 0x07, subcommand 0 = configure); returns its seq."""
        a, g, m, pl = self.me_cal
        seq = self.cmd_seq
        self.cmd_seq = (seq + 1) & 0xFF
        self._queue(CH_CONTROL, bytes([0xF2, seq, 0x07, a, g, m, 0x00, pl, 0, 0, 0, 0]), ("cmd", 0x07))
        return seq

    def _send_feature(self, name):
        self.cfg_time = time.monotonic()
        iv, batch = self.config[name]
        self._queue(CH_CONTROL, struct.pack("<BBBHIII", 0xFD, _ID[name], 0, 0, iv, batch, 0),
                    ("feature", _ID[name]))

    def _queue(self, ch, payload, kind):
        """kind = what answers the packet: ("ping",) ("feature", id) ("product",) ("frs", rec)
        ("cmd", cmd), ("reset",) (the hub reboots and talks again) or ("free",) (nothing)."""
        self.txq.append((ch, bytes(payload), kind))

    # ---- link ------------------------------------------------------------------------------
    def _int_low(self):
        return self._gpio.get_value(self.int_pin) == self._V.INACTIVE

    def _drain_edges(self):
        g = self._gpio
        if g.wait_edge_events(0):
            for ev in g.read_edge_events():
                self.last_edge = ev.timestamp_ns * 1e-9

    def _wait_int(self, timeout):
        if self._int_low():
            return True
        if self._gpio.wait_edge_events(timeout):
            self._drain_edges()
            return self._int_low()
        return False

    def _careful(self, now):
        """Data may stop at any moment (idle, booting, right after a write): protect the chain."""
        return not self.ready or now < self.chain_until or now - self.last_data > DATA_IDLE

    def _step(self):
        now = time.monotonic()
        if self.request_reset:
            why = self.request_reset
            self.boot(why)
            return
        periodic = any(n in _PERIODIC for n in self.config)
        if periodic and self.ready and now - max(self.last_data, self.cfg_time) > WATCHDOG:
            self.boot("watchdog")            # hub silent although it should be streaming
            return
        if self._int_low():
            if self.partial and now < self.quiet_until and self._careful(now):
                time.sleep(self.quiet_until - now)   # the anchor read must carry a ping: wait
                return
            if (self.ready and CH_CONTROL in self.partial and len(self.partial) == 1
                    and self._careful(now) and not self.txq and now - self.last_xfer < IDLE_HOLD
                    and (self.out is None or self._completes(CH_CONTROL))):
                # idle: the pending INT is an anchor (rest of a peeked control packet, i.e. a ping
                # answer). Leave it unread until a command arrives or the hold expires.
                if self._ctl_poll.poll(max(1, int((IDLE_HOLD - (now - self.last_xfer)) * 1000))):
                    self.ctl_ready = True
                return
            self._transfer()
            return
        if self.txq and now - self.last_xfer > STALL:
            self.boot("stall")               # writes pending but nothing will ever assert INT
            return
        # sleep until INT or a command
        for fd, _ in self._both_poll.poll(10 if self.txq else 100):
            if fd == self._gpio.fd:
                self._drain_edges()
            else:
                self.ctl_ready = True

    def _completes(self, ch):
        """Is the pending anchor on `ch` the (last) answer to the request in flight?"""
        p = self.partial.get(ch)
        o = self.out
        if p is None or o is None or ch != CH_CONTROL or len(p[1]) < 3:
            return o is not None and o[0] == "reset"
        rid, b1, b2 = p[1][0], p[1][1], p[1][2]
        k = o[0]
        return ((k == "ping" and rid == 0xFC and b1 == PING_ID) or
                (k == "feature" and rid == 0xFC and b1 == o[1]) or
                (k == "product" and rid == 0xF8 and len(self.product) + p[0] // 16 >= 4) or
                (k == "frs" and rid == 0xF3 and (b1 & 0x0F) != 0) or
                (k == "cmd" and rid == 0xF1 and (b2 & 0x7F) == o[1]))

    def _pick_tx(self, now, anchor_ch=None):
        """What rides on this transfer: a queued write, a ping, or nothing.

        anchor_ch: this transfer reads the rest of a peeked packet while data may stop, so
        something must be left in flight afterwards.
        """
        if now < self.quiet_until:
            return None                     # let the hub digest a command nobody answers
        o = self.out
        if o is not None and now - o[-1] > OUT_TIMEOUT.get(o[0], 1.0):
            self.counters["timeout_" + o[0]] += 1           # the answer never came
            self.out = o = None
        free = o is None or (anchor_ch is not None and self._completes(anchor_ch))
        if self.txq and self.ready and self.txq[0][2][0] == "free" and o is None and anchor_ch is None:
            # nothing answers it: only while data flows and no answer is pending (a control
            # write would make the hub drop it); when careful see the head-read case in _transfer
            self.quiet_until = now + FREE_QUIET
            ch, payload, kind = self.txq.popleft()
            self._sent_free(kind)
            return self._packet(ch, payload)
        if not free:
            return None                                    # its answer keeps INT coming
        if self.txq and self.ready and self.txq[0][2][0] != "free":
            ch, payload, kind = self.txq.popleft()
            # while data flows it keeps INT coming, so a Set Feature need not wait for its ack
            # (in batched mode acks can take >0.5 s); when careful the ack is the next INT
            self.out = None if kind[0] == "feature" and anchor_ch is None else tuple(kind) + (now,)
            if kind[0] != "ping":
                self.chain_until = now + CHAIN_HOLD        # data may stop (disable, reset...)
            return self._packet(ch, payload)
        if not self.advert:
            return None
        if anchor_ch is not None or now - self.last_ping > PING_PERIOD:
            self.last_ping = now
            self.out = ("ping", now)
            return self._packet(CH_CONTROL, bytes([0xFE, PING_ID]))
        return None

    def _sent_free(self, kind):
        if len(kind) > 1:
            self._reply(kind[1], True, None)

    def _packet(self, ch, payload):
        hdr = struct.pack("<HBB", len(payload) + 4, ch, self.tx_seq[ch])
        self.tx_seq[ch] = (self.tx_seq[ch] + 1) & 0xFF
        return hdr + payload

    def _transfer(self):
        self._drain_edges()
        t = time.monotonic()
        # INT for this packet fell after the previous transfer started (its CS cleared the old one)
        fresh = self.last_edge is not None and self.last_edge > self.prev_xfer_start
        t_int = self.last_edge if fresh else t
        if fresh:
            self.lat.append(t - self.last_edge)
        self.prev_xfer_start = t
        careful = self._careful(t)
        sent = None
        if careful and not self.partial:
            # new packet while data may stop: read only its head. The rest stays pending as a
            # continuation ("anchor"), so INT is guaranteed to come back; the anchor is then read
            # together with the next request, so something is always in flight.
            tx, n = None, PEEK
            if (self.txq and self.ready and self.txq[0][2][0] == "free" and self.out
                    and self.out[0] == "ping" and len(self.txq[0][1]) + 4 < 21):
                # a command nobody answers rides on the head of the ping's answer (21 bytes):
                # the read is just long enough for the command, the rest stays as the anchor
                ch, payload, kind = self.txq.popleft()
                self._sent_free(kind)
                tx = self._packet(ch, payload)
                n = len(tx)
                self.quiet_until = t + FREE_QUIET
        else:
            anchor = next(iter(self.partial)) if careful and self.partial else None
            before = self.out
            tx = self._pick_tx(t, anchor)
            if self.out is not before:
                sent = self.out                # a request went out with the read of the previous
                                               # answer: parsing that answer must not clear it
            n = self.read_size
            if self.partial:                   # continuation: read exactly what is left
                n = max(16, min(n, 4 + max(p[0] - len(p[1]) for p in self.partial.values())))
        if tx is not None:
            n = max(n, len(tx))
            rx = self._spi.xfer2(list(tx) + [0] * (n - len(tx)))
            self.counters["writes"] += 1
        else:
            rx = self._spi.readbytes(n)
        self.last_xfer = time.monotonic()
        if self.trace:
            self.trace.write("%.4f n=%d tx=%s rx=%s out=%s careful=%d partial=%s\n" % (
                t, n, tx[4:7].hex() if tx else "-", bytes(rx[:min(n, 8)]).hex(), self.out and self.out[0],
                careful, list(self.partial)))
        self.counters["transfers"] += 1
        self.counters["bytes"] += n
        self._rx(rx, n, t_int)
        if sent is not None and self.out is None:
            self.out = sent

    def _rx(self, rx, n, t):
        length = rx[0] | (rx[1] << 8)
        if length in (0, 0xFFFF):
            self.counters["empty_reads"] += 1
            return
        cont = bool(length & 0x8000)
        length &= 0x7FFF
        ch, seq = rx[2], rx[3]
        if ch > 5 or length < 4:
            self.counters["bad_headers"] += 1
            return
        last = self.rx_seq.get(ch)
        if last is not None and ((seq - last) & 0xFF) != 1:
            self.counters["shtp_lost"] += (seq - last - 1) & 0xFF
        self.rx_seq[ch] = seq
        got = min(length, n)
        body = bytes(rx[4:got])
        if not cont:
            if ch in self.partial:
                self.counters["dropped_partial"] += 1
                del self.partial[ch]
            self.recent_len.append(length)
            if len(self.recent_len) >= 16:     # size reads to cover recent packets in one go
                self.read_size = min(512, max(32, (max(self.recent_len) + 15) & ~15))
                self.recent_len = []
            if got == length:
                self._dispatch(ch, body, t, seq)
            else:
                self.partial[ch] = [length - 4, bytearray(body), t, seq]
        else:
            p = self.partial.get(ch)
            if p is None:
                self.counters["orphan_continuations"] += 1
                return
            p[1] += body
            if len(p[1]) >= p[0]:
                del self.partial[ch]
                self._dispatch(ch, bytes(p[1][:p[0]]), p[2], p[3])

    def _dispatch(self, ch, d, t, seq=0):
        self.counters["packets"] += 1
        if ch == CH_COMMAND:
            if d and d[0] == 0x00:
                self._parse_advert(d[1:])
                if not self.booting and self.ready and not self.request_reset and seq == 0:
                    self.request_reset = "unexpected"      # the hub rebooted by itself
                self.advert = True
            elif d and d[0] == 0x01:
                self.counters["shtp_errors"] += 1
        elif ch == CH_EXEC:
            if d and d[0] == 0x01:
                if self.booting:
                    self.exec_reset = True
                elif not self.request_reset and seq == 0:   # the hub reset by itself
                    self.request_reset = "unexpected"
        elif ch == CH_CONTROL:
            self._control(d)
        elif ch in (CH_REPORTS, CH_WAKE):
            self.last_data = time.monotonic()
            self._reports(d, t)
        elif ch == CH_GIRV:
            self.last_data = time.monotonic()
            s = self.sstate.get("girv")
            for off in range(0, len(d) - 13, 14):
                v = struct.unpack_from("<7h", d, off)
                vals = [x / 16384.0 for x in v[:4]] + [x / 1024.0 for x in v[4:]]
                cnt = s["count"] if s else 0
                if s:
                    s["count"] += 1
                self._emit(0x2A, 3, cnt & 0xFF, t, vals)

    def _parse_advert(self, data):
        i = 0
        while i + 2 <= len(data):
            tag, ln = data[i], data[i + 1]
            val = data[i + 2:i + 2 + ln]
            i += 2 + ln
            if tag == 0x81:
                for j in range(0, len(val) - 1, 2):
                    self.report_len[val[j]] = val[j + 1]

    def _control(self, d):
        if not d:
            return
        rid = d[0]
        if rid == 0xFC and len(d) >= 17:
            sid, interval = d[1], struct.unpack_from("<I", d, 5)[0]
            o = self.out
            if o and ((o[0] == "ping" and sid == PING_ID) or (o[0] == "feature" and sid == o[1])):
                self.out = None
            name = _NAME.get(sid)
            if name in self.sstate and interval:
                self.sstate[name]["applied_us"] = interval
        elif rid == 0xF1 and len(d) >= 16:
            if d[2] == 0x84:                    # initialize (also unsolicited after reset)
                self.init_seen = True
            if self.out and self.out[0] == "cmd" and (d[2] & 0x7F) == self.out[1]:
                self.out = None
            self.responses.append((d[2] & 0x7F, d[3], bytes(d[5:16])))
        elif rid == 0xF8 and len(d) >= 16:
            for off in range(0, len(d) - 15, 16):
                _, cause, vmaj, vmin, part, build, patch = struct.unpack_from("<BBBBIIH", d, off)
                self.product.append({"part": part, "version": "%d.%d.%d" % (vmaj, vmin, patch),
                                     "build": build, "reset_cause": cause})
            if self.out and self.out[0] == "product" and len(self.product) >= 4:
                self.out = None
        elif rid == 0xF3 and len(d) >= 16:      # FRS read response
            ln, status = d[1] >> 4, d[1] & 0x0F
            off, w0, w1, rec_id = struct.unpack_from("<HIIH", d, 2)
            rec = self.frs.setdefault(rec_id, {"words": {}, "status": None, "frags": 0})
            for k, w in enumerate((w0, w1)[:ln]):
                rec["words"][off + k] = w
            rec["frags"] += 1
            if self.out and self.out[0] == "frs" and self.out[1] == rec_id:
                self.out = ("frs", rec_id, time.monotonic()) if not status else None   # still alive
            if status:
                rec["status"] = status

    def _reports(self, d, t):
        i, base, n = 0, 0.0, len(d)
        while i < n:
            rid = d[i]
            if rid == 0xFB:                     # base timestamp: signed delta back from INT, 100 us
                base = struct.unpack_from("<i", d, i + 1)[0] * 100e-6
                i += 5
                continue
            if rid == 0xFA:                     # timestamp rebase
                base -= struct.unpack_from("<i", d, i + 1)[0] * 100e-6
                i += 5
                continue
            ln = self.report_len.get(rid)
            if not ln or i + ln > n:
                self.counters["unknown_reports"] += 1
                break
            name = _NAME.get(rid)
            if name:
                seq, status = d[i + 1], d[i + 2]
                ts = t - base + (((status >> 2) << 8) | d[i + 3]) * 100e-6
                s = self.sstate.get(name)
                if s is not None:
                    if s["seq"] is not None and name in _PERIODIC:   # events skip seq numbers
                        gap = (seq - s["seq"] - 1) & 0xFF
                        if gap:
                            s["settle_gaps" if t < self.settle_until else "lost"] += gap
                    s["seq"] = seq
                    s["count"] += 1
                self._emit(rid, status & 3, seq, ts, self._decode(name, rid, d, i))
            i += ln

    @staticmethod
    def _decode(name, rid, d, i):
        p = _PERIODIC.get(name)
        if p:
            nf, q = p[1], p[2]
            v = struct.unpack_from("<%dh" % nf, d, i + 4)
            if q == 0:
                return list(v)
            if nf == 5:                         # quaternion Q14 + heading accuracy Q12
                return [x / 16384.0 for x in v[:4]] + [v[4] / 4096.0]
            sc = 1.0 / (1 << q)
            return [x * sc for x in v]
        if rid == 0x10:                         # tap: flags x, x+, y, y+, z, z+, double
            return [d[i + 4]]
        if rid == 0x11:                         # step counter: steps
            return [struct.unpack_from("<H", d, i + 8)[0]]
        if rid in (0x12, 0x19):                 # significant motion / shake bits
            return [struct.unpack_from("<H", d, i + 4)[0]]
        if rid == 0x13:                         # stability classification
            return [d[i + 4]]
        if rid == 0x18:                         # step detector latency (us)
            return [struct.unpack_from("<I", d, i + 4)[0]]
        return [0]

    def _emit(self, rid, acc, seq, ts, vals):
        vals = list(vals) + [0.0] * (7 - len(vals))
        off = _RHDR + (self.widx % self.cap) * _RSZ
        _REC.pack_into(self.ring, off + 8, rid, acc, seq, ts, *vals[:7])
        self.widx += 1
        _STAMP.pack_into(self.ring, off, self.widx)          # publish after the body
        _STAMP.pack_into(self.ring, 0, self.widx)
        self.notify_due = True


def _worker_main(cfg):
    from multiprocessing.connection import Connection
    signal.signal(signal.SIGINT, signal.SIG_IGN)            # Ctrl-C belongs to the parent
    conn = Connection(cfg["ctl_fd"])
    ring = mmap.mmap(cfg["ring_fd"], _RHDR + cfg["ring_records"] * _RSZ)
    rt = False
    if cfg.get("realtime"):
        try:   # needs CAP_SYS_NICE or an rtprio limit (scripts/setup_spi0_mode3.sh)
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(20))
            rt = True
        except (PermissionError, OSError, AttributeError):
            pass
    try:
        hub = _Hub(cfg, conn, cfg["notify_fd"], ring)
        hub.ppid = cfg["ppid"]
        hub.boot("start")
    except Exception as e:                                   # noqa: BLE001
        conn.send((0, False, "%s: %s" % (type(e).__name__, e)))
        return 1
    conn.send((0, True, {"rt": rt}))
    hub.run()
    return 0


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "--worker":
    sys.exit(_worker_main(json.loads(sys.argv[2])))
