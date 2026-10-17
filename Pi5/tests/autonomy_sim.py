#!/usr/bin/env python3
"""Autonomy safety logic of drive_server.Control against a simulated car (no hardware, runs anywhere).

    python3 tests/autonomy_sim.py

Scenarios (each must pass):
  1 armed + plan 1.0 m/s       -> drives, throttle never above the page's level (0.23)
  2 plan target drops to 0.5   -> brakes (negative throttle) until near 0.5 m/s
  3 SPACE released             -> brakes to a stop, then neutral
  4 arming messages stop (WiFi)-> brakes to a stop within ~ARM_S + braking
  5 planner goes quiet         -> brakes to a stop
  6 planner says path lost (0) -> brakes to a stop
  7 manual W/A/S/D while armed -> manual command applied at once, autonomy off
  8 plan without arming        -> nothing moves
  9 a second page (driving enabled, SPACE not held: "not armed" + neutral commands at 20 Hz)
                               -> cannot cancel the page that holds SPACE
 10 smooth throttle            -> steady speed near the target, few throttle reversals
"""
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import drive_server as ds  # noqa: E402

MM_PER_TURN = 55.0


class FakeCar:
    """Speed follows throttle (full throttle ~5 m/s, 0.23 -> ~1.15 m/s), brakes at 3 m/s^2 per unit."""

    class _P:
        position = 0.0
        current_us = 1500.0

    def __init__(self):
        self.v = 0.0                    # mm/s
        self.t = 0.0
        self.steering, self.esc = self._P(), self._P()
        self.turns = 0.0
        self.max_t_seen = 0.0
        self.min_t_seen = 0.0
        self.log = []                   # (time, throttle)
        self._run = True
        threading.Thread(target=self._sim, daemon=True).start()

    def forward(self, f):
        self.t = f
        self.log.append((time.monotonic(), f))
        self.max_t_seen = max(self.max_t_seen, f)

    def brake(self, b):
        self.t = -b
        self.log.append((time.monotonic(), -b))
        self.min_t_seen = min(self.min_t_seen, -b)

    def stop(self):
        self.t = 0.0

    def steer(self, s):
        self.steering.position = s

    @property
    def rps(self):
        return -self.v / MM_PER_TURN    # this car's encoder counts forward as negative

    def _sim(self):
        dt = 0.005
        while self._run:
            if self.t > 0:
                self.v += (self.t * 5000 - self.v) * 3.0 * dt
            elif self.t < 0:
                self.v = max(0.0, self.v + self.t * 3000 * dt)
            else:
                self.v = max(0.0, self.v - 300 * dt)          # rolling resistance
            self.turns += self.v * dt / MM_PER_TURN
            time.sleep(dt)


class State:
    a = np.zeros(len(ds.FIELDS))


def run():
    car = FakeCar()
    ctl = ds.Control(car, State())
    ctl.mm_per_turn = MM_PER_TURN
    ds._set_rt = lambda name: False
    for f in (ctl.auto_loop, ctl.watchdog):
        threading.Thread(target=f, daemon=True).start()
    level = 0.23
    stop_feed = {"arm": False, "plan": False, "page2": False}
    plan = {"s": 0.2, "v": 1000.0, "m": "gates"}
    held = {"space": False}

    def page():                         # the page's worker: 20 Hz arm messages while SPACE is held
        while True:
            if not stop_feed["arm"]:
                ctl.arm("ws", held["space"], level)
            time.sleep(0.05)

    def planner():                      # the map worker: 20 Hz plans
        while True:
            if not stop_feed["plan"]:
                ctl.set_plan(plan["s"], plan["v"], plan["m"])
            time.sleep(0.05)
    def page2():                        # a second tab: driving enabled, autonomous mode, SPACE not held
        while True:
            if stop_feed["page2"]:
                ctl.arm("ws", False, level, client=2)
                ctl.command("ws", 0.0, 0.0)
            time.sleep(0.05)
    threading.Thread(target=page, daemon=True).start()
    threading.Thread(target=page2, daemon=True).start()
    threading.Thread(target=planner, daemon=True).start()
    results = []

    def check(name, ok, info):
        results.append(ok)
        print("%-4s %-46s %s" % ("PASS" if ok else "FAIL", name, info))

    def wait_stop(timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if ctl.stopped and car.v < ds.STOP_MM_S and ctl.auto_state == ds.AUTO_OFF:
                return time.monotonic() - t0
            time.sleep(0.01)
        return None

    def throttle_stats(t0, t1):
        xs = [x for tt, x in car.log if t0 <= tt <= t1]
        rev = sum(1 for a, b in zip(xs, xs[1:]) if abs(a - b) > 0.05)
        return rev / max(t1 - t0, 1e-6), xs

    # 8 plan without arming
    time.sleep(0.5)
    check("8 plan, not armed: nothing moves", car.v == 0 and car.max_t_seen == 0, "v %.0f" % car.v)
    # 1 armed
    held["space"] = True
    time.sleep(2.5)
    check("1 armed: drives at ~1.0 m/s", 850 < car.v < 1200 and ctl.auto_state == ds.AUTO_DRIVING,
          "v %.0f mm/s, state %d" % (car.v, ctl.auto_state))
    check("1 throttle never above the page level", car.max_t_seen <= level + 1e-9, "max %.3f" % car.max_t_seen)
    # 2 slower target -> brakes
    car.min_t_seen = 0.0
    plan["v"] = 500.0
    time.sleep(1.5)
    check("2 target 0.5 m/s: brakes, then holds ~0.5", car.min_t_seen < 0 and 350 < car.v < 650,
          "min throttle %.2f, v %.0f" % (car.min_t_seen, car.v))
    # 3 space released
    plan["v"] = 1000.0
    time.sleep(2.0)
    held["space"] = False
    car.min_t_seen = 0.0
    dt = wait_stop(3.0)
    check("3 SPACE released: brakes to a stop", dt is not None and car.min_t_seen <= -ds.AUTO_STOP_BRAKE + 1e-9,
          "stopped in %s s, brake %.2f" % (dt and round(dt, 2), car.min_t_seen))
    # 4 arming lost (link drop)
    held["space"] = True
    time.sleep(2.0)
    stop_feed["arm"] = True
    dt = wait_stop(3.0)
    check("4 arm messages stop (WiFi): brakes to a stop", dt is not None, "stopped in %s s" % (dt and round(dt, 2)))
    stop_feed["arm"] = False
    # 5 planner quiet
    time.sleep(2.0)
    stop_feed["plan"] = True
    dt = wait_stop(3.0)
    check("5 planner quiet: brakes to a stop", dt is not None, "stopped in %s s" % (dt and round(dt, 2)))
    stop_feed["plan"] = False
    # 6 path lost
    time.sleep(2.0)
    plan["v"], plan["m"] = 0.0, "lost"
    dt = wait_stop(3.0)
    check("6 planner: path lost -> brakes to a stop", dt is not None, "stopped in %s s" % (dt and round(dt, 2)))
    plan["v"], plan["m"] = 1000.0, "gates"
    # 9 second page sending "not armed" + neutral commands
    held["space"] = True
    time.sleep(1.5)
    stop_feed["page2"] = True
    a = time.monotonic()
    n_bad = 0
    while time.monotonic() - a < 2.0:
        if ctl.auto_state != ds.AUTO_DRIVING:
            n_bad += 1
        time.sleep(0.02)
    stop_feed["page2"] = False
    check("9 second page cannot cancel autonomy", n_bad == 0 and car.v > 700,
          "%d samples not driving, v %.0f" % (n_bad, car.v))
    # 10 smooth throttle: 3 s at a 1.0 m/s target that the ceiling can reach
    time.sleep(1.0)
    a = time.monotonic()
    vs = []
    while time.monotonic() - a < 3.0:
        vs.append(car.v)
        time.sleep(0.02)
    rate, xs = throttle_stats(a, time.monotonic())
    check("10 smooth throttle: steady speed, few big steps", rate < 2.0 and 850 < np.mean(vs) < 1150 and np.std(vs) < 80,
          "big throttle steps %.1f/s, v %.0f +- %.0f mm/s, throttle %.2f..%.2f" % (rate, np.mean(vs), np.std(vs), min(xs), max(xs)))
    # 7 manual override while armed
    time.sleep(2.0)
    was = ctl.auto_state
    stop_feed["arm"] = True             # like the page: while A is held it sends manual commands, no arm
    ctl.command("ws", 0.1, -1.0)
    ok = ctl.active == "ws" and ctl.steer == -1.0 and ctl.auto_state == ds.AUTO_OFF
    for _ in range(4):                  # A still held
        ctl.command("ws", 0.1, -1.0)
        time.sleep(0.05)
    ok = ok and ctl.active == "ws" and ctl.steer == -1.0
    stop_feed["arm"] = False
    check("7 manual command while armed: takes over", was == ds.AUTO_DRIVING and ok,
          "active %s, steer %.1f, auto %d" % (ctl.active, ctl.steer, ctl.auto_state))
    # 11 autonomous mode, W pressed then released with SPACE not held ("not armed" messages only)
    held["space"] = False
    time.sleep(1.5)
    for _ in range(10):                 # W held 0.5 s: manual commands
        ctl.command("ws", level, 0.0)
        time.sleep(0.05)
    a = time.monotonic()
    for _ in range(6):                  # released: the page sends {"arm": false} only
        ctl.arm("ws", False, level)
        time.sleep(0.05)
    check("11 W released in autonomous mode: throttle off at once", car.t == 0.0 and ctl.throttle == 0.0,
          "throttle now %.2f" % car.t)
    # 12 W released, SPACE held: autonomy takes over at once (no 0.5 s of leftover W throttle)
    for _ in range(10):
        ctl.command("ws", level, -1.0)
        time.sleep(0.05)
    held["space"] = True
    time.sleep(0.15)
    check("12 W released + SPACE held: autonomy at once", ctl.active == "auto",
          "active %s after 0.15 s" % ctl.active)
    held["space"] = False
    car._run = False
    ctl._run = False
    print("ALL PASS" if all(results) else "SOME FAILED")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
