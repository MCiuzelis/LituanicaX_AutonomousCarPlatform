"""Run a driving algorithm on the simulated car.

    uv run python source/rc_sim/run.py                                # gate_follower, GUI + viewer
    uv run python source/rc_sim/run.py --headless --episodes 5        # batch, prints a summary
    uv run python source/rc_sim/run.py --algo manual --real-time      # drive by hand (WASD)
    uv run python source/rc_sim/run.py --param v_max=2.5 --set detector.fp_rate_per_frame=1.0
    uv run python source/rc_sim/run.py --reverse --laps 1 --set episode.spawn_x=1.5

Every run writes runs/<timestamp>_<algo>/: config.json, log.jsonl (one line per control step:
ground truth pose, command, what was delivered to the algorithm) and summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # source/

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LituanicaX car simulator: run a driving algorithm.")
parser.add_argument("--algo", default="gate_follower")
parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="algorithm parameter")
parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE", help="config override (see config.py)")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--laps", type=int, default=None)
parser.add_argument("--reverse", action="store_true", help="drive the track the other way round")
parser.add_argument("--no-view", action="store_true", help="no camera/map window")
parser.add_argument("--view-hz", type=float, default=20.0)
parser.add_argument("--record", action="store_true", help="save every viewer frame to runs/<run>/frames/")
parser.add_argument("--real-time", action="store_true", help="pace the loop to wall-clock time")
parser.add_argument("--log-sensors", action="store_true", help="also log every box / IMU / encoder sample")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")        # one car: CPU PhysX beats GPU PhysX (see config.SimCfg.device)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

# ---- everything below needs the running app ---------------------------------------------------------
import dataclasses                                   # noqa: E402

from rc_sim.algorithms import make                   # noqa: E402
from rc_sim.config import RUNS_DIR, SimCfg           # noqa: E402
from rc_sim.interface import Command, SensorFrame    # noqa: E402
from rc_sim.sim import CarSim                        # noqa: E402
from rc_sim.track import TrackMap                    # noqa: E402


def parse_kv(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def main():
    cfg = SimCfg(seed=args.seed)
    for k, v in parse_kv(args.set).items():
        cfg.set(k, v)
    if args.laps is not None:
        cfg.episode.laps = args.laps
    if args.reverse:
        cfg.episode.reverse = True

    track = TrackMap()
    cfg.device = args.device
    sim = CarSim(cfg, track, gui=not args.headless)
    algo = make(args.algo, **parse_kv(args.param))
    run_dir = os.path.join(RUNS_DIR, time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.algo}")
    os.makedirs(run_dir, exist_ok=True)
    viewer = None
    if not args.no_view or args.record:
        from rc_sim.viewer import Viewer
        from rc_sim.sim import lens_remap
        viewer = Viewer(track, sim.car_info.camera, show=not args.no_view, remap=lens_remap(sim.lens, sim.render_model),
                        record_dir=os.path.join(run_dir, "frames") if args.record else None)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"sim": cfg.as_dict(), "algo": args.algo, "params": algo.params, "argv": sys.argv}, f, indent=1)
    log = open(os.path.join(run_dir, "log.jsonl"), "w")
    results = []
    view_every = max(1, round(cfg.control_hz / args.view_hz))

    for ep in range(args.episodes):
        info = sim.reset(seed=args.seed + ep)
        algo.reset(info)
        if viewer:
            viewer.reset()
        sensors, cmd, k = SensorFrame(0.0), Command(), 0
        wall0 = time.monotonic()
        while app.is_running():
            cmd = algo.step(sensors)
            sensors, snap = sim.step(cmd)
            st = snap.status
            rec = {"ep": ep, "t": round(st.t, 4), "x": round(st.x, 4), "y": round(st.y, 4), "yaw": round(st.yaw, 4),
                   "v": round(st.speed, 3), "progress": round(st.progress, 3), "lateral": round(st.lateral, 3),
                   "cmd": [round(cmd.throttle, 3), round(cmd.steer, 3)],
                   "applied": [round(snap.applied.throttle, 3), round(snap.applied.steer, 3)],
                   "boxes": [len(d.detections) for d in sensors.detections]}
            if args.log_sensors:
                rec["sensors"] = dataclasses.asdict(sensors)
            log.write(json.dumps(rec) + "\n")
            if viewer and (k % view_every == 0 or st.done):
                viewer.update(snap, cmd, algo.debug(), extra=f"ep {ep + 1}/{args.episodes}")
            sim.chase_camera(st)
            if args.real_time:
                time.sleep(max(0.0, st.t - (time.monotonic() - wall0)))
            k += 1
            if st.done:
                break
        wall = time.monotonic() - wall0
        res = {"episode": ep, "reason": st.reason, "time_s": round(st.t, 2), "laps": st.laps,
               "lap_times_s": [round(x, 3) for x in st.lap_times], "progress_m": round(st.progress, 2),
               "track_m": round(track.length, 2), "wall_clock_s": round(wall, 1),
               "sim_speed_x": round(st.t / max(wall, 1e-6), 2)}
        results.append(res)
        print(f"[run] episode {ep}: {st.reason} after {st.t:.1f} s, {st.progress:.1f} m, laps {st.laps} "
              f"{res['lap_times_s']}  ({res['sim_speed_x']}x real time)")
        if not app.is_running():
            break

    log.close()
    summary = {"algo": args.algo, "episodes": results,
               "completed": sum(r["reason"] == "laps" for r in results),
               "crashes": sum(r["reason"] == "crash" for r in results)}
    laps = [t for r in results for t in r["lap_times_s"]]
    if laps:
        summary["best_lap_s"], summary["mean_lap_s"] = min(laps), round(sum(laps) / len(laps), 3)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[run] {summary['completed']}/{len(results)} completed, {summary['crashes']} crashes"
          + (f", best lap {summary['best_lap_s']:.2f} s" if laps else "") + f" -> {run_dir}")
    algo.close()
    if viewer:
        viewer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[run] interrupted")
    finally:
        # Kit's app.close() hangs or aborts (malloc) on this setup and leaves the GPU memory held,
        # so exit hard once the logs are flushed.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
