"""Verifies `--log`: that a run directory is complete and, above all, that the
stored ToF frames, observations, actions and telemetry line up exactly."""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS_DIR / "stubs"))
sys.path.insert(0, str(TESTS_DIR.parent))

from lituanicax.modules import drive as drive_mod


class FakePWM:
    def __init__(self, channel, frequency_hz, initial_pulse_ns=0):
        self.channel, self.period_ns = channel, int(1e9 / frequency_hz)
        self.pulse_ns = initial_pulse_ns

    def set_duty_fraction(self, f): self.pulse_ns = int(self.period_ns * max(0.0, min(1.0, f)))
    def set_pulse_ns(self, ns): self.pulse_ns = int(ns)
    def stop(self): pass


drive_mod.SysfsHardwarePWM = FakePWM
drive_mod.Drive._pinctrl_set = staticmethod(lambda *a: None)

from lituanicax.main import parse_args, settings_from_args
from lituanicax.modules.tof_camera import DEPTH_RAW_H, DEPTH_RAW_W, OBS_DIM, process_for_policy
from lituanicax.robot import Robot

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


RUN_DIR = TESTS_DIR / "logrun"
shutil.rmtree(RUN_DIR, ignore_errors=True)

# --- the flag itself ---
check("--log enables the recorder", settings_from_args(parse_args(["--log"])).recorder)
check("--no-log wins over --log", not settings_from_args(parse_args(["--log", "--no-log"])).recorder)
check("recorder off by default", not settings_from_args(parse_args([])).recorder)
check("--log-dir/--log-max-mb reach settings",
      settings_from_args(parse_args(["--log", "--log-dir", "/tmp/x", "--log-max-mb", "7"]))
      .log_dir == "/tmp/x" and
      settings_from_args(parse_args(["--log", "--log-max-mb", "7"])).log_max_mb == 7)

# --- record a short run with the policy driving ---
settings = settings_from_args(parse_args(
    ["--log", "--log-dir", str(RUN_DIR), "--disable", "dashboard"]))
robot = Robot(settings)
check("recorder starts before the rest", [m.name for m in robot.modules][:3] ==
      ["tof_camera_left", "tof_camera_right", "recorder"],
      str([m.name for m in robot.modules]))
robot.start()
time.sleep(1.0)
robot.set_autopilot_enabled(True)
time.sleep(2.5)
robot.set_autopilot_enabled(False)
robot.stop()

# --- files ---
expected = ["meta.json", "summary.json", "README.md", "policy.jsonl", "telemetry.jsonl",
            "frames.jsonl", "obs.bin", "console.log",
            "tof_left_depth.bin", "tof_left_conf.bin",
            "tof_right_depth.bin", "tof_right_conf.bin"]
missing = [name for name in expected if not (RUN_DIR / name).exists()]
check("run directory complete", not missing, f"missing={missing}")

meta = json.loads((RUN_DIR / "meta.json").read_text())
summary = json.loads((RUN_DIR / "summary.json").read_text())
steps = [json.loads(line) for line in (RUN_DIR / "policy.jsonl").read_text().splitlines()]
telemetry = [json.loads(line) for line in (RUN_DIR / "telemetry.jsonl").read_text().splitlines()]
frames = [json.loads(line) for line in (RUN_DIR / "frames.jsonl").read_text().splitlines()]

check("meta describes the clocks and obs layout",
      meta["observation"]["dim"] == OBS_DIM and abs(meta["clock_offset"]) > 0
      and meta["config"]["POLICY_HZ"] == 30 and meta["images"]["shape"] == [180, 240],
      f"schema={meta['schema']} commit={meta['git_commit']!r}")
check("policy rows recorded", len(steps) > 60, f"{len(steps)} rows in {summary['duration_sec']:.1f} s")
check("telemetry rows recorded", len(telemetry) > 60, f"{len(telemetry)} rows")
check("nothing dropped", summary["records_dropped"] == 0, str(summary))

# --- alignment: policy row -> frames.jsonl -> blob -> obs.bin ---
depth_blobs = {
    side: np.memmap(RUN_DIR / f"tof_{side}_depth.bin", dtype=np.float32, mode="r").reshape(
        -1, DEPTH_RAW_H, DEPTH_RAW_W)
    for side in ("left", "right")
}
conf_blobs = {
    side: np.memmap(RUN_DIR / f"tof_{side}_conf.bin", dtype=np.uint8, mode="r").reshape(
        -1, DEPTH_RAW_H, DEPTH_RAW_W)
    for side in ("left", "right")
}
obs_all = np.memmap(RUN_DIR / "obs.bin", dtype=np.float32, mode="r").reshape(-1, OBS_DIM)

by_index = {(row["side"], row["index"]): row for row in frames}
check("frames.jsonl indexes every stored frame",
      len(frames) == summary["frames_stored"]
      and all(len(depth_blobs[side]) == sum(1 for r in frames if r["side"] == side)
              for side in ("left", "right")),
      f"{len(frames)} rows, blobs={[len(depth_blobs[s]) for s in ('left', 'right')]}")
check("obs.bin has one record per stored observation",
      len(obs_all) == summary["obs_stored"], f"{len(obs_all)} vs {summary['obs_stored']}")

driven = [row for row in steps if row["skipped"] is None]
check("driven steps recorded", len(driven) > 40, f"{len(driven)} of {len(steps)}")

frame_id_mismatch = []
content_mismatch = []
obs_mismatch = []
for row in driven:
    for side in ("left", "right"):
        index = row[f"{side}_image_index"]
        entry = by_index.get((side, index))
        if entry is None or entry["frame_id"] != row[f"{side}_frame_id"] \
                or entry["t"] != row[f"{side}_frame_t"]:
            frame_id_mismatch.append((row["step"], side))
            continue
        # The stub paints frame n as 800 + n % 100, so the pixels themselves say
        # which frame this is: the blob record must be that exact frame.
        if depth_blobs[side][index][0, 0] != 800.0 + (entry["frame_id"] % 100):
            content_mismatch.append((row["step"], side, float(depth_blobs[side][index][0, 0])))

    rebuilt = np.concatenate([
        process_for_policy(depth_blobs["left"][row["left_image_index"]],
                           conf_blobs["left"][row["left_image_index"]]),
        process_for_policy(depth_blobs["right"][row["right_image_index"]],
                           conf_blobs["right"][row["right_image_index"]]),
        np.array([row["wheel_vel_norm"]], dtype=np.float32),
        np.array(row["action_history_in"], dtype=np.float32),
    ])
    if not np.array_equal(rebuilt, obs_all[row["obs_index"]]):
        obs_mismatch.append(row["step"])

check("policy row -> frames.jsonl agree on frame id and time",
      not frame_id_mismatch, str(frame_id_mismatch[:5]))
check("stored image is the frame the row names", not content_mismatch, str(content_mismatch[:5]))
check("obs.bin == images + wheel vel + action history from the same row",
      not obs_mismatch, f"{len(obs_mismatch)} mismatched steps {obs_mismatch[:5]}")

# --- the action history chains one step to the next ---
chained = [
    (a["step"], b["step"]) for a, b in zip(driven, driven[1:])
    if b["step"] == a["step"] + 1
    and b["action_history_in"][-2:] != [a["action_throttle_raw"], a["action_steer"]]
]
check("action history carries the previous action", not chained, str(chained[:5]))

# --- outputs and drive state ---
check("actions recorded", any(row["action_steer"] != 0.0 for row in driven)
      and all(-1.0 <= row["action_throttle_raw"] <= 1.0 for row in driven))
check("drive state on every row",
      all("drive_duty" in row and "drive_enabled" in row for row in steps))
check("inference timing recorded",
      all(row["infer_ms"] is not None and row["infer_ms"] < 50 for row in driven),
      f"max={max(row['infer_ms'] for row in driven):.1f} ms")
check("skipped steps say why",
      all(row["skipped"] == "autopilot disabled" for row in steps if row["skipped"]),
      str({row["skipped"] for row in steps}))

# --- telemetry: dashboard numbers plus the raw state behind them ---
last = telemetry[-1]
check("telemetry has the sensor numbers",
      all(k in last for k in ("battery_v", "battery_min_v", "current_a", "current_peak_a",
                              "cpu_temp_c", "velocity_m_s", "distance_m", "drive_duty",
                              "drive_current_scale", "camera_left_frames")),
      str(sorted(last)[:6]))
check("telemetry timestamps are monotonic and in the run window",
      all(b["t"] >= a["t"] for a, b in zip(telemetry, telemetry[1:]))
      and telemetry[0]["t"] >= meta["started_mono"],
      f"span={telemetry[-1]['t'] - telemetry[0]['t']:.1f} s")
check("telemetry and policy share one clock",
      min(t["t"] for t in telemetry) < np.median([s["t"] for s in steps])
      < max(t["t"] for t in telemetry))

# a dashboard-enabled run must carry its payload verbatim
settings = settings_from_args(parse_args(
    ["--log", "--log-dir", str(RUN_DIR / "dash"), "--disable", "tof_cameras",
     "--disable", "policy", "--port", "8098"]))
dash_robot = Robot(settings)
dash_robot.start()
time.sleep(1.0)
dash_robot.stop()
dash_rows = [json.loads(line)
             for line in (RUN_DIR / "dash" / "telemetry.jsonl").read_text().splitlines()]
check("dashboard payload mirrored into telemetry.jsonl",
      dash_rows and all(k in dash_rows[-1] for k in ("dash_battery", "dash_cpu_temp",
                                                     "dash_throttle", "dash_active")),
      str([k for k in dash_rows[-1] if k.startswith("dash_")]) if dash_rows else "no rows")

# cameras without the policy: the telemetry tick has to store the frames itself
settings = settings_from_args(parse_args(
    ["--log", "--log-dir", str(RUN_DIR / "nopolicy"), "--disable", "policy",
     "--disable", "dashboard"]))
cam_robot = Robot(settings)
cam_robot.start()
time.sleep(1.5)
cam_robot.stop()
cam_rows = [json.loads(line)
            for line in (RUN_DIR / "nopolicy" / "telemetry.jsonl").read_text().splitlines()]
cam_frames = [json.loads(line)
              for line in (RUN_DIR / "nopolicy" / "frames.jsonl").read_text().splitlines()]
stored = [row for row in cam_rows if row.get("left_image_index") is not None]
check("frames still recorded with the policy off",
      len(cam_frames) > 20 and len(stored) > 20
      and max(row["left_image_index"] for row in stored) == max(
          f["index"] for f in cam_frames if f["side"] == "left"),
      f"{len(cam_frames)} frames, {len(stored)} telemetry rows naming one")

console = (RUN_DIR / "console.log").read_text()
check("console mirrored with both clocks", "[policy_runner]" in console
      and console.count("[INFO]") > 3, f"{len(console.splitlines())} lines")

size_mb = sum(f.stat().st_size for f in RUN_DIR.rglob("*")) / 1e6
print(f"\nrun dir: {RUN_DIR} ({size_mb:.1f} MB, {summary['frames_stored']} frames)")
shutil.rmtree(RUN_DIR, ignore_errors=True)

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
