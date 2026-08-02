"""Flight recorder: writes everything the stack sees and decides to disk.

Enabled with `--log` (the `run` script passes it by default, and either
calibration test turns it on by itself).  One directory per run under
`Deployment/Logs/`, holding

    meta.json         run constants: clocks, config snapshot, obs layout, files
    policy.jsonl      one row per policy tick: inputs, outputs, what drive did
    telemetry.jsonl   one row per telemetry tick: every number the dashboard shows
    frames.jsonl      one row per stored ToF frame: index -> timestamp, camera
    test.jsonl        one row per calibration-test tick (300 Hz), when one runs
    events.jsonl      test phase changes and final results, machine-readable
    obs.bin           float32 [n_steps, OBS_DIM] — the exact vectors fed to ONNX
    tof_<side>_depth.bin / tof_<side>_conf.bin   raw sensor frames, fixed-size
    console.log       the console stream, with both clocks on every line
    summary.json      counts and byte totals, written at shutdown

Alignment
---------
Every row in every file carries `t` (CLOCK_MONOTONIC, the same clock the camera
subprocesses stamp frames with) and `t_wall` (unix time).  `t` is the join key:
`meta.json.clock_offset` converts it to wall time.  A policy row names the exact
image records it was built from (`left_image_index`, `right_image_index`) and
the exact obs record it fed the network (`obs_index`), so "which image produced
this action" is a lookup, never a nearest-timestamp guess.

Cost and safety
---------------
Full frames are 216 kB per camera per tick — ~13 MB/s with two cameras at 30 Hz.
The control path must never pay for that, so callers only hand arrays to a
bounded queue and a writer thread does the I/O.  If the queue backs up the step
is dropped whole (row, images and obs together) rather than in pieces, so the
files can never disagree about what happened; drops are counted in summary.json.
Blob writes stop once `--log-max-mb` is reached, but the scalar rows keep going.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .. import config
from ..config import TELEMETRY_HZ
from ..logging_utils import add_file_handler
from ..module import Module
from .tof_camera import (
    ACTION_HIST,
    ACTION_HIST_LEN,
    CROP_BOTTOM_ROWS,
    CROP_TOP_ROWS,
    DEPTH_RAW_H,
    DEPTH_RAW_W,
    OBS_DIM,
    PER_CAM_DIM,
    POLICY_H,
    POLICY_W,
    WHEEL_VEL_OBS,
)

LOG_ROOT = Path(__file__).resolve().parents[2] / "Logs"

#: Schema version of the run directory. Bump when a field changes meaning.
SCHEMA_VERSION = 1

#: Queued items, each holding at most one tick's frames (~449 kB). Deep enough
#: to ride out an SD-card write stall, shallow enough to bound memory.
QUEUE_MAX = 24

DEFAULT_MAX_MB = 2048.0
#: Never let the recorder fill the card: the budget is clamped to this fraction
#: of what is actually free when the run starts.
FREE_SPACE_FRACTION = 0.8

_DEPTH_BYTES = DEPTH_RAW_H * DEPTH_RAW_W * 4
_CONF_BYTES = DEPTH_RAW_H * DEPTH_RAW_W
_OBS_BYTES = OBS_DIM * 4


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _git_commit() -> str:
    """Best-effort commit of the checkout this run came from; "" if unknown."""
    try:
        result = subprocess.run(
            ["git", "-C", str(LOG_ROOT.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _config_snapshot() -> dict:
    """Every tunable in config.py, so two runs can be diffed field by field."""
    return {
        name: value
        for name, value in vars(config).items()
        if name.isupper() and isinstance(value, (int, float, bool, str))
    }


def _free_bytes(path: Path) -> int:
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except OSError:
        return 0


class Recorder(Module):
    """Owns the run directory; ticks at the telemetry rate for scalar rows."""

    def __init__(
        self,
        robot,
        log_dir: str = "",
        max_mb: float = DEFAULT_MAX_MB,
        rate_hz: float = TELEMETRY_HZ,
    ) -> None:
        super().__init__("recorder", rate_hz)

        self.robot = robot
        self.t0_mono = time.monotonic()
        self.t0_wall = time.time()

        # The directory has to exist before anything can be logged into it, and
        # console.log must start capturing while the other modules are still
        # being built — both earlier than setup(), which runs at start().
        self.run_dir = Path(log_dir).expanduser() if log_dir else LOG_ROOT / _run_name()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        add_file_handler(self.run_dir / "console.log")

        self.max_bytes = self._budget(max_mb)

        self.steps = 0  # policy ticks submitted (not necessarily written)
        self.ticks = 0  # telemetry rows submitted
        self.test_ticks = 0  # calibration-test rows submitted
        self.events = 0  # test phase changes and results submitted
        self.dropped = 0  # items the queue had no room for
        self.bytes_written = 0

        self._queue: "queue.Queue[dict | None]" = queue.Queue(maxsize=QUEUE_MAX)
        self._writer: threading.Thread | None = None
        self._text_files: dict[str, object] = {}
        self._blob_files: dict[str, object] = {}
        self._frame_count = 0  # image records written, across both cameras
        self._obs_count = 0
        self._side_counts: dict[str, int] = {}  # side -> records in its blob
        self._last_frame: dict[str, tuple[int, int]] = {}  # side -> (frame_id, index)
        self._last_drop_warn = 0.0
        self._budget_warned = False

    # --- lifecycle ---

    def _budget(self, max_mb: float) -> int:
        requested = int(max(1.0, float(max_mb)) * 1024 * 1024)
        free = _free_bytes(self.run_dir)
        if free and requested > free * FREE_SPACE_FRACTION:
            return int(free * FREE_SPACE_FRACTION)
        return requested

    def setup(self) -> None:
        names = ["policy", "telemetry", "frames"]
        if self._tests():
            names += ["test", "events"]
        for name in names:
            self._text_files[name] = open(self.run_dir / f"{name}.jsonl", "w", buffering=1 << 16)
        self._blob_files["obs"] = open(self.run_dir / "obs.bin", "wb", buffering=1 << 16)
        for side in self.robot.cameras:
            for kind in ("depth", "conf"):
                self._blob_files[f"{side}_{kind}"] = open(
                    self.run_dir / f"tof_{side}_{kind}.bin", "wb", buffering=1 << 16
                )

        self._write_meta()
        self._write_readme()

        self._writer = threading.Thread(target=self._writer_main, name="recorder_writer")
        self._writer.start()

        rate = self._estimated_bytes_per_sec()
        seconds = self.max_bytes / rate if rate else float("inf")
        self.log.info(
            f"Recording to {self.run_dir} "
            f"(~{rate / 1e6:.1f} MB/s, budget {self.max_bytes / 1e6:.0f} MB "
            f"≈ {seconds:.0f} s of frames)"
        )

    def shutdown(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            self._queue.put(None)
            writer.join(timeout=10.0)
            if writer.is_alive():
                self.log.warning("Recorder writer did not drain in 10 s; log may be truncated")

        self._write_summary()
        for handle in list(self._text_files.values()) + list(self._blob_files.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._text_files.clear()
        self._blob_files.clear()
        self.log.info(
            f"Recorded {self.steps} policy steps, {self._frame_count} frames, "
            f"{self.bytes_written / 1e6:.0f} MB to {self.run_dir}"
        )

    # --- producers ---

    def _tests(self) -> list:
        """The calibration tests in this run, if any. They drive the car directly
        and are the reason most logged runs exist, so they get their own streams."""
        return [
            test
            for test in (self.robot.acceleration_test, self.robot.skid_pad_test)
            if test is not None
        ]

    def log_test_tick(self, row: dict) -> None:
        """One row per calibration-test tick (300 Hz, ten times the telemetry rate).

        The encoder samples at 300 Hz too, so this is the full-resolution record of
        the accel/brake ramp and the settled circle — sampling it at the 30 Hz
        telemetry rate throws away most of a braking event.
        """
        self.test_ticks += 1
        self._submit({"file": "test", "row": row, "frames": {}, "obs": None})

    def log_event(self, row: dict) -> None:
        """A test phase change or its final result.

        A handful of rows per run, and the whole point of the run, so this one
        waits for room in the queue rather than being dropped like a sample.
        """
        self.events += 1
        self._submit({"file": "events", "row": row, "frames": {}, "obs": None}, wait_s=1.0)

    def log_policy_step(self, row: dict, frames: dict, obs: np.ndarray | None) -> None:
        """Called by PolicyRunner once per tick, before it returns.

        `frames` maps side -> (frame_id, frame_t, depth, conf); the arrays are
        the camera's own buffers, which are replaced rather than mutated on the
        next frame, so the writer thread can keep reading them.
        """
        row["step"] = self.steps
        self.steps += 1
        self._submit({"file": "policy", "row": row, "frames": frames, "obs": obs})

    def update(self) -> None:
        """Telemetry tick: every number the dashboard shows, plus the raw state
        behind it.  Also stores camera frames when the policy is not running, so
        `--log --disable-policy` still records what the cameras saw."""
        row = self._telemetry_row()
        self.ticks += 1
        frames = {} if self.robot.policy is not None else self._camera_frames()
        self._submit({"file": "telemetry", "row": row, "frames": frames, "obs": None})

    def _camera_frames(self) -> dict:
        frames = {}
        for side, camera in self.robot.cameras.items():
            depth, conf, frame_t, frame_id = camera.snapshot_with_id()
            if depth is not None and conf is not None:
                frames[side] = (frame_id, frame_t, depth, conf)
        return frames

    def _telemetry_row(self) -> dict:
        now = time.monotonic()
        row = {
            "t": now,
            "t_wall": time.time(),
            "tick": self.ticks,
            "autopilot": self.robot.autopilot_enabled,
            "policy_steps": self.steps,
        }

        monitoring = self.robot.monitoring
        if monitoring is not None:
            row.update(
                {
                    "battery_v": monitoring.battery_voltage,
                    "battery_min_v": monitoring.battery_min,
                    "current_a": monitoring.motor_current,
                    "current_peak_a": monitoring.motor_current_peak,
                    "undervoltage": monitoring.undervoltage,
                    "overcurrent": monitoring.overcurrent,
                    "cpu_usage": monitoring.cpu_usage,
                    "cpu_temp_c": monitoring.cpu_temperature,
                    "monitoring_age": now - monitoring.last_update_time,
                }
            )

        encoder = self.robot.encoder
        if encoder is not None:
            row.update(
                {
                    "velocity_m_s": encoder.linear_velocity,
                    "distance_m": encoder.distance_travelled,
                    "encoder_age": now - encoder.last_update_time,
                }
            )

        drive = self.robot.drive
        if drive is not None:
            # The limiter state is private to Drive; read it defensively so the
            # recorder can never be what breaks the drive module.
            row.update(
                {
                    "drive_enabled": drive.enabled,
                    "drive_throttle": drive.last_throttle,
                    "drive_steer": drive.last_steer,
                    "drive_duty": drive.last_duty_fraction,
                    "drive_command_age": now - drive.last_command_time,
                    "drive_ramped_throttle": getattr(drive, "_ramped_throttle", None),
                    "drive_current_scale": getattr(drive, "_current_scale", None),
                    "drive_limiter_engaged": getattr(drive, "_limiter_engaged", None),
                    "drive_blocked": list(getattr(drive, "_blocked_reasons", [])),
                }
            )

        policy = self.robot.policy
        if policy is not None:
            row["policy_wheel_vel_norm"] = policy.wheel_vel_norm
            row["policy_enabled"] = policy.enabled

        for side, camera in self.robot.cameras.items():
            _, _, frame_t, frame_count = camera.snapshot_with_id()
            row[f"camera_{side}_frames"] = frame_count
            row[f"camera_{side}_age"] = now - frame_t if frame_t else None

        for test in self._tests():
            # Which phase the test was in, so telemetry.jsonl can be sliced into
            # the accel / brake / settle / measure windows without replaying the
            # state machine from console.log.
            row.update(test.telemetry())

        dashboard = self.robot.dashboard
        if dashboard is not None:
            # Verbatim copy of the websocket payload, so a row holds exactly what
            # the browser was told at that instant, next to the unfiltered truth.
            for key, value in dashboard.telemetry.items():
                row[f"dash_{key}"] = value
        return row

    def _submit(self, item: dict, wait_s: float = 0.0) -> None:
        try:
            if wait_s > 0.0:
                self._queue.put(item, timeout=wait_s)
            else:
                self._queue.put_nowait(item)
        except queue.Full:
            self.dropped += 1
            now = time.monotonic()
            if now - self._last_drop_warn > 1.0:
                self._last_drop_warn = now
                self.log.warning(
                    f"Recorder queue full — dropped {self.dropped} records so far "
                    "(disk cannot keep up; the log stays consistent, just sparser)"
                )

    # --- writer thread ---

    def _writer_main(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._write_item(item)
            except Exception as exc:
                self.log.error(f"Recorder write failed: {exc}", exc_info=True)
            if self._queue.empty():
                self._flush()
        self._flush()

    def _write_item(self, item: dict) -> None:
        row = item["row"]
        for side, (frame_id, frame_t, depth, conf) in item["frames"].items():
            index = self._store_frame(side, frame_id, frame_t, depth, conf)
            if row is not None:
                row[f"{side}_image_index"] = index

        obs = item["obs"]
        if row is not None and obs is not None:
            row["obs_index"] = self._store_obs(obs)

        handle = self._text_files.get(item["file"])
        if row is not None and handle is not None:
            handle.write(json.dumps(row, default=_json_default) + "\n")

    def _store_frame(self, side: str, frame_id: int, frame_t: float, depth, conf) -> int | None:
        """Append one sensor frame, or reuse the record if it is already stored.

        The policy reads the *newest* frame each tick, so at 30 Hz policy vs
        ~30 fps camera the same frame is often seen twice.  Storing it once and
        pointing both steps at the same index keeps the blobs honest about how
        many distinct frames the run actually saw.
        """
        previous = self._last_frame.get(side)
        if previous is not None and previous[0] == frame_id:
            return previous[1]

        depth_file = self._blob_files.get(f"{side}_depth")
        conf_file = self._blob_files.get(f"{side}_conf")
        if depth_file is None or conf_file is None or not self._has_budget(_DEPTH_BYTES + _CONF_BYTES):
            return None

        np.ascontiguousarray(depth, dtype=np.float32).tofile(depth_file)
        np.ascontiguousarray(conf, dtype=np.uint8).tofile(conf_file)
        self.bytes_written += _DEPTH_BYTES + _CONF_BYTES

        # Per-side record numbers: index i is the i-th frame in that side's blob.
        index = self._side_counts.get(side, 0)
        self._side_counts[side] = index + 1
        self._last_frame[side] = (frame_id, index)
        self._frame_count += 1

        self._text_files["frames"].write(
            json.dumps(
                {
                    "index": index,
                    "side": side,
                    "frame_id": frame_id,
                    "t": frame_t,
                    "t_wall": frame_t + self.clock_offset,
                    "stored_t": time.monotonic(),
                }
            )
            + "\n"
        )
        return index

    def _store_obs(self, obs: np.ndarray) -> int | None:
        handle = self._blob_files.get("obs")
        if handle is None or not self._has_budget(_OBS_BYTES):
            return None
        np.ascontiguousarray(obs, dtype=np.float32).tofile(handle)
        self.bytes_written += _OBS_BYTES
        index, self._obs_count = self._obs_count, self._obs_count + 1
        return index

    def _has_budget(self, extra: int) -> bool:
        if self.bytes_written + extra <= self.max_bytes:
            return True
        if not self._budget_warned:
            self._budget_warned = True
            self.log.warning(
                f"Log budget of {self.max_bytes / 1e6:.0f} MB reached — frames and "
                "observations stop here; the scalar rows keep recording "
                "(raise it with --log-max-mb)"
            )
        return False

    def _flush(self) -> None:
        for handle in list(self._text_files.values()) + list(self._blob_files.values()):
            try:
                handle.flush()
            except Exception:
                pass

    # --- run metadata ---

    @property
    def clock_offset(self) -> float:
        """Add to any `t` in this run to get unix wall time."""
        return self.t0_wall - self.t0_mono

    def _estimated_bytes_per_sec(self) -> float:
        per_step = len(self.robot.cameras) * (_DEPTH_BYTES + _CONF_BYTES) + _OBS_BYTES
        rate = self.robot.settings.policy_hz if self.robot.policy is not None else self.rate_hz
        return per_step * float(rate)

    def _write_meta(self) -> None:
        settings = self.robot.settings
        meta = {
            "schema": SCHEMA_VERSION,
            "started_wall": self.t0_wall,
            "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0_wall)),
            "started_mono": self.t0_mono,
            "clock": "t is CLOCK_MONOTONIC seconds; t_wall = t + clock_offset",
            "clock_offset": self.clock_offset,
            "host": platform.node(),
            "git_commit": _git_commit(),
            "modules": settings.enabled_modules(),
            "rates_hz": {
                "policy": settings.policy_hz,
                "telemetry": settings.telemetry_hz,
                "encoder": settings.encoder_hz,
                "recorder": self.rate_hz,
            },
            "policy_path": str(self.robot.policy.policy_path) if self.robot.policy else "",
            # The hardcoded constants a calibration test was run with. Without
            # these a swept THROTTLE makes every run in Logs/ look alike.
            "tests": {test.name: test.parameters() for test in self._tests()},
            "observation": {
                "dim": OBS_DIM,
                "layout": [
                    {"name": "depth_left", "start": 0, "size": PER_CAM_DIM},
                    {"name": "depth_right", "start": PER_CAM_DIM, "size": PER_CAM_DIM},
                    {"name": "wheel_vel", "start": 2 * PER_CAM_DIM, "size": WHEEL_VEL_OBS},
                    {
                        "name": "action_history",
                        "start": 2 * PER_CAM_DIM + WHEEL_VEL_OBS,
                        "size": ACTION_HIST,
                        "note": f"{ACTION_HIST_LEN} × (throttle, steer), oldest first",
                    },
                ],
                "depth_shape": [POLICY_H, POLICY_W],
                "depth_encoding": "1 - distance/2 m, clipped to [0, 1]; 1 = at the lens",
            },
            "images": {
                "shape": [DEPTH_RAW_H, DEPTH_RAW_W],
                "depth_dtype": "float32",
                "depth_units": "mm (NaN where the sensor did not measure)",
                "conf_dtype": "uint8",
                "policy_crop_rows": [CROP_TOP_ROWS, DEPTH_RAW_H - CROP_BOTTOM_ROWS],
                "record_bytes": {"depth": _DEPTH_BYTES, "conf": _CONF_BYTES},
                "sides": sorted(self.robot.cameras),
            },
            "budget_bytes": self.max_bytes,
            "config": _config_snapshot(),
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))

    def _write_summary(self) -> None:
        summary = {
            "schema": SCHEMA_VERSION,
            "stopped_wall": time.time(),
            "duration_sec": time.monotonic() - self.t0_mono,
            "policy_steps": self.steps,
            "telemetry_ticks": self.ticks,
            "test_ticks": self.test_ticks,
            "test_events": self.events,
            "frames_stored": self._frame_count,
            "obs_stored": self._obs_count,
            "records_dropped": self.dropped,
            "bytes_written": self.bytes_written,
            "budget_reached": self._budget_warned,
        }
        try:
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        except OSError as exc:
            self.log.warning(f"Could not write summary.json: {exc}")

    def _write_readme(self) -> None:
        sides = sorted(self.robot.cameras) or ["left", "right"]
        tests = self._tests()
        (self.run_dir / "README.md").write_text(
            README_TEMPLATE.format(
                schema=SCHEMA_VERSION,
                obs_dim=OBS_DIM,
                depth_h=DEPTH_RAW_H,
                depth_w=DEPTH_RAW_W,
                side=sides[0],
                sides=", ".join(sides),
                test_files=(
                    TEST_FILES_TEMPLATE.format(hz=int(tests[0].rate_hz)) if tests else ""
                ),
                test_section=(
                    TEST_SECTION_TEMPLATE.format(name=tests[0].name) if tests else ""
                ),
            )
        )


def _run_name() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


README_TEMPLATE = """# LituanicaX run log (schema {schema})

Written by `lituanicax.modules.recorder`. Every file is append-only and every
row carries the same two clocks:

* `t` — `CLOCK_MONOTONIC` seconds, the join key across all files. The ToF
  frame timestamps come from the camera processes on this same system-wide
  clock, so they are directly comparable.
* `t_wall` — unix time. `t_wall = t + meta.json["clock_offset"]`.

## Files

| File | One row per | Key fields |
|------|-------------|------------|
| `meta.json` | run | clocks, enabled modules, full `config.py` snapshot, obs layout |
| `policy.jsonl` | policy tick (30 Hz) | `step`, `t`, `{side}_image_index`, `obs_index`, `action_*`, `drive_*` |
| `telemetry.jsonl` | telemetry tick (30 Hz) | `t`, battery/current/CPU/encoder, `drive_*`, `dash_*` |
| `frames.jsonl` | stored ToF frame | `index`, `side`, `frame_id`, `t` (capture time) |{test_files}
| `obs.bin` | policy tick | float32 `[n, {obs_dim}]`, row `n` = `obs_index` |
| `tof_<side>_depth.bin` | stored frame | float32 `[n, {depth_h}, {depth_w}]`, mm |
| `tof_<side>_conf.bin` | stored frame | uint8 `[n, {depth_h}, {depth_w}]` |
| `console.log` | log line | `[LEVEL] [t_wall] [t] [module]: message` |
| `summary.json` | run | counts, bytes, dropped records |

Cameras: {sides}.
{test_section}
## Which image produced which action

`policy.jsonl` names the record it used — no timestamp matching required:

```python
import json, numpy as np

meta  = json.load(open("meta.json"))
steps = [json.loads(line) for line in open("policy.jsonl")]
depth = np.memmap("tof_{side}_depth.bin", dtype=np.float32,
                  mode="r").reshape(-1, {depth_h}, {depth_w})
obs   = np.memmap("obs.bin", dtype=np.float32, mode="r").reshape(-1, {obs_dim})

s = steps[100]
frame  = depth[s["{side}_image_index"]]        # exactly what the camera saw
vector = obs[s["obs_index"]]                   # exactly what ONNX was given
print(s["t"], s["action_throttle_raw"], s["action_steer"], s["drive_duty"])
```

`*_image_index` is `null` when that camera had no fresh frame (the policy then
commands zero) or when the size budget was reached. Consecutive steps often
share an index: the policy reads the newest frame, and the camera does not
always have a new one.

## Lining a step up with telemetry

`telemetry.jsonl` is a separate 30 Hz stream, so match on `t`:

```python
import bisect
tel = [json.loads(line) for line in open("telemetry.jsonl")]
ts  = [row["t"] for row in tel]
near = tel[min(bisect.bisect_left(ts, s["t"]), len(tel) - 1)]
```

`dash_*` fields are the exact websocket payload the browser received (EMA
filtered); the unprefixed fields next to them are the unfiltered module state.

## Reproducing an observation

`process_for_policy(depth, conf)` in `modules/tof_camera.py` turns a raw frame
into the `depth_left` / `depth_right` slices of `obs.bin`, using the crop rows
and constants recorded in `meta.json`.
"""

#: Extra rows in the file table above, only for a run that had a test in it.
TEST_FILES_TEMPLATE = """
| `test.jsonl` | calibration-test tick ({hz} Hz) | `test_state`, `test_velocity`, `test_distance`, `test_throttle_cmd`, `test_steer_cmd` |
| `events.jsonl` | phase change / result | `event`, `test`, plus the numbers that phase produced |"""

#: Extra section, same condition.
TEST_SECTION_TEMPLATE = """
## The calibration test in this run

`{name}` drove the car here, not the policy. Its constants are in
`meta.json["tests"]`; the result it printed to `console.log` is also in
`events.jsonl` as machine-readable numbers:

```python
events = [json.loads(line) for line in open("events.jsonl")]
result = next(e for e in events if e["event"] == "finished")
```

`test.jsonl` is the full-resolution trace behind that result — one row per
300 Hz tick, so a braking event is ~1200 samples rather than the ~120 that
`telemetry.jsonl` would give it. Slice it by `test_state` to get each phase:

```python
rows  = [json.loads(line) for line in open("test.jsonl")]
brake = [r for r in rows if r["test_state"] == "BRAKING"]
decel = (brake[0]["test_velocity"] - brake[-1]["test_velocity"]) / (
    brake[-1]["t"] - brake[0]["t"])
```

`telemetry.jsonl` carries the same `test_*` fields at 30 Hz, next to the
battery, current and drive state — that is where to look for the voltage sag or
the current limiter engaging under the phase you are measuring.
"""
