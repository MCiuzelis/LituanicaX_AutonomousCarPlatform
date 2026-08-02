# LituanicaX Deployment — vanilla Python

The same control stack as `Deployment/Ros2`, rewritten as plain Python classes.
No ROS, no DDS, no colcon: one process, one thread per module, and modules that
call each other's methods directly.

The ROS workspace is untouched and still works — this is a parallel
implementation, not a replacement.

## Run

```bash
Deployment/Python/run                       # everything on, run recorded
Deployment/Python/run --no-log              # ...without recording
Deployment/Python/run --list-modules        # show what is on/off
```

Then open `http://<raspberry-pi-ip>:8080` and hold SPACE to drive.

The camera panes stream the full 240×180 sensor frame so you can aim the
cameras, with the rows outside `CROP_TOP_FRAC`/`CROP_BOTTOM_FRAC` dimmed and the
boundaries drawn in white — the bright band between the lines is exactly what
reaches the policy. It is a preview overlay only; the observation is built from
the raw frame (`CROP_DIM_FACTOR`, `CROP_EDGE_COLOR` in `modules/tof_camera.py`).

It uses the same `Deployment/.venv` and system packages that `Deployment/install.sh`
creates, so no extra install step is needed.

## Enabling and disabling modules

Every module is a flag on `RobotSettings`, which is what the ROS launch file's
`condition=IfCondition(...)` arguments used to be:

```bash
Deployment/Python/run --disable-tof --disable-policy      # bench test, no cameras
Deployment/Python/run --disable policy --enable acceleration_test
Deployment/Python/run --disable dashboard --port 9090
```

| Flag | Shorthand | Default |
|------|-----------|---------|
| `--enable/--disable tof_cameras` | `--disable-tof` | on |
| `--enable/--disable encoder` | | on |
| `--enable/--disable monitoring` | | on |
| `--enable/--disable drive` | | on |
| `--enable/--disable policy` | `--disable-policy` | on |
| `--enable/--disable dashboard` | | on |
| `--enable/--disable screenshot_grabber` | `--enable-screenshot` | off |
| `--enable/--disable acceleration_test` | `--enable-accel-test` | off |
| `--enable/--disable recorder` | `--log` / `--no-log` | on via `run` |

In code it is the same thing:

```python
from lituanicax.robot import Robot, RobotSettings

Robot(RobotSettings(policy=False, acceleration_test=True)).run()
```

The acceleration test takes over the drive, so disable the policy when you run
it (`--disable-policy --enable-accel-test`); `Robot` warns if both are on.

## Logging a run (`--log`)

`run` passes `--log` by default, so every run leaves a complete record in
`Deployment/Logs/run_<timestamp>/` — full ToF frames, the exact observation
vectors handed to ONNX, the actions that came back, what the drive did with
them, and every number the dashboard shows. `run --no-log` turns it off;
`--log-dir DIR` and `--log-max-mb N` (default 2048, also clamped to 80 % of the
free space) change where it goes and how much it may write.

| File | One row per | Key fields |
|------|-------------|------------|
| `meta.json` | run | clocks, enabled modules, obs layout, full `config.py` snapshot, git commit |
| `policy.jsonl` | policy tick | `step`, `t`, `<side>_image_index`, `obs_index`, `action_*`, `drive_*`, `infer_ms` |
| `telemetry.jsonl` | telemetry tick | battery/current/CPU/encoder/drive state, plus `dash_*` — the websocket payload verbatim |
| `frames.jsonl` | stored ToF frame | `index`, `side`, `frame_id`, `t` (capture time) |
| `obs.bin` | policy tick | float32 `[n, 4331]` |
| `tof_<side>_depth.bin` / `_conf.bin` | stored frame | float32 / uint8 `[n, 180, 240]` |
| `console.log` | log line | `[LEVEL] [t_wall] [t] [module]: message`, subprocesses included |
| `summary.json` | run | counts, bytes, dropped records |

**Everything is keyed on one clock.** Every row carries `t` (`CLOCK_MONOTONIC`,
the same clock the camera subprocesses stamp frames with) and `t_wall`
(`t + meta.json["clock_offset"]`). Which image produced which action is a
lookup, not a nearest-timestamp guess — a policy row names its image and
observation records directly:

```python
s = steps[100]                                  # a row from policy.jsonl
depth[s["left_image_index"]]                    # exactly what the camera saw
obs[s["obs_index"]]                             # exactly what ONNX was given
s["action_throttle_raw"], s["drive_duty"]       # what came back, what was applied
```

Each run directory gets its own `README.md` with the schema and loading
snippets, so an analysis script or agent can work from the directory alone.
`tests/test_logging.py` checks the alignment end to end: it rebuilds `obs.bin`
from the stored frames and the row's own wheel velocity and action history, and
requires bit-exact equality.

Full frames cost ~13 MB/s with both cameras at 30 Hz. The control path never
pays for it: producers hand arrays to a bounded queue and a writer thread does
the I/O. If the disk cannot keep up, a whole tick is dropped — row, images and
observation together — so the files can never disagree about what happened, and
the count lands in `summary.json`.

## Layout

```
Deployment/Python/
├── run                      # launcher (activates the venv, runs main)
├── lituanicax/
│   ├── main.py              # CLI: flags -> RobotSettings -> Robot
│   ├── robot.py             # Robot: builds, wires, starts and stops modules
│   ├── module.py            # Module base class: thread + fixed-rate loop
│   ├── config.py            # pins, calibration, limits, rates (was common.py)
│   ├── realtime.py          # SCHED_RR helper for the latency-critical threads
│   ├── logging_utils.py     # ROS-style console logging (+ the --log file copy)
│   ├── dashboard.html       # unchanged from the ROS package
│   ├── policy.onnx          # unchanged from the ROS package
│   └── modules/             # one class per former node, plus recorder.py
└── tests/                   # off-hardware verification (stubbed sensors)

Deployment/Logs/run_<timestamp>/   # written by --log, git-ignored
```

## Modules

| Class | Was | Rate | Reads | Writes |
|-------|-----|------|-------|--------|
| `TofCamera` (×2) | `tof_camera_node` | camera-driven | Arducam CSI | latest depth frame, preview JPEG |
| `PolicyRunner` | `policy_runner_node` | 30 Hz | both cameras, encoder | `Drive.command()` |
| `Drive` | `drive_node` | 30 Hz watchdog | monitoring | motor H-bridge + servo PWM |
| `Encoder` | `encoder_node` | 300 Hz | AS5600 over I²C | velocity, distance |
| `Monitoring` | `monitoring_node` | 30 Hz | SPI ADC, `/proc` | battery, current, CPU, fault flags |
| `Dashboard` | `dashboard_node` | 15 Hz | whole robot | HTTP/WebSocket/MJPEG on :8080 |
| `AccelerationTest` | `acceleration_test_node` | 300 Hz | monitoring | `Drive.command()` |
| `ScreenshotGrabber` | `screenshot_grabber` | 1 Hz | both cameras | PNGs in `Deployment/Screenshots` |
| `Recorder` | — (new) | 30 Hz + policy ticks | whole robot | a run directory in `Deployment/Logs` |

## How it replaces ROS

| ROS 2 | Here |
|-------|------|
| Node | `Module` subclass, one thread |
| `create_timer(period, cb)` | `Module.update()` at `rate_hz` |
| Publish / subscribe | direct attribute read or method call |
| Topic message | plain attributes (`monitoring.battery_voltage`) |
| Node parameters | `RobotSettings` fields / constructor arguments |
| Launch file + conditions | `Robot._build()` + `RobotSettings` flags |
| `ros2 launch` | `Deployment/Python/run` |

Safety behaviour is unchanged. `Drive` still refuses to move unless autopilot is
enabled, monitoring telemetry is fresh (0.5 s), there is no undervoltage or
overcurrent, and commands keep arriving (0.25 s timeout). Losing the dashboard
websocket still disables autopilot.

### Two deliberate differences

* **Cameras stay in their own processes, like ROS.** They were briefly threads,
  which does not work: `ArducamDepthCamera.requestFrame()` blocks for the next
  frame *without releasing the GIL*, so it freezes every other Python thread in
  the process for a whole frame period. Measured on the Pi, a pure-Python
  heartbeat that should tick every 1 ms stalled to 33 ms median with one camera
  grabbing, and with two the 30 Hz policy loop ran at 12 Hz and the 300 Hz
  encoder at 52 Hz. Each `TofCamera` now forks a worker that owns the SDK and
  publishes frames through `multiprocessing.shared_memory`; the parent keeps a
  poll thread that copies the newest frame out. `snapshot()`, `frame_count` and
  `preview_jpeg` are unchanged, so nothing downstream had to move. Frames are
  still deep-copied before `releaseFrame()` — the same guard the ROS code used
  against the SDK's shared buffers. **Do not move the cameras back in-process.**
* **Real-time priority is per-thread.** The ROS drive process ran under
  `SCHED_RR`; the control modules now share one process, so `Drive` and
  `PolicyRunner` raise only their own threads (`realtime = True`). It needs
  `CAP_SYS_NICE`; without it you get one warning and normal scheduling, as
  before.

Loop rates on a Pi 5 with both cameras live, measured with the whole stack up:
`policy_runner` 30.0/30 Hz, `monitoring` 30.0/30 Hz, `encoder` 299.7/300 Hz,
cameras ~30 fps each. A policy loop meaningfully below 30 Hz is a bug, not a
slow log — the policy was trained at 30 Hz and carries a 5-step action history,
so the tick rate is part of its input.

## Adding a module

```python
from lituanicax.module import Module

class Beeper(Module):
    def __init__(self, monitoring):
        super().__init__("beeper", rate_hz=5)
        self.monitoring = monitoring

    def setup(self): ...        # open hardware
    def update(self): ...       # called at 5 Hz
    def shutdown(self): ...     # release hardware
```

Then add a flag to `RobotSettings`, a name to `MODULE_NAMES`, and construct it
in `Robot._build()`. Modules that block on their own source (a camera, a web
server) override `run()` instead of `update()`.

## Tests

`tests/` runs the whole stack with stubbed GPIO, SPI, I²C and Arducam SDK, so it
works on a laptop. It exercises real ONNX inference, the dashboard, the safety
gate and the acceleration test.

```bash
cd Deployment/Python/tests
python test_stack.py                              # full stack, 20 checks
FAKE_ENCODER_STOP_AFTER=3 python test_modules.py  # module switches, 14 checks
python test_logging.py                            # --log run directory, 27 checks
```

Needs `numpy`, `opencv-python`, `aiohttp` and `onnxruntime`; the stubs supply
everything Pi-specific.
