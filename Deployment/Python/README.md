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
| `--enable/--disable skid_pad_test` | `--enable-skid-test` | off |
| `--enable/--disable recorder` | `--log` / `--no-log` | on via `run`, and via either test |

In code it is the same thing:

```python
from lituanicax.robot import Robot, RobotSettings

Robot(RobotSettings(policy=False, acceleration_test=True)).run()
```

The calibration tests take over the drive, so disable the policy when you run one
(`--disable-policy --enable-accel-test`); `Robot` warns if more than one thing
that commands the drive is enabled. Enabling either test also turns the recorder
on, `--log` or not — see [Logging a run](#logging-a-run---log).

## Matching the car to the sim

Three tests, one per axis of the actuator model the policy was trained against.
Everything they measure feeds back into `Training/source/cone_track/env.py`.

| Test | Measures | Feeds |
|------|----------|-------|
| `--enable-accel-test` | straight-line accel + braking distance | `drive_torque`, `brake_torque_base`, `brake_torque_gain`, `max_velocity_m_s` |
| `--enable-skid-test` | steady-state turn radius at full lock | `max_steer_rad`, `ground_*_friction`, wheelbase |
| `HardwareTests/steering_angle_test.py` | road-wheel angle at steer ±1 | `max_steer_rad`, `SERVO_AMPLITUDE` |

The longitudinal axis was matched long ago (`Training/real_car_acceleration_log.txt`)
and behaves. The lateral axis never was, and that is the one that fails: yaw rate
recovered from the depth-camera image flow in a logged run puts the car at
1.1–1.7 1/m of curvature at full steer against the sim's 2.37 1/m — a minimum turn
radius 40–100 % larger than the line the policy plans. The skid pad and the
steering-angle script are the direct measurements of that number.

```bash
sudo Deployment/Python/HardwareTests/steering_angle_test.py   # protractor on the front wheels
Deployment/Python/run --disable-policy --enable-skid-test     # count laps, radius follows
```

`SkidPadTest` cannot count its own laps — there is no gyro on this car — so it
reports the distance the wheels covered over a settled window and prints the
radius for each plausible lap count. Sweep its `THROTTLE` across runs: a radius
that grows with speed is the understeer gradient, and it separates a steering-
geometry mismatch (radius too big but constant) from a grip mismatch (radius
growing), which are two very different edits to `env.py`.

Every run of either test is recorded whether or not you asked for it. A run you
have to repeat because the numbers only reached the terminal costs a battery
charge and a clear floor, so `--enable-accel-test` and `--enable-skid-test` imply
`--log` (`--no-log` still overrides). On top of the usual files each run gets:

| File | Holds |
|------|-------|
| `test.jsonl` | one row per test tick — **300 Hz**, the rate the encoder itself samples at, so a braking event is ~1200 samples and not the ~120 `telemetry.jsonl` would give it. `test_state`, `test_velocity`, `test_distance`, and the throttle/steer the test asked for next to what `Drive` did with it |
| `events.jsonl` | every phase change and the final result as numbers, not console text — including the whole lap-count table for the skid pad |
| `meta.json["tests"]` | the hardcoded constants the run was made with, which is what makes a `THROTTLE` sweep comparable after the fact |

`telemetry.jsonl` carries the same `test_*` fields at 30 Hz next to battery,
current, CPU and drive state, so "was the current limiter engaged during the
window I am measuring" is one file away.

```python
result = next(e for e in events if e["event"] == "finished")   # events.jsonl
brake  = [r for r in rows if r["test_state"] == "BRAKING"]     # test.jsonl
```

## Throttle and brake scaling

`MAX_THROTTLE` and `BRAKE_MAGNITUDE` sit on opposite sides of zero and must never
stack. They used to: `Drive._set_throttle` applied `MAX_THROTTLE` after the sign
was decided, so at `MAX_THROTTLE = 0.4` the policy's hardest brake came out as a
`1.0 × 0.4 × 0.4 = 0.16` duty worth ~5.4 m/s², against the 12.9 m/s² the sim
models for the same action. Capping top speed silently removed most of the brakes.

| Constant | Side | Applied in | Mirrors |
|----------|------|-----------|---------|
| `MAX_THROTTLE` | forward only | `Drive._set_throttle` | nothing — a hardware ceiling with no sim counterpart |
| `BRAKE_MAGNITUDE` | braking only | `PolicyRunner.update` | sim `brake_throttle_scale` |
| `SAFETY_BRAKE_MAGNITUDE` | braking only | `Drive.set_autopilot_enabled` | — |

Because `BRAKE_MAGNITUDE` lives in `PolicyRunner`, a direct `Drive.command()` from
a calibration test still means raw duty — which is what the braking numbers in
`real_car_acceleration_log.txt` were measured at.

`THROTTLE_SLEW_ENABLED` turns off the rising-throttle ramp. The ramp has no
counterpart in the sim and is not a minor correction: in the logged run it was
below the policy's request on 61 % of ticks and cut mean delivered throttle to
73 % of what the network asked for. With it off, the closed-loop current limiter
is the only protection — a real backstop that never engaged in that run (peak
19.1 A against a 20 A limit).

## Logging a run (`--log`)

`run` passes `--log` by default — and either calibration test turns it on by
itself — so every run leaves a complete record in
`Deployment/Logs/run_<timestamp>/`: full ToF frames, the exact observation
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
| `test.jsonl` | calibration-test tick (300 Hz) | `test_state`, `test_velocity`, `test_distance`, `test_throttle_cmd`, `test_steer_cmd` — only when a test ran |
| `events.jsonl` | test phase change / result | `event`, `test`, and the numbers that phase produced — only when a test ran |
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
| `PolicyRunner` | `policy_runner_node` | frame-driven (~30 Hz) | both cameras, encoder | `Drive.command()` |
| `Drive` | `drive_node` | 30 Hz watchdog | monitoring | motor H-bridge + servo PWM |
| `Encoder` | `encoder_node` | 300 Hz | AS5600 over I²C | velocity, distance |
| `Monitoring` | `monitoring_node` | 30 Hz | SPI ADC, `/proc` | battery, current, CPU, fault flags |
| `Dashboard` | `dashboard_node` | 15 Hz | whole robot | HTTP/WebSocket/MJPEG on :8080 |
| `AccelerationTest` | `acceleration_test_node` | 300 Hz | monitoring | `Drive.command()`, `test.jsonl` + `events.jsonl` |
| `SkidPadTest` | — (new) | 300 Hz | monitoring | `Drive.command()`, `test.jsonl` + `events.jsonl` |
| `ScreenshotGrabber` | `screenshot_grabber` | 1 Hz | both cameras | PNGs in `Deployment/Screenshots` |
| `Recorder` | — (new) | 30 Hz + policy and test ticks | whole robot | a run directory in `Deployment/Logs` |

### Why the policy tick is frame-driven

`PolicyRunner` waits for **both** cameras to deliver before it runs, rather than
ticking on a free-running 30 Hz timer. With the sensors free-running near
29.33 Hz and the loop at 30 Hz, the two settle into a fixed arbitrary phase and
the loop reads late: measured frame age at inference was 23.2 ms (left) and
14.3 ms (right), both with a std of only 1.4 ms. Waiting for the *second* arrival
puts inference right behind the newer frame and leaves the older one stale only
by the inter-sensor skew, taking mean age from ~18.8 ms to ~4.5 ms.

That is ~14 ms off a loop whose measured command-to-yaw latency is 100 ms against
the 33 ms `action_delay_ms` the policy was trained with. The remaining 8.9 ms of
left-vs-right skew is the phase difference between two free-running sensors with
no shared trigger — not fixable in software without dropping frames.

A stalled camera cannot wedge the loop: the wait times out after 1.5 nominal
periods and ticks anyway, so `update()` takes its "no fresh frame" branch,
commands zero, and keeps feeding the drive watchdog.

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
gate, and both calibration tests down to the run directory each one leaves.

```bash
cd Deployment/Python/tests
python test_stack.py                              # full stack, 20 checks
FAKE_ENCODER_STOP_AFTER=3 python test_modules.py  # module switches, 19 checks
python test_control_path.py                       # duty path + skid pad, 31 checks
python test_logging.py                            # --log run directory, 27 checks
```

Needs `numpy`, `opencv-python`, `aiohttp` and `onnxruntime`; the stubs supply
everything Pi-specific.
