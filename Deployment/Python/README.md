# LituanicaX Deployment — vanilla Python

The same control stack as `Deployment/Ros2`, rewritten as plain Python classes.
No ROS, no DDS, no colcon: one process, one thread per module, and modules that
call each other's methods directly.

The ROS workspace is untouched and still works — this is a parallel
implementation, not a replacement.

## Run

```bash
Deployment/Python/run                       # everything on
Deployment/Python/run --list-modules        # show what is on/off
```

Then open `http://<raspberry-pi-ip>:8080` and hold SPACE to drive.

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

In code it is the same thing:

```python
from lituanicax.robot import Robot, RobotSettings

Robot(RobotSettings(policy=False, acceleration_test=True)).run()
```

The acceleration test takes over the drive, so disable the policy when you run
it (`--disable-policy --enable-accel-test`); `Robot` warns if both are on.

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
│   ├── logging_utils.py     # ROS-style console logging
│   ├── dashboard.html       # unchanged from the ROS package
│   ├── policy.onnx          # unchanged from the ROS package
│   └── modules/             # one class per former node
└── tests/                   # off-hardware verification (stubbed sensors)
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

* **Cameras are threads, not processes.** ROS ran each `tof_camera_node` as its
  own process. Here they are threads that deep-copy every frame before calling
  `releaseFrame()` — the same guard the ROS code relied on against the SDK's
  shared buffers. If the two cameras ever interfere, split `TofCamera` back out
  into a subprocess rather than reintroducing a message bus.
* **Real-time priority is per-thread.** The ROS drive process ran under
  `SCHED_RR`; everything now shares one process, so `Drive` and `PolicyRunner`
  raise only their own threads (`realtime = True`). It needs `CAP_SYS_NICE`;
  without it you get one warning and normal scheduling, as before.

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
```

Needs `numpy`, `opencv-python`, `aiohttp` and `onnxruntime`; the stubs supply
everything Pi-specific.
