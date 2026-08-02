# LituanicaX Deployment

Control stack for the RL-trained autonomous RC car, running on a **Raspberry Pi 5**.

It exists in two interchangeable forms with identical behaviour:

- **`Ros2/`** — the ROS 2 node graph (documented below).
- **`Python/`** — the same stack as plain Python classes, no ROS. See
  [Python/README.md](./Python/README.md).

## Overview

```
Deployment/
├── install.sh                 # Full system setup: deps, venv, colcon build
├── launch                     # One-shot script: build + source + launch ROS stack
├── hardware_pwm_guide.md      # Native PWM setup for GPIO12/GPIO13 (motor + servo)
├── ros2_jazzy_build_guide.md  # Build ROS 2 Jazzy from source on Pi 5 / Debian Trixie
├── Ros2/                      # ROS 2 colcon workspace
│   └── src/LituanicaX_RC_Car/ # ROS package with all deployment nodes
├── Python/                    # Same stack without ROS: one process, plain classes
│   ├── run                    # Launcher: Deployment/Python/run [--disable-tof ...]
│   └── lituanicax/            # Robot + one module class per ROS node
└── old_python/                # Legacy standalone Python deployment (deprecated)
    ├── autopilot.py
    ├── hardware.py
    └── HardwareTests/
```

## Quick Start

### Prerequisites

- Raspberry Pi 5 running Raspberry Pi OS (Debian Trixie, kernel 6.12+)
- ROS 2 Jazzy built from source at `~/ros2_jazzy` (see [ros2_jazzy_build_guide.md](./ros2_jazzy_build_guide.md))
- PWM0 overlay enabled for GPIO12/GPIO13 (see [hardware_pwm_guide.md](./hardware_pwm_guide.md))

### Install

```bash
Deployment/install.sh
```

This installs system dependencies (Arducam SDK, OpenCV, lgpio, spidev), creates a Python venv at `Deployment/.venv`, and runs `colcon build` on the ROS workspace.

### Run

```bash
Deployment/launch
```

Or manually:

```bash
source ~/ros2_jazzy/install/setup.bash
source Deployment/Ros2/install/setup.bash
ros2 launch lituanicax_rc_car ros_autopilot.launch.py
```

Open the dashboard at `http://<raspberry-pi-ip>:8080`.

To run the same stack without ROS (no colcon build, no sourcing):

```bash
Deployment/Python/run            # records the run to Deployment/Logs/
Deployment/Python/run --no-log   # without recording
```

Each recorded run holds the full ToF frames, the policy's inputs and outputs and
every dashboard number, all keyed on one clock — see
[`Python/README.md`](Python/README.md#logging-a-run---log).

## ROS Nodes

The launch file (`ros_autopilot.launch.py`) starts six nodes:

| Node | Description |
|------|-------------|
| `tof_camera_node` (×2) | Dual ArduCam ToF depth cameras (CSI index 0 and 8). Each runs as a separate process for SDK isolation. Publishes depth/confidence packets and JPEG previews. |
| `policy_runner_node` | Subscribes to both camera topics, runs the RL policy (`policy.onnx`) at 30 Hz, publishes `/cmd_drive`. |
| `drive_node` | Owns GPIO/PWM. Drives motor (H-bridge via GPIO) and servo (PWM sysfs on GPIO12/13). Enforces safety: blocks drive on undervoltage, overcurrent, or command timeout. |
| `monitoring_node` | Reads battery voltage and motor current via SPI ADC, monitors CPU usage and temperature. Publishes `/monitoring/telemetry`. |
| `dashboard_node` | Serves a browser dashboard on port 8080. Hold SPACE to enable autopilot. Streams MJPEG camera feeds and relays telemetry over WebSocket. |

### Key Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/{left,right}/tof_packet` | `UInt8MultiArray` | Packed depth (float32) + confidence (uint8) frames |
| `/camera/{left,right}/preview/compressed` | `CompressedImage` | JPEG depth previews for dashboard |
| `/cmd_drive` | `Float32MultiArray` | `[throttle, steering]` from policy |
| `/autopilot/enabled` | `Bool` | Dashboard SPACE gate |
| `/monitoring/telemetry` | `Float32MultiArray` | `[battery_v, motor_current, cpu_usage, cpu_temp, undervoltage, overcurrent]` |
| `/drive/telemetry` | `Float32MultiArray` | `[throttle, steering, enabled, safety_blocked]` |

## Hardware Configuration

All pin assignments and calibration live in `lituanicax_rc_car/common.py`:

- **Motor:** H-bridge via GPIO23/24, PWM on sysfs `pwmchip0/pwm0` (20 kHz)
- **Servo:** PWM on sysfs `pwmchip0/pwm1` (50 Hz, 500-2500 µs, center 1500 µs with -130 µs offset)
- **ADC:** SPI bus 0, MCP3008-style, reads battery voltage (divider: 180k/63.4k) and motor current (1.5 Ω sense, 7000:1 ratio)
- **Safety:** Undervoltage < 6.0 V, overcurrent > 5.0 A — both block drive commands
- **Max throttle:** clamped to 6% via `MAX_THROTTLE = 0.06`

## Depth Camera Pipeline

The policy observation is built from dual ToF cameras (Arducam, 240×180 raw, 2 m range):

1. Orient frame (flip + optional vertical flip)
2. Low-confidence pixels → max range; crop top 35% / bottom 45% (180 → 36 rows)
3. Bilinear downsample to 120×18 = 2160 values per camera
4. Normalize: `1.0 - depth_m / 2.0` (1.0 = obstacle at 0 m, 0.0 = clear at 2 m)
5. Concatenate → **4331**-dim observation:

```
[ left depth 2160 | right depth 2160 | wheel speed 1 | action history 5×2 ]
```

The action history holds the last 5 raw `(throttle, steer)` commands, oldest
first, and is zeroed whenever autopilot is disabled.

Every number above must match `ConeTrackEnvCfg` in
`Training/source/cone_track/env.py` — the policy is a fixed-width MLP, so a
disagreement is either a startup shape error or, worse, a silently misread
observation. The constants block at the top of
`Python/lituanicax/modules/tof_camera.py` is the single place to change them,
and `PolicyRunner.setup()` checks them against the loaded ONNX file at startup.

## Policy

The ONNX policy file lives at `lituanicax_rc_car/policy.onnx` and is included in the ROS package. It is exported from the trained PPO model via `Training/play.py`.
