# LituanicaX ROS 2 Deployment Workspace

This workspace is rebuilt from the current top-level `LituanicaX_RC_Car`
package and uses the source-built ROS 2 Jazzy underlay at `~/ros2_jazzy` by
default.

## Rebuild

From the repository root:

```bash
Deployment/install.sh
```

If ROS 2 is installed somewhere else:

```bash
ROS_UNDERLAY=/path/to/ros2_jazzy Deployment/install.sh
```

The installer:

- installs Raspberry Pi, Arducam, OpenCV, GPIO, and SPI system dependencies;
- creates `Deployment/.venv` and installs Arducam, OpenCV, aiohttp, NumPy, and ONNX Runtime Python packages there;
- uses `Deployment/Ros2/src/LituanicaX_RC_Car/` as the package source, or synchronizes the top-level `LituanicaX_RC_Car/` into it when that mirror exists;
- runs `colcon build --symlink-install` in `Deployment/Ros2`.

## Run

From the repository root, use the helper command:

```bash
launch
```

Or run the steps manually:

```bash
source Deployment/.venv/bin/activate
export PYTHONPATH="$(Deployment/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])'):${PYTHONPATH:-}"
export ROS_LOG_DIR="$PWD/Deployment/Ros2/logs"
source ~/ros2_jazzy/install/setup.bash
source Deployment/Ros2/install/setup.bash
ros2 launch lituanicax_rc_car ros_autopilot.launch.py
```

Then open:

```text
http://<raspberry-pi-ip>:8080
```

## Hardware Setup

The drive node expects Pi 5 PIO PWM overlays for GPIO12 and GPIO13. Keep the
PWM setup from `Deployment/README.md` in place before running the full launch.

## Notes

Generated `build/`, `install/`, and `log/` directories are rebuildable. The
deployment package source lives in `Deployment/Ros2/src/LituanicaX_RC_Car/`.
