# lituanicax_rc_car ROS 2 Deployment

This package contains the standalone ROS 2 runtime for the RC car.

## Nodes

- `tof_camera_node`: launched twice, once for CSI index `0`/left and once for CSI index `8`/right. Each launch entry is a separate OS process, preserving the Arducam SDK isolation used by the original code.
- `policy_runner_node`: subscribes to both ToF packet topics, runs the package-local `lituanicax_rc_car/policy.onnx` at 30 Hz, and publishes `/cmd_drive`.
- `monitoring_node`: owns SPI/ADC only, reads battery/current, monitors CPU usage and temperature, and publishes `/monitoring/telemetry`.
- `drive_node`: owns GPIO/PWM only, subscribes to `/cmd_drive`, `/autopilot/enabled`, and `/monitoring/telemetry`, then drives the motor/servo only when undervoltage and overcurrent are clear.
- `dashboard_node`: hosts the browser dashboard on port `8080`, publishes `/autopilot/enabled` from SPACE key events, serves MJPEG previews, and relays telemetry over WebSocket.

## Topics

- `/camera/left/tof_packet` (`std_msgs/UInt8MultiArray`): atomic packed depth/confidence frame for policy.
- `/camera/right/tof_packet` (`std_msgs/UInt8MultiArray`): atomic packed depth/confidence frame for policy.
- `/camera/left/preview/compressed` (`sensor_msgs/CompressedImage`): JPEG dashboard preview.
- `/camera/right/preview/compressed` (`sensor_msgs/CompressedImage`): JPEG dashboard preview.
- `/autopilot/enabled` (`std_msgs/Bool`): dashboard hold-to-run gate.
- `/cmd_drive` (`std_msgs/Float32MultiArray`): `[throttle_motor, steering]`.
- `/policy/action` (`std_msgs/Float32MultiArray`): `[throttle_motor, steering, throttle_raw, enabled]`.
- `/monitoring/telemetry` (`std_msgs/Float32MultiArray`): `[battery_voltage, motor_current, cpu_usage, cpu_temp_c, undervoltage, overcurrent]`.
- `/drive/telemetry` (`std_msgs/Float32MultiArray`): `[throttle, steering, enabled, safety_blocked]`.

## Build

From `Deployment/Ros2`:

```bash
colcon build --symlink-install
source install/setup.bash
```

The package installs its own policy artifact, so no old `Deployment` Python files are required at runtime.

Then open:

```text
http://<raspberry-pi-ip>:8080
```

## Camera Isolation Note

Do not run the two camera nodes as composable nodes in one process. The original implementation used separate OS processes because the Arducam SDK can share global frame buffers internally. These nodes preserve the same safety property: each camera process imports `ArducamDepthCamera`, opens one CSI index, deep-copies `depth_data` and `confidence_data`, releases the SDK frame, then publishes the copied arrays.
