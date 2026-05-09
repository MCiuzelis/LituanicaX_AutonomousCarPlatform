from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PACKAGE_NAME = "lituanicax_rc_car"


def generate_launch_description() -> LaunchDescription:
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
            SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "0"),
            DeclareLaunchArgument("host", default_value="0.0.0.0"),
            DeclareLaunchArgument("port", default_value="8080"),
            Node(
                package=PACKAGE_NAME,
                executable="tof_camera_node",
                name="tof_camera_left",
                output="screen",
                parameters=[
                    {
                        "side": "left",
                        "camera_index": 0,
                        "frame_id": "tof_left",
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="tof_camera_node",
                name="tof_camera_right",
                output="screen",
                parameters=[
                    {
                        "side": "right",
                        "camera_index": 8,
                        "frame_id": "tof_right",
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="policy_runner_node",
                name="policy_runner",
                output="screen",
                parameters=[
                    {
                        "policy_hz": 30,
                        "camera_timeout_sec": 0.5,
                        "onnx_threads": 4,
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="drive_node",
                name="drive",
                output="screen",
                parameters=[
                    {
                        "command_timeout_sec": 0.25,
                        "monitoring_timeout_sec": 0.5,
                        "telemetry_hz": 15,
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="monitoring_node",
                name="monitoring",
                output="screen",
                parameters=[
                    {
                        "telemetry_hz": 15,
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="dashboard_node",
                name="dashboard",
                output="screen",
                parameters=[
                    {
                        "host": host,
                        "port": port,
                        "stream_fps": 15,
                    }
                ],
            ),
        ]
    )
