from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PACKAGE_NAME = "lituanicax_rc_car"


def _default_deployment_dir() -> str:
    env_dir = os.environ.get("LITUANICAX_DEPLOYMENT_DIR")
    if env_dir:
        return env_dir

    launch_file = Path(__file__).resolve()
    for parent in launch_file.parents:
        if parent.name == "Ros2":
            return str(parent.parent)

    return str(Path.cwd() / "Deployment")


def generate_launch_description() -> LaunchDescription:
    deployment_dir = LaunchConfiguration("deployment_dir")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "deployment_dir",
                default_value=_default_deployment_dir(),
                description="Path to the existing Deployment directory containing hardware.py and Policy/policy.onnx.",
            ),
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
                        "deployment_dir": deployment_dir,
                        "policy_hz": 30,
                        "camera_timeout_sec": 0.5,
                        "onnx_threads": 4,
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="hardware_node",
                name="hardware",
                output="screen",
                parameters=[
                    {
                        "deployment_dir": deployment_dir,
                        "command_timeout_sec": 0.25,
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
