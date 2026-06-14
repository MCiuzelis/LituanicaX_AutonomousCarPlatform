from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PACKAGE_NAME = "lituanicax_rc_car"


def generate_launch_description() -> LaunchDescription:
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    disable_tof = LaunchConfiguration("disable_tof")
    disable_policy = LaunchConfiguration("disable_policy")
    enable_accel_test = LaunchConfiguration("enable_accel_test")
    enable_screenshot = LaunchConfiguration("enable_screenshot")

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
            SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "0"),
            DeclareLaunchArgument("host", default_value="0.0.0.0"),
            DeclareLaunchArgument("port", default_value="8080"),
            DeclareLaunchArgument("disable_tof", default_value="false"),
            DeclareLaunchArgument("disable_policy", default_value="false"),
            DeclareLaunchArgument("enable_accel_test", default_value="false"),
            DeclareLaunchArgument("enable_screenshot", default_value="false"),
            Node(
                package=PACKAGE_NAME,
                executable="tof_camera_node",
                name="tof_camera_left",
                output="screen",
                condition=UnlessCondition(disable_tof),
                parameters=[
                    {
                        "side": "left",
                        "frame_id": "tof_left",
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="tof_camera_node",
                name="tof_camera_right",
                output="screen",
                condition=UnlessCondition(disable_tof),
                parameters=[
                    {
                        "side": "right",
                        "frame_id": "tof_right",
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="policy_runner_node",
                name="policy_runner",
                output="screen",
                condition=UnlessCondition(disable_policy),
                parameters=[
                    {
                        "policy_hz": 30,
                        "camera_timeout_sec": 0.5,
                        "onnx_threads": 1,
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
                        "telemetry_hz": 30,
                    }
                ],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="encoder_node",
                name="encoder",
                output="screen",
                parameters=[
                    {
                        "encoder_hz": 300.0,
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
                        "telemetry_hz": 30,
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
            Node(
                package=PACKAGE_NAME,
                executable="screenshot_grabber",
                name="screenshot_grabber",
                output="screen",
                condition=IfCondition(enable_screenshot),
            ),
            Node(
                package=PACKAGE_NAME,
                executable="acceleration_test_node",
                name="acceleration_test",
                output="screen",
                condition=IfCondition(enable_accel_test),
            ),
        ]
    )
