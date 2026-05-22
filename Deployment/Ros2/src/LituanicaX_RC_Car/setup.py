from glob import glob
from setuptools import find_packages, setup


PACKAGE_NAME = "lituanicax_rc_car"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
    ],
    package_data={PACKAGE_NAME: ["dashboard.html", "policy.onnx"]},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="LituanicaX",
    maintainer_email="robot@example.com",
    description="ROS 2 nodes for the LituanicaX RC car deployment stack.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tof_camera_node = lituanicax_rc_car.tof_camera_node:main",
            "policy_runner_node = lituanicax_rc_car.policy_runner_node:main",
            "drive_node = lituanicax_rc_car.drive_node:main",
            "monitoring_node = lituanicax_rc_car.monitoring_node:main",
            "dashboard_node = lituanicax_rc_car.dashboard_node:main",
        ],
    },
)
