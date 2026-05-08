from __future__ import annotations

import os
import sys
from pathlib import Path


def discover_deployment_dir(configured: str = "") -> Path:
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())

    env_dir = os.environ.get("LITUANICAX_DEPLOYMENT_DIR", "")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "Ros2":
            candidates.append(parent.parent)
        candidates.append(parent / "Deployment")

    candidates.append(Path.cwd() / "Deployment")
    candidates.append(Path.cwd())

    for candidate in candidates:
        if (candidate / "hardware.py").exists() and (candidate / "Policy" / "policy.onnx").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Deployment directory. Set the 'deployment_dir' parameter "
        "or LITUANICAX_DEPLOYMENT_DIR."
    )


def discover_policy_path(configured_policy_path: str, deployment_dir: Path) -> Path:
    if configured_policy_path:
        return Path(configured_policy_path).expanduser()
    return deployment_dir / "Policy" / "policy.onnx"


def import_robot_class(deployment_dir: Path):
    deployment_str = str(deployment_dir)
    if deployment_str not in sys.path:
        sys.path.insert(0, deployment_str)
    from hardware import Robot

    return Robot
