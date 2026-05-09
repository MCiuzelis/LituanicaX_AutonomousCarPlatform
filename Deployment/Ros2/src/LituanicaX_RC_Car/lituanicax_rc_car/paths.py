from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


PACKAGE_NAME = "lituanicax_rc_car"


def package_share_dir() -> Path:
    candidates: list[Path] = []

    try:
        candidates.append(Path(get_package_share_directory(PACKAGE_NAME)))
    except PackageNotFoundError:
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "package.xml").exists() and (parent / "policy").is_dir():
            candidates.append(parent)

    for candidate in candidates:
        if (candidate / "policy" / "policy.onnx").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the lituanicax_rc_car package policy directory. "
        "Rebuild and source the ROS workspace, or set the 'policy_path' parameter explicitly."
    )


def discover_policy_path(configured_policy_path: str = "") -> Path:
    if configured_policy_path:
        return Path(configured_policy_path).expanduser()
    return package_share_dir() / "policy" / "policy.onnx"
