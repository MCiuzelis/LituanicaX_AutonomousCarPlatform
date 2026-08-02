"""Entry point: parse the module switches, build a Robot, run it.

    python -m lituanicax.main                                # everything on
    python -m lituanicax.main --disable-tof --disable-policy # bench test
    python -m lituanicax.main --disable policy --enable acceleration_test
    python -m lituanicax.main --log                          # record the run

A calibration test (`acceleration_test`, `skid_pad_test`) turns the recorder on
by itself, so its numbers always land in `Deployment/Logs/run_<timestamp>/`
whether or not `--log` was passed.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .logging_utils import configure, get_logger
from .modules.recorder import DEFAULT_MAX_MB
from .robot import MODULE_NAMES, Robot, RobotSettings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lituanicax",
        description="LituanicaX RC car deployment stack (vanilla Python).",
    )

    parser.add_argument(
        "--enable",
        action="append",
        default=[],
        metavar="MODULE",
        choices=MODULE_NAMES,
        help="Turn a module on (repeatable). Choices: " + ", ".join(MODULE_NAMES),
    )
    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        metavar="MODULE",
        choices=MODULE_NAMES,
        help="Turn a module off (repeatable).",
    )
    parser.add_argument(
        "--list-modules", action="store_true", help="Print the module list and exit."
    )

    # Shorthands matching the old launch arguments.
    parser.add_argument("--disable-tof", action="store_true", help="Same as --disable tof_cameras")
    parser.add_argument("--disable-policy", action="store_true", help="Same as --disable policy")
    parser.add_argument(
        "--enable-accel-test", action="store_true", help="Same as --enable acceleration_test"
    )
    parser.add_argument(
        "--enable-skid-test", action="store_true", help="Same as --enable skid_pad_test"
    )
    parser.add_argument(
        "--enable-screenshot", action="store_true", help="Same as --enable screenshot_grabber"
    )

    parser.add_argument("--host", default=None, help="Dashboard bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Dashboard port (default 8080)")
    parser.add_argument("--policy-path", default=None, help="Override the ONNX policy file")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")

    # --- flight recorder ---
    parser.add_argument(
        "--log",
        action="store_true",
        help="Record the whole run to Deployment/Logs/run_<timestamp>/: full ToF "
        "frames, policy observations and actions, and every dashboard number, "
        "all keyed on one monotonic clock. Same as --enable recorder. Implied by "
        "the calibration tests.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Cancel --log (the run script and the calibration tests turn it on, "
        "so this turns it back off).",
    )
    parser.add_argument("--log-dir", default=None, help="Write the run into this directory instead")
    parser.add_argument(
        "--log-max-mb",
        type=float,
        default=None,
        help="Stop writing frames/observations after this many MB "
        f"(default {DEFAULT_MAX_MB:.0f}; scalar rows always continue)",
    )

    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> RobotSettings:
    settings = RobotSettings()

    if args.disable_tof:
        settings.tof_cameras = False
    if args.disable_policy:
        settings.policy = False
    if args.enable_accel_test:
        settings.acceleration_test = True
    if args.enable_skid_test:
        settings.skid_pad_test = True
    if args.enable_screenshot:
        settings.screenshot_grabber = True

    for name in args.disable:
        setattr(settings, name, False)
    for name in args.enable:
        setattr(settings, name, True)

    # A calibration test exists to produce numbers, and a number nobody can go
    # back and check is worth very little — so enabling one turns the recorder
    # on exactly the way --log does, however the module was switched on.
    # --no-log (and an explicit `--disable recorder`) still wins, so `run` can
    # pass --log unconditionally and stay overridable on the command line:
    # `Deployment/Python/run --no-log`.
    log_off = args.no_log or "recorder" in args.disable
    if (args.log or settings.acceleration_test or settings.skid_pad_test) and not log_off:
        settings.recorder = True
    if log_off:
        settings.recorder = False

    if args.host is not None:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    if args.policy_path is not None:
        settings.policy_path = args.policy_path
    if args.log_dir is not None:
        settings.log_dir = args.log_dir
    if args.log_max_mb is not None:
        settings.log_max_mb = args.log_max_mb

    return settings


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure(level=logging.DEBUG if args.debug else logging.INFO)
    log = get_logger("main")

    settings = settings_from_args(args)

    if args.list_modules:
        for name in MODULE_NAMES:
            state = "enabled" if getattr(settings, name) else "disabled"
            print(f"  {name:<20} {state}")
        return 0

    robot = Robot(settings)
    try:
        robot.run()
    except Exception as exc:
        log.error(f"Fatal: {exc}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
