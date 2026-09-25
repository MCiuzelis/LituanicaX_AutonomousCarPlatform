"""Driving algorithms. Register new ones in ALGORITHMS; run with: run.py --algo <name>."""
from .base import Algorithm
from .gate_follower import GateFollower
from .manual import Manual

ALGORITHMS: dict[str, type[Algorithm]] = {cls.name: cls for cls in (GateFollower, Manual)}


def make(name: str, **params) -> Algorithm:
    if name not in ALGORITHMS:
        raise SystemExit(f"unknown algorithm '{name}', have: {', '.join(sorted(ALGORITHMS))}")
    return ALGORITHMS[name](**params)
