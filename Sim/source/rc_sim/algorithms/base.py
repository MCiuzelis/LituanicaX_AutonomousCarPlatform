"""Base class every driving algorithm implements.

An algorithm only sees rc_sim.interface types (plain Python + numpy), exactly what the car can
provide, so the same file can be copied to the Pi and fed from Drive / AppleDetector / the IMU.

    class MyAlgo(Algorithm):
        name = "my_algo"
        def reset(self, car): ...                       # called at the start of every run
        def step(self, frame) -> Command: ...           # called at car.control_hz
        def debug(self) -> dict: ...                    # optional, drawn by the viewer

debug() keys the viewer understands (all points in the CAR frame: x forward, y left, metres,
origin at the rear-axle centre on the ground):
    "landmarks": [(x, y), ...]   "left": [...]   "right": [...]   "path": [...]   "target": (x, y)
    "text": "one line of status"
"""
from __future__ import annotations

from ..interface import CarInfo, Command, SensorFrame


class Algorithm:
    name = "base"

    def __init__(self, **params):
        self.params = params
        self.car: CarInfo | None = None

    def reset(self, car: CarInfo) -> None:
        self.car = car

    def step(self, frame: SensorFrame) -> Command:
        raise NotImplementedError

    def debug(self) -> dict:
        return {}

    def close(self) -> None:
        pass
