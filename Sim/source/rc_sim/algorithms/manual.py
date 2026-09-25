"""Drive by hand (W throttle, S brake, A/D steer, needs pynput + a display) to look at the boxes."""
from __future__ import annotations

import threading

from ..interface import Command, SensorFrame
from .base import Algorithm


class Manual(Algorithm):
    name = "manual"

    def __init__(self, throttle=0.25, **params):
        super().__init__(throttle=float(throttle), **params)
        self._down: set = set()
        self._lock = threading.Lock()
        self._listener = None
        try:
            from pynput import keyboard

            def key(k):
                return getattr(k, "char", None) and k.char.lower()

            def press(k):
                with self._lock:
                    self._down.add(key(k))

            def release(k):
                with self._lock:
                    self._down.discard(key(k))

            self._listener = keyboard.Listener(on_press=press, on_release=release)
            self._listener.start()
            print("[manual] W throttle, S brake, A/D steer")
        except Exception as e:  # noqa: BLE001 - no X server / no pynput
            print(f"[manual] keyboard unavailable ({e}); the car will sit still")

    def step(self, frame: SensorFrame) -> Command:
        with self._lock:
            d = set(self._down)
        throttle = self.params["throttle"] if "w" in d else (-1.0 if "s" in d else 0.0)
        steer = (1.0 if "d" in d else 0.0) - (1.0 if "a" in d else 0.0)
        return Command(throttle, steer)

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
