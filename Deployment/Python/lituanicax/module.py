"""Base class for every robot module.

A module is what a ROS node was: a named object with its own thread that ticks
at a fixed rate.  The differences are that modules talk to each other by
holding references and calling methods, and that `Robot` owns them all.

Subclasses implement:
    setup()     once, before the thread starts
    update()    every tick, at `rate_hz`
    shutdown()  once, after the thread stops

Modules that block on an external source (the ToF cameras, the dashboard web
server) override `run()` instead of `update()`.
"""

from __future__ import annotations

import threading
import time

from .logging_utils import get_logger
from .realtime import set_thread_realtime


class Module:
    #: Raise this module's thread to SCHED_RR so its timing survives CPU load.
    realtime = False

    def __init__(self, name: str, rate_hz: float) -> None:
        self.name = name
        self.rate_hz = float(rate_hz)
        self.log = get_logger(name)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # --- lifecycle hooks for subclasses ---

    def setup(self) -> None:
        """Acquire hardware and other resources. Raise to abort startup."""

    def update(self) -> None:
        """One tick of work, called at `rate_hz`."""

    def shutdown(self) -> None:
        """Release whatever `setup()` acquired. Always called after stop()."""

    # --- lifecycle, driven by Robot ---

    def start(self) -> None:
        try:
            self.setup()
        except Exception:
            self.shutdown()  # release whatever setup() managed to acquire
            raise
        self._thread = threading.Thread(target=self._thread_main, name=self.name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self.shutdown()

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def sleep(self, seconds: float) -> None:
        """Interruptible sleep: returns early when the module is stopping."""
        if seconds > 0:
            self._stop_event.wait(seconds)

    def _thread_main(self) -> None:
        if self.realtime and not set_thread_realtime():
            self.log.warning("Could not raise thread to real-time priority")
        try:
            self.run()
        except Exception as exc:
            self.log.error(f"Module thread died: {exc}", exc_info=True)

    def run(self) -> None:
        """Fixed-rate loop. Override for modules that block on their own source."""
        period = 1.0 / self.rate_hz
        next_tick = time.monotonic()

        while not self.stopping:
            try:
                self.update()
            except Exception as exc:
                self.log.error(f"update() failed: {exc}", exc_info=True)

            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                self.sleep(remaining)
            else:
                # Fell behind: drop the missed ticks instead of bursting.
                next_tick = time.monotonic()
