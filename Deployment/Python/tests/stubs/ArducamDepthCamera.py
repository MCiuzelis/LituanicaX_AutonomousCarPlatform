"""Fake Arducam ToF SDK: 240x180 synthetic depth frames at ~30 fps."""
import time

import numpy as np


class Connection:
    CSI = "CSI"


class FrameType:
    DEPTH = "DEPTH"


class Control:
    RANGE = "RANGE"


class DepthData:
    def __init__(self, depth, confidence):
        self.depth_data = depth
        self.confidence_data = confidence


class ArducamCamera:
    def __init__(self):
        self._range = 2000
        self._n = 0

    def open(self, connection, index):
        self.index = index
        return 0

    def start(self, frame_type): pass
    def setControl(self, control, value): self._range = value
    def getControl(self, control): return self._range

    def requestFrame(self, timeout_ms):
        time.sleep(1 / 30)
        self._n += 1
        depth = np.full((180, 240), 800.0 + (self._n % 100), dtype=np.float32)
        confidence = np.full((180, 240), 200, dtype=np.uint8)
        return DepthData(depth, confidence)

    def releaseFrame(self, frame): pass
    def stop(self): pass
    def close(self): pass
