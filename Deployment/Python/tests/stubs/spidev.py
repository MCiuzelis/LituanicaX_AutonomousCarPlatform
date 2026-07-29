"""Fake spidev: battery ~7.6 V, low current."""


class SpiDev:
    def __init__(self):
        self.max_speed_hz = 0
        self._closed = False

    def open(self, bus, device): pass

    def xfer2(self, data):
        config = data[1]
        raw = 40 if config == 0x80 else 2457  # current channel, battery channel (~7.6 V)
        return [0, (raw >> 8) & 0x0F, raw & 0xFF]

    def close(self): self._closed = True
