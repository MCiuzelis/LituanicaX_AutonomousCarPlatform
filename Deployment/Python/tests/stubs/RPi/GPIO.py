"""Fake RPi.GPIO for off-Pi smoke testing."""
BCM = "BCM"
OUT = "OUT"
HIGH = 1
LOW = 0

state = {}


def setmode(mode): state["mode"] = mode
def setup(pin, mode): state[pin] = 0
def output(pin, value): state[pin] = value
def cleanup(): state.clear()
