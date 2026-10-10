"""hardware - drivers for everything on the Raspberry Pi 5 RC car.

Every class derives from hardware.base.Subsystem (available() / status() / close(), usable
as a context manager) and raises hardware.HardwareError on failure.

Board / sensors
    CpuMonitor      SoC temperature, ARM clock, throttling flags
    BoardPower      PMIC rail currents/voltages, total board power; PowerLogger samples it
    HailoBoard      Hailo-8 26 TOPS accelerator (identify / benchmark / measure-power)
    Camera          IMX708 via picamera2: frame acquisition, resolution + fps switching
    MCP3202         12-bit SPI ADC through the kernel IIO driver
    BatteryMonitor  battery voltage on MCP3202 CH1 behind a 15k/4.7k divider
    AS5600          12-bit magnetic rotary encoder on the motor shaft (I2C bus 1, 0x36)
    BNO085          9-DOF IMU with sensor fusion on SPI0 CE1 (worker process, SHTP/SH-2)
    CarFrame        IMU board frame -> car frame (mount axes + level reference, config/imu.json)

Actuators
    HardwarePWM     RP1 hardware PWM on GPIO 12/13 (18/19)
    Servo           50 Hz RC servo on top of HardwarePWM
    Steering        steering servo on GPIO12 with saved center/amplitude calibration
    HobbywingESC    QuicRun 1080 G2 brushed ESC: throttle pulse on GPIO13 (ESC is powered by hand)
    Drive           motor + encoder + steering: forward()/brake()/stop()/steer(), rps/rpm/turns

RP5Board            facade holding one lazily created instance of each board subsystem
"""
from .base import HardwareError, Subsystem
from .cpu import CpuMonitor
from .power import BoardPower, PowerLogger, PowerSample
from .hailo import HailoBoard, BenchmarkResult
from .camera import Camera, CameraMode, SensorMode
from .adc import MCP3202, BatteryMonitor, BatteryReading
from .pwm import HardwarePWM, Servo, PWM_PINS
from .encoder import AS5600, EncoderReading
from .bno085 import BNO085, Reading as ImuReading
from .imu_frame import CarFrame
from .esc import HobbywingESC
from .steering import Steering
from .drive import Drive
from .board import RP5Board

__all__ = [
    "HardwareError", "Subsystem",
    "CpuMonitor", "BoardPower", "PowerLogger", "PowerSample",
    "HailoBoard", "BenchmarkResult",
    "Camera", "CameraMode", "SensorMode",
    "MCP3202", "BatteryMonitor", "BatteryReading",
    "HardwarePWM", "Servo", "PWM_PINS",
    "AS5600", "EncoderReading",
    "BNO085", "ImuReading", "CarFrame",
    "HobbywingESC", "Steering", "Drive",
    "RP5Board",
]
__version__ = "0.2.0"
