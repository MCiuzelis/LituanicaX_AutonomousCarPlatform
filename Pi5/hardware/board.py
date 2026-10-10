"""RP5Board: one object that owns every subsystem, created lazily on first use."""
from .adc import MCP3202, BatteryMonitor
from .bno085 import BNO085
from .camera import Camera
from .cpu import CpuMonitor
from .encoder import AS5600
from .hailo import HailoBoard
from .power import BoardPower
from .pwm import Servo, HardwarePWM

# GPIO13 also has a PWM channel but it is the ESC throttle on this car (see esc.py / drive.py);
# opening it as a plain Servo would drop the ESC's neutral signal on close(), so it is not a
# default servo pin. Pass servo_pins=(12, 13) explicitly for a bench setup without the ESC.
SERVO_PINS = (12,)


class RP5Board:
    """
        from hardware import RP5Board
        with RP5Board() as b:
            print(b.cpu.temperature_c(), b.power.sample().total_w)
            print(b.battery.voltage())           # ~7.0 V
            b.servo(12).set_angle(30)            # steering servo (raw angle, no calibration)
            b.camera.set_mode(1280, 720, 60)
            frame = b.camera.capture_frame()
            print(b.hailo.benchmark(seconds=5))
    """

    def __init__(self, vref=3.3, battery_channel=1, r_top=15_000, r_bottom=4_700,
                 servo_pins=SERVO_PINS, servo_kwargs=None):
        self._vref = vref
        self._batt_cfg = dict(channel=battery_channel, r_top=r_top, r_bottom=r_bottom)
        self._servo_pins = tuple(servo_pins)
        self._servo_kwargs = servo_kwargs or {}
        self._cpu = self._power = self._hailo = self._camera = self._adc = self._battery = None
        self._encoder = None
        self._imu = None
        self._servos = {}

    @property
    def cpu(self) -> CpuMonitor:
        if self._cpu is None:
            self._cpu = CpuMonitor()
        return self._cpu

    @property
    def power(self) -> BoardPower:
        if self._power is None:
            self._power = BoardPower()
        return self._power

    @property
    def hailo(self) -> HailoBoard:
        if self._hailo is None:
            self._hailo = HailoBoard()
        return self._hailo

    @property
    def camera(self) -> Camera:
        if self._camera is None:
            self._camera = Camera()
        return self._camera

    @property
    def adc(self) -> MCP3202:
        if self._adc is None:
            self._adc = MCP3202(vref=self._vref)
        return self._adc

    @property
    def battery(self) -> BatteryMonitor:
        if self._battery is None:
            self._battery = BatteryMonitor(self.adc, **self._batt_cfg)
        return self._battery

    @property
    def encoder(self) -> AS5600:
        if self._encoder is None:
            self._encoder = AS5600()
        return self._encoder

    @property
    def imu(self) -> BNO085:
        """BNO085 (starts its worker process on first use; enable sensors yourself)."""
        if self._imu is None:
            self._imu = BNO085()
        return self._imu

    def servo(self, gpio) -> Servo:
        if gpio not in self._servo_pins:
            raise ValueError("servo pins configured: %s" % (self._servo_pins,))
        if gpio not in self._servos:
            self._servos[gpio] = Servo(gpio, **self._servo_kwargs)
        return self._servos[gpio]

    @property
    def servos(self):
        return {g: self.servo(g) for g in self._servo_pins}

    def pwm(self, gpio, frequency_hz=50.0) -> HardwarePWM:
        """Raw PWM on a pin instead of the servo abstraction (pin must not be in use as a servo)."""
        if gpio in self._servos:
            raise ValueError("GPIO%d is already opened as a servo" % gpio)
        return HardwarePWM(gpio, frequency_hz)

    def summary(self):
        """Status of every subsystem; failures are captured instead of raised."""
        out = {}
        for name, getter in (("cpu", lambda: self.cpu), ("power", lambda: self.power),
                             ("hailo", lambda: self.hailo), ("adc", lambda: self.adc),
                             ("battery", lambda: self.battery), ("encoder", lambda: self.encoder),
                             ("camera", lambda: self.camera)):
            try:
                out[name] = getter().status()
            except Exception as e:  # noqa: BLE001 - report, don't abort the summary
                out[name] = {"error": str(e).splitlines()[0]}
        out["servos"] = {g: s.status() for g, s in self._servos.items()}
        return out

    def close(self):
        for s in self._servos.values():
            s.close()
        self._servos.clear()
        if self._camera is not None:
            self._camera.close()
        if self._encoder is not None:
            self._encoder.close()
        if self._imu is not None:
            self._imu.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
