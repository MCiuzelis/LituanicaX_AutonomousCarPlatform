import os
import time
import subprocess
import RPi.GPIO as GPIO

PWM_CHIP_PATH = "/sys/class/pwm/pwmchip0"


class Robot:
    def __init__(
        self,
        pin_ina=24,
        pin_inb=23,
        motor_gpio=12,
        servo_gpio=13,
        motor_pwm_freq=20000,
        servo_pwm_freq=50,
        servo_min_us=500,
        servo_max_us=2500,
        servo_center_us=1500,
        servo_offset_us=-130,
        servo_amplitude=0.25,
        servo_direction=1,
        motor_direction=-1,
        max_throttle=0.06,
        adc_channel_current=0,
        adc_channel_battery=1,
        adc_v_ref=3.3,
        r_divider_top=180000,
        r_divider_bottom=63400,
        r_sense=1500,
        k_ratio=7000,
        adc_max=4095,
    ):
        self.pin_ina = pin_ina
        self.pin_inb = pin_inb
        self.motor_gpio = motor_gpio
        self.servo_gpio = servo_gpio
        self.motor_pwm_freq = motor_pwm_freq
        self.servo_pwm_freq = servo_pwm_freq
        self.servo_min_us = servo_min_us
        self.servo_max_us = servo_max_us
        self.servo_center_us = servo_center_us
        self.servo_offset_us = servo_offset_us
        self.servo_amplitude = servo_amplitude
        self.servo_direction = servo_direction
        self.motor_direction = motor_direction
        self.max_throttle = max(0.0, min(1.0, max_throttle))

        self.adc_channel_current = adc_channel_current
        self.adc_channel_battery = adc_channel_battery
        self.adc_v_ref = adc_v_ref
        self.r_divider_top = r_divider_top
        self.r_divider_bottom = r_divider_bottom
        self.r_sense = r_sense
        self.k_ratio = k_ratio
        self.adc_max = adc_max

        self._setup_motor_hardware_pwm()
        self._setup_servo_software_pwm()

    # --- pinctrl helpers ---

    def _pinctrl_set(self, pin, *args):
        subprocess.run(["pinctrl", "set", str(pin), *args], check=True)

    def _pinctrl_output(self, pin, high):
        self._pinctrl_set(pin, "op", "dh" if high else "dl")

    # --- Motor: hardware PWM via sysfs (channel 0, 20 kHz) ---

    def _export_pwm(self, channel):
        path = f"{PWM_CHIP_PATH}/pwm{channel}"
        if not os.path.exists(path):
            with open(f"{PWM_CHIP_PATH}/export", "w") as f:
                f.write(str(channel))
            time.sleep(0.1)
        return path

    def _pwm_write(self, path, attr, value):
        with open(f"{path}/{attr}", "w") as f:
            f.write(str(value))

    def _setup_motor_hardware_pwm(self):
        self._pinctrl_set(self.motor_gpio, "a0", "pd")
        period_ns = int(1_000_000_000 / self.motor_pwm_freq)
        path = self._export_pwm(channel=0)
        self._pwm_write(path, "enable", 0)
        self._pwm_write(path, "period", period_ns)
        self._pwm_write(path, "duty_cycle", 0)
        self._pwm_write(path, "enable", 1)
        self.motor_pwm_path = path
        self.motor_period_ns = period_ns

        self._pinctrl_output(self.pin_ina, high=False)
        self._pinctrl_output(self.pin_inb, high=False)

    # --- Servo: software PWM via RPi.GPIO (50 Hz) ---

    def _setup_servo_software_pwm(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_gpio, GPIO.OUT)
        self.servo_pwm = GPIO.PWM(self.servo_gpio, self.servo_pwm_freq)
        self.servo_pwm.start(0)

    # --- ADC stubs ---

    def _read_adc(self, channel=0):
        return 0

    def _adc_to_voltage(self, raw):
        return (raw / self.adc_max) * self.adc_v_ref

    def get_motor_current(self):
        raw = self._read_adc(self.adc_channel_current)
        v_sense = self._adc_to_voltage(raw)
        return (v_sense / self.r_sense) * self.k_ratio

    def get_battery_voltage(self):
        raw = self._read_adc(self.adc_channel_battery)
        v_adc = self._adc_to_voltage(raw)
        return v_adc * (self.r_divider_top + self.r_divider_bottom) / self.r_divider_bottom

    # --- Motor ---

    def set_throttle(self, value):
        value = max(-1.0, min(1.0, value)) * self.motor_direction
        magnitude = abs(value) * self.max_throttle

        if value >= 0:
            self._pinctrl_output(self.pin_ina, high=True)
            self._pinctrl_output(self.pin_inb, high=False)
        else:
            self._pinctrl_output(self.pin_ina, high=False)
            self._pinctrl_output(self.pin_inb, high=True)

        duty_ns = int(magnitude * self.motor_period_ns)
        self._pwm_write(self.motor_pwm_path, "duty_cycle", duty_ns)

    # --- Servo ---

    def set_steering(self, value):
        value = max(-1.0, min(1.0, value))
        actual = value * self.servo_direction * self.servo_amplitude

        span = (self.servo_max_us - self.servo_min_us) / 2.0
        pulse_us = self.servo_center_us + self.servo_offset_us + (actual * span)
        pulse_us = max(self.servo_min_us, min(self.servo_max_us, pulse_us))

        duty = (pulse_us / 20000.0) * 100.0
        self.servo_pwm.ChangeDutyCycle(duty)

    # --- Cleanup ---

    def stop(self):
        self.servo_pwm.stop()
        GPIO.cleanup()
        self._pinctrl_set(self.motor_gpio, "a0", "pd")
        try:
            self._pwm_write(self.motor_pwm_path, "duty_cycle", 0)
            self._pwm_write(self.motor_pwm_path, "enable", 0)
        except Exception:
            pass
