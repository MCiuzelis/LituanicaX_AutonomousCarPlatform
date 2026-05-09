import os, spidev
import RPi.GPIO as GPIO
from rpi_hardware_pwm import HardwarePWM

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

        motor_chip = self._find_pwmchip(motor_gpio)
        servo_chip = self._find_pwmchip(servo_gpio)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin_ina, GPIO.OUT)
        GPIO.setup(self.pin_inb, GPIO.OUT)

        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 5000

        self.motor_pwm = HardwarePWM(pwm_channel=0, hz=self.motor_pwm_freq, chip=motor_chip)
        self.servo_pwm = HardwarePWM(pwm_channel=0, hz=self.servo_pwm_freq, chip=servo_chip)

        self.motor_pwm.start(0)
        self.servo_pwm.start(7.5)

    @staticmethod
    def _find_pwmchip(gpio_pin):
        for entry in os.listdir("/sys/class/pwm"):
            if not entry.startswith("pwmchip"):
                continue
            chip = int(entry[7:])
            try:
                with open(f"/sys/class/pwm/{entry}/device/uevent") as f:
                    for line in f:
                        if line.startswith("OF_FULLNAME="):
                            addr = line.strip().split("@")[-1]
                            if int(addr, 16) == gpio_pin:
                                return chip
            except OSError:
                continue
        raise RuntimeError(f"No pwmchip found for GPIO{gpio_pin}")

    def _read_adc(self, channel=0):
        config = 0x80 if channel == 0 else 0xC0
        adc_resp = self.spi.xfer2([1, config, 0])
        return ((adc_resp[1] & 0x0F) << 8) + adc_resp[2]

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

    def set_throttle(self, value):
        value = max(-1.0, min(1.0, value))
        value *= self.max_throttle
        value *= self.motor_direction

        GPIO.output(self.pin_ina, GPIO.HIGH if value >= 0 else GPIO.LOW)
        GPIO.output(self.pin_inb, GPIO.HIGH if value < 0 else GPIO.LOW)

        self.motor_pwm.change_duty_cycle(abs(value) * 100.0)

    def set_steering(self, value):
        value = max(-1.0, min(1.0, value))
        value = value * self.servo_direction * self.servo_amplitude

        span = (self.servo_max_us - self.servo_min_us) / 2
        pulse = self.servo_center_us + self.servo_offset_us + (value * span)
        pulse = max(self.servo_min_us, min(self.servo_max_us, pulse))

        duty = (pulse / 20000.0) * 100.0
        self.servo_pwm.change_duty_cycle(duty)

    def stop(self):
        try:
            self.motor_pwm.stop()
            self.servo_pwm.stop()
        finally:
            self.spi.close()
            GPIO.cleanup()
