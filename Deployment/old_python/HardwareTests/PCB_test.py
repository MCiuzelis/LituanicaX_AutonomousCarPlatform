import time, math, os, spidev
import RPi.GPIO as GPIO
from rpi_hardware_pwm import HardwarePWM

PIN_INA = 24
PIN_INB = 23
MOTOR_GPIO = 12
SERVO_GPIO = 13

V_REF = 3.3
K_RATIO = 7000
R_SENSE = 1500


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


MOTOR_PWM_CHIP = _find_pwmchip(MOTOR_GPIO)
SERVO_PWM_CHIP = _find_pwmchip(SERVO_GPIO)

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_INA, GPIO.OUT)
GPIO.setup(PIN_INB, GPIO.OUT)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 5000

motor_pwm = HardwarePWM(pwm_channel=0, hz=20000, chip=MOTOR_PWM_CHIP)
motor_pwm.start(0)

servo_pwm = HardwarePWM(pwm_channel=0, hz=20000, chip=SERVO_PWM_CHIP)
servo_pwm.start(7.5)


def read_adc(channel=0):
    config = 0x80 if channel == 0 else 0xC0
    resp = spi.xfer2([1, config, 0])
    return ((resp[1] & 0x0F) << 8) + resp[2]


def get_motor_current(raw_val):
    return ((raw_val / 4095.0) * V_REF / R_SENSE) * K_RATIO


try:
    print("Initiating Sine Sweep. Press Ctrl+C to exit.")
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time

        servo_val = 7.5 + 2.5 * math.sin(2 * math.pi * 0.5 * elapsed)
        servo_pwm.change_duty_cycle(servo_val)

        motor_sine = math.sin(2 * math.pi * 0.2 * elapsed)
        if motor_sine >= 0:
            GPIO.output(PIN_INA, GPIO.HIGH)
            GPIO.output(PIN_INB, GPIO.LOW)
            duty = motor_sine * 100
        else:
            GPIO.output(PIN_INA, GPIO.LOW)
            GPIO.output(PIN_INB, GPIO.HIGH)
            duty = abs(motor_sine) * 100
        motor_pwm.change_duty_cycle(duty)

        raw_ch0 = read_adc(0)
        raw_ch1 = read_adc(1)
        v_ch1 = (raw_ch1 / 4095.0) * V_REF
        print(
            f"Time: {elapsed:.1f}s | Speed: {motor_sine*100:>5.1f}% | "
            f"CH0: {get_motor_current(raw_ch0):.3f}A | CH1: {v_ch1:.3f}V",
            end="\r",
        )
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nShutting down safely...")
finally:
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
    spi.close()
