import time
import math
import os
import subprocess

PWM_CHIP_PATH = "/sys/class/pwm/pwmchip0"
PIN_MOTOR_PWM = 12
PIN_INA = 24
PIN_INB = 23
FREQ = 20000
PERIOD_NS = int(1_000_000_000 / FREQ)

def pinctrl_set(pin, *args):
    subprocess.run(["pinctrl", "set", str(pin), *args], check=True)

def pinctrl_output(pin, high):
    pinctrl_set(pin, "op", "dh" if high else "dl")

def export_pwm(channel):
    path = f"{PWM_CHIP_PATH}/pwm{channel}"
    if not os.path.exists(path):
        with open(f"{PWM_CHIP_PATH}/export", "w") as f:
            f.write(str(channel))
        time.sleep(0.1)
    return path

def pwm_write(path, attr, value):
    with open(f"{path}/{attr}", "w") as f:
        f.write(str(value))

pinctrl_set(PIN_MOTOR_PWM, "a0", "pd")
motor_path = export_pwm(channel=0)
pwm_write(motor_path, "enable", 0)
pwm_write(motor_path, "period", PERIOD_NS)
pwm_write(motor_path, "duty_cycle", 0)
pwm_write(motor_path, "enable", 1)

pinctrl_output(PIN_INA, high=False)
pinctrl_output(PIN_INB, high=False)

def set_motor(speed_fraction):
    speed = max(-1.0, min(1.0, speed_fraction)) *0.5
    if speed >= 0:
        pinctrl_output(PIN_INA, high=True)
        pinctrl_output(PIN_INB, high=False)
    else:
        pinctrl_output(PIN_INA, high=False)
        pinctrl_output(PIN_INB, high=True)
    duty_ns = int(abs(speed) * PERIOD_NS)
    pwm_write(motor_path, "duty_cycle", duty_ns)

try:
    print("Hardware PWM motor test (20 kHz). Press Ctrl+C to exit.")
    t = 0.0
    while True:
        motor_val = math.sin(t)
        set_motor(motor_val)
        print(f"Motor: {motor_val:>5.1f} | Duty: {motor_val*100:>5.1f}%", end="\r")
        t += 0.05
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping motor...")
finally:
    pwm_write(motor_path, "duty_cycle", 0)
    pwm_write(motor_path, "enable", 0)
