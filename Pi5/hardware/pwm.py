"""Hardware PWM on the Pi 5 (RP1 PWM0 block, sysfs /sys/class/pwm) + RC servo helper.

GPIO -> PWM0 channel mapping on the Pi 5 header:
    GPIO12 -> PWM0_CHAN0 (alt0)      GPIO18 -> PWM0_CHAN2 (alt3)
    GPIO13 -> PWM0_CHAN1 (alt0)      GPIO19 -> PWM0_CHAN3 (alt3)

RP1 has two PWM blocks. PWM1 (1f0009c000.pwm) is always enabled and drives the fan header;
PWM0 (1f00098000.pwm) is the one routed to the GPIO header and is *disabled* in the base
device tree. It is enabled (and the pins muxed) by the overlay

    dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4

which scripts/setup_pwm_overlay.sh adds to /boot/firmware/config.txt. If the overlay is not
present at runtime, HardwarePWM loads it on the fly with `sudo dtoverlay ...` (passwordless
sudo is configured on this box). The pin mux is additionally re-asserted with `pinctrl`.
Access to /sys/class/pwm is granted to the gpio group by /lib/udev/rules.d/99-com.rules.
"""
import glob
import os
import time

from .base import Subsystem, read_sysfs, write_sysfs, run_cmd, HardwareError

RP1_PWM0 = "1f00098000.pwm"
PWM_OVERLAY = "pwm-2chan,pin=12,func=4,pin2=13,func2=4"
PWM_PINS = {  # gpio: (channel, pinctrl alt function, function name reported by pinctrl)
    12: (0, "a0", "PWM0_CHAN0"),
    13: (1, "a0", "PWM0_CHAN1"),
    18: (2, "a3", "PWM0_CHAN2"),
    19: (3, "a3", "PWM0_CHAN3"),
}


def _lookup_chip(block):
    for chip in sorted(glob.glob("/sys/class/pwm/pwmchip*")):
        if block in os.path.realpath(os.path.join(chip, "device")):
            return chip
    return None


def find_pwm_chip(block=RP1_PWM0, autoload=True):
    """sysfs pwmchip directory for the RP1 PWM0 block, loading the overlay if needed."""
    chip = _lookup_chip(block)
    if chip is None and autoload:
        rc, out = run_cmd(["sudo", "-n", "dtoverlay", PWM_OVERLAY], check=False)
        if rc != 0:
            raise HardwareError("PWM0 not enabled and overlay load failed (%s). Add "
                                "'dtoverlay=%s' to /boot/firmware/config.txt and reboot."
                                % (out.strip()[:120], PWM_OVERLAY))
        deadline = time.monotonic() + 3.0
        while chip is None and time.monotonic() < deadline:
            time.sleep(0.1)
            chip = _lookup_chip(block)
    if chip is None:
        raise HardwareError("no pwmchip for %s under /sys/class/pwm" % block)
    return chip


class HardwarePWM(Subsystem):
    """One hardware PWM output on a header GPIO."""
    name = "pwm"

    def __init__(self, gpio, frequency_hz=50.0, restore_pin=True, keep_on_close=False):
        if gpio not in PWM_PINS:
            raise ValueError("GPIO%d has no PWM0 channel; use one of %s" % (gpio, sorted(PWM_PINS)))
        self.gpio = gpio
        self.channel, self._alt, self._func = PWM_PINS[gpio]
        self.chip = find_pwm_chip()
        self.path = os.path.join(self.chip, "pwm%d" % self.channel)
        self.restore_pin = restore_pin
        self.keep_on_close = keep_on_close  # True: leave the output running on close()
        self._period_ns = 0
        self._duty_ns = 0
        self._enabled = False
        self._export()
        self._mux_pin()
        self.set_frequency(frequency_hz)

    # ---- low level -------------------------------------------------------
    def _export(self):
        if not os.path.isdir(self.path):
            try:
                write_sysfs(os.path.join(self.chip, "export"), self.channel)
            except HardwareError as e:
                if "busy" not in str(e).lower():
                    raise
        # udev chgrp's the new pwmN dir asynchronously; wait until we can write it
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if os.access(os.path.join(self.path, "period"), os.W_OK):
                return
            time.sleep(0.02)
        raise HardwareError("%s not writable (is the user in the gpio group?)" % self.path)

    def _mux_pin(self):
        run_cmd(["pinctrl", "set", str(self.gpio), self._alt])
        _, out = run_cmd(["pinctrl", "get", str(self.gpio)])
        if self._func not in out:
            raise HardwareError("GPIO%d did not switch to %s: %s" % (self.gpio, self._func, out.strip()))

    def _w(self, attr, value):
        write_sysfs(os.path.join(self.path, attr), int(value))

    def _r(self, attr):
        return read_sysfs(os.path.join(self.path, attr))

    def available(self):
        return os.path.isdir(self.path)

    # ---- configuration ---------------------------------------------------
    @property
    def frequency_hz(self):
        return 1e9 / self._period_ns if self._period_ns else 0.0

    @property
    def period_ns(self):
        return self._period_ns

    @property
    def duty_ns(self):
        return self._duty_ns

    @property
    def duty_cycle(self):
        return self._duty_ns / self._period_ns if self._period_ns else 0.0

    @property
    def pulse_width_us(self):
        return self._duty_ns / 1000.0

    def set_frequency(self, hz):
        period = int(round(1e9 / float(hz)))
        if period <= 0:
            raise ValueError("frequency must be > 0")
        duty = min(self._duty_ns, period)
        # the kernel rejects duty > period, so shrink duty first when the period shrinks
        if duty < self._duty_ns:
            self._w("duty_cycle", duty)
            self._duty_ns = duty
        self._w("period", period)
        self._period_ns = period
        return self

    def set_duty_ns(self, ns):
        ns = int(max(0, min(ns, self._period_ns)))
        self._w("duty_cycle", ns)
        self._duty_ns = ns
        return self

    def set_duty_cycle(self, fraction):
        return self.set_duty_ns(float(fraction) * self._period_ns)

    def set_pulse_width_us(self, us):
        return self.set_duty_ns(float(us) * 1000.0)

    def enable(self):
        self._w("enable", 1)
        self._enabled = True
        return self

    def disable(self):
        self._w("enable", 0)
        self._enabled = False
        return self

    @property
    def enabled(self):
        return self._enabled

    def close(self):
        if not os.path.isdir(self.path):
            return
        if self.keep_on_close:
            return  # leave period/duty/enable and the pin mux exactly as they are
        try:
            self._w("duty_cycle", 0)
            self._w("enable", 0)
        except HardwareError:
            pass
        try:
            write_sysfs(os.path.join(self.chip, "unexport"), self.channel)
        except HardwareError:
            pass
        if self.restore_pin:
            run_cmd(["pinctrl", "set", str(self.gpio), "no"], check=False)
        self._enabled = False

    def status(self):
        return {
            "gpio": self.gpio, "chip": os.path.basename(self.chip), "channel": self.channel,
            "frequency_hz": round(self.frequency_hz, 3), "period_ns": self._period_ns,
            "duty_ns": self._duty_ns, "duty_cycle": round(self.duty_cycle, 4),
            "pulse_width_us": self.pulse_width_us, "enabled": self._enabled,
            "sysfs_enable": self._r("enable"), "sysfs_period": self._r("period"),
            "sysfs_duty": self._r("duty_cycle"),
        }


class Servo(Subsystem):
    """Standard RC servo: 50 Hz frame, 1.0-2.0 ms pulse (adjustable), angle -90..+90."""
    name = "servo"

    def __init__(self, gpio, min_us=1000.0, max_us=2000.0, min_angle=-90.0, max_angle=90.0,
                 frequency_hz=50.0, neutral_us=None, start_at_neutral=False, keep_on_close=False):
        self.pwm = HardwarePWM(gpio, frequency_hz, keep_on_close=keep_on_close)
        self.min_us, self.max_us = float(min_us), float(max_us)
        self.min_angle, self.max_angle = float(min_angle), float(max_angle)
        self.neutral_us = float(neutral_us) if neutral_us is not None else (self.min_us + self.max_us) / 2
        self._pulse_us = None
        self.pwm.enable()
        if start_at_neutral:
            self.center()

    @property
    def gpio(self):
        return self.pwm.gpio

    @property
    def pulse_us(self):
        return self._pulse_us

    @property
    def angle(self):
        if self._pulse_us is None:
            return None
        span = self.max_us - self.min_us
        return self.min_angle + (self._pulse_us - self.min_us) / span * (self.max_angle - self.min_angle)

    def set_pulse_us(self, us):
        us = max(self.min_us, min(float(us), self.max_us))
        self.pwm.set_pulse_width_us(us)
        self._pulse_us = us
        return us

    def set_angle(self, degrees):
        d = max(self.min_angle, min(float(degrees), self.max_angle))
        frac = (d - self.min_angle) / (self.max_angle - self.min_angle)
        return self.set_pulse_us(self.min_us + frac * (self.max_us - self.min_us))

    def set_fraction(self, f):
        """0.0 = min_us, 1.0 = max_us (handy for ESCs / throttle)."""
        return self.set_pulse_us(self.min_us + max(0.0, min(1.0, f)) * (self.max_us - self.min_us))

    def center(self):
        return self.set_pulse_us(self.neutral_us)

    def release(self):
        """Stop sending pulses (servo goes limp / holds no position)."""
        self.pwm.set_duty_ns(0)
        self._pulse_us = None

    def available(self):
        return self.pwm.available()

    def close(self):
        self.pwm.close()

    def status(self):
        st = self.pwm.status()
        st.update({"pulse_us": self._pulse_us, "angle": None if self.angle is None else round(self.angle, 1),
                   "range_us": (self.min_us, self.max_us)})
        return st
