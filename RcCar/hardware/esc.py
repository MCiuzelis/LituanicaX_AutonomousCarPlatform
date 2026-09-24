"""Hobbywing QuicRun 1080 G2 brushed ESC - throttle signal only.

Wiring on this car:
    GPIO13 -> ESC throttle/PWM input (hardware PWM0_CHAN1, 50 Hz servo pulses)

The ESC is switched on by hand (its own power button); the Pi only generates the throttle
pulse. esc-neutral.service puts the pin at 1500 us neutral at boot and this class never
removes that signal on close(), so the ESC always sees a valid neutral when idle.

Calibrated 2026-09-20 with the throttle REVERSED (motor wiring): 1000 us = full forward,
2000 us = full brake/reverse. throttle(+1) is still forward. Reverse is disabled in the ESC
running mode (forward/brake), so negative throttle only sets brake strength.
"""
from .base import Subsystem
from .pwm import Servo

ESC_PWM_GPIO = 13
NEUTRAL_US, FULL_FWD_US, FULL_BRAKE_US = 1500.0, 1000.0, 2000.0


class HobbywingESC(Subsystem):
    name = "esc"

    def __init__(self, pwm_gpio=ESC_PWM_GPIO, neutral_us=NEUTRAL_US, full_us=FULL_FWD_US,
                 brake_us=FULL_BRAKE_US, signal_on=True):
        self.neutral_us, self.full_us, self.brake_us = float(neutral_us), float(full_us), float(brake_us)
        # the servo pulse range covers both endpoints (whichever order) so throttle(-1..1) maps onto it
        lo_us, hi_us = sorted((self.brake_us, self.full_us))
        self.servo = Servo(pwm_gpio, min_us=lo_us, max_us=hi_us, min_angle=-1.0, max_angle=1.0,
                           neutral_us=self.neutral_us, keep_on_close=True)
        if signal_on:
            self.neutral()
        else:
            self.servo.release()

    def neutral(self):
        self.servo.set_pulse_us(self.neutral_us)

    def throttle(self, fraction):
        """-1.0 (full brake) .. 0 (neutral) .. +1.0 (full forward). Returns the pulse in us."""
        f = max(-1.0, min(1.0, float(fraction)))
        if f >= 0:
            us = self.neutral_us + f * (self.full_us - self.neutral_us)
        else:
            us = self.neutral_us + f * (self.neutral_us - self.brake_us)
        return self.servo.set_pulse_us(us)

    def pulse_us(self, us):
        """Raw pulse width (clamped to the brake..forward range)."""
        return self.servo.set_pulse_us(us)

    @property
    def current_us(self):
        return self.servo.pulse_us

    def signal_off(self):
        """Stop sending pulses entirely (the ESC treats it as signal loss -> failsafe/stop)."""
        self.servo.release()

    # ---- Subsystem -----------------------------------------------------------
    def available(self):
        return self.servo.available()

    def status(self):
        return {"pwm_gpio": self.servo.gpio, "pulse_us": self.current_us,
                "neutral_us": self.neutral_us, "full_us": self.full_us, "brake_us": self.brake_us}

    def close(self):
        """Back to neutral; the PWM output keeps running (keep_on_close)."""
        try:
            self.neutral()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            pass
        self.servo.close()
