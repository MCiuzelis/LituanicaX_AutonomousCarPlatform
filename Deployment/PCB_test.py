import time
import math
import spidev
import RPi.GPIO as GPIO
from hardwarepwm import HardwarePWM # Source 1

# --- PIN CONFIGURATION ---
PIN_DIR_1 = 24  # INA
PIN_DIR_2 = 23  # INB
ADC_CS_PIN = 8  # SPI CE0
SERVO_PWM_CHAN = 1 # PWM_1 (GPIO13)
MOTOR_PWM_CHAN = 0 # PWM_0 (GPIO12)

# --- CONSTANTS ---
V_REF = 3.3      # RP5 Logic Voltage
K_RATIO = 7000   # Typical VNH5019 Current Sense Ratio
R_SENSE = 1500   # 1.5k Ohm resistor on CS pin

# --- INITIALIZATION ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_DIR_1, GPIO.OUT)
GPIO.setup(PIN_DIR_2, GPIO.OUT)

# SPI for MCP3202
spi = spidev.SpiDev()
spi.open(0, 0) # Bus 0, Device 0 (GPIO8)
spi.max_speed_hz = 1000000 # 1MHz

# Hardware PWM Setup
# Motor: 20kHz to avoid audible noise
motor_pwm = HardwarePWM(pwm_channel=MOTOR_PWM_CHAN, hz=20000)
motor_pwm.start(0)

# Servo: 50Hz standard
servo_pwm = HardwarePWM(pwm_channel=SERVO_PWM_CHAN, hz=50)
servo_pwm.start(7.5) # Neutral (1.5ms)

def read_motor_current():
    """
    Reads MCP3202 CH0 and calculates motor current in Amps.
    Command: Start(1), SGL(1), CH0(0), MSBF(1) -> 0xA0
    """
    #
    reply = spi.xfer2([0x01, 0xA0, 0x00])
    # Extract 12 bits from the 3-byte response
    adc_raw = ((reply[1] & 0x0F) << 8) + reply[2]
    
    # Calculate Voltage
    v_sense = (adc_raw / 4095.0) * V_REF
    
    # Calculate Motor Current: I_out = (V_sense / R_sense) * K
    i_out = (v_sense / R_SENSE) * K_RATIO
    return i_out

try:
    print("Starting Sine Sweep... Press Ctrl+C to stop.")
    start_time = time.time()
    
    while True:
        t = time.time() - start_time
        
        # 1. Servo Sine Sweep (Range: 5% to 10% duty cycle)
        # Using a 0.5Hz frequency for the sweep
        servo_duty = 7.5 + 2.5 * math.sin(2 * math.pi * 0.5 * t)
        servo_pwm.change_duty_cycle(servo_duty)
        
        # 2. Motor Sine Sweep (Range: -100% to 100% speed)
        motor_val = math.sin(2 * math.pi * 0.2 * t) # 0.2Hz sweep
        
        if motor_val >= 0:
            GPIO.output(PIN_DIR_1, GPIO.HIGH) # INA=1
            GPIO.output(PIN_DIR_2, GPIO.LOW)  # INB=0 (CW)
            duty = motor_val * 100
        else:
            GPIO.output(PIN_DIR_1, GPIO.LOW)  # INA=0
            GPIO.output(PIN_DIR_2, GPIO.HIGH) # INB=1 (CCW)
            duty = abs(motor_val) * 100
            
        motor_pwm.change_duty_cycle(duty)
        
        # 3. Read and Print Current
        current = read_motor_current()
        print(f"Time: {t:.2f}s | Servo Duty: {servo_duty:.1f}% | Motor: {motor_val*100:.1f}% | Amps: {current:.3f}A")
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
    spi.close()