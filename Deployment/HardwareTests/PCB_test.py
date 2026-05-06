import time
import math
import spidev
import RPi.GPIO as GPIO
from rpi_hardware_pwm import HardwarePWM

# --- PIN CONFIGURATION ---
PIN_INA = 24  
PIN_INB = 23  
MOTOR_PWM_CHAN = 1  # GPIO12 (PWM0)
SERVO_PWM_CHAN = 0  # GPIO13 (PWM1)

# --- CONSTANTS ---
V_REF = 3.3       
K_RATIO = 7000    
R_SENSE = 1500    

# --- INITIALIZATION ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_INA, GPIO.OUT)
GPIO.setup(PIN_INB, GPIO.OUT)

spi = spidev.SpiDev()
spi.open(0, 0)  
spi.max_speed_hz = 5000

motor_pwm = HardwarePWM(pwm_channel=MOTOR_PWM_CHAN, hz=20000)
motor_pwm.start(0)

servo_pwm = HardwarePWM(pwm_channel=SERVO_PWM_CHAN, hz=20000)
servo_pwm.start(7.5) 

def read_adc(channel=0):
    """
    Reads MCP3202 specified channel (0 or 1).
    Returns the raw 12-bit value (0-4095).
    """
    # Start bit(1), SGL/DIFF(1). 
    # For CH0: 1 0 (0x80), For CH1: 1 1 (0xC0)
    config = 0x80 if channel == 0 else 0xC0
    adc_resp = spi.xfer2([1, config, 0])
    
    # Extract 12-bit value from the 3 bytes returned
    adc_raw = ((adc_resp[1] & 0x0F) << 8) + adc_resp[2]
    return adc_raw

def get_motor_current(raw_val):
    """ Converts raw ADC value to Amperage """
    v_sense = (raw_val / 4095.0) * V_REF
    i_out = (v_sense / R_SENSE) * K_RATIO
    return i_out

try:
    print("Initiating Sine Sweep. Press Ctrl+C to exit.")
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        # 1. Servo Sine Sweep
        servo_val = 7.5 + 2.5 * math.sin(2 * math.pi * 0.5 * elapsed)
        servo_pwm.change_duty_cycle(servo_val)
        
        # 2. Motor Sine Sweep
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
        
        # 3. Telemetry - Reading Both Channels
        raw_ch0 = read_adc(0)
        raw_ch1 = read_adc(1)
        
        amps_ch0 = get_motor_current(raw_ch0)
        # Assuming CH1 is a raw voltage reading, but you can apply 
        # get_motor_current here too if it's a second motor driver.
        v_ch1 = (raw_ch1 / 4095.0) * V_REF 
        
        print(f"Time: {elapsed:.1f}s | Speed: {motor_sine*100:>5.1f}% | CH0: {amps_ch0:.3f}A | CH1: {v_ch1:.3f}V", end="\r")
        
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nShutting down safely...")
finally:
    motor_pwm.stop()
    servo_pwm.stop()
    GPIO.cleanup()
    spi.close()
