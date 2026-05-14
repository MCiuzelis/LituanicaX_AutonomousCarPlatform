import smbus2
import time

# AS5600 I2C address
AS5600_ADDR = 0x36 # 

# Register addresses for the 12-bit Angle
# 0x0E is the High Byte (bits 11:8), 0x0F is the Low Byte (bits 7:0)
REG_ANGLE_HIGH = 0x0E # [cite: 147]

# Initialize I2C bus 1 (standard on Raspberry Pi)
bus = smbus2.SMBus(1)

def read_angle():
    try:
        # Read 2 bytes starting from the high byte register
        # The AS5600 transmits the MSB first [cite: 107, 115]
        data = bus.read_i2c_block_data(AS5600_ADDR, REG_ANGLE_HIGH, 2)
        
        # Combine the two bytes into a single 12-bit value
        # high_byte << 8 | low_byte
        raw_value = (data[0] << 8) | data[1]
        
        # Convert the 12-bit raw value (0-4095) to degrees (0-360)
        degrees = (raw_value * 360.0) / 4096.0 # 
        
        return degrees
    except Exception as e:
        print(f"Error reading from sensor: {e}")
        return None

print("Reading AS5600 Position... Press Ctrl+C to stop.")

try:
    while True:
        angle = read_angle()
        if angle is not None:
            print(f"Angle: {angle:.2f}°")
        time.sleep(0.05)  # Read 10 times per second
except KeyboardInterrupt:
    print("\nMeasurement stopped by user.")
finally:
    bus.close()