import smbus2
import time

# AS5600 I2C address
AS5600_ADDR = 0x36 

# Register addresses for the 12-bit Angle
REG_ANGLE_HIGH = 0x0E 

# Initialize I2C bus 1 (standard on Raspberry Pi)
bus = smbus2.SMBus(1)

def read_angle():
    try:
        # Read 2 bytes starting from the high byte register
        data = bus.read_i2c_block_data(AS5600_ADDR, REG_ANGLE_HIGH, 2)
        
        # Combine the two bytes into a single 12-bit value
        raw_value = (data[0] << 8) | data[1]
        
        # Convert the 12-bit raw value (0-4095) to degrees (0-360)
        degrees = -(raw_value * 360.0) / 4096.0 
        
        return degrees
    except Exception as e:
        print(f"Error reading from sensor: {e}")
        return None

print("Reading AS5600 Position continuously... Press Ctrl+C to stop.")

try:
    # Initialize tracking variables
    last_raw_angle = read_angle()
    while last_raw_angle is None:
        time.sleep(0.01)
        last_raw_angle = read_angle()

    start_angle = last_raw_angle
    total_interated_angle = 0.0

    while True:
        current_raw_angle = read_angle()
        
        if current_raw_angle is not None:
            # Calculate the shortest difference between current and last reading
            delta = current_raw_angle - last_raw_angle
            
            # Handle the wrap-around (e.g., crossing the 0 / 360 boundary)
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0
            
            # Accumulate the true relative movement
            total_interated_angle += delta
            last_raw_angle = current_raw_angle
            
            print(f"Relative Angle: {total_interated_angle:.2f}°")
            
        time.sleep(1/500)  # Read interval
        
except KeyboardInterrupt:
    print("\nMeasurement stopped by user.")
finally:
    bus.close()