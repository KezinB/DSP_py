import time
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
import adafruit_ahtx0
import csv
import bluetooth
from adafruit_ads1x15.analog_in import AnalogIn
from datetime import datetime

# Initialize I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize two ADS1115 ADCs
ads1 = ADS.ADS1115(i2c, address=0x48)  # First ADC (0x48)
ads2 = ADS.ADS1115(i2c, address=0x4A)  # Second ADC (0x4A)

# Initialize AHT11 sensor (temperature & humidity)
aht11 = adafruit_ahtx0.AHTx0(i2c)

# Generate filename with start time
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_filename = f"sensor_data_{start_time}.csv"
header = [
    "Timestamp", "A0 (V)", "A1 (V)", "A2 (V)", "A3 (V)",
    "B0 (V)", "B1 (V)", "B2 (V)", "B3 (V)",
    "Temperature (C)", "Humidity (%)"
]

# Create or append to the CSV file
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header only if file is empty
    if file.tell() == 0:
        writer.writerow(header)

# Bluetooth setup with reconnection handling
def setup_bluetooth():
    global client_socket, server_socket
    server_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_socket.bind(("", 1))
    server_socket.listen(1)
    print("Waiting for Bluetooth connection...")
    client_socket, address = server_socket.accept()
    print(f"Accepted connection from {address}")

setup_bluetooth()

# Infinite loop to read values
while True:
    try:
        # Get current timestamp
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Read from ADS1115 #1 (0x48)
        A0 = AnalogIn(ads1, ADS.P0).voltage
        A1 = AnalogIn(ads1, ADS.P1).voltage
        A2 = AnalogIn(ads1, ADS.P2).voltage
        A3 = AnalogIn(ads1, ADS.P3).voltage

        # Read from ADS1115 #2 (0x4A)
        B0 = AnalogIn(ads2, ADS.P0).voltage
        B1 = AnalogIn(ads2, ADS.P1).voltage
        B2 = AnalogIn(ads2, ADS.P2).voltage
        B3 = AnalogIn(ads2, ADS.P3).voltage

        # Read AHT11 temperature & humidity
        temperature = aht11.temperature
        humidity = aht11.relative_humidity

        # Create single string data
        data_string = f"{now},{A0:.3f},{A1:.3f},{A2:.3f},{A3:.3f},{B0:.3f},{B1:.3f},{B2:.3f},{B3:.3f},{temperature:.2f},{humidity:.2f}"

        # Print formatted sensor readings with timestamp
        print("=" * 60)
        print(f"Timestamp: {now}")
        print(f"ADC1 (0x48) - A0: {A0:.3f}V, A1: {A1:.3f}V, A2: {A2:.3f}V, A3: {A3:.3f}V")
        print(f"ADC2 (0x4A) - B0: {B0:.3f}V, B1: {B1:.3f}V, B2: {B2:.3f}V, B3: {B3:.3f}V")
        print(f"AHT11 - Temperature: {temperature:.2f}°C, Humidity: {humidity:.2f}%")
        print("=" * 60)

        # Save data to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([now, A0, A1, A2, A3, B0, B1, B2, B3, temperature, humidity])

        # Send data via Bluetooth
        try:
            client_socket.send(data_string + "\n")
        except bluetooth.btcommon.BluetoothError:
            print("Bluetooth connection lost. Reconnecting...")
            client_socket.close()
            setup_bluetooth()

        # Wait before next reading
        time.sleep(1)

    except Exception as e:
        print(f"Error reading sensors: {e}")
        time.sleep(1)
