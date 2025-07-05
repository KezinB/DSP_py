import sys
import serial
import time
import signal
import struct
import os
from datetime import datetime
from collections import deque

class ECGRecorder:
    def __init__(self):
        self.serial_port = None
        self.com_port = 'COM6'
        self.baud_rate = 9600
        self.sample_rate = 1000  # Hz
        self.buffer = deque(maxlen=3600*self.sample_rate)  # 1 hour buffer
        
        # Create unique output directory
        # self.folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_name = r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\1Mtech\ECG-Arrhythmia\realtime\saveData" + "\\" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_name = "realData"
        os.makedirs(self.folder_name, exist_ok=True)

        # WFDB format parameters
        self.adc_gain = 200  # mm/mV
        self.adc_resolution = 16  # bits
        self.adc_zero = 0
        self.voltage_range = 5.0  # V
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.connect_serial()

    def connect_serial(self):
        try:
            self.serial_port = serial.Serial(
                port=self.com_port,
                baudrate=self.baud_rate,
                timeout=0
            )
            print(f"Connected to {self.com_port}")
            self.read_serial()
        except serial.SerialException as e:
            print(f"Connection failed: {str(e)}")
            sys.exit(1)

    def read_serial(self):
        try:
            while True:
                if self.serial_port.in_waiting > 0:
                    raw = self.serial_port.readline().decode('ascii', errors='ignore').strip()
                    val = self.process_sample(raw)
                    if val is not None:
                        self.buffer.append(val)
                
                time.sleep(0.0001)  # Minimal delay

        except KeyboardInterrupt:
            pass

    def process_sample(self, raw):
        try:
            clean = raw.split('\x00')[0].strip()
            return int(clean)
        except (ValueError, IndexError):
            return None

    def signal_handler(self, signum, frame):
        print("\nSaving data...")
        self.save_wfdb()
        sys.exit(0)

    def save_wfdb(self):
        # Create full file paths
        dat_path = os.path.join(self.folder_name, f"{self.base_name}.dat")
        hea_path = os.path.join(self.folder_name, f"{self.base_name}.hea")
        atr_path = os.path.join(self.folder_name, f"{self.base_name}.atr")
        xws_path = os.path.join(self.folder_name, f"{self.base_name}.xws")

        # Save .dat file (16-bit binary)
        with open(dat_path, 'wb') as f:
            for sample in self.buffer:
                f.write(struct.pack('<h', sample))  # 16-bit little-endian

        # Create .hea header file
        header = f"""{self.base_name} 1 {len(self.buffer)/self.sample_rate:.3f}
{self.base_name}.dat 16 1000/mV 16 0 0 0 I
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ADC Resolution: {self.adc_resolution} bits
# Voltage Range: ±{self.voltage_range}V
# Sample Rate: {self.sample_rate} Hz
# Lead Configuration: Lead II"""
        with open(hea_path, 'w') as f:
            f.write(header)

        # Create empty annotation file
        with open(atr_path, 'w') as f:
            f.write("# No annotations in real-time recording\n")

        # Create .xws workbench signal file
        xws_content = f"""<?xml version="1.0"?>
<workbench_signals>
    <signal index="0">
        <name>ECG</name>
        <file>{self.base_name}.dat</file>
        <format>16</format>
        <sampling_rate>{self.sample_rate}</sampling_rate>
    </signal>
</workbench_signals>"""
        with open(xws_path, 'w') as f:
            f.write(xws_content)

        print(f"Saved {len(self.buffer)} samples to:")
        print(f" - {dat_path}")
        print(f" - {hea_path}")
        print(f" - {atr_path}")
        print(f" - {xws_path}")

if __name__ == "__main__":
    recorder = ECGRecorder()