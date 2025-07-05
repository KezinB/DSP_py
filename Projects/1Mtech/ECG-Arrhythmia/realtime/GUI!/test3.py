import sys
import serial
import csv
import datetime
import time
import re
from collections import deque
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, 
                             QWidget, QLabel, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import pyqtgraph.exporters

class ECGVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon("ecg_icon.png"))
        
        # Initialize UI components
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Status labels
        self.status_label = QLabel("Status: Disconnected")
        self.lead_status = QLabel("Lead Status: OK")
        self.timer_label = QLabel("Elapsed Time: 00:00")
        for label in [self.status_label, self.lead_status, self.timer_label]:
            label.setFont(QFont("Arial", 12))

        # Plot setup
        self.graph = PlotWidget()
        self.graph.setTitle("ECG Signal")
        self.graph.setLabel('left', "Amplitude (ADC)")
        self.graph.setLabel('bottom', "Time (seconds)")
        self.plot = self.graph.plot(pen='r')

        # Buttons
        self.button_layout = QHBoxLayout()
        self.create_buttons()

        # Assemble UI
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.lead_status)
        self.layout.addWidget(self.timer_label)
        self.layout.addWidget(self.graph)
        self.layout.addLayout(self.button_layout)

        # Serial configuration
        self.serial_port = None
        self.baud_rate = 115200  # Increased baud rate
        self.com_port = "COM6"
        self.serial_buffer = bytearray()
        self.auto_connect()

        # Data buffers
        self.max_data_points = 2000
        self.ecg_data = deque(maxlen=self.max_data_points)
        self.timestamps = deque(maxlen=self.max_data_points)
        self.start_time = time.time()
        self.paused = False
        self.dynamic_y = True

        # Timers
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_plot)
        self.data_timer.start(1)  # 1ms update interval
        
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def create_buttons(self):
        buttons = [
            ('Connect', self.toggle_connection),
            ('Pause/Resume', self.toggle_pause),
            ('Toggle Y-Axis', self.toggle_y_axis),
            ('Save CSV', self.save_csv),
            ('Save Image', self.save_image),
            ('Reset', self.reset)
        ]
        
        for text, handler in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(120, 40)
            btn.setStyleSheet(
                "QPushButton {"
                "background-color: #4CAF50;"
                "border: none;"
                "color: white;"
                "padding: 8px;"
                "font-size: 14px;"
                "}"
                "QPushButton:hover {background-color: #45a049;}"
            )
            btn.clicked.connect(handler)
            self.button_layout.addWidget(btn)

    def auto_connect(self):
        try:
            self.serial_port = serial.Serial(
                port=self.com_port,
                baudrate=self.baud_rate,
                timeout=0,  # Non-blocking
                write_timeout=0,
                exclusive=True
            )
            self.status_label.setText(f"Connected to {self.com_port}")
            self.serial_port.reset_input_buffer()
        except Exception as e:
            self.status_label.setText(f"Failed to connect to {self.com_port}")
            QMessageBox.critical(self, "Error", f"Connection failed: {str(e)}")

    def read_serial(self):
        if self.serial_port and self.serial_port.is_open and not self.paused:
            try:
                # Read all available bytes
                data = self.serial_port.read_all()
                if data:
                    self.serial_buffer.extend(data)
                    
                    # Process complete lines
                    while b'\r' in self.serial_buffer:
                        line, self.serial_buffer = self.serial_buffer.split(b'\r', 1)
                        self.process_line(line)

            except Exception as e:
                self.handle_serial_error(e)
                return

    def process_line(self, line):
        try:
            raw = line.decode('ascii', errors='ignore').strip()
            # Remove all non-digit characters except '!'
            clean = re.sub(r'[^\d!]', '', raw)
            
            if not clean:
                return
                
            if clean == '!':
                self.lead_status.setText("Lead Status: DISCONNECTED!")
                self.lead_status.setStyleSheet("color: red;")
                return
                
            # Extract first valid number
            numbers = re.findall(r'\d+', clean)
            if numbers:
                value = int(numbers[0])
                self.ecg_data.append(value)
                self.timestamps.append(time.time() - self.start_time)
                self.lead_status.setText("Lead Status: OK")
                self.lead_status.setStyleSheet("color: green;")

        except Exception as e:
            print(f"Processing error: {e}")

    def update_plot(self):
        try:
            self.read_serial()
        except Exception as e:
            self.handle_serial_error(e)
        
        if not self.ecg_data:
            return
        
        # Convert deques to lists for plotting
        ts = list(self.timestamps)
        ecg = list(self.ecg_data)
        
        self.plot.setData(ts, ecg)
        
        # Dynamic axis scaling
        if ts:
            current_time = ts[-1]
            self.graph.setXRange(max(0, current_time - 10), current_time + 0.1)
            
            if self.dynamic_y:
                visible_data = ecg[-500:] or [0]
                y_min = min(visible_data)
                y_max = max(visible_data)
                margin = (y_max - y_min) * 0.2
                self.graph.setYRange(y_min - margin, y_max + margin)
            else:
                self.graph.setYRange(0, 1023)

    def handle_serial_error(self, error):
        if "handle is invalid" in str(error) or not self.serial_port.is_open:
            self.status_label.setText("Port disconnected")
            self.serial_port = None
        print(f"Serial error: {error}")

    def toggle_connection(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.status_label.setText("Disconnected")
        else:
            self.auto_connect()

    def toggle_y_axis(self):
        self.dynamic_y = not self.dynamic_y
        self.update_plot()

    def toggle_pause(self):
        self.paused = not self.paused

    def save_csv(self):
        if not self.ecg_data:
            QMessageBox.warning(self, "Warning", "No data to save!")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"ecg_data_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp (s)", "ECG Value"])
                for t, v in zip(self.timestamps, self.ecg_data):
                    writer.writerow([t, v])
            QMessageBox.information(self, "Success", f"Data saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

    def save_image(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"ecg_plot_{timestamp}.png"
        
        try:
            exporter = pg.exporters.ImageExporter(self.graph.plotItem)
            exporter.parameters()['width'] = 1920  # Higher resolution
            exporter.parameters()['height'] = 1080
            exporter.export(filename)
            QMessageBox.information(self, "Success", f"Plot saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")

    def reset(self):
        self.ecg_data.clear()
        self.timestamps.clear()
        self.start_time = time.time()
        self.plot.clear()
        self.update_plot()

    def update_clock(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        self.timer_label.setText(f"Elapsed Time: {mins:02}:{secs:02}")

    def closeEvent(self, event):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ECGVisualizer()
    window.show()
    sys.exit(app.exec_())