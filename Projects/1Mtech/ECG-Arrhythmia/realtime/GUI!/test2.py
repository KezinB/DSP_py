import sys
import serial
import csv
import datetime
import time
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
        
        # Initialize UI components first
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
        self.baud_rate = 9600
        self.com_port = "COM6"
        self.auto_connect()

        # Data buffers
        self.max_data_points = 2000
        self.ecg_data = []
        self.timestamps = []
        self.start_time = time.time()
        self.paused = False
        self.dynamic_y = True

        # Timers
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_plot)
        self.data_timer.start(50)
        
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
                timeout=1
            )
            self.status_label.setText(f"Connected to {self.com_port}")
        except serial.SerialException as e:
            self.status_label.setText(f"Failed to connect to {self.com_port}")
            QMessageBox.critical(self, "Connection Error", 
                                f"Could not connect to {self.com_port}:\n{str(e)}")

    def toggle_connection(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.status_label.setText("Disconnected")
        else:
            self.auto_connect()

    def read_serial(self):
        if self.serial_port and self.serial_port.in_waiting and not self.paused:
            try:
                raw_data = self.serial_port.readline().decode().strip()
                
                if raw_data == '!':
                    self.lead_status.setText("Lead Status: DISCONNECTED!")
                    self.lead_status.setStyleSheet("color: red;")
                    return
                
                self.lead_status.setText("Lead Status: OK")
                self.lead_status.setStyleSheet("color: green;")
                
                ecg_value = int(raw_data)
                self.ecg_data.append(ecg_value)
                self.timestamps.append(time.time() - self.start_time)
                
                # Keep data buffers within limit
                if len(self.ecg_data) > self.max_data_points:
                    self.ecg_data.pop(0)
                    self.timestamps.pop(0)
                    
            except (UnicodeDecodeError, ValueError, serial.SerialException) as e:
                print(f"Serial error: {e}")

    def update_plot(self):
        self.read_serial()
        
        if not self.ecg_data:
            return
        
        self.plot.setData(self.timestamps, self.ecg_data)
        
        # X-axis scaling
        if self.timestamps:
            current_time = self.timestamps[-1]
            self.graph.setXRange(max(0, current_time - 10), current_time + 0.1)
        
        # Y-axis scaling
        if self.dynamic_y:
            y_min = min(self.ecg_data[-500:] or [0])
            y_max = max(self.ecg_data[-500:] or [1023])
            margin = (y_max - y_min) * 0.1
            self.graph.setYRange(y_min - margin, y_max + margin)
        else:
            self.graph.setYRange(0, 1023)

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