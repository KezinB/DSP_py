import sys
import serial
import csv
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

class SerialPlotter(QtWidgets.QMainWindow):
    def __init__(self, serial_port, baud_rate=115200):
        super().__init__()

        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(self.serial_port, self.baud_rate, timeout=1)

        self.setWindowTitle("AHT11 & ADS1115 Data Plotter")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Labels for temperature and humidity
        self.temp_label = QtWidgets.QLabel("Temperature: -- °C")
        self.humidity_label = QtWidgets.QLabel("Humidity: -- %")
        self.layout.addWidget(self.temp_label)
        self.layout.addWidget(self.humidity_label)

        # Plot widget
        self.plot_widget = pg.PlotWidget(title="ADC Values")
        self.plot_widget.setYRange(0, 4.096)
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        
        # Set X-axis label
        self.plot_widget.setLabel('bottom', "Time", "Samples")

        self.layout.addWidget(self.plot_widget)

        # ADC value labels
        self.adc_value_labels = []
        for i in range(8):
            label = QtWidgets.QLabel(f"Sensor {i}: -- V")
            self.adc_value_labels.append(label)
            self.layout.addWidget(label)

        # Button Layout
        self.button_layout = QtWidgets.QHBoxLayout()

        # Auto-scale button
        self.auto_scale_button = QtWidgets.QPushButton("Toggle Y Auto-Scale")
        self.auto_scale_button.setFixedWidth(200)
        self.auto_scale_button.clicked.connect(self.toggle_auto_scaling)
        self.button_layout.addWidget(self.auto_scale_button)

        # Pause button
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setFixedWidth(100)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.button_layout.addWidget(self.pause_button)
        self.paused = False

        # Reset Graph Button
        self.reset_button = QtWidgets.QPushButton("Reset Graph")
        self.reset_button.setFixedWidth(150)
        self.reset_button.clicked.connect(self.reset_graph)
        self.button_layout.addWidget(self.reset_button)

        # Save data button
        self.save_button = QtWidgets.QPushButton("Save Data")
        self.save_button.setFixedWidth(120)
        self.save_button.clicked.connect(self.save_data_to_file)
        self.button_layout.addWidget(self.save_button)

        self.layout.addLayout(self.button_layout)

        # Data storage
        self.data = [[] for _ in range(8)]
        self.time_data = []
        self.auto_scale = False  # Track auto-scaling state

        # Line colors
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

    def update_data(self):
        if not self.paused and self.serial.in_waiting:
            line = self.serial.readline().decode('utf-8').strip()
            values = line.split('|')
            if len(values) == 10:
                try:
                    temp = float(values[0])
                    humidity = float(values[1])
                    voltages = [float(v) for v in values[2:]]

                    # Update labels
                    self.temp_label.setText(f"Temperature: {temp:.2f} °C")
                    self.humidity_label.setText(f"Humidity: {humidity:.2f} %")

                    for i in range(8):
                        self.adc_value_labels[i].setText(f"Sensor {i}: {voltages[i]:.5f} V")

                    # Update plot data
                    self.time_data.append(len(self.time_data))
                    for i in range(8):
                        self.data[i].append(voltages[i])
                        if len(self.data[i]) > 10000:  # Fixed time window
                            self.data[i].pop(0)

                    # Clear and redraw plot
                    self.plot_widget.clear()
                    for i in range(8):
                        self.plot_widget.plot(self.time_data, self.data[i], pen=self.colors[i], name=f"Sensor {i}")

                    if len(self.time_data) > 10000:
                        self.time_data.pop(0)

                except ValueError:
                    print("Invalid data received")

    def toggle_auto_scaling(self):
        self.auto_scale = not self.auto_scale
        if self.auto_scale:
            self.plot_widget.enableAutoRange(axis='y')
            self.auto_scale_button.setText("Disable Y Auto-Scale")
        else:
            self.plot_widget.disableAutoRange(axis='y')
            self.plot_widget.setYRange(0, 4.096)
            self.auto_scale_button.setText("Enable Y Auto-Scale")

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def reset_graph(self):
        """Clear all data and reset the graph."""
        self.data = [[] for _ in range(8)]
        self.time_data = []
        self.plot_widget.clear()
        if not self.auto_scale:
            self.plot_widget.setYRange(0, 4.096)  # Reset fixed range

    def save_data_to_file(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Temperature (°C)", "Humidity (%)"] + [f"Sensor {i} (V)" for i in range(8)])
                for t, temp, hum, *adc_values in zip(self.time_data, self.data[0], self.data[1], *self.data[2:]):
                    writer.writerow([t, temp, hum] + adc_values)

    def closeEvent(self, event):
        self.serial.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    serial_port = "COM8"  # Replace with your serial port
    plotter = SerialPlotter(serial_port)
    plotter.show()
    sys.exit(app.exec_())
