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

        # Single plot widget for all ADC values
        self.plot_widget = pg.PlotWidget(title="ADC Values")
        self.plot_widget.setYRange(0, 4.096)
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)  # Enable grid
        self.layout.addWidget(self.plot_widget)

        # Create labels to display current ADC values
        self.adc_value_labels = []
        for i in range(8):
            label = QtWidgets.QLabel(f"Sensor {i}: -- V")
            self.adc_value_labels.append(label)
            self.layout.addWidget(label)

        # Adjustable time window (slider)
        self.time_window_label = QtWidgets.QLabel("Time Window: 10000 points")
        self.layout.addWidget(self.time_window_label)

        self.time_window_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_window_slider.setMinimum(100)
        self.time_window_slider.setMaximum(20000)
        self.time_window_slider.setValue(10000)
        self.time_window_slider.valueChanged.connect(self.update_time_window)
        self.layout.addWidget(self.time_window_slider)

        # Auto-scaling for Y-axis (checkbox)
        self.auto_scale_checkbox = QtWidgets.QCheckBox("Enable Y-Axis Auto-Scaling")
        self.auto_scale_checkbox.stateChanged.connect(self.toggle_auto_scaling)
        self.layout.addWidget(self.auto_scale_checkbox)

        # Pause/Resume button
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.layout.addWidget(self.pause_button)
        self.paused = False

        # Save data to file button
        self.save_button = QtWidgets.QPushButton("Save Data to CSV")
        self.save_button.clicked.connect(self.save_data_to_file)
        self.layout.addWidget(self.save_button)

        # Data storage for plotting
        self.data = [[] for _ in range(8)]  # 8 ADC channels
        self.time_data = []

        # Colors for each ADC line
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

        # Timer to update the plots and labels
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # Update every 100 ms

    def update_data(self):
        if not self.paused and self.serial.in_waiting:
            line = self.serial.readline().decode('utf-8').strip()
            values = line.split('|')
            if len(values) == 10:
                try:
                    temp = float(values[0])
                    humidity = float(values[1])
                    voltages = [float(v) for v in values[2:]]

                    # Update temperature and humidity labels
                    self.temp_label.setText(f"Temperature: {temp:.2f} °C")
                    self.humidity_label.setText(f"Humidity: {humidity:.2f} %")

                    # Update ADC value labels
                    for i in range(8):
                        self.adc_value_labels[i].setText(f"Sensor {i}: {voltages[i]:.5f} V")

                    # Update plot data
                    self.time_data.append(len(self.time_data))
                    for i in range(8):
                        self.data[i].append(voltages[i])
                        if len(self.data[i]) > self.time_window_slider.value():  # Adjustable time window
                            self.data[i].pop(0)

                    # Clear the plot and redraw all lines
                    self.plot_widget.clear()
                    for i in range(8):
                        self.plot_widget.plot(
                            self.time_data, self.data[i], pen=self.colors[i], name=f"Sensor {i}"
                        )

                    if len(self.time_data) > self.time_window_slider.value():
                        self.time_data.pop(0)

                except ValueError:
                    print("Invalid data received")

    def update_time_window(self):
        """Update the time window based on the slider value."""
        self.time_window_label.setText(f"Time Window: {self.time_window_slider.value()} points")

    def toggle_auto_scaling(self):
        """Enable or disable Y-axis auto-scaling."""
        if self.auto_scale_checkbox.isChecked():
            self.plot_widget.enableAutoRange(axis='y')
        else:
            self.plot_widget.disableAutoRange(axis='y')

    def toggle_pause(self):
        """Pause or resume plotting."""
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def save_data_to_file(self):
        """Save the received data to a CSV file."""
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