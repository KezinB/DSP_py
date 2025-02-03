import sys
import serial
import serial.tools.list_ports
import csv
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

class SerialPlotter(QtWidgets.QMainWindow):
    def __init__(self, serial_port=None, baud_rate=115200):
        super().__init__()

        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.serial = None

        self.setWindowTitle("Data Plotter")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Font setup
        self.font = QtGui.QFont("Arial", 10)

        # Combo box for COM port selection
        self.com_port_label = QtWidgets.QLabel("Select COM Port:")
        self.com_port_label.setFont(self.font)
        self.layout.addWidget(self.com_port_label)

        self.com_port_combo = QtWidgets.QComboBox()
        self.com_port_combo.setFont(self.font)
        self.layout.addWidget(self.com_port_combo)

        # Populate the combo box with available COM ports
        self.populate_com_ports()

        # Connect button
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.setFont(self.font)
        self.connect_button.clicked.connect(self.connect_serial)
        self.layout.addWidget(self.connect_button)

        # Labels for temperature and humidity
        self.temp_label = QtWidgets.QLabel("Temperature: -- °C")
        self.temp_label.setFont(self.font)
        self.humidity_label = QtWidgets.QLabel("Humidity: -- %")
        self.humidity_label.setFont(self.font)
        self.layout.addWidget(self.temp_label)
        self.layout.addWidget(self.humidity_label)

        # Plot widget
        self.plot_widget = pg.PlotWidget(title="ADC Values")
        self.plot_widget.setYRange(0, 5)
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('bottom', "Time", "Samples")

        self.layout.addWidget(self.plot_widget)

        # Grid layout for ADC value labels (2 rows, 4 columns)
        self.adc_grid_layout = QtWidgets.QGridLayout()
        self.adc_value_labels = []

        for i in range(8):
            label = QtWidgets.QLabel(f"Sensor {i}: -- V")
            label.setFont(self.font)
            self.adc_value_labels.append(label)

            # Arrange in 2 rows (0-3 in row 0, 4-7 in row 1)
            row = 0 if i < 4 else 1
            col = i % 4
            self.adc_grid_layout.addWidget(label, row, col)

        self.layout.addLayout(self.adc_grid_layout)

        # Button Layout
        self.button_layout = QtWidgets.QHBoxLayout()

        # Button style with black borders and bigger text
        button_style = """
            QPushButton {
                font-size: 16px;
                border: 2px solid black;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #d3d3d3;
            }
        """

        # Auto-scale button
        self.auto_scale_button = QtWidgets.QPushButton("Toggle Y Auto-Scale")
        self.auto_scale_button.setFixedWidth(250)
        self.auto_scale_button.setStyleSheet(button_style)
        self.auto_scale_button.clicked.connect(self.toggle_auto_scaling)
        self.button_layout.addWidget(self.auto_scale_button)

        # Pause button
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.setFixedWidth(150)
        self.pause_button.setStyleSheet(button_style)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.button_layout.addWidget(self.pause_button)
        self.paused = False

        # Reset Graph Button
        self.reset_button = QtWidgets.QPushButton("Reset Graph")
        self.reset_button.setFixedWidth(180)
        self.reset_button.setStyleSheet(button_style)
        self.reset_button.clicked.connect(self.reset_graph)
        self.button_layout.addWidget(self.reset_button)

        # Save data button
        self.save_button = QtWidgets.QPushButton("Save Data")
        self.save_button.setFixedWidth(160)
        self.save_button.setStyleSheet(button_style)
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

    def populate_com_ports(self):
        """Populate the combo box with available COM ports."""
        ports = serial.tools.list_ports.comports()
        for port, desc, hwid in sorted(ports):
            self.com_port_combo.addItem(f"{port}: {desc}")

    def connect_serial(self):
        """Connect to the selected COM port."""
        selected_port = self.com_port_combo.currentText().split(":")[0]
        try:
            self.serial = serial.Serial(selected_port, self.baud_rate, timeout=1)
            self.serial_port = selected_port
            self.connect_button.setEnabled(False)
            self.com_port_combo.setEnabled(False)
            print(f"Connected to {selected_port}")
        except serial.SerialException as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to connect to {selected_port}: {e}")

    def update_data(self):
        if self.serial and not self.paused and self.serial.in_waiting:
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
            self.plot_widget.setYRange(0, 5)
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
            self.plot_widget.setYRange(0, 5)  # Reset fixed range

    def save_data_to_file(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if filename:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Temperature (°C)", "Humidity (%)"] + [f"Sensor {i} (V)" for i in range(8)])
                for t, temp, hum, *adc_values in zip(self.time_data, self.data[0], self.data[1], *self.data[2:]):
                    writer.writerow([t, temp, hum] + adc_values)

    def closeEvent(self, event):
        if self.serial:
            self.serial.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    plotter = SerialPlotter()
    plotter.show()
    sys.exit(app.exec_())