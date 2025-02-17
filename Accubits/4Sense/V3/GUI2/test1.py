import sys
import serial
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
        self.plot_widget.setYRange(0, 5)
        self.plot_widget.addLegend()
        self.layout.addWidget(self.plot_widget)

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
        if self.serial.in_waiting:
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

                    # Update plot data
                    self.time_data.append(len(self.time_data))
                    for i in range(8):
                        self.data[i].append(voltages[i])
                        if len(self.data[i]) > 10000:  # Keep last 100 data points
                            self.data[i].pop(0)

                    # Clear the plot and redraw all lines
                    self.plot_widget.clear()
                    for i in range(8):
                        self.plot_widget.plot(
                            self.time_data, self.data[i], pen=self.colors[i], name=f"sensor {i}"
                        )

                    if len(self.time_data) > 10000:
                        self.time_data.pop(0)

                except ValueError:
                    print("Invalid data received")

    def closeEvent(self, event):
        self.serial.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    serial_port = "COM8"  # Replace with your serial port
    plotter = SerialPlotter(serial_port)
    plotter.show()
    sys.exit(app.exec_())