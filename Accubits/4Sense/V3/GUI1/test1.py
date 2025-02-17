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

        # Plot widgets for ADC values
        self.plot_widgets = []
        for i in range(8):
            plot_widget = pg.PlotWidget(title=f"ADC {i}")
            plot_widget.setYRange(0, 4.096)
            self.layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)

        # Data storage for plotting
        self.data = [[] for _ in range(8)]
        self.time_data = []

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
                        if len(self.data[i]) > 100:  # Keep last 100 data points
                            self.data[i].pop(0)
                        self.plot_widgets[i].plot(self.time_data, self.data[i], clear=True)

                    if len(self.time_data) > 100:
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