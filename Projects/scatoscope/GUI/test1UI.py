import sys
import serial
import serial.tools.list_ports
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QComboBox, QWidget
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget
import pyqtgraph as pg

def list_com_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

class SerialPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPU6050 Bluetooth Data Plotter")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.com_selector = QComboBox()
        self.com_selector.addItems(list_com_ports())
        self.layout.addWidget(self.com_selector)

        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_serial)
        self.layout.addWidget(self.connect_button)

        self.graph = PlotWidget()
        self.layout.addWidget(self.graph)

        self.graph.setTitle("MPU6050 Sensor Data")
        self.graph.setLabel('left', "Sensor Value")
        self.graph.setLabel('bottom', "Time")
        self.graph.addLegend()

        self.accel_x = self.graph.plot(pen='r', name='Accel X')
        self.accel_y = self.graph.plot(pen='g', name='Accel Y')
        self.accel_z = self.graph.plot(pen='b', name='Accel Z')
        self.gyro_x = self.graph.plot(pen='y', name='Gyro X')
        self.gyro_y = self.graph.plot(pen='m', name='Gyro Y')
        self.gyro_z = self.graph.plot(pen='c', name='Gyro Z')

        self.serial_port = None
        self.data_buffer = {"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "time": []}
        self.time_counter = 0
        self.running = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial)

    def connect_serial(self):
        port = self.com_selector.currentText()
        try:
            self.serial_port = serial.Serial(port, 115200, timeout=1)
            self.running = True
            self.timer.start(50)
        except serial.SerialException:
            print("Failed to connect to port")

    def read_serial(self):
        if self.serial_port and self.serial_port.in_waiting:
            try:
                data = self.serial_port.readline().decode().strip()
                values = data.split(',')
                if len(values) == 6:
                    clean_values = []
                    for v in values:
                        try:
                            clean_values.append(float(v))
                        except ValueError:
                            print("Invalid value detected:", v)
                            return

                    self.data_buffer["ax"].append(clean_values[0])
                    self.data_buffer["ay"].append(clean_values[1])
                    self.data_buffer["az"].append(clean_values[2])
                    self.data_buffer["gx"].append(clean_values[3])
                    self.data_buffer["gy"].append(clean_values[4])
                    self.data_buffer["gz"].append(clean_values[5])
                    self.data_buffer["time"].append(self.time_counter)
                    self.time_counter += 1
                    self.update_plot()
            except Exception as e:
                print("Error reading data:", e)

    def update_plot(self):
        self.accel_x.setData(self.data_buffer["time"], self.data_buffer["ax"])
        self.accel_y.setData(self.data_buffer["time"], self.data_buffer["ay"])
        self.accel_z.setData(self.data_buffer["time"], self.data_buffer["az"])
        self.gyro_x.setData(self.data_buffer["time"], self.data_buffer["gx"])
        self.gyro_y.setData(self.data_buffer["time"], self.data_buffer["gy"])
        self.gyro_z.setData(self.data_buffer["time"], self.data_buffer["gz"])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialPlotter()
    window.show()
    sys.exit(app.exec_())
