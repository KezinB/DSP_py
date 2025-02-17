import sys
import serial
import serial.tools.list_ports
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QComboBox, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer, QTime
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

        self.timer_label = QLabel("Time: 00:00")
        self.layout.addWidget(self.timer_label)
        self.start_time = QTime.currentTime()

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

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.reset_button = QPushButton("Reset Graph")
        self.reset_button.clicked.connect(self.reset_graph)
        self.button_layout.addWidget(self.reset_button)

        self.pause_button = QPushButton("Pause/Play")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.button_layout.addWidget(self.pause_button)

        self.save_img_button = QPushButton("Save as Image")
        self.save_img_button.clicked.connect(self.save_as_image)
        self.button_layout.addWidget(self.save_img_button)

        self.save_csv_button = QPushButton("Save as CSV")
        self.save_csv_button.clicked.connect(self.save_as_csv)
        self.button_layout.addWidget(self.save_csv_button)

        self.serial_port = None
        self.data_buffer = {"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "time": []}
        self.time_counter = 0
        self.running = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_serial)
        self.time_update_timer = QTimer()
        self.time_update_timer.timeout.connect(self.update_time)
        self.time_update_timer.start(1000)

    def connect_serial(self):
        port = self.com_selector.currentText()
        try:
            self.serial_port = serial.Serial(port, 115200, timeout=1)
            self.running = True
            self.timer.start(50)
        except serial.SerialException:
            print("Failed to connect to port")

    def read_serial(self):
        if self.serial_port and self.serial_port.in_waiting and self.running:
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

    def reset_graph(self):
        self.data_buffer = {"ax": [], "ay": [], "az": [], "gx": [], "gy": [], "gz": [], "time": []}
        self.time_counter = 0
        self.update_plot()

    def toggle_pause(self):
        self.running = not self.running

    def save_as_image(self):
        exporter = pg.exporters.ImageExporter(self.graph.plotItem)
        exporter.export("plot.png")

    def save_as_csv(self):
        with open("data.csv", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"])
            for i in range(len(self.data_buffer["time"])):
                writer.writerow([self.data_buffer["time"][i], self.data_buffer["ax"][i], self.data_buffer["ay"][i], self.data_buffer["az"][i],
                                 self.data_buffer["gx"][i], self.data_buffer["gy"][i], self.data_buffer["gz"][i]])

    def update_time(self):
        elapsed_time = self.start_time.secsTo(QTime.currentTime())
        self.timer_label.setText(f"Time: {elapsed_time//60:02}:{elapsed_time%60:02}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialPlotter()
    window.show()
    sys.exit(app.exec_())
