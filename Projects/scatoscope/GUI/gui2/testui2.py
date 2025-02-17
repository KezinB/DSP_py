import sys
import serial
import serial.tools.list_ports
import csv
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout,QVBoxLayout, QPushButton, QComboBox, QWidget, QLabel, QHBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, QTime
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import pyqtgraph.exporters

def list_com_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

class SerialPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scatoscope Data Plotter")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        grid_layout = QGridLayout()
        
        self.com_selector = QComboBox()
        self.com_selector.addItems(list_com_ports())
        #self.com_selector.setFixedSize(1200, 40)
        self.com_selector.setFixedHeight(45)
        self.com_selector.setStyleSheet(
            "background-color: #E2E0C8; "
            "color: black; "
            "border: 1px solid black; "  # Set border
            "border-radius: 5px;"  # Optional: Rounded corners
            "font-size: 13px;"
        )
        self.layout.addWidget(self.com_selector)

        self.connect_button = QPushButton("Connect")
        self.connect_button.setFixedSize(150, 50)
        self.connect_button.setStyleSheet(
            "background-color: #E2E0C8; "
            "color: black; "
            "border: 1px solid black; "  # Set border
            "border-radius: 5px;"  # Optional: Rounded corners
            "font-size: 13px;"
        )
        self.connect_button.clicked.connect(self.connect_serial)
        self.layout.addWidget(self.connect_button)

        grid_layout.addWidget(self.com_selector, 0, 0)
        grid_layout.addWidget(self.connect_button, 0, 1)
        self.layout.addLayout(grid_layout)

        
        self.timer_label = QLabel("Time: 00:00")
        font = QFont()
        font.setPointSize(20)  # Set the desired font size
        self.timer_label.setFont(font)
        self.layout.addWidget(self.timer_label)
        self.start_time = None

        self.graph = PlotWidget()
        self.layout.addWidget(self.graph)

        self.graph.setTitle("Scatoscope Data")
        self.graph.setLabel('left', "Data")
        self.graph.setLabel('bottom', "Time")
        self.graph.addLegend()

        self.accel_x = self.graph.plot(pen='r', name='Accel X')
        self.accel_y = self.graph.plot(pen='g', name='Accel Y')
        self.accel_z = self.graph.plot(pen='b', name='Accel Z')
        self.gyro_x = self.graph.plot(pen='y', name='Gyro X')
        self.gyro_y = self.graph.plot(pen='m', name='Gyro Y')
        self.gyro_z = self.graph.plot(pen='c', name='Gyro Z')
        
        grid_layout = QGridLayout()
        font = QFont()
        font.setPointSize(12)  # Set the desired font size
        
        self.ax_label = QLabel("Accel X: ")
        self.ay_label = QLabel("Accel Y: ")
        self.az_label = QLabel("Accel Z: ")
        self.gx_label = QLabel("Gyro X: ")
        self.gy_label = QLabel("Gyro Y: ")
        self.gz_label = QLabel("Gyro Z: ")
        
        self.ax_label.setFont(font)
        self.ay_label.setFont(font)
        self.az_label.setFont(font)
        self.gx_label.setFont(font)
        self.gy_label.setFont(font)
        self.gz_label.setFont(font)

        grid_layout.addWidget(self.ax_label, 0, 0)
        grid_layout.addWidget(self.ay_label, 0, 1)
        grid_layout.addWidget(self.az_label, 0, 2)
        grid_layout.addWidget(self.gx_label, 1, 0)
        grid_layout.addWidget(self.gy_label, 1, 1)
        grid_layout.addWidget(self.gz_label, 1, 2)

        self.layout.addLayout(grid_layout)

        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        self.reset_button = QPushButton("Reset Graph")
        self.reset_button.setFixedSize(150, 50)
        self.reset_button.setStyleSheet(
            "background-color: #E2E0C8; "
            "color: black; "
            "border: 1px solid black; "  # Set border
            "border-radius: 5px;"  # Optional: Rounded corners
            "font-size: 13px;"
        )
        self.reset_button.clicked.connect(self.reset_graph)
        self.button_layout.addWidget(self.reset_button)

        self.pause_button = QPushButton("Pause/Play")
        self.pause_button.setFixedSize(150, 50)
        self.pause_button.setStyleSheet(            
            "background-color: #E2E0C8; "
            "color: black; "
            "border: 1px solid black; "  # Set border
            "border-radius: 5px;"  # Optional: Rounded corners
            "font-size: 13px;"
        )
        self.pause_button.clicked.connect(self.toggle_pause)
        self.button_layout.addWidget(self.pause_button)

        self.save_img_button = QPushButton("Save as Image")
        self.save_img_button.setFixedSize(150, 50)
        self.save_img_button.setStyleSheet(
            "background-color: #E2E0C8; "
            "color: black; "
            "border: 1px solid black; "  # Set border
            "border-radius: 5px;"  # Optional: Rounded corners
            "font-size: 13px;"
        )
        self.save_img_button.clicked.connect(self.save_as_image)
        self.button_layout.addWidget(self.save_img_button)

        self.save_csv_button = QPushButton("Save as CSV")
        self.save_csv_button.setFixedSize(150, 50)
        self.save_csv_button.setStyleSheet(
            "background-color: #E2E0C8; "
            "color: black; "
            "border: 1px solid black; "  # Set border
            "border-radius: 5px;"  # Optional: Rounded corners
            "font-size: 13px;"
        )
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

    def toggle_pause(self):
        self.running = not self.running

    def read_serial(self):
        if self.serial_port and self.serial_port.in_waiting and self.running:
            if self.start_time is None:
                self.start_time = QTime.currentTime()
                self.connect_button.setText("Connected")
                self.connect_button.setStyleSheet(
                    "background-color: red; "
                    "color: white; "
                    "border: 1px solid black; "  # Set border
                    "border-radius: 5px;"  # Optional: Rounded corners
                    "font-size: 13px;"
                )
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
                    self.ax_label.setText(f"Accel X: {clean_values[0]}")
                    
                    self.data_buffer["ay"].append(clean_values[1])
                    self.ay_label.setText(f"Accel Y: {clean_values[1]}")
                    
                    self.data_buffer["az"].append(clean_values[2])
                    self.az_label.setText(f"Accel Z: {clean_values[2]}")
                    
                    self.data_buffer["gx"].append(clean_values[3])
                    self.gx_label.setText(f"Gyro X: {clean_values[3]}")
                    
                    self.data_buffer["gy"].append(clean_values[4])
                    self.gy_label.setText(f"Gyro Y: {clean_values[4]}")
                    
                    self.data_buffer["gz"].append(clean_values[5])
                    self.gz_label.setText(f"Gyro Z: {clean_values[5]}")
                    
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
        self.start_time = None
        self.update_plot()

    def save_as_image(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"C:/Users/kezin/OneDrive/Documents/Codes/python/Projects/scatoscope/datas/Images/ScatoscopeData_{timestamp}.png"
        exporter = pg.exporters.ImageExporter(self.graph.plotItem)
        exporter.export(filename)

    def save_as_csv(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"C:/Users/kezin/OneDrive/Documents/Codes/python/Projects/scatoscope/datas/logs/ScatoscopeData_{timestamp}.csv"

        with open(filename, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"])
            for i in range(len(self.data_buffer["time"])):
                writer.writerow([self.data_buffer["time"][i], self.data_buffer["ax"][i], self.data_buffer["ay"][i], self.data_buffer["az"][i],
                                 self.data_buffer["gx"][i], self.data_buffer["gy"][i], self.data_buffer["gz"][i]])
        print(f"CSV file saved as {filename}")

    def update_time(self):
        if self.start_time:
            elapsed_time = self.start_time.secsTo(QTime.currentTime())
            self.timer_label.setText(f"Time: {elapsed_time//60:02}:{elapsed_time%60:02}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialPlotter()
    window.show()
    sys.exit(app.exec_())