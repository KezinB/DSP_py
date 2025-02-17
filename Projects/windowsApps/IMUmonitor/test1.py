import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QPushButton)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import serial  # Import the serial library
from serial.tools import list_ports

class SerialReader(QObject):
    data_received = pyqtSignal(str)

    def __init__(self, parent=None):
        super(SerialReader, self).__init__(parent)
        self.serial_port = None
        self.timer = None
        self.status_label = parent.status_label

    def connect_to_port(self, port, baudrate):
        try:
            self.serial_port = serial.Serial(port, baudrate)
            self.timer = QTimer()
            self.timer.timeout.connect(self.read_data)
            self.timer.start(100)  # Read data every 100ms
            self.status_label.setText(f"Connected to {port}")
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            self.status_label.setText(f"Error: {e}")

    def read_data(self):
        if self.serial_port and self.serial_port.isOpen():
            try:
                data = self.serial_port.readline().decode('utf-8').strip()
                self.data_received.emit(data)
            except UnicodeDecodeError:
                pass

    def close_port(self):
        if self.serial_port and self.serial_port.isOpen():
            self.serial_port.close()
        if self.timer:
            self.timer.stop()
        self.status_label.setText("Disconnected")

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Accelerometer & Gyro Data")

        # Create widgets
        self.port_label = QLabel("COM Port:")
        self.port_combo = QComboBox()
        self.baudrate_label = QLabel("Baud Rate:")
        self.baudrate_combo = QComboBox()
        self.status_label = QLabel()
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        self.connect_button = QPushButton("Connect")

        # Populate combo boxes
        self.port_combo.addItem("Auto")
        self.baudrate_combo.addItem("9600")
        self.baudrate_combo.addItem("115200")

        # Create layouts
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.port_label)
        top_layout.addWidget(self.port_combo)
        top_layout.addWidget(self.baudrate_label)
        top_layout.addWidget(self.baudrate_combo)
        top_layout.addWidget(self.status_label)
        top_layout.addWidget(self.connect_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.data_display)

        # Set layout
        self.setLayout(main_layout)

        # Create serial reader
        self.serial_reader = SerialReader(self)
        self.serial_reader.data_received.connect(self.update_display)

        # Connect signals
        self.connect_button.clicked.connect(self.update_port)

    def update_port(self):
        port = self.port_combo.currentText()
        baudrate = int(self.baudrate_combo.currentText())

        if port == "Auto":
            # Use list_ports to find available ports
            available_ports = list_ports.comports()
            if available_ports:
                self.port_combo.clear()
                self.port_combo.addItems([port.device for port in available_ports])
                self.port_combo.setCurrentIndex(0)  # Select the first available port
            else:
                self.status_label.setText("No serial ports found.")
                return
        else:
            self.serial_reader.close_port()
            self.serial_reader.connect_to_port(port, baudrate)

    def update_display(self, data):
        self.data_display.append(data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())