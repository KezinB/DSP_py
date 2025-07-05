import sys
import cv2
import numpy as np
import requests
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                            QLabel, QLineEdit, QPushButton, 
                            QVBoxLayout, QHBoxLayout)

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('ESP32-CAM Login')
        self.setGeometry(100, 100, 400, 200)
        
        layout = QVBoxLayout()
        
        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText('Stream URL')
        self.user_input = QLineEdit(self)
        self.user_input.setPlaceholderText('Username')
        self.pass_input = QLineEdit(self)
        self.pass_input.setPlaceholderText('Password')
        self.pass_input.setEchoMode(QLineEdit.Password)
        
        self.submit_btn = QPushButton('Connect', self)
        self.submit_btn.clicked.connect(self.authenticate)
        
        layout.addWidget(QLabel('Stream URL:'))
        layout.addWidget(self.url_input)
        layout.addWidget(QLabel('Username:'))
        layout.addWidget(self.user_input)
        layout.addWidget(QLabel('Password:'))
        layout.addWidget(self.pass_input)
        layout.addWidget(self.submit_btn)
        
        self.setLayout(layout)
    
    def authenticate(self):
        url = self.url_input.text()
        user = self.user_input.text()
        password = self.pass_input.text()
        
        self.stream_window = StreamWindow(url, user, password)
        self.stream_window.show()
        self.close()

class StreamWindow(QMainWindow):
    def __init__(self, url, user, password):
        super().__init__()
        self.url = url
        self.user = user
        self.password = password
        self.recording = False
        self.video_writer = None
        self.initUI()
        self.initStream()
        
    def initUI(self):
        self.setWindowTitle('ESP32-CAM Stream')
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.snapshot_btn = QPushButton('Take Snapshot', self)
        self.snapshot_btn.clicked.connect(self.save_snapshot)
        self.record_btn = QPushButton('Start Recording', self)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.quit_btn = QPushButton('Quit', self)
        self.quit_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(self.snapshot_btn)
        btn_layout.addWidget(self.record_btn)
        btn_layout.addWidget(self.quit_btn)
        
        layout.addLayout(btn_layout)
        central_widget.setLayout(layout)
        
        # Timer for video updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def initStream(self):
        self.response = requests.get(self.url, 
                                   auth=(self.user, self.password), 
                                   stream=True)
        self.bytes_data = bytes()
        self.timer.start(30)  # Update every 30ms
        
    def update_frame(self):
        try:
            for chunk in self.response.iter_content(chunk_size=8192):
                self.bytes_data += chunk
                a = self.bytes_data.find(b'\xff\xd8')
                b = self.bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg_frame = self.bytes_data[a:b+2]
                    self.bytes_data = self.bytes_data[b+2:]
                    
                    # Process frame
                    img = cv2.imdecode(np.frombuffer(jpg_frame, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        # Convert to Qt format
                        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, 
                                              QtGui.QImage.Format_RGB888)
                        pixmap = QtGui.QPixmap.fromImage(qt_image)
                        self.video_label.setPixmap(pixmap.scaled(
                            self.video_label.size(), Qt.KeepAspectRatio))
                        
                        # Save frame if recording
                        if self.recording:
                            if self.video_writer is None:
                                self.init_video_writer(img.shape)
                            self.video_writer.write(img)
                    break
        except Exception as e:
            print(f"Error: {e}")
            self.timer.stop()
            
    def init_video_writer(self, frame_shape):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            'recording.avi', fourcc, 20.0, 
            (frame_shape[1], frame_shape[0])
        )
        
    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.record_btn.setText('Stop Recording')
        else:
            self.record_btn.setText('Start Recording')
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                
    def save_snapshot(self):
        pixmap = self.video_label.pixmap()
        if pixmap:
            pixmap.save(f'snapshot_{QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")}.jpg')
            
    def closeEvent(self, event):
        self.timer.stop()
        if self.video_writer is not None:
            self.video_writer.release()
        self.response.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login = LoginWindow()
    login.show()
    sys.exit(app.exec_())