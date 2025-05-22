import sys
import cv2
import numpy as np
import requests
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex,QMutexLocker

class StreamWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, url, auth):
        super().__init__()
        self.url = url
        self.auth = auth
        self.running = False
        self.mutex = QMutex()
        self.buffer = bytes()

    def run(self):
        self.running = True
        session = requests.Session()
        try:
            response = session.get(self.url, auth=self.auth, stream=True, timeout=5)
            for chunk in response.iter_content(chunk_size=16384):
                with QMutexLocker(self.mutex):
                    if not self.running:
                        break
                
                if chunk:
                    self.buffer += chunk
                    self.process_frames()
                    
        except Exception as e:
            self.error_signal.emit(f"Stream error: {str(e)}")
        finally:
            session.close()
            self.running = False

    def process_frames(self):
        while True:
            start = self.buffer.find(b'\xff\xd8')
            end = self.buffer.find(b'\xff\xd9')
            
            if start == -1 or end == -1:
                break
                
            if start < end:
                jpg_data = self.buffer[start:end+2]
                self.buffer = self.buffer[end+2:]
                
                try:
                    frame = cv2.imdecode(np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame_ready.emit(frame)
                except Exception as e:
                    self.error_signal.emit(f"Decode error: {str(e)}")

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
        self.wait()

class LoginWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Camera Login')
        self.setFixedSize(400, 200)

        layout = QtWidgets.QVBoxLayout()
        
        self.url_input = QtWidgets.QLineEdit('http://192.168.4.1/sustain?stream=0')
        self.user_input = QtWidgets.QLineEdit('ESPCAM')
        self.pass_input = QtWidgets.QLineEdit('1234')
        self.pass_input.setEchoMode(QtWidgets.QLineEdit.Password)
        
        connect_btn = QtWidgets.QPushButton('Connect')
        connect_btn.clicked.connect(self.connect_camera)

        layout.addWidget(QtWidgets.QLabel('Stream URL:'))
        layout.addWidget(self.url_input)
        layout.addWidget(QtWidgets.QLabel('Username:'))
        layout.addWidget(self.user_input)
        layout.addWidget(QtWidgets.QLabel('Password:'))
        layout.addWidget(self.pass_input)
        layout.addWidget(connect_btn)

        self.setLayout(layout)

    def connect_camera(self):
        url = self.url_input.text()
        user = self.user_input.text()
        password = self.pass_input.text()

        self.stream_window = StreamWindow(url, user, password)
        self.stream_window.show()
        self.close()

class StreamWindow(QtWidgets.QMainWindow):
    def __init__(self, url, user, password):
        super().__init__()
        self.url = url
        self.user = user
        self.password = password
        self.recording = False
        self.video_writer = None
        self.init_ui()
        self.start_stream()

    def init_ui(self):
        self.setWindowTitle('Live Stream')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout()
        
        # Video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Control buttons
        control_layout = QtWidgets.QHBoxLayout()
        self.snapshot_btn = QtWidgets.QPushButton('Capture Image')
        self.record_btn = QtWidgets.QPushButton('Start Recording')
        self.quit_btn = QtWidgets.QPushButton('Exit')

        self.snapshot_btn.clicked.connect(self.capture_image)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.quit_btn.clicked.connect(self.close)

        control_layout.addWidget(self.snapshot_btn)
        control_layout.addWidget(self.record_btn)
        control_layout.addWidget(self.quit_btn)

        layout.addLayout(control_layout)
        central_widget.setLayout(layout)

    def start_stream(self):
        self.worker = StreamWorker(self.url, (self.user, self.password))
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.error_signal.connect(self.show_error)
        self.worker.start()

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, 
                               QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self.recording:
            if self.video_writer is None:
                self.init_video_writer(frame)
            self.video_writer.write(frame)

    def init_video_writer(self, frame):
        self.video_writer = cv2.VideoWriter(
            f'recording_{QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")}.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            20.0,
            (frame.shape[1], frame.shape[0])
        )

    def toggle_recording(self):
        self.recording = not self.recording
        self.record_btn.setText('Stop Recording' if self.recording else 'Start Recording')
        if not self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def capture_image(self):
        pixmap = self.video_label.pixmap()
        if pixmap:
            filename = f'capture_{QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")}.jpg'
            pixmap.save(filename)
            QtWidgets.QMessageBox.information(self, 'Success', f'Image saved as {filename}')

    def show_error(self, message):
        QtWidgets.QMessageBox.critical(self, 'Error', message)
        self.close()

    def closeEvent(self, event):
        self.worker.stop()
        if self.video_writer is not None:
            self.video_writer.release()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())