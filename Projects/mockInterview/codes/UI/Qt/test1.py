import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout,QHBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
import random
import cv2

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.cap = None
        self._running = True

    def run(self):
        # Capture from webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video device")
            return

        while self._running:
            ret, cv_img = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(320, 240, Qt.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            else:
                print("Failed to capture frame")
            self.msleep(30)  # Add sleep to avoid maxing out CPU

    def stop(self):
        self._running = False
        if self.cap:
            self.cap.release()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interview Preparation App")
        self.setGeometry(100, 100, 800, 600)  # Set initial window size

    # List of questions
        self.questions = [
        "Question: Tell me about your project.",
        "Question: Why do you want this job?",
        "Question: What are your strengths?",
        "Question: What are your weaknesses?",
        "Question: Where do you see yourself in 5 years?",
        "Question: Why should we hire you?",
        "Question: Tell me about a challenge you faced at work.",
        "Question: How do you handle stress and pressure?",
        "Question: Describe a time you worked in a team.",
        "Question: What motivates you?"
        ]
        self.initUI()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        
    def initUI(self):
        # Set layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header
        self.header = QLabel("Interview Preparation App")
        self.header.setFont(QFont("Verdana", 16, QFont.Bold))
        self.layout.addWidget(self.header, alignment=Qt.AlignCenter)

        # Name entry with placeholder
        # self.name_entry = QLineEdit(self)
        # self.name_entry.setPlaceholderText("Name")
        # self.layout.addWidget(self.name_entry, alignment=Qt.AlignCenter)
        
        self.name_entry = QLineEdit(self)
        self.name_entry.setPlaceholderText("  Name")
        self.name_entry.setFixedSize(250, 40)  # Increase length (width) and breadth (height)
        self.name_entry.setStyleSheet("border-radius: 10px; border: 1px solid black;")
        self.layout.addWidget(self.name_entry, alignment=Qt.AlignCenter)


        # Email entry with placeholder
        self.email_entry = QLineEdit(self)
        self.email_entry.setPlaceholderText("  Email")
        self.email_entry.setFixedSize(250, 40)  # Increase length (width) and breadth (height)
        self.email_entry.setStyleSheet("border-radius: 10px; border: 1px solid black;")
        self.layout.addWidget(self.email_entry, alignment=Qt.AlignCenter)

        # Add a horizontal layout
                
        # Resume upload
        self.upload_button = QPushButton("Upload Resume", self)
        self.upload_button.clicked.connect(self.upload_resume)
        self.upload_button.setFixedSize(150, 40)
        # button_layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)
        
        # save button
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.saveName)
        self.save_button.setFixedSize(150, 40)
        # button_layout.addWidget(self.save_button)
        self.layout.addWidget(self.save_button, alignment=Qt.AlignCenter)

        # Resume label
        self.resume_label = QLabel("")
        self.layout.addWidget(self.resume_label, alignment=Qt.AlignCenter)

        # Start interview button
        self.start_interview_button = QPushButton("Start Interview", self)
        self.start_interview_button.clicked.connect(self.show_interview_ui)
        self.start_interview_button.setFixedSize(150, 40)
        self.layout.addWidget(self.start_interview_button, alignment=Qt.AlignCenter)

        # Video label
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(320, 240)
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)


    def upload_resume(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "PDF Files (*.pdf)")
        if file_name:
            self.resume_label.setText(f"Resume uploaded: {file_name.split('/')[-1]}")

    def show_interview_ui(self):
    # Clear main UI and show interview-related elements
        for i in reversed(range(self.layout.count())):
            widget_to_remove = self.layout.itemAt(i).widget()
            if (widget_to_remove is not None):
                self.layout.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)

    # Back button
        self.back_button = QPushButton("Back", self)
        self.back_button.setIcon(QIcon(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\mockInterview\icons\back1.png"))  # Update this path to your icon file
        self.back_button.clicked.connect(self.restore_main_ui)
        self.back_button.setFixedSize(150, 40)
        self.back_button.setStyleSheet("border-radius: 10px; border: 1px solid black;")
        self.layout.addWidget(self.back_button)

    # Add a horizontal layout for video control
        button_layout = QHBoxLayout()

    # Start video button
        self.start_video_button = QPushButton("Start Video", self)
        self.start_video_button.clicked.connect(self.start_video)
        self.start_video_button.setFixedSize(150, 40)
        self.start_video_button.setStyleSheet("background-color: green; color: white;border-radius: 10px;")
        button_layout.addWidget(self.start_video_button)

    # End video button
        self.end_video_button = QPushButton("End Video", self)
        self.end_video_button.clicked.connect(self.end_video)
        self.end_video_button.setFixedSize(150, 40)
        self.end_video_button.setStyleSheet("background-color: red; color: white;border-radius: 10px;")
        button_layout.addWidget(self.end_video_button)
    
        self.layout.addLayout(button_layout)

        # Add the video label to the interview UI
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        self.start_video()  # Start the video feed when switching to the interview screen
        
    # Question label
        self.question_label = QLabel("Question: Tell me about yourself.", self)
        self.question_label.setFont(QFont("Verdana", 16))  # Increase text size
        self.layout.addWidget(self.question_label, alignment=Qt.AlignCenter)


    # Next Question button
        self.NextQus_button = QPushButton("Next Question", self)
        self.NextQus_button.setIcon(QIcon(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\mockInterview\icons\next1.png"))
        self.NextQus_button.setLayoutDirection(Qt.RightToLeft)
        self.NextQus_button.clicked.connect(self.nextQus)
        self.NextQus_button.setFixedSize(150, 40)
        self.NextQus_button.setStyleSheet("border-radius: 10px; border: 1px solid black;")
        self.layout.addWidget(self.NextQus_button, alignment=Qt.AlignCenter)

    def restore_main_ui(self):
    # Clear the current layout by removing all widgets
        for i in reversed(range(self.layout.count())):
            widget_to_remove = self.layout.itemAt(i).widget()
            if widget_to_remove is not None:
                self.layout.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)

    # Re-add the main UI elements without reinitializing the layout
    # Header
        self.layout.addWidget(self.header, alignment=Qt.AlignCenter)

    # Name entry
        self.layout.addWidget(self.name_entry, alignment=Qt.AlignCenter)

    # Email entry
        self.layout.addWidget(self.email_entry, alignment=Qt.AlignCenter)

    # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.save_button)
        self.layout.addLayout(button_layout)

    # Resume label
        self.layout.addWidget(self.resume_label, alignment=Qt.AlignCenter)

    # Start interview button
        self.layout.addWidget(self.start_interview_button, alignment=Qt.AlignCenter)

    # Video label
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

    def saveName(self):
        name = self.name_entry.text()
        email = self.email_entry.text()
        print(f"Name: {name}, Email: {email}")

    def start_video(self):
        self.thread.start()

    def end_video(self):
        self.thread.stop()
        
    def nextQus(self):
        random_question = random.choice(self.questions)
        self.question_label.setText(random_question)

    def update_image(self, cv_img):
        qt_img = QPixmap.fromImage(cv_img)
        self.video_label.setPixmap(qt_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
