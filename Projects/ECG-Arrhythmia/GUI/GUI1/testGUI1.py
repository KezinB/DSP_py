import sys
import wfdb
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                            QTabWidget, QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class WelcomeScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('ECG Analysis System')
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        self.label = QLabel("Welcome to ECG Analysis System", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        self.start_btn = QPushButton("Get Started", self)
        self.start_btn.setStyleSheet("font-size: 18px; padding: 15px;")
        self.start_btn.clicked.connect(self.open_main_window)
        
        layout.addWidget(self.label)
        layout.addWidget(self.start_btn)
        self.setLayout(layout)
    
    def open_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ecg_data = None
        self.fs = None
        
    def initUI(self):
        self.setWindowTitle('ECG Analysis')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        live_menu = menubar.addMenu('Live Test')
        
        # File menu actions
        file_menu.addAction('Open File', self.open_file_dialog)
        file_menu.addAction('Exit', self.close)
        
        # Central widget with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Initialize empty tabs
        self.file_tab = QWidget()
        self.live_tab = QWidget()
        self.tabs.addTab(self.file_tab, "File Analysis")
        self.tabs.addTab(self.live_tab, "Live Monitoring")
        
        # Setup file analysis tab
        self.setup_file_tab()
        
    def setup_file_tab(self):
        layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 600)
        self.sample_spin.setValue(10)
        self.sample_spin.setSuffix(" seconds")
        control_layout.addWidget(QLabel("Analysis Duration:"))
        control_layout.addWidget(self.sample_spin)
        
        # Plot area
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Results display
        self.results_label = QLabel("Analysis Results Will Appear Here")
        self.results_label.setStyleSheet("font-size: 16px;")
        
        layout.addLayout(control_layout)
        layout.addWidget(self.canvas)
        layout.addWidget(self.results_label)
        self.file_tab.setLayout(layout)
        
    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open ECG File", "", 
            "ECG Files (*.dat *.hea *.mat *.csv)", options=options)
            
        if file_name:
            self.process_file(file_name)
            
    def process_file(self, file_path):
        try:
            # Load and process ECG data using your existing code
            record = wfdb.rdrecord(file_path.split('.')[0])
            self.ecg_data = record.p_signal[:, 0]
            self.fs = record.fs
            
            # Apply processing chain
            self.update_plots()
            self.perform_analysis()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process file:\n{str(e)}")
            
    def update_plots(self):
        self.figure.clear()
        
        # Get number of samples to display
        samp_sec = self.sample_spin.value()
        num_samples = int(samp_sec * self.fs)
        ecg_signal = self.ecg_data[:num_samples]
        
        # Plot raw signal
        ax = self.figure.add_subplot(111)
        ax.plot(ecg_signal)
        ax.set_title("Raw ECG Signal")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        
        self.canvas.draw()
        
    def perform_analysis(self):
        # Add your analysis code here
        # This should include all the processing steps from your existing code
        # and update the results_label accordingly
        
        # Example results:
        results = [
            "Heart Rate: 72 BPM",
            "HRV: 24.5 ms",
            "Classification: Normal"
        ]
        self.results_label.setText("\n".join(results))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    welcome = WelcomeScreen()
    welcome.show()
    sys.exit(app.exec_()) 