import sys
import wfdb
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                            QTabWidget, QSpinBox, QMessageBox, QLineEdit, QFormLayout)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, find_peaks

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
        self.filtered_signal = None
        self.r_peaks = None
        self.q_peaks = None
        self.s_peaks = None
        self.p_peaks = None
        self.t_peaks = None
        
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
        self.sample_spin.valueChanged.connect(self.update_plots_and_analysis)
        
        self.reload_btn = QPushButton("Reload", self)
        self.reload_btn.clicked.connect(self.update_plots_and_analysis)
        
        control_layout.addWidget(QLabel("Analysis Duration:"))
        control_layout.addWidget(self.sample_spin)
        control_layout.addWidget(self.reload_btn)
        
        # Editable parameters
        param_layout = QFormLayout()
        self.bandpass_low = QLineEdit("0.5")
        self.bandpass_high = QLineEdit("40")
        self.pan_tompkins_threshold = QLineEdit("0.5")
        self.p_wave_prominence = QLineEdit("0.1")
        self.t_wave_prominence = QLineEdit("0.01")
        
        param_layout.addRow("Bandpass Low (Hz):", self.bandpass_low)
        param_layout.addRow("Bandpass High (Hz):", self.bandpass_high)
        param_layout.addRow("Pan-Tompkins Threshold:", self.pan_tompkins_threshold)
        param_layout.addRow("P-Wave Prominence:", self.p_wave_prominence)
        param_layout.addRow("T-Wave Prominence:", self.t_wave_prominence)
        
        # Plot area
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Results display
        self.results_label = QLabel("Analysis Results Will Appear Here")
        self.results_label.setStyleSheet("font-size: 16px;")
        
        layout.addLayout(control_layout)
        layout.addLayout(param_layout)
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
            self.update_plots_and_analysis()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process file:\n{str(e)}")
            
    def update_plots_and_analysis(self):
        if self.ecg_data is None:
            return
        
        # Get number of samples to display
        samp_sec = self.sample_spin.value()
        num_samples = int(samp_sec * self.fs)
        ecg_signal = self.ecg_data[:num_samples]
        
        # Apply preprocessing and feature extraction
        self.filtered_signal = self.bandpass_filter(ecg_signal)
        self.r_peaks = self.detect_r_peaks(self.filtered_signal)
        self.q_peaks, self.s_peaks = self.detect_qs_peaks(self.filtered_signal, self.r_peaks)
        self.p_peaks, self.t_peaks = self.detect_pt_peaks(self.filtered_signal, self.r_peaks)
        
        # Update plots
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.filtered_signal, label='Filtered ECG', alpha=0.7)
        ax.plot(self.r_peaks, self.filtered_signal[self.r_peaks], 'ro', label='R Peaks')
        ax.plot(self.q_peaks, self.filtered_signal[self.q_peaks], 'go', label='Q Peaks')
        ax.plot(self.s_peaks, self.filtered_signal[self.s_peaks], 'bo', label='S Peaks')
        ax.plot(self.p_peaks, self.filtered_signal[self.p_peaks], 'yo', label='P Peaks')
        ax.plot(self.t_peaks, self.filtered_signal[self.t_peaks], 'mo', label='T Peaks')
        ax.set_title("ECG Signal with Detected Peaks")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        self.canvas.draw()
        
        # Update results
        self.perform_analysis()
        
    def bandpass_filter(self, signal):
        lowcut = float(self.bandpass_low.text())
        highcut = float(self.bandpass_high.text())
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, signal)
    
    def detect_r_peaks(self, signal):
        threshold = float(self.pan_tompkins_threshold.text())
        peaks, _ = find_peaks(signal, height=np.mean(signal), distance=self.fs*0.6)
        return peaks
    
    def detect_qs_peaks(self, signal, r_peaks):
        q_peaks, s_peaks = [], []
        search_window = int(0.05 * self.fs)
        
        for r in r_peaks:
            q_start = max(0, r - search_window)
            q_window = signal[q_start:r]
            if len(q_window) > 0:
                q_peaks.append(np.argmin(q_window) + q_start)
            else:
                q_peaks.append(np.nan)
            
            s_end = min(len(signal), r + search_window)
            s_window = signal[r:s_end]
            if len(s_window) > 0:
                s_peaks.append(np.argmin(s_window) + r)
            else:
                s_peaks.append(np.nan)
        
        return np.array(q_peaks), np.array(s_peaks)
    
    def detect_pt_peaks(self, signal, r_peaks):
        p_peaks, t_peaks = [], []
        p_prominence = float(self.p_wave_prominence.text())
        t_prominence = float(self.t_wave_prominence.text())
        
        for i, r in enumerate(r_peaks):
            if i > 0:
                prev_t = t_peaks[i-1] if i > 0 else 0
                p_start = max(prev_t + int(0.1 * self.fs), r - int(0.3 * self.fs))
            else:
                p_start = max(0, r - int(0.3 * self.fs))
            
            p_window = signal[p_start:r]
            p_peak, _ = find_peaks(p_window, prominence=p_prominence * np.max(signal))
            if len(p_peak) > 0:
                p_peaks.append(p_peak[0] + p_start)
            else:
                p_peaks.append(np.nan)
            
            t_start = r + int(0.2 * self.fs)
            t_end = min(len(signal), r + int(0.5 * self.fs))
            t_window = signal[t_start:t_end]
            t_peak, _ = find_peaks(t_window, prominence=t_prominence * np.max(signal))
            if len(t_peak) > 0:
                t_peaks.append(t_peak[0] + t_start)
            else:
                t_peaks.append(np.nan)
        
        return np.array(p_peaks), np.array(t_peaks)
    
    def perform_analysis(self):
        if self.r_peaks is None or len(self.r_peaks) < 2:
            self.results_label.setText("Insufficient R-peaks detected for analysis.")
            return
        
        # Calculate heart rate and HRV
        rr_intervals = np.diff(self.r_peaks) / self.fs
        heart_rate = 60 / np.mean(rr_intervals)
        hrv = (np.max(rr_intervals) - np.min(rr_intervals)) * 100
        
        # Classification
        classification = []
        if heart_rate < 60:
            classification.append("Bradycardia")
        elif heart_rate > 100:
            classification.append("Tachycardia")
        
        rr_std = np.std(rr_intervals)
        if rr_std > 0.15 and len(self.p_peaks) < 0.5 * len(self.r_peaks):
            classification.append("Atrial Fibrillation (Suspected)")
        
        if not classification:
            classification.append("Normal - healthy")
        
        # Display results
        results = [
            f"Heart Rate: {heart_rate:.2f} BPM",
            f"HRV: {hrv:.2f} ms",
            f"Classification: {', '.join(classification)}"
        ]
        self.results_label.setText("\n".join(results))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    welcome = WelcomeScreen()
    welcome.show()
    sys.exit(app.exec_())