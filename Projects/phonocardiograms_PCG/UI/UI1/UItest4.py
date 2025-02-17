import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, 
                             QTabWidget, QSpinBox, QLabel, QDoubleSpinBox, QHBoxLayout)
from PyQt5.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

class PCGProcessor(QWidget):
    def __init__(self):
        super().__init__()

        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f0f0f0"))
        self.setPalette(palette)

        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)

        # Tabs for steps
        self.tabs = QTabWidget(self)

        # Step 1: File Selection & Settings Tab
        self.file_param_tab = QWidget()
        file_param_layout = QVBoxLayout()

        # File Selection
        self.file_button = QPushButton("Select WAV File")
        self.file_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")

        # Noise Variance
        self.noise_label = QLabel("Noise Variance:")
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.01, 0.1)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.05)

        # Savitzky-Golay Window
        self.window_label = QLabel("Savitzky-Golay Window Size:")
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 101)
        self.window_spin.setSingleStep(2)
        self.window_spin.setValue(51)

        # Proceed Button
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.setEnabled(False)
        self.proceed_button.clicked.connect(self.process_signal)

        # Waveform Display (Hidden Initially)
        self.figure1, self.ax1 = plt.subplots(figsize=(8, 4))
        self.canvas1 = FigureCanvas(self.figure1)
        self.canvas1.hide()

        # Align Buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.file_button)
        button_layout.addWidget(self.proceed_button)

        # Add Widgets to Layout
        file_param_layout.addLayout(button_layout)
        file_param_layout.addWidget(self.file_label)
        file_param_layout.addWidget(self.noise_label)
        file_param_layout.addWidget(self.noise_spin)
        file_param_layout.addWidget(self.window_label)
        file_param_layout.addWidget(self.window_spin)
        file_param_layout.addWidget(self.canvas1)
        self.file_param_tab.setLayout(file_param_layout)

        # Step 2: Processing Tab
        self.process_tab = QWidget()
        process_layout = QVBoxLayout()
        Testing_layout = QVBoxLayout()

        self.figure2, self.ax2 = plt.subplots(4, 1, figsize=(10, 8))
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas2.hide()
        
        # Step 3: testing Tab
        self.Testing_tab = QWidget()
        Testing_layout = QVBoxLayout()

        self.figure2, self.ax2 = plt.subplots(4, 1, figsize=(10, 8))
        self.canvas3 = FigureCanvas(self.figure2)
        self.canvas3.hide()

        # Buttons for Saving
        save_button_layout = QHBoxLayout()
        self.save_img_button = QPushButton("Save Image")
        self.save_img_button.clicked.connect(self.save_plot)
        self.save_img_button.setEnabled(False)

        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.clicked.connect(self.save_csv)
        self.save_csv_button.setEnabled(False)

        save_button_layout.addWidget(self.save_img_button)
        save_button_layout.addWidget(self.save_csv_button)

        process_layout.addWidget(self.canvas2)
        process_layout.addLayout(save_button_layout)
        self.process_tab.setLayout(process_layout)
        self.Testing_tab.setLayout(Testing_layout)

        # Add tabs
        self.tabs.addTab(self.file_param_tab, "1. File & Settings")
        self.tabs.addTab(self.process_tab, "2. Processing")
        self.tabs.addTab(self.Testing_tab, "3. Testing")
        self.tabs.setTabEnabled(1, False)

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.file_path = None
        self.rate = None
        self.maternal_pcg = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV Files (*.wav)")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected: {file_path.split('/')[-1]}")

            # Load and Normalize WAV file
            self.rate, self.maternal_pcg = wav.read(self.file_path)
            self.maternal_pcg = self.maternal_pcg / np.max(np.abs(self.maternal_pcg))  

            # Show Original Waveform
            self.ax1.clear()
            self.ax1.plot(self.maternal_pcg, color='b')
            self.ax1.set_title("Original Maternal PCG Signal")
            self.figure1.tight_layout()
            self.canvas1.draw()
            self.canvas1.show()

            # Enable Proceed Button
            self.proceed_button.setEnabled(True)

    def process_signal(self):
        noise_variance = self.noise_spin.value()
        window_length = self.window_spin.value()
        
        # Add Gaussian Noise
        wgn = np.random.normal(0, noise_variance, self.maternal_pcg.shape)
        noisy_pcg = self.maternal_pcg + wgn

        # Ensure valid window length
        window_length = min(window_length, len(noisy_pcg))
        if window_length % 2 == 0:
            window_length -= 1

        # Apply Savitzky-Golay filter
        filtered_pcg = savgol_filter(noisy_pcg, window_length=window_length, polyorder=2, mode='wrap')

        # Apply FastICA
        ica = FastICA(n_components=1)
        recovered_pcg = ica.fit_transform(filtered_pcg.reshape(-1, 1)).flatten()
        recovered_pcg = recovered_pcg / np.max(np.abs(recovered_pcg)) * np.max(np.abs(self.maternal_pcg))
        recovered_pcg = recovered_pcg[:len(self.maternal_pcg)]

        # Update Plot
        self.ax2[0].cla()
        self.ax2[1].cla()
        self.ax2[2].cla()
        self.ax2[3].cla()

        self.ax2[0].plot(self.maternal_pcg, color='b')
        self.ax2[0].set_title("Original Maternal PCG Signal")

        self.ax2[1].plot(noisy_pcg, color='r')
        self.ax2[1].set_title("Maternal PCG with White Gaussian Noise")

        self.ax2[2].plot(recovered_pcg, color='g')
        self.ax2[2].set_title("Recovered PCG after SG Filtering & FastICA")

        self.ax2[3].plot(recovered_pcg, color='g', label='Recovered PCG')
        self.ax2[3].plot(noisy_pcg, color='r', label='Noisy PCG')
        self.ax2[3].plot(self.maternal_pcg, color='b', label='Original PCG')
        self.ax2[3].set_title("Comparison of Original, Noisy, and Recovered PCG")
        self.ax2[3].legend()

        self.figure2.tight_layout()
        self.canvas2.draw()
        self.canvas2.show()

        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)

        self.save_img_button.setEnabled(True)
        self.save_csv_button.setEnabled(True)

    def save_plot(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if save_path:
            self.figure2.savefig(save_path)
            print(f"Plot saved as {save_path}")

    def save_csv(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if save_path:
            df = pd.DataFrame({"Original": self.maternal_pcg})
            df.to_csv(save_path, index=False)
            print(f"CSV saved as {save_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PCGProcessor()
    window.setWindowTitle("PCG Signal Processing")
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec_())
