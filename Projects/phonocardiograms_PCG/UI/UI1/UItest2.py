import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QFileDialog, QTabWidget, QSpinBox, QLabel, QDoubleSpinBox)
from PyQt5.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

        # Tabs for file selection and parameter settings
        self.tabs = QTabWidget(self)
        
        # File Selection Tab
        self.file_tab = QWidget()
        file_layout = QVBoxLayout()
        self.file_button = QPushButton("Select WAV File")
        self.file_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label)
        self.file_tab.setLayout(file_layout)
        
        # Parameter Settings Tab
        self.param_tab = QWidget()
        param_layout = QVBoxLayout()
        
        # Noise Variance
        self.noise_label = QLabel("Noise Variance:")
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.01, 0.1)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(0.05)  # Default
        
        # Savitzky-Golay Window
        self.window_label = QLabel("Savitzky-Golay Window Size:")
        self.window_spin = QSpinBox()
        self.window_spin.setRange(5, 101)
        self.window_spin.setSingleStep(2)
        self.window_spin.setValue(51)  # Default
        
        param_layout.addWidget(self.noise_label)
        param_layout.addWidget(self.noise_spin)
        param_layout.addWidget(self.window_label)
        param_layout.addWidget(self.window_spin)
        
        self.param_tab.setLayout(param_layout)
        
        # Add tabs
        self.tabs.addTab(self.file_tab, "File Selection")
        self.tabs.addTab(self.param_tab, "Settings")
        layout.addWidget(self.tabs)

        # Proceed Button
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self.process_signal)
        layout.addWidget(self.proceed_button)

        # Save Image Button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_plot)
        self.save_button.setEnabled(False)  # Initially disabled
        layout.addWidget(self.save_button)

        # Matplotlib Canvas (Initially Hidden)
        self.figure, self.ax = plt.subplots(4, 1, figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.hide()
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # Variables
        self.file_path = None
        self.rate = None
        self.maternal_pcg = None

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV Files (*.wav)")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected: {file_path.split('/')[-1]}")

    def process_signal(self):
        if not self.file_path:
            self.file_label.setText("Please select a file first!")
            return
        
        # Load WAV file
        self.rate, self.maternal_pcg = wav.read(self.file_path)
        self.maternal_pcg = self.maternal_pcg / np.max(np.abs(self.maternal_pcg))  # Normalize
        
        # Get user-defined parameters
        noise_variance = self.noise_spin.value()
        window_length = self.window_spin.value()
        
        # Add Gaussian noise
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

        # Hide canvas before plotting
        self.canvas.hide()
        
        # Update plot
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[2].cla()
        self.ax[3].cla()

        self.ax[0].plot(self.maternal_pcg, color='b')
        self.ax[0].set_title("Original Maternal PCG Signal")

        self.ax[1].plot(noisy_pcg, color='r')
        self.ax[1].set_title("Maternal PCG with White Gaussian Noise")

        self.ax[2].plot(recovered_pcg, color='g')
        self.ax[2].set_title("Recovered PCG after SG Filtering & FastICA")

        self.ax[3].plot(recovered_pcg, color='g', label='Recovered PCG')
        self.ax[3].plot(noisy_pcg, color='r', label='Noisy PCG')
        self.ax[3].plot(self.maternal_pcg, color='b', label='Original PCG')
        self.ax[3].set_title("Comparison of Original, Noisy, and Recovered PCG")
        self.ax[3].legend()

        # Adjust layout and show plot
        self.figure.tight_layout()
        self.canvas.draw()
        self.canvas.show()

        # Enable "Save Image" button
        self.save_button.setEnabled(True)

        # Save recovered signal
        save_path = self.file_path.replace(".wav", "_recovered.wav")
        wav.write(save_path, self.rate, np.int16(recovered_pcg * 32767))
        print(f"Recovered PCG saved as {save_path}")

    def save_plot(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if save_path:
            self.figure.savefig(save_path)
            print(f"Plot saved as {save_path}")

# Run Application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PCGProcessor()
    window.setWindowTitle("PCG Signal Processing")
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
