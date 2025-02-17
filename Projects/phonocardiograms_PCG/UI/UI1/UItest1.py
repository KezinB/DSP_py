import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTabWidget, QLabel, QLineEdit, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PCGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("PCG Signal Processing")
        self.setGeometry(100, 100, 1900, 1050)
        
        layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        
        # Tab 1: File Selection
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout()
        self.file_label = QLabel("Selected File: None")
        self.file_button = QPushButton("Select WAV File")
        self.file_button.clicked.connect(self.select_file)
        self.tab1_layout.addWidget(self.file_label)
        self.tab1_layout.addWidget(self.file_button)
        self.tab1.setLayout(self.tab1_layout)
        
        # Tab 2: Parameter Settings
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout()
        
        self.noise_label = QLabel("Noise Variance (0.01 - 0.1):")
        self.noise_input = QLineEdit("0.05")
        self.window_label = QLabel("Savitzky-Golay Window Length (odd, >5):")
        self.window_input = QLineEdit("51")
        
        self.tab2_layout.addWidget(self.noise_label)
        self.tab2_layout.addWidget(self.noise_input)
        self.tab2_layout.addWidget(self.window_label)
        self.tab2_layout.addWidget(self.window_input)
        self.tab2.setLayout(self.tab2_layout)
        
        # Add tabs
        self.tabs.addTab(self.tab1, "File Selection")
        self.tabs.addTab(self.tab2, "Settings")
        
        layout.addWidget(self.tabs)
        
        # Proceed Button
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self.process_signal)
        layout.addWidget(self.proceed_button)
        
        # Matplotlib Canvas
        self.figure, self.ax = plt.subplots(4, 1, figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
        self.file_path = None
    
    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV Files (*.wav)", options=options)
        if file_name:
            self.file_path = file_name
            self.file_label.setText(f"Selected File: {file_name}")
    
    def process_signal(self):
        if not self.file_path:
            self.file_label.setText("Please select a file first!")
            return
        
        try:
            noise_var = float(self.noise_input.text())
            window_len = int(self.window_input.text())
            if window_len % 2 == 0:
                window_len -= 1  # Ensure it's odd
            
            # Step 1: Load the signal
            rate, maternal_pcg = wav.read(self.file_path)
            maternal_pcg = maternal_pcg / np.max(np.abs(maternal_pcg))
            
            # Step 2: Add noise
            wgn = np.random.normal(0, noise_var, maternal_pcg.shape)
            noisy_pcg = maternal_pcg + wgn
            
            # Step 3: Apply Savitzky-Golay Filter
            filtered_pcg = savgol_filter(noisy_pcg, window_length=window_len, polyorder=2, mode='wrap')
            
            # Step 4: Apply FastICA
            ica = FastICA(n_components=1)
            recovered_pcg = ica.fit_transform(filtered_pcg.reshape(-1, 1)).flatten()
            recovered_pcg = recovered_pcg / np.max(np.abs(recovered_pcg)) * np.max(np.abs(maternal_pcg))
            recovered_pcg = recovered_pcg[:len(maternal_pcg)]
            
            # Step 5: Plot results
            self.ax[0].clear()
            self.ax[0].plot(maternal_pcg, color='b')
            self.ax[0].set_title("Original Maternal PCG Signal")
            
            self.ax[1].clear()
            self.ax[1].plot(noisy_pcg, color='r')
            self.ax[1].set_title("Maternal PCG with White Gaussian Noise")
            
            self.ax[2].clear()
            self.ax[2].plot(recovered_pcg, color='g')
            self.ax[2].set_title("Recovered PCG after SG Filtering & FastICA")
            
            self.ax[3].clear()
            self.ax[3].plot(maternal_pcg, color='b', label='Original')
            self.ax[3].plot(noisy_pcg, color='r', label='Noisy')
            self.ax[3].plot(recovered_pcg, color='g', label='Recovered')
            self.ax[3].set_title("Comparison of Signals")
            self.ax[3].legend()
            
            self.canvas.draw()
            
            # Step 6: Save recovered signal
            output_path = self.file_path.replace(".wav", "_recovered.wav")
            wav.write(output_path, rate, np.int16(recovered_pcg * 32767))
            self.file_label.setText(f"Processed! Saved: {output_path}")
            
        except Exception as e:
            self.file_label.setText(f"Error: {e}")
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PCGApp()
    window.show()
    sys.exit(app.exec_())
