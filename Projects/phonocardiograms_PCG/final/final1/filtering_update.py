import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
from scipy.signal import savgol_filter
import os
from datetime import datetime

# Step 1: Load the maternal PCG signal
rate, maternal_pcg = wav.read(r"C:\Users\kezin\Downloads\dataset\maternal\m64.wav")

# Normalize the signal
maternal_pcg = maternal_pcg / np.max(np.abs(maternal_pcg))  # Scale between -1 and 1

# Step 2: Add White Gaussian Noise (WGN)
# wgn = np.random.normal(0, 0.05, maternal_pcg.shape)  # 0.05 is noise variance, adjust as needed
wgn = np.random.normal(0, 0.01, maternal_pcg.shape)  # 0.05 is noise variance, adjust as needed
noisy_pcg = maternal_pcg + wgn

# Step 3: Apply Savitzky-Golay Filter for Denoising
# Ensure valid window length (must be <= signal length and an odd number)
window_length = min(51, len(noisy_pcg))  # Ensure it's not larger than signal length
if window_length % 2 == 0:
    window_length -= 1  # Ensure it's odd

# Apply Savitzky-Golay filter with safe mode
filtered_pcg = savgol_filter(noisy_pcg, window_length=window_length, polyorder=2, mode='wrap') #nearest,mirror,wrap

# Step 4: Apply FastICA
ica = FastICA(n_components=1)  #source
recovered_pcg = ica.fit_transform(filtered_pcg.reshape(-1, 1)).flatten()

# Normalize recovered signal to match original amplitude
recovered_pcg = recovered_pcg / np.max(np.abs(recovered_pcg)) * np.max(np.abs(maternal_pcg))
recovered_pcg = recovered_pcg[:len(maternal_pcg)]

# Step 5: Plot the combined results

# Create output directory with timestamp
output_dir = datetime.now().strftime(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\output_%Y-%m-%d_%H-%M-%S")
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(maternal_pcg, color='b')
plt.title("Original Maternal PCG Signal")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 2)
plt.plot(noisy_pcg, color='r')
plt.title("Maternal PCG with White Gaussian Noise")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 3)
plt.plot(recovered_pcg, color='g')
plt.title("Recovered PCG after SG Filtering & FastICA")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")

plt.subplot(4, 1, 4)
plt.plot(recovered_pcg, color='g', label='Recovered PCG after SG Filtering & FastICA')
plt.plot(noisy_pcg, color='r', label='Maternal PCG with White Gaussian Noise')
plt.plot(maternal_pcg, color='b', label='Original Maternal PCG Signal')

plt.title("Comparison of Original, Noisy, and Recovered PCG Signals")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")

plt.tight_layout()

# Save plot before showing
plot_path = os.path.join(output_dir, "signal_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()

# Step 6: Save the recovered signal as a WAV file
# wav.write(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\recovered_pcg3.wav", rate, np.int16(recovered_pcg * 32767))
# Save both original and recovered signals
wav.write(os.path.join(output_dir, "original_maternal.wav"), rate, np.int16(maternal_pcg * 32767))
wav.write(os.path.join(output_dir, "recovered_pcg.wav"), rate, np.int16(recovered_pcg * 32767))
print("file saved")   