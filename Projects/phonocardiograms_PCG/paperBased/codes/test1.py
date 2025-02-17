import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from sklearn.decomposition import FastICA
from scipy.signal import savgol_filter

# Step 1: Load the maternal PCG signal
rate, maternal_pcg = wav.read(r"C:\Users\kezin\Downloads\dataset\maternal\m89.wav")

# Normalize the signal
maternal_pcg = maternal_pcg / np.max(np.abs(maternal_pcg))  # Scale between -1 and 1

# Step 2: Add White Gaussian Noise (WGN)
wgn = np.random.normal(0, 0.03, maternal_pcg.shape)  # 0.02 is noise variance, adjust as needed
noisy_pcg = maternal_pcg + wgn

# Step 3: Apply Savitzky-Golay Filter for Denoising
# filtered_pcg = savgol_filter(noisy_pcg, window_length=5, polyorder=3)  # Adjust window and order if needed

# Ensure valid window length (must be <= signal length and an odd number)
window_length = min(51, len(noisy_pcg))  # Ensure it's not larger than signal length
if window_length % 2 == 0:
    window_length -= 1  # Ensure it's odd

# Apply Savitzky-Golay filter with safe mode
filtered_pcg = savgol_filter(noisy_pcg, window_length=window_length, polyorder=2, mode='nearest')


# Step 4: Apply FastICA
ica = FastICA(n_components=1)  # Since we're working with a single source
recovered_pcg = ica.fit_transform(filtered_pcg.reshape(-1, 1)).flatten()

# Step 5: Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(maternal_pcg, color='b')
plt.title("Original Maternal PCG Signal")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(noisy_pcg, color='r')
plt.title("Maternal PCG with White Gaussian Noise")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.plot(recovered_pcg, color='g')
plt.title("Recovered PCG after SG Filtering & FastICA")
plt.xlabel("Time Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
