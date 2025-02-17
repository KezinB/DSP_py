import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import librosa.display
from scipy.signal import welch
from sklearn.decomposition import FastICA
from scipy.signal import savgol_filter

# Step 1: Load the maternal PCG signal
rate, maternal_pcg = wav.read(r"C:\Users\kezin\Downloads\dataset\maternal\m64.wav")

# Normalize the signal
maternal_pcg = maternal_pcg / np.max(np.abs(maternal_pcg))  # Scale between -1 and 1

# Step 2: Add White Gaussian Noise (WGN)
wgn = np.random.normal(0, 0.05, maternal_pcg.shape)  
noisy_pcg = maternal_pcg + wgn

# Step 3: Apply Savitzky-Golay Filter for Denoising
window_length = min(51, len(noisy_pcg))
if window_length % 2 == 0:
    window_length -= 1  # Ensure it's odd

filtered_pcg = savgol_filter(noisy_pcg, window_length=window_length, polyorder=2, mode='wrap')

# Step 4: Apply FastICA
ica = FastICA(n_components=1)
recovered_pcg = ica.fit_transform(filtered_pcg.reshape(-1, 1)).flatten()

# Normalize recovered signal
recovered_pcg = recovered_pcg / np.max(np.abs(recovered_pcg)) * np.max(np.abs(maternal_pcg))
recovered_pcg = recovered_pcg[:len(maternal_pcg)]

# Function to compute and plot log frequency power spectrum using Welch's method
def plot_welch_log_spectrum(signal, sr, title):
    # Ensure nperseg is valid
    nperseg = min(1024, len(signal) // 2)  # Ensure nperseg is not greater than signal length

    # Compute Power Spectral Density (PSD) using Welch's method
    freqs, psd = welch(signal, fs=sr, nperseg=nperseg)

    # Convert PSD to dB scale (avoid log(0) issue)
    log_psd = 10 * np.log10(psd + 1e-10)  # Small offset to avoid log(0)

    # Plot log frequency power spectrum
    plt.figure(figsize=(10, 4))
    plt.semilogx(freqs, log_psd, color='b')  # Logarithmic frequency axis
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Step 5: Plot log spectrograms using Welch's method
plot_welch_log_spectrum(maternal_pcg, rate, "Welch Log Spectrum - Original PCG")
plot_welch_log_spectrum(noisy_pcg, rate, "Welch Log Spectrum - Noisy PCG")
plot_welch_log_spectrum(recovered_pcg, rate, "Welch Log Spectrum - Recovered PCG")

# Step 6: Save the recovered signal as a WAV file
wav.write(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\recovered_pcg1.wav", rate, np.int16(recovered_pcg * 32767))
print("Recovered PCG signal saved as recovered_pcg1.wav")
