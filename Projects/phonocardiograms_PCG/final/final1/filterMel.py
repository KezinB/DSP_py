import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import librosa.display
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

# Function to compute and plot Mel-frequency spectrogram
def plot_mel_spectrogram(signal, sr, title):
    # Ensure n_fft is at most half the signal length
    n_fft = min(1024, len(signal) // 2)
    hop_length = n_fft // 4  # Standard hop size

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).squeeze()  # Ensure proper shape

    # Plot Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.show()

# Step 5: Plot Mel spectrograms
plot_mel_spectrogram(maternal_pcg, rate, "Mel Spectrogram - Original PCG")
plot_mel_spectrogram(noisy_pcg, rate, "Mel Spectrogram - Noisy PCG")
plot_mel_spectrogram(recovered_pcg, rate, "Mel Spectrogram - Recovered PCG")

# Step 6: Save the recovered signal as a WAV file
wav.write(r"C:\Users\kezin\OneDrive\Documents\Codes\python\Projects\phonocardiograms_PCG\paperBased\audio\recovered_pcg1.wav", rate, np.int16(recovered_pcg * 32767))
print("Recovered PCG signal saved as recovered_pcg1.wav")
