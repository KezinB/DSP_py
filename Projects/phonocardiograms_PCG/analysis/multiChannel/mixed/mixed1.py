import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import FastICA
import soundfile as sf

# Acknowledgment messages
def acknowledge(message):
    print(f"=== {message} ===")

# Load audio signal
def load_audio(file_path, sr=16000):
    acknowledge(f"Loading audio file: {file_path}")
    signal, _ = librosa.load(file_path, sr=sr, mono=True)
    return signal

# Apply Savitzky-Golay filter for noise reduction
def apply_savgol_filter(signal, window_size=11, poly_order=3):
    acknowledge("Applying Savitzky-Golay filter for noise reduction")
    return savgol_filter(signal, window_length=window_size, polyorder=poly_order)

# Mix two sound signals together
def mix_signals(fetal_signal, maternal_signal):
    acknowledge("Mixing fetal and maternal heart sounds to create a mixed signal")

    # Ensure both signals are the same length
    min_length = min(len(fetal_signal), len(maternal_signal))
    fetal_signal = fetal_signal[:min_length]
    maternal_signal = maternal_signal[:min_length]

    # Mix signals (weighted sum to simulate real-world PCG)
    mixed_signal = 0.6 * fetal_signal + 0.4 * maternal_signal  # Adjust weights as needed

    # Create a stereo-like signal for ICA
    mixed_sources = np.vstack((mixed_signal, maternal_signal)).T  
    return mixed_sources

# Perform FastICA for Blind Source Separation
def apply_fastica(mixed_signal, components=2):
    acknowledge("Performing FastICA for Blind Source Separation")

    # Apply ICA
    ica = FastICA(n_components=components, fun="logcosh", random_state=42)
    separated_sources = ica.fit_transform(mixed_signal)  # Apply ICA

    return separated_sources[:, 0], separated_sources[:, 1]  # Return separated components

# Plot the signals
def plot_signals(original_fetal, original_maternal, mixed, fetal, maternal, sr=16000):
    time = np.linspace(0, len(mixed) / sr, num=len(mixed))

    plt.figure(figsize=(12, 10))

    plt.subplot(5, 1, 1)
    plt.plot(time, original_fetal, label="Original Fetal Signal", color='red')
    plt.title("Original Fetal Heart Sound")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(time, original_maternal, label="Original Maternal Signal", color='green')
    plt.title("Original Maternal Heart Sound")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(time, mixed, label="Mixed Signal", color='black')
    plt.title("Mixed Signal (Fetal + Maternal)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(time, fetal, label="Extracted Fetal Signal (ICA)", color='red')
    plt.title("Extracted Fetal Signal (FastICA)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.plot(time, maternal, label="Extracted Maternal Signal (ICA)", color='green')
    plt.title("Extracted Maternal Signal (FastICA)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Save extracted fetal and maternal heart sounds
def save_audio(signal, file_name, sr=16000):
    acknowledge(f"Saving extracted sound: {file_name}")
    sf.write(file_name, signal, sr)

# Main function
def process_fetal_maternal_signals(fetal_file, maternal_file):
    # Load individual signals
    fetal_signal = load_audio(fetal_file)
    maternal_signal = load_audio(maternal_file)

    # Apply filtering
    fetal_signal = apply_savgol_filter(fetal_signal)
    maternal_signal = apply_savgol_filter(maternal_signal)

    # Mix the signals
    mixed_signal = mix_signals(fetal_signal, maternal_signal)

    # Apply FastICA for Blind Source Separation
    extracted_fetal, extracted_maternal = apply_fastica(mixed_signal)

    # Plot the signals
    plot_signals(fetal_signal, maternal_signal, mixed_signal[:, 0], extracted_fetal, extracted_maternal)

    # Save the extracted sounds
    save_audio(extracted_fetal, "extracted_fetal_heart_sound.wav")
    # save_audio(extracted_maternal, "extracted_maternal_heart_sound.wav")

# Run the process
fetal_audio_path = r"C:\Users\kezin\Downloads\dataset\fetal\f88.wav"  # Replace with actual file path
maternal_audio_path = r"C:\Users\kezin\Downloads\dataset\maternal\m88.wav" # Replace with actual file path
process_fetal_maternal_signals(fetal_audio_path, maternal_audio_path)
